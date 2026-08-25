package tech.qdrant.glasses.fleet

import android.util.Log
import io.qdrant.edge.unpackSnapshotAsync
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.File
import kotlin.coroutines.coroutineContext
import tech.qdrant.glasses.storage.MomentStore

/**
 * Orchestrates BOTH halves of the fleet-sync PoC (Spec §3/§5/§9): [pull] is the DOWN half — pulls a
 * curated fleet corpus from the private Qdrant into a second, read-only [FleetShardStore] that
 * [tech.qdrant.glasses.search.MomentSearcher] can merge beside local memory. The native snapshot
 * path (Spec §2 — proven compatible), not an API-scroll fallback: [client] creates+downloads a shard
 * snapshot, [unpackSnapshotAsync] (top-level suspend, `io.qdrant.edge`) unpacks it on disk, and
 * [FleetShardStore.load] opens it.
 *
 * [syncOnce]/[syncLoop] are the UP half — the flag-on-store design (Spec §5), NOT a queue: the
 * durable local [momentStore] IS the upload backlog. There is no separate queue file, no
 * enqueue-on-capture, and [tech.qdrant.glasses.pipeline.MomentCapture] is completely decoupled from
 * this class.
 *
 * Fail-soft per Spec §7: every entry point here wraps its whole sequence in one try/catch and
 * returns a safe default (`null`/`0`) on any `Throwable` (unreachable server, HTTP error,
 * malformed/incompatible snapshot, native load failure, …) — never propagates. `GlassesComponents`
 * only constructs this when `Config.FLEET_URL` is set; a null/0 result there just means "no fleet
 * tier this pass", the exact same shape as every other nullable-optional-feature in that file.
 */
class FleetSync(
    private val client: FleetQdrantClient,
    private val filesDir: File,
    private val clipDim: Int,
    // UP half deps (Spec §5) — the local store doubling as the upload backlog, and the same
    // idle-gate signal MomentCapture already reads (GlassesComponents' `isRecording` lambda).
    private val momentStore: MomentStore,
    private val isRecording: () -> Boolean,
) {
    /**
     * Pulls [collection] down as a snapshot and opens it read-only. Never throws.
     *
     * `fleet_snap.bin` is only ever an intermediate — [unpackSnapshotAsync] consumes it into `dir`,
     * so it's deleted in a `finally` regardless of outcome. `fleet_shard` (`dir`) is different: on
     * success it becomes the returned [FleetShardStore]'s on-disk backing and must survive; on
     * failure nothing owns it, so the catch path deletes it too, leaving no orphan on disk.
     */
    suspend fun pull(collection: String = "fleet_curated"): FleetShardStore? {
        val snap = File(filesDir, "fleet_snap.bin")
        val dir = File(filesDir, "fleet_shard")
        var name: String? = null   // hoisted so `finally` can delete the server snapshot on EVERY exit path
        return try {
            name = client.createShardSnapshot(collection)
            snap.delete()
            client.downloadSnapshot(collection, 0, name, snap)
            dir.deleteRecursively(); dir.mkdirs()   // EdgeShard/unpack needs the target dir to EXIST (empty) — it does NOT create parents (os error 2 otherwise)
            unpackSnapshotAsync(snap.absolutePath, dir.absolutePath)
            FleetShardStore.load(dir.absolutePath, clipDim).also { Log.i(TAG, "fleet pulled: $collection") }
        } catch (e: Throwable) {
            // Clean the (possibly half-unpacked) shard dir on EVERY non-success exit — do it BEFORE the
            // cancellation rethrow so a cancelled pull leaves no orphan either. On success this line is
            // never reached, so the loaded store's on-disk backing survives.
            dir.deleteRecursively()
            // Structured concurrency: a CancellationException means the load coroutine itself was
            // cancelled (app closing) — propagate it, don't swallow it as a fleet-pull failure. The
            // fail-soft contract below is for real errors only; `finally` still runs on this path.
            if (e is kotlinx.coroutines.CancellationException) throw e
            Log.w(TAG, "fleet pull failed (non-fatal): ${e.message}")
            null
        } finally {
            snap.delete()
            // The server-side snapshot is only ever an intermediate — delete it on ALL paths (success,
            // download failure, cancellation), or created-but-undownloaded snapshots accumulate on the
            // fleet hub. Best-effort: a failed delete never affects the pull's outcome.
            name?.let { n ->
                runCatching { client.deleteSnapshot(collection, 0, n) }
                    .onFailure { Log.w(TAG, "fleet snapshot delete failed (non-fatal): ${it.message}") }
            }
        }
    }

    /**
     * One upstream pass (Spec §5 "up"): scrolls the LOCAL store for up to [UP_BATCH_SIZE] points
     * whose `synced` flag is not yet `true` ([MomentStore.scrollUnsyncedFrames] — the durable local
     * store IS the backlog, no separate queue), and — only if that batch is non-empty — upserts it
     * to [collection] and, ONLY AFTER that upsert returns successfully, flips every point's local
     * `synced` flag via [MomentStore.markSynced]. An empty backlog sends no request at all (mirrors
     * [FleetQdrantClient.upsertPoints]'s own empty-batch no-op).
     *
     * Crash-safe by construction (Spec §5): a crash between the upsert and the flag-flip leaves the
     * batch `synced=false`, so the NEXT [syncOnce] just re-uploads it — safely, because upsert-by-id
     * is idempotent (overwrite, never duplicate). Fail-soft (Spec §7): any `Throwable` from the
     * scroll or the upsert (unreachable hub, HTTP error, unparseable point, …) is caught and logged;
     * [MomentStore.markSynced] is only ever reached on the success path, so a failed pass leaves
     * every point in the batch `synced=false` for retry — never partially/incorrectly flagged.
     *
     * @return the number of points actually synced this pass (`0` on an empty backlog OR a failure).
     */
    suspend fun syncOnce(collection: String = UP_COLLECTION): Int = try {
        val batch = momentStore.scrollUnsyncedFrames(UP_BATCH_SIZE)
        if (batch.isEmpty()) {
            0
        } else {
            client.upsertPoints(collection, batch)
            val ids = batch.map { it.id }
            momentStore.markSynced(ids)
            Log.i(TAG, "syncOnce: synced ${ids.size} point(s) to $collection")
            ids.size
        }
    } catch (e: CancellationException) {
        throw e   // structured concurrency: propagate, don't swallow as a sync failure (see pull()).
    } catch (e: Throwable) {
        Log.w(TAG, "syncOnce failed (non-fatal, retried next idle pass): ${e.message}")
        0
    }

    /**
     * The background upstream loop (Spec §3/§5 "up"): while this coroutine is active, runs
     * [syncOnce] ONLY when the app is idle (`!isRecording()` — Spec §3's "ONLY when the app is idle
     * … and the hub is reachable"; reachability is [syncOnce]'s own fail-soft concern, not this
     * loop's), then sleeps [SYNC_INTERVAL_MS] before the next pass. A recording session simply
     * skips a pass rather than pausing the loop outright — the very next check after the delay
     * picks sync back up the moment recording stops, no separate resume signal needed.
     *
     * Never returns normally — exits only via cancellation of the coroutine that calls it (the
     * caller is expected to `scope.launch(fleetLane) { fleetSync.syncLoop() }`, same shape as
     * [AppStateHolder]'s recording ticker). [coroutineContext]'s [ensureActive] at the top of every
     * iteration turns that cancellation into a clean exit (a thrown [CancellationException]) instead
     * of looping past it or running one extra pass after cancel.
     */
    suspend fun syncLoop() {
        while (true) {
            coroutineContext.ensureActive()
            if (!isRecording()) syncOnce()
            delay(SYNC_INTERVAL_MS)
        }
    }

    companion object {
        private const val TAG = "FleetSync"
        // Spec §6: the "up" collection — raw device contributions land here (curation, Spec §8, is
        // a later, separate step that copies approved points into fleet_curated).
        private const val UP_COLLECTION = "fleet_inbox"
        // Matches MomentStore.scrollUnsyncedFrames's own default — one idle pass moves at most this
        // many points; a bigger backlog just takes multiple passes (SYNC_INTERVAL_MS apart).
        private const val UP_BATCH_SIZE = 100
        /** How long [syncLoop] sleeps between passes. Not Config-gated (Spec §7's fail-soft dial is
         *  per-op, not per-timer) — a plain constant, like [AppStateHolder]'s 1000ms ticker. */
        const val SYNC_INTERVAL_MS = 30_000L
    }
}
