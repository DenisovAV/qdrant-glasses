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
            Log.w(TAG, "fleet pull failed (non-fatal)", e)
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
     * One upstream pass (Spec §5 "up"): ONLY when the app is idle (`!isRecording()` — Spec §3's
     * "ONLY when the app is idle … and the hub is reachable"; reachability is this function's own
     * fail-soft concern below), scrolls the LOCAL store for up to [UP_BATCH_SIZE] points whose
     * `synced` flag is not yet `true` ([MomentStore.scrollUnsyncedFrames] — the durable local store
     * IS the backlog, no separate queue), and — only if that batch is non-empty — upserts it to
     * [collection] and, ONLY AFTER that upsert returns successfully, flips every point's local
     * `synced` flag via [MomentStore.markSynced]. An empty backlog, or a recording session, sends no
     * request at all (mirrors [FleetQdrantClient.upsertPoints]'s own empty-batch no-op).
     *
     * The idle check lives HERE, not just in [syncLoop]'s caller-side gate, because [syncOnce] is a
     * public entry point: any caller reaching it directly (test, future call site, …) must get the
     * same "idle + online only" invariant [syncLoop] enforces, not one that only holds when entered
     * through the loop.
     *
     * The entry check alone isn't enough: [momentStore.scrollUnsyncedFrames] and — far more so —
     * [client.upsertPoints]'s network round trip take real wall-clock time, during which recording
     * can start on another thread (this runs on its own dedicated lane, `isRecording` reads shared
     * state written elsewhere). So [isRecording] is re-checked immediately before the upsert/flag
     * write, right after the batch is known non-empty: a recording session that starts in that gap
     * makes this pass a no-op (`0`, no request sent, no flag flipped) instead of uploading — or
     * flagging as uploaded — while capture is active (Spec §3/§5). The batch simply stays
     * `synced=false` and is retried on the very next idle pass.
     *
     * This re-check NARROWS but does not fully close the window: a recording that starts DURING the
     * `upsertPoints` network call itself still lets this batch's `markSynced` fire once the call
     * returns. That is deliberate and safe — the idle-gate is a resource-contention heuristic (keep
     * network/flush off the cores capture needs), NOT a data-safety invariant. The invariant that
     * actually matters — confirmed-implies-uploaded — holds regardless of recording state: `markSynced`
     * is reached only after a durable, confirmed upsert of exactly these ids.
     *
     * Crash-safe by construction (Spec §5): a crash between the upsert and the flag-flip leaves the
     * batch `synced=false`, so the NEXT [syncOnce] just re-uploads it — safely, because upsert-by-id
     * is idempotent (overwrite, never duplicate). Fail-soft (Spec §7): any `Throwable` from the
     * scroll or the upsert (unreachable hub, HTTP error, unparseable point, …) is caught and logged;
     * [MomentStore.markSynced] is only ever reached on the success path, so a failed pass leaves
     * every point in the batch `synced=false` for retry — never partially/incorrectly flagged.
     *
     * @return the number of points actually synced this pass (`0` while recording — at entry OR
     *   just before the upload, on an empty backlog, OR on a failure).
     */
    suspend fun syncOnce(collection: String = UP_COLLECTION): Int = try {
        if (isRecording()) {
            0
        } else {
            val batch = momentStore.scrollUnsyncedFrames(UP_BATCH_SIZE)
            if (batch.isEmpty()) {
                0
            } else if (isRecording()) {
                // Recording started while we were scrolling the local store — bail before the
                // upload/flag write ever happen (see the re-check note above). Nothing was sent,
                // nothing was flagged; the batch is picked up again next idle pass.
                0
            } else {
                client.upsertPoints(collection, batch)
                val ids = batch.map { it.id }
                momentStore.markSynced(ids)
                Log.i(TAG, "syncOnce: synced ${ids.size} point(s) to $collection")
                ids.size
            }
        }
    } catch (e: CancellationException) {
        throw e   // structured concurrency: propagate, don't swallow as a sync failure (see pull()).
    } catch (e: Throwable) {
        // Log the throwable itself, not just `.message`: an NPE/IllegalState from a REAL bug has a
        // null message, which would print "...: null" and look identical to a benign "hub unreachable",
        // then retry silently every 30s forever. The stack trace keeps a genuine defect diagnosable.
        Log.w(TAG, "syncOnce failed (non-fatal, retried next idle pass)", e)
        0
    }

    /**
     * The background upstream loop (Spec §3/§5 "up"): while this coroutine is active, runs
     * [syncOnce] every [SYNC_INTERVAL_MS] — [syncOnce] itself is what enforces "idle only" (see its
     * doc), so a recording session simply gets a `0`-result no-op pass rather than the loop pausing
     * outright; the very next pass after recording stops finds real work again, no separate resume
     * signal needed.
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
            syncOnce()
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
