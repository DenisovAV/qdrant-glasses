package tech.qdrant.glasses.fleet

import android.util.Log
import io.qdrant.edge.unpackSnapshotAsync
import java.io.File

/**
 * Orchestrates the DOWN half of the fleet-sync PoC (plan Task 6, Spec §3/§9 P1): pull a curated
 * fleet corpus from the private Qdrant into a second, read-only [FleetShardStore] that
 * [tech.qdrant.glasses.search.MomentSearcher] can merge beside local memory. The native snapshot
 * path (Spec §2 — proven compatible), not an API-scroll fallback: [client] creates+downloads a shard
 * snapshot, [unpackSnapshotAsync] (top-level suspend, `io.qdrant.edge`) unpacks it on disk, and
 * [FleetShardStore.load] opens it.
 *
 * Fail-soft per Spec §7: [pull] wraps the WHOLE sequence in one try/catch and returns null on any
 * `Throwable` (unreachable server, HTTP error, malformed/incompatible snapshot, native load
 * failure, …) — never propagates. `GlassesComponents` (Task 6 step 2) only constructs this when
 * `Config.FLEET_URL` is set; a null result there just means "no fleet tier this session", the exact
 * same shape as every other nullable-optional-feature in that file.
 */
class FleetSync(
    private val client: FleetQdrantClient,
    private val filesDir: File,
    private val clipDim: Int,
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
        return try {
            val name = client.createShardSnapshot(collection)
            snap.delete()
            client.downloadSnapshot(collection, 0, name, snap)
            // Cleanup, best-effort: every pull creates a NEW server-side snapshot (createShardSnapshot
            // above) that otherwise never gets deleted and accumulates on the fleet hub. A failed
            // delete must never fail the pull itself — the local copy already downloaded successfully.
            runCatching { client.deleteSnapshot(collection, 0, name) }
                .onFailure { Log.w(TAG, "fleet snapshot delete failed (non-fatal): ${it.message}") }
            dir.deleteRecursively(); dir.mkdirs()   // EdgeShard/unpack needs the target dir to EXIST (empty) — it does NOT create parents (os error 2 otherwise)
            unpackSnapshotAsync(snap.absolutePath, dir.absolutePath)
            FleetShardStore.load(dir.absolutePath, clipDim).also { Log.i(TAG, "fleet pulled: $collection") }
        } catch (e: Throwable) {
            // Structured concurrency: a CancellationException here means the load coroutine itself
            // was cancelled (e.g. the app closing) — it must propagate to cancel the coroutine, not
            // be swallowed as a fleet-pull failure. The fail-soft contract below is for real errors
            // only. `finally { snap.delete() }` still runs on this path.
            if (e is kotlinx.coroutines.CancellationException) throw e
            Log.w(TAG, "fleet pull failed (non-fatal): ${e.message}")
            dir.deleteRecursively()
            null
        } finally {
            snap.delete()
        }
    }

    companion object {
        private const val TAG = "FleetSync"
    }
}
