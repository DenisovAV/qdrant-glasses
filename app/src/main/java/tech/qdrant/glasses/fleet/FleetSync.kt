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

    companion object {
        private const val TAG = "FleetSync"
    }
}
