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
    /** Pulls [collection] down as a snapshot and opens it read-only. Never throws. */
    suspend fun pull(collection: String = "fleet_curated"): FleetShardStore? = try {
        val name = client.createShardSnapshot(collection)
        val snap = File(filesDir, "fleet_snap.bin").apply { delete() }
        client.downloadSnapshot(collection, 0, name, snap)
        val dir = File(filesDir, "fleet_shard").apply { deleteRecursively() }
        unpackSnapshotAsync(snap.absolutePath, dir.absolutePath)
        FleetShardStore.load(dir.absolutePath, clipDim).also { Log.i(TAG, "fleet pulled: $collection") }
    } catch (e: Throwable) {
        Log.w(TAG, "fleet pull failed (non-fatal): ${e.message}")
        null
    }

    companion object {
        private const val TAG = "FleetSync"
    }
}
