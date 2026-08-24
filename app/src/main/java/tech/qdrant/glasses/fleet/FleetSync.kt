package tech.qdrant.glasses.fleet

import android.util.Base64
import android.util.Log
import io.qdrant.edge.unpackSnapshotAsync
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
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
    // UP half (plan Task 11, Spec §4/§5 dual-write): the persistent queue [MomentCapture] appends a
    // copy of each stored moment to. Null when no fleet tier is configured — [pushDrain] is then a
    // no-op. Only [pushDrain] touches it here; [MomentCapture] owns the enqueue side.
    private val uploadQueue: UploadQueue? = null,
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
     * UP half (plan Task 11, Spec §4/§5 dual-write): drain the [uploadQueue] to the private Qdrant
     * in batches — [FleetQdrantClient.upsertPoints] the batch, then [UploadQueue.ack] it ONLY after
     * the upsert is confirmed (drain is a non-destructive peek), so a failed or interrupted upsert
     * leaves those points queued for the next drain — no loss (Decision C: the LOCAL moment is never
     * touched by any of this either way). No-op when no fleet tier is configured or the queue is
     * empty. Fail-soft per Spec §7: any error stops this pass and is retried on the next call; this
     * runs on a fleet lane, never the capture path, so capture is never affected.
     *
     * **No-progress guard (round-1 fix):** [UploadQueue.ack] can itself fail fail-soft (returns
     * `false` rather than throwing — a full disk, a stuck rename, …). The old code ignored that
     * return value, so a PERSISTENTLY failing [ack] against a reachable server produced an unbounded
     * hot loop: [UploadQueue.drain] keeps returning the SAME un-acked batch, [FleetQdrantClient.upsertPoints]
     * keeps succeeding, [UploadQueue.ack] keeps failing, repeat — a tight spin of full-batch HTTP PUTs with no
     * suspension point in between. This now checks [UploadQueue.ack]'s return and BREAKS the loop the
     * first time it's `false`, leaving that batch queued for the next [pushDrain] call rather than
     * retrying it inside this one.
     *
     * **Cancellation (round-1 fix, partial):** the loop body ([UploadQueue.drain], the blocking
     * OkHttp [FleetQdrantClient.upsertPoints] call, [UploadQueue.ack]) has no suspending calls of its
     * own, so a coroutine cancellation can't interrupt an in-flight HTTP request — that limitation is
     * NOT fixed here (would need `runInterruptible`/a cancellable OkHttp call wrapper, out of scope
     * for this pass). What IS fixed: `coroutineContext.ensureActive()` at the top of every iteration
     * means a cancellation IS observed BETWEEN batches, not only after the whole backlog drains or an
     * unrelated network error happens to throw — previously true only in the sense that a thrown
     * [kotlinx.coroutines.CancellationException] gets rethrown, which nothing in the loop body could
     * ever produce on its own.
     *
     * **Single-flight (round-2 review fix, Finding 2):** the whole body runs under [pushMutex], so
     * at most one drain/upsert/ack cycle is ever in flight across every caller of this method —
     * required now that this is no longer called exactly once per process (see Finding 1's fix:
     * [tech.qdrant.glasses.pipeline.MomentCapture] fires this after every enqueue too). See
     * [pushMutex]'s own doc for why concurrent [ack]s specifically are unsafe without it.
     */
    suspend fun pushDrain(collection: String = "fleet_inbox") {
        val queue = uploadQueue ?: return
        pushMutex.withLock {
            try {
                while (true) {
                    coroutineContext.ensureActive()
                    val batch = queue.drain(BATCH)
                    if (batch.isEmpty()) break
                    // Round-4 review fix: read+base64-encode each point's thumbnail HERE, right
                    // before the PUT — not back at [tech.qdrant.glasses.pipeline.MomentCapture]'s
                    // enqueue call, and not stored in the queue file at all. See [withThumbB64]'s
                    // own doc for why. This is I/O (a JPEG read per point in the batch) but it runs
                    // on whatever dispatcher called [pushDrain] (a dedicated `Dispatchers.IO` launch
                    // or the fleet lane, per [MomentCapture]'s `onFleetEnqueued` wiring / the startup
                    // flush) — never [MomentCapture.embedLane] — so it can never block capture.
                    val withThumbs = batch.map(::withThumbB64)
                    client.upsertPoints(collection, withThumbs)   // throws on HTTP failure → caught below; batch stays queued
                    if (!queue.ack(batch.map { it.id })) {
                        // ack failed (already logged by UploadQueue) — stop this pass instead of
                        // re-upserting the same still-queued batch in a tight loop; retried whole on the
                        // next pushDrain call.
                        Log.w(TAG, "fleet pushDrain: ack failed, stopping this pass (batch stays queued, will retry)")
                        break
                    }
                }
            } catch (e: Throwable) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                Log.w(TAG, "fleet pushDrain failed (non-fatal, will retry): ${e.message}")
            }
        }
    }

    /**
     * Returns a copy of [point] whose payload additionally carries `thumb_b64` — the on-device
     * thumbnail JPEG (found via the payload's own `thumb_path`, stamped at capture time by
     * [tech.qdrant.glasses.pipeline.MomentCapture]), base64-encoded — for the upcoming upsert PUT.
     * The thumbnail bytes have to travel because `thumb_path` is a device-local path, meaningless
     * off-device (Spec §6); reading them HERE, right before the HTTP call, rather than back at
     * enqueue time, is what keeps [tech.qdrant.glasses.pipeline.MomentCapture.embedLane] from ever
     * touching the JPEG a second time.
     *
     * Round-4 review fix (blocker + major): until this round, [tech.qdrant.glasses.pipeline.
     * MomentCapture.confirmAndStore] did this read+encode INLINE, synchronously, as part of the
     * enqueue call it makes on `embedLane` — so `busy` stayed set (dropping capture candidates) for
     * however long a tens-to-hundreds-of-KB disk read + base64 encode took, violating Spec §7's
     * "fleet work must never block capture". It also meant every [UploadQueue] JSONL line embedded
     * that same blob, so [UploadQueue.ack]'s `writeAtomic` — which runs INSIDE the queue's `lock`,
     * the same lock `enqueue()` needs — could hold that lock for however long it took to
     * re-serialize a whole thumb-bearing backlog to disk. Moving the read+encode HERE fixes both:
     * [UploadQueue.enqueue]'s line is back to just the clip vector + small payload (the class doc's
     * original "multi-KB clip vector" framing, from before Task 10 added thumbnails), and this runs
     * off [tech.qdrant.glasses.pipeline.MomentCapture.embedLane] entirely — [pushDrain] already does
     * blocking network I/O under [pushMutex] on a fleet lane / `Dispatchers.IO`, so one more
     * blocking disk read per point changes nothing about what it's allowed to block.
     *
     * Fail-soft per-point (Spec §7): a missing or unreadable thumbnail (deleted, corrupted, a stale
     * queue entry whose `thumb_path` no longer resolves) degrades that ONE point to `thumb_b64=""`
     * — mirrors [tech.qdrant.glasses.pipeline.buildUploadPayloadJson]'s existing "" convention for a
     * thumbnail write that failed at capture time — rather than failing the whole batch's upsert.
     */
    private fun withThumbB64(point: QueuedPoint): QueuedPoint {
        val payload = JSONObject(point.payloadJson)
        val thumbPath = payload.optString("thumb_path")
        val thumbB64 = if (thumbPath.isNotBlank()) {
            try {
                Base64.encodeToString(File(thumbPath).readBytes(), Base64.NO_WRAP)
            } catch (e: Throwable) {
                Log.w(TAG, "fleet pushDrain: thumb read failed for id=${point.id}, uploading without it (non-fatal): ${e.message}")
                ""
            }
        } else ""
        return point.copy(payloadJson = payload.put("thumb_b64", thumbB64).toString())
    }

    // Single-flight guard (round-2 review fix, Finding 2): UploadQueue.ack's "fold in appends that
    // landed mid-ack" logic is correct against ONE concurrent enqueue() but NOT against TWO
    // concurrent ack() calls — two independently-snapshotted, independently-filtered ack()s can
    // race their own writeAtomic() swaps and either resurrect already-acked ids (harmless, upsert-
    // by-id is idempotent) or, worse, undercount and silently drop entries enqueued between the two
    // snapshots (see UploadQueue's class doc for the full mechanism). That race was unreachable
    // while pushDrain had exactly one caller per process (the startup flush in
    // GlassesComponents.load); it becomes reachable now that MomentCapture ALSO fires a pushDrain
    // after every enqueue (this round's other fix, Finding 1 — the UP half must be an ongoing loop,
    // Spec §4, not a restart-gated one). Wrapping the WHOLE pushDrain body in this Mutex is the
    // simplest correct fix: it guarantees at most one drain()/upsertPoints()/ack() triple ever runs
    // at a time, which is what the UploadQueue class doc already claims as its invariant. A call
    // that lands while another is in flight loses nothing — it just suspends (no busy spin) until
    // the lock frees, then drains whatever the file holds at THAT point (already reflecting the
    // prior run's acks and any fresh enqueue()s), so it's either a normal drain of newly-queued work
    // or an immediate no-op (drain() returns empty right away).
    private val pushMutex = Mutex()

    companion object {
        private const val TAG = "FleetSync"
        /** Upsert batch size for [pushDrain] — matches the plan's Task 11 (BATCH_SIZE). */
        private const val BATCH = 20
    }
}
