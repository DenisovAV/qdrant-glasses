package tech.qdrant.glasses.fleet

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * One point queued for upload to the private Qdrant fleet hub (Spec §4 UP flow / §5 dual-write,
 * plan Task 8): the clip vector plus its already-serialized payload JSON. [payloadJson] is opaque
 * to [UploadQueue] — the caller ([tech.qdrant.glasses.pipeline.MomentCapture], Task 10) stamps in
 * `sync_ts`/`thumb_b64` before enqueuing; this class only persists and replays what it's given.
 */
data class QueuedPoint(val id: String, val clip: FloatArray, val payloadJson: String)

/**
 * Persistent, append-only upload queue for the UP half of fleet sync (plan Task 8, Spec §4:
 * "new local moment --> upload queue --> (online) batch upsert --> private Qdrant [fleet_inbox]").
 * Backed by one JSONL file (`filesDir/fleet_queue.jsonl`, per Spec §4/plan) — one line per queued
 * point, `{"id":..,"clip":[..],"payload":{..}}` — so the queue survives a process restart: a moment
 * captured offline stays queued until the next successful [FleetSync.pushDrain], across reboots.
 *
 * [drain] is a peek, not a pop: it returns up to [max] points without removing them, so a crash
 * between drain and a successful server upsert never loses a point — the caller only removes a
 * batch via [ack] AFTER the upsert actually lands (see plan Task 11). This mirrors Decision C for
 * the OTHER direction: exactly as a local moment is never deleted after a successful upload, a
 * queued upload is never speculatively removed before one.
 *
 * Every call is synchronized on one monitor — [tech.qdrant.glasses.pipeline.MomentCapture] enqueues
 * from the capture lane while [FleetSync.pushDrain] drains/acks from the fleet lane (Spec §7: fleet
 * ops run off the inference lane), same discipline as [FleetShardStore]/`QdrantEdgeMomentStore`.
 */
class UploadQueue(private val file: File) {
    private val lock = Any()

    /** Appends one point to the queue (durable — a plain file append, no in-memory-only state). */
    fun enqueue(id: String, clipVec: FloatArray, payloadJson: String) = synchronized(lock) {
        val line = JSONObject()
            .put("id", id)
            .put("clip", JSONArray(clipVec.map { it.toDouble() }))
            .put("payload", JSONObject(payloadJson))
            .toString()
        file.parentFile?.mkdirs()
        file.appendText(line + "\n")
    }

    /**
     * Returns up to [max] queued points, oldest-first (FIFO — append order on disk). Does NOT
     * remove them from the queue; call [ack] once they've actually been upserted server-side.
     * A missing file (nothing ever queued) is an empty queue, not an error.
     */
    fun drain(max: Int): List<QueuedPoint> = synchronized(lock) {
        if (!file.exists()) return@synchronized emptyList()
        file.readLines().asSequence().filter { it.isNotBlank() }.take(max).map(::parseLine).toList()
    }

    /**
     * Durably removes the given [ids] from the queue (rewrites the file without them) — call after
     * a batch [drain] has been confirmed upserted server-side. A missing file or an empty [ids] is a
     * no-op, not an error (mirrors [drain]'s soft-fail-safe shape — an ack after a queue already
     * emptied by a prior run must never throw).
     */
    fun ack(ids: Collection<String>) = synchronized(lock) {
        if (!file.exists() || ids.isEmpty()) return@synchronized
        val acked = ids.toSet()
        val remaining = file.readLines().filter { it.isNotBlank() && parseLine(it).id !in acked }
        if (remaining.isEmpty()) file.delete() else file.writeText(remaining.joinToString("\n") + "\n")
    }

    private fun parseLine(line: String): QueuedPoint {
        val o = JSONObject(line)
        val arr = o.getJSONArray("clip")
        val clip = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
        return QueuedPoint(id = o.getString("id"), clip = clip, payloadJson = o.getJSONObject("payload").toString())
    }
}
