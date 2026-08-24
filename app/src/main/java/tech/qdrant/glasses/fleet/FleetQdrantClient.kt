package tech.qdrant.glasses.fleet

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Thin REST client to the private Qdrant fleet hub (Sovereign Fleet Memory PoC). Reuses OkHttp (same
 * stack as MacEndpointEncoder). SYNCHRONOUS — call OFF the main thread. Fleet is optional: callers
 * wrap these in try/catch and fall back to local-only (see FleetSync).
 */
class FleetQdrantClient(
    private val baseUrl: String,
    private val http: OkHttpClient = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS).build(),
) {
    /** POST create a shard snapshot; returns the snapshot file name. */
    fun createShardSnapshot(collection: String, shard: Int = 0): String {
        val req = Request.Builder()
            .url("$baseUrl/collections/$collection/shards/$shard/snapshots")
            .post("".toRequestBody(null)).build()
        http.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            require(resp.isSuccessful) { "snapshot create ${resp.code}: $body" }
            return JSONObject(body).getJSONObject("result").getString("name")
        }
    }

    /** GET the snapshot file to [dest]. */
    fun downloadSnapshot(collection: String, shard: Int, name: String, dest: File) {
        val req = Request.Builder()
            .url("$baseUrl/collections/$collection/shards/$shard/snapshots/$name").get().build()
        http.newCall(req).execute().use { resp ->
            require(resp.isSuccessful) { "snapshot download ${resp.code}" }
            val body = resp.body ?: throw IOException(
                "snapshot download ${resp.code}: empty response body for $collection/shards/$shard/snapshots/$name"
            )
            dest.outputStream().use { out -> body.byteStream().copyTo(out) }
        }
    }

    /**
     * DELETE the server-side shard snapshot [name] — cleanup after [downloadSnapshot] pulls it
     * locally, so snapshots don't accumulate on the fleet hub across every [FleetSync.pull]. Purely
     * a cleanup op: callers soft-fail this, so the result is ignored here too (no return value).
     */
    fun deleteSnapshot(collection: String, shard: Int, name: String) {
        val req = Request.Builder()
            .url("$baseUrl/collections/$collection/shards/$shard/snapshots/$name").delete().build()
        http.newCall(req).execute().use { resp ->
            require(resp.isSuccessful) { "snapshot delete ${resp.code}" }
        }
    }

    /**
     * PUT-upsert a batch of already-queued points into [collection] (the UP half of fleet sync —
     * Spec §4/§5, plan Task 9; [FleetSync.pushDrain] is the caller, off the fleet lane after
     * [UploadQueue.drain]). `wait=true` so the call only returns once Qdrant has applied the whole
     * batch — [FleetSync.pushDrain] only [UploadQueue.ack]s after this returns successfully (Task 11),
     * so a partial/failed upsert must never look like a success here.
     *
     * A [QueuedPoint.payloadJson] is a serialized JSON OBJECT of point payload fields (see
     * [UploadQueue]'s doc) — it is parsed and used directly as the point's `payload`, not nested
     * under another key. [points] empty is a no-op (no request sent): nothing to upsert, and an
     * empty `points` array would just be a wasted round trip.
     */
    fun upsertPoints(collection: String, points: List<QueuedPoint>) {
        if (points.isEmpty()) return
        val body = JSONObject().put("points", JSONArray(points.map { p ->
            JSONObject()
                .put("id", p.id)
                .put("vector", JSONObject().put("clip", JSONArray(p.clip.map { it.toDouble() })))
                .put("payload", JSONObject(p.payloadJson))
        }))
        val req = Request.Builder()
            .url("$baseUrl/collections/$collection/points?wait=true")
            .put(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(req).execute().use { resp ->
            val respBody = resp.body?.string().orEmpty()
            require(resp.isSuccessful) { "upsert points ${resp.code}: $respBody" }
        }
    }
}
