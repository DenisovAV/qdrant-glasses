package tech.qdrant.glasses.fleet

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * One point to upsert to a private-Qdrant fleet collection over REST (upstream flag design, Spec
 * §5/§6): the wire shape [FleetQdrantClient.upsertPoints] needs, distinct from the native
 * `io.qdrant.edge.Point` the local shard writes use ([tech.qdrant.glasses.storage.QdrantEdgeMomentStore]).
 * [id] is the point's PointId as a string — every local write is a `PointId.Uuid`, so this is always
 * a UUID string in practice, sent as a bare JSON string the server parses as its UUID id form.
 * [vector] is the raw "clip" named-vector floats. [payload] is a JSON OBJECT string — the exact
 * output of [tech.qdrant.glasses.storage.MomentPayload.toJson] plus the `synced` flag stripped/added
 * by the caller — kept as a String (not `org.json.JSONObject`) so this class carries no JSON-library
 * dependency of its own; [upsertPoints] parses it back into the request body.
 */
data class FleetPoint(
    val id: String,
    val vector: FloatArray,
    val payload: String,
)

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
    companion object {
        // Matches QdrantEdgeMomentStore/FleetShardStore's named-vector key — every collection this
        // client talks to (fleet_inbox, fleet_curated) is provisioned with the SAME schema (Spec §6).
        private const val CLIP_FIELD = "clip"
    }

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
     * PUT-upserts [points] to [collection] in ONE batch (the "up" half of the upstream flag design,
     * Spec §5): `?wait=true` so the call only returns after the server has durably applied the
     * write — [tech.qdrant.glasses.fleet.FleetSync]'s local `synced` flag-flip (Spec §5/§6) must
     * happen ONLY after a confirmed upsert, and an async ack here would let it flip the flag on a
     * write the server hadn't actually committed yet. A no-op for an empty [points] (an idle pass
     * that finds nothing to sync must not fire a request). Synchronous, same fail-soft contract as
     * every other method here — throws on a non-2xx/network error; callers wrap and fall back to
     * "stays unsynced, retried next idle pass" (never a crash).
     *
     * Every point's [FleetPoint.payload] is parsed BEFORE any request is built or sent — an
     * unparseable payload throws immediately and skips the network call entirely, so a batch can
     * never upsert a substitute empty `{}` in place of a point's real payload. That would otherwise
     * let a "successful" upsert of empty JSON get treated as confirmed by the caller and flip the
     * point's local `synced` flag despite its real data never reaching the hub (Spec §5/§6's
     * crash-safe invariant is confirmed-implies-uploaded, not confirmed-implies-something-uploaded).
     * Failing the whole call leaves every point in the batch `synced=false`, retried next idle pass.
     */
    fun upsertPoints(collection: String, points: List<FleetPoint>) {
        if (points.isEmpty()) return
        val pointsJson = JSONArray()
        points.forEach { p ->
            val payloadObj = try {
                JSONObject(p.payload)
            } catch (e: JSONException) {
                throw IllegalArgumentException(
                    "upsertPoints: unparseable payload for point id=${p.id}, aborting batch " +
                        "(leaving it — and the rest of this batch — unsynced for retry)",
                    e,
                )
            }
            val vectorJson = JSONArray()
            p.vector.forEach { vectorJson.put(it.toDouble()) }
            pointsJson.put(
                JSONObject()
                    .put("id", p.id)
                    .put("vector", JSONObject().put(CLIP_FIELD, vectorJson))
                    .put("payload", payloadObj)
            )
        }
        val body = JSONObject().put("points", pointsJson).toString()
        val req = Request.Builder()
            .url("$baseUrl/collections/$collection/points?wait=true")
            .put(body.toRequestBody("application/json".toMediaType())).build()
        http.newCall(req).execute().use { resp ->
            val respBody = resp.body?.string().orEmpty()
            require(resp.isSuccessful) { "upsert points ${resp.code}: $respBody" }
            // Confirmed-implies-uploaded (Spec §5/§6): a bare HTTP 2xx is NOT proof the write reached
            // Qdrant — a reverse proxy / captive portal / misrouted host in front of the hub can
            // synthesize a 200 it never forwarded. Qdrant's own upsert(?wait=true) response carries a
            // top-level {"status":"ok", ...}; require that ack too, so a 200-from-something-else can't
            // false-confirm and let the caller flip `synced` on frames that never actually landed.
            val ackOk = try {
                JSONObject(respBody).optString("status") == "ok"
            } catch (_: JSONException) {
                false
            }
            require(ackOk) { "upsert points: HTTP ${resp.code} but Qdrant ack not ok: $respBody" }
        }
    }
}
