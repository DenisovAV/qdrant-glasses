package tech.qdrant.glasses.fleet

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
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
}
