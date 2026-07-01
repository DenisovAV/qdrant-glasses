package tech.qdrant.glasses.embedding

import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Crop/text embedding via the Mac SigLIP2 endpoint (reach over USB with
 * `adb reverse tcp:9000 tcp:9000`). Throws IOException on ANY failure (non-2xx, timeout,
 * empty body, malformed JSON) so the caller can leave the track un-embedded and retry on a
 * later sighting.
 *
 * Must be called OFF the main thread (synchronous OkHttp call → NetworkOnMainThreadException
 * otherwise). The caller (ViewModel crop-embed lane) already runs off-main.
 */
class MacEndpointEncoder(
    override val dim: Int = 768,
    private val baseUrl: String = "http://localhost:9000",
) : CropEncoder {
    // Calibrated for SigLIP2 (Mac endpoint) — its text↔image cosine scale is ~2× LOWER than
    // TinyCLIP's. Measured on a room frame: relevant text (laptop) ~0.10, "a room" ~0.08;
    // irrelevant (kitchen when absent) ~0.015, elephant ~0.02. Object CROPS (denser than a full
    // frame) score higher than that — a real crop match ("my phone"→cell phone) logged ~0.14.
    // 0.12 sits above the irrelevant floor and below real crop matches, so an absent query
    // ("kitchen") returns nothing instead of surfacing the nearest junk. TinyCLIP's 0.20 gate was
    // for a different modality-gap scale and rejected everything on SigLIP2.
    override val visionMinScore: Float = 0.12f
    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .writeTimeout(5, TimeUnit.SECONDS)
        .build()

    override fun encode(crop: Bitmap): FloatArray {
        val baos = ByteArrayOutputStream()
        crop.compress(Bitmap.CompressFormat.JPEG, 85, baos)
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart(
                "crop", "crop.jpg",
                baos.toByteArray().toRequestBody("image/jpeg".toMediaType())
            ).build()
        val req = Request.Builder().url("$baseUrl/embed_image").post(body).build()
        return runAndParse(req)
    }

    override fun encodeText(query: String): FloatArray {
        val json = JSONObject().put("query", query).toString()
        val req = Request.Builder().url("$baseUrl/embed_text")
            .post(json.toRequestBody("application/json".toMediaType())).build()
        return runAndParse(req)
    }

    private fun runAndParse(req: Request): FloatArray {
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw IOException("embed HTTP ${resp.code}")
            val payload = (resp.body ?: throw IOException("embed: empty response body")).string()
            try {
                val arr: JSONArray = JSONObject(payload).getJSONArray("vector")
                return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
            } catch (e: JSONException) {
                throw IOException("embed: malformed response", e)
            }
        }
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
}
