package tech.qdrant.glasses.search

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.Base64
import java.util.concurrent.TimeUnit

class GoogleSpeechRecognizer(private val apiKey: String) : SpeechRecognizer {
    companion object {
        private const val TAG = "VoiceSearch"
        private const val SAMPLE_RATE = 16000
        private const val ENDPOINT = "https://speech.googleapis.com/v1/speech:recognize"
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    @Volatile private var running = false
    private var onPartialCallback: ((String) -> Unit)? = null
    private var onResultCallback: ((String) -> Unit)? = null
    private var onErrorCallback: ((String) -> Unit)? = null

    override fun startListening(
        onPartial: (String) -> Unit,
        onResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        onPartialCallback = onPartial
        onResultCallback = onResult
        onErrorCallback = onError
        running = true
        Log.i(TAG, "google stt: startListening")
    }

    // Called by VoiceSearchManager with the complete recorded audio (raw PCM 16-bit LE, 16kHz)
    fun recognize(pcmBytes: ByteArray) {
        if (!running) return
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val audioB64 = Base64.getEncoder().encodeToString(pcmBytes)
                val body = JSONObject().apply {
                    put("config", JSONObject().apply {
                        put("encoding", "LINEAR16")
                        put("sampleRateHertz", SAMPLE_RATE)
                        put("languageCode", "en-US")
                        put("model", "command_and_search")
                        put("enableAutomaticPunctuation", false)
                    })
                    put("audio", JSONObject().apply {
                        put("content", audioB64)
                    })
                }.toString()

                val request = Request.Builder()
                    .url("$ENDPOINT?key=$apiKey")
                    .post(body.toRequestBody("application/json".toMediaType()))
                    .build()

                val httpStart = System.currentTimeMillis()
                val response = client.newCall(request).execute()
                val responseBody = response.body?.string() ?: ""
                val httpMs = System.currentTimeMillis() - httpStart
                Log.d(TAG, "google stt response: ${response.code} (${httpMs}ms)")

                if (!response.isSuccessful) {
                    onErrorCallback?.invoke("Google STT error: ${response.code}")
                    return@launch
                }

                val json = JSONObject(responseBody)
                val results = json.optJSONArray("results")
                val transcript = results
                    ?.getJSONObject(0)
                    ?.getJSONArray("alternatives")
                    ?.getJSONObject(0)
                    ?.optString("transcript", "")
                    ?.trim()
                    ?: ""

                Log.i(TAG, "google stt result: \"$transcript\"")
                if (transcript.isNotEmpty()) onResultCallback?.invoke(transcript)
                else onErrorCallback?.invoke("Empty result")
            } catch (e: Exception) {
                Log.e(TAG, "google stt exception", e)
                onErrorCallback?.invoke("Google STT error: ${e.message}")
            }
        }
    }

    override fun stopListening() { running = false }
    override fun destroy() { running = false }
}
