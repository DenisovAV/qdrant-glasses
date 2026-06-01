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
        if (pcmBytes.size < 2) { onErrorCallback?.invoke("No speech detected"); return }
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val audioB64 = Base64.getEncoder().encodeToString(pcmBytes)
                val body = JSONObject().apply {
                    put("config", JSONObject().apply {
                        put("encoding", "LINEAR16")
                        put("sampleRateHertz", SAMPLE_RATE)
                        put("languageCode", "en-US")
                        // latest_short: current short-form model, far more robust to silence
                        // than command_and_search. useEnhanced improves accuracy.
                        put("model", "latest_short")
                        put("useEnhanced", true)
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
                    Log.e(TAG, "google stt error body: $responseBody")
                    onErrorCallback?.invoke("Google STT error: ${response.code}")
                    return@launch
                }

                val json = JSONObject(responseBody)
                val results = json.optJSONArray("results")
                // Concatenate transcripts across all result segments (long audio can split
                // the utterance across results[0], results[1], ...). opt* avoids throwing
                // when a segment lacks alternatives, so an empty result reads as "" not error.
                val sb = StringBuilder()
                if (results != null) for (i in 0 until results.length()) {
                    val t = results.optJSONObject(i)
                        ?.optJSONArray("alternatives")
                        ?.optJSONObject(0)
                        ?.optString("transcript", "")
                        ?.trim().orEmpty()
                    if (t.isNotEmpty()) { if (sb.isNotEmpty()) sb.append(' '); sb.append(t) }
                }
                val transcript = sb.toString()

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
