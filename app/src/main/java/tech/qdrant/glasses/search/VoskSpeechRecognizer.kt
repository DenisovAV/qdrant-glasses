package tech.qdrant.glasses.search

import android.content.Context
import android.util.Log
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.StorageService

class VoskSpeechRecognizer(context: Context) : SpeechRecognizer {
    companion object {
        private const val TAG = "VoiceSearch"
        private const val SAMPLE_RATE = 16000
    }

    private var model: Model? = null
    private var recognizer: Recognizer? = null
    @Volatile private var running = false
    @Volatile private var gotResult = false

    init {
        StorageService.unpack(
            context, "model", "vosk-model",
            { m ->
                model = m
                // Pre-create recognizer so startListening has zero latency
                recognizer = try { Recognizer(m, SAMPLE_RATE.toFloat()) }
                catch (e: Exception) { Log.e(TAG, "vosk recognizer init failed", e); null }
                Log.i(TAG, "vosk model ready")
            },
            { e -> Log.e(TAG, "vosk model load failed", e) }
        )
    }

    override fun startListening(
        onPartial: (String) -> Unit,
        onResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        if (model == null) { onError("VOSK model not ready"); return }
        running = true
        gotResult = false
        Log.i(TAG, "vosk: startListening")
        recognizer?.close()
        recognizer = try { Recognizer(model!!, SAMPLE_RATE.toFloat()) }
        catch (e: Exception) { onError("VOSK recognizer error: ${e.message}"); null }
    }

    // Called by VoiceSearchManager with each audio chunk
    fun acceptChunk(bytes: ByteArray, onPartial: (String) -> Unit, onResult: (String) -> Unit) {
        val rec = recognizer ?: return
        if (rec.acceptWaveForm(bytes, bytes.size)) {
            val text = extractText(rec.result)
            Log.i(TAG, "vosk auto result: \"$text\"")
            if (text.isNotEmpty()) { gotResult = true; onResult(text) }
        } else {
            val partial = extractPartial(rec.partialResult)
            if (partial.isNotEmpty()) onPartial(partial)
        }
    }

    // Called by VoiceSearchManager when recording stops
    fun finalize(onResult: (String) -> Unit, onError: (String) -> Unit) {
        val rec = recognizer ?: return
        val text = extractText(rec.finalResult)
        Log.i(TAG, "vosk finalResult: \"$text\"")
        when {
            text.isNotEmpty() -> { gotResult = true; onResult(text) }
            // If acceptChunk already delivered a result, don't error — we're done
            gotResult -> Log.i(TAG, "vosk finalResult: empty but already got result, ignoring")
            else -> onError("Empty result")
        }
    }

    override fun stopListening() {
        running = false
    }

    override fun destroy() {
        running = false
        recognizer?.close()
        recognizer = null
        model?.close()
        model = null
    }

    val isReady get() = model != null

    private fun extractText(json: String) =
        Regex("\"text\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1)?.trim() ?: ""

    private fun extractPartial(json: String) =
        Regex("\"partial\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1)?.trim() ?: ""
}
