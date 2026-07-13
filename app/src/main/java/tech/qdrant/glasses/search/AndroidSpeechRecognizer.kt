package tech.qdrant.glasses.search

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * On-device speech recognition via Android's standard SpeechRecognizer (Google).
 * Simple, stock usage: the recognizer owns the mic, VAD and endpointing itself.
 */
class AndroidSpeechRecognizer(
    private val context: Context,
    private val preferOffline: Boolean = true,
    private val label: String = "query",  // distinguishes ambient vs query instances in logcat
) : tech.qdrant.glasses.search.SpeechRecognizer {
    companion object {
        private const val TAG = "VoiceSearch"
        // The device default RecognitionService is RayNeo's (which hangs silently),
        // so bind explicitly to Google's service instead.
        private const val GOOGLE_PKG = "com.google.android.tts"
        private const val GOOGLE_SERVICE =
            "com.google.android.apps.speech.tts.googletts.service.GoogleTTSRecognitionService"
    }

    private var recognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var lastPartial = ""

    private fun buildIntent() = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
        putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, preferOffline)
        // DIAG test #3: widen the utterance window against early endpointing.
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 2500)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 1500)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1500)
    }

    /**
     * Warm-up: run one short throwaway recognition at app start (calm moment, usually on USB).
     * The first real search otherwise pays service bind + offline model load + audio-path
     * power-up INSIDE the search peak — on battery that cold-start transient stacks onto
     * ASR+embed+render and has brownout the glasses below UVLO. ~600ms, cancelled before any
     * result; the startup beep is acceptable off-stage.
     */
    fun warmUp() {
        mainHandler.post {
            Log.i(TAG, "[$label] ASR warm-up: binding service + loading model")
            val r = SpeechRecognizer.createSpeechRecognizer(
                context, ComponentName(GOOGLE_PKG, GOOGLE_SERVICE)
            )
            r.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle) {}
                override fun onPartialResults(partialResults: Bundle) {}
                override fun onError(error: Int) { Log.i(TAG, "[$label] warm-up done (err=$error is fine)") }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            r.startListening(buildIntent())
            mainHandler.postDelayed({
                runCatching { r.cancel(); r.destroy() }
                Log.i(TAG, "[$label] ASR warm-up complete")
            }, 600)
        }
    }

    override fun startListening(
        onPartial: (String) -> Unit,
        onResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        mainHandler.post {
            lastPartial = ""  // reset on main, same thread as onPartialResults writes
            recognizer?.destroy()
            recognizer = SpeechRecognizer.createSpeechRecognizer(
                context, ComponentName(GOOGLE_PKG, GOOGLE_SERVICE)
            ).apply {
                setRecognitionListener(object : RecognitionListener {
                    override fun onResults(results: Bundle) {
                        var text = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            ?.firstOrNull()?.trim() ?: ""
                        // The Google SODA engine sometimes returns text only in partials and
                        // an empty final bundle — fall back to the last partial we saw.
                        if (text.isEmpty()) text = lastPartial
                        Log.i(TAG, "[$label] android stt result: \"$text\" (lastPartial=\"$lastPartial\")")
                        if (text.isNotEmpty()) onResult(text) else onError("Empty result")
                    }
                    override fun onPartialResults(partialResults: Bundle) {
                        val text = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            ?.firstOrNull()?.trim() ?: ""
                        if (text.isNotEmpty()) { lastPartial = text; onPartial(text) }
                    }
                    override fun onError(error: Int) {
                        Log.w(TAG, "[$label] android stt error: $error")
                        onError("STT error: $error")
                    }
                    override fun onReadyForSpeech(params: Bundle) { Log.i(TAG, "[$label] android stt: ready") }
                    override fun onBeginningOfSpeech() { Log.i(TAG, "[$label] android stt: speech started") }
                    override fun onEndOfSpeech() { Log.i(TAG, "[$label] android stt: speech ended") }
                    override fun onRmsChanged(rmsdB: Float) {}
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })
                startListening(buildIntent())
            }
            Log.i(TAG, "[$label] android stt: startListening")
        }
    }

    override fun stopListening() {
        mainHandler.post { recognizer?.stopListening() }
    }

    override fun destroy() {
        mainHandler.post {
            recognizer?.destroy()
            recognizer = null
        }
    }
}
