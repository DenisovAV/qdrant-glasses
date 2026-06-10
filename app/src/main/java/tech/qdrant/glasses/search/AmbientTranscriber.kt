package tech.qdrant.glasses.search

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Continuously transcribes ambient speech during a recording session using the
 * offline Google engine (the only STT path that works on this device). Android's
 * SpeechRecognizer is one-shot, so we re-arm it on every terminal callback to make
 * it continuous ("ambient"). Exactly ONE mic owner — the recognizer's own — so there
 * is no AudioRecord contention. Emits each non-empty utterance as (text, tStart, tEnd).
 */
class AmbientTranscriber(
    private val context: Context,
    private val onSegment: (text: String, tStartMs: Long, tEndMs: Long) -> Unit,
) {
    companion object {
        private const val TAG = "AmbientTranscriber"
        private const val MIN_CHARS = 2          // drop trivial noise tokens
        private const val RESTART_DELAY_MS = 250L // brief settle before re-arming
        private const val MAX_CONSECUTIVE_ERRORS = 8
    }

    // Offline Google recognizer (its own mic + VAD); preferOffline defaults to true.
    private val recognizer = AndroidSpeechRecognizer(context, preferOffline = true)
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var running = false
    private var cycleStartMs = 0L
    private var consecutiveErrors = 0

    /** Always "ready" — the Google engine has no model-load gate like Vosk did. */
    val isReady get() = true

    fun start() {
        if (running) { Log.w(TAG, "start skipped (already running)"); return }
        running = true
        Log.i(TAG, "ambient transcription started (offline Google, auto-restart)")
        arm()
    }

    private fun arm() {
        if (!running) return
        cycleStartMs = System.currentTimeMillis()
        recognizer.startListening(
            onPartial = { /* ignored — we only store finalized utterances */ },
            onResult = { text ->
                // A returned result (even empty/too-short) means the recognizer is healthy.
                consecutiveErrors = 0
                val t = text.trim()
                val tEnd = System.currentTimeMillis()
                if (t.length >= MIN_CHARS) {
                    Log.i(TAG, "ambient segment: \"${t.take(60)}\"")
                    onSegment(t, cycleStartMs, tEnd)
                    consecutiveErrors = 0
                }
                rearm()
            },
            onError = { err ->
                consecutiveErrors++
                if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                    Log.w(TAG, "ambient stopping: $consecutiveErrors consecutive STT errors (last: $err) — offline model missing or recognizer unavailable?")
                    running = false
                } else {
                    Log.d(TAG, "ambient cycle ended ($err), re-arming [$consecutiveErrors/$MAX_CONSECUTIVE_ERRORS]")
                    rearm()
                }
            },
        )
    }

    private fun rearm() {
        if (!running) return
        mainHandler.postDelayed({ arm() }, RESTART_DELAY_MS)
    }

    fun stop() {
        if (!running) return
        running = false
        recognizer.stopListening()
        Log.i(TAG, "ambient transcription stopped")
    }

    fun destroy() {
        running = false
        recognizer.destroy()
    }
}
