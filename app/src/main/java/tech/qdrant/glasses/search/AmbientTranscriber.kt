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
        private const val MIN_CHARS = 2            // drop trivial noise tokens
        private const val RESTART_DELAY_MS = 250L  // brief settle before re-arming
        private const val MAX_CONSECUTIVE_ERRORS = 8
        // Let an in-flight final result land after stopListening before releasing the
        // service connection — destroying immediately would drop the last utterance.
        private const val DESTROY_GRACE_MS = 2000L
        // SpeechRecognizer.ERROR_SPEECH_TIMEOUT / ERROR_NO_MATCH: the ROUTINE outcome of
        // a listen cycle in a quiet room — a healthy engine reporting "nobody spoke".
        // They must NOT count toward the kill threshold, or ambient dies after ~1 minute
        // of normal silence. "Empty result" (empty final, no partials) is the same case.
        private val BENIGN_ERROR_CODES = setOf(6, 7)
    }

    // Offline Google recognizer (its own mic + VAD); labeled to tell its logcat lines
    // apart from the tap-to-talk query recognizer (both log under the same TAG).
    private val recognizer = AndroidSpeechRecognizer(context, preferOffline = true, label = "ambient")
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var running = false
    private var cycleStartMs = 0L
    private var consecutiveErrors = 0

    fun start() {
        if (running) { Log.w(TAG, "start skipped (already running)"); return }
        running = true
        consecutiveErrors = 0
        Log.i(TAG, "ambient transcription started (offline Google, auto-restart)")
        arm()
    }

    private fun arm() {
        if (!running) return
        cycleStartMs = System.currentTimeMillis()
        recognizer.startListening(
            onPartial = { /* ignored — we only store finalized utterances */ },
            onResult = { text ->
                // A real recognized result proves the engine is healthy.
                consecutiveErrors = 0
                val t = text.trim()
                val tEnd = System.currentTimeMillis()
                if (t.length >= MIN_CHARS) {
                    Log.i(TAG, "ambient segment: \"${t.take(60)}\"")
                    onSegment(t, cycleStartMs, tEnd)
                }
                rearm()
            },
            onError = { err ->
                if (isBenign(err)) {
                    // Quiet cycle (timeout / no-match / empty final) — normal for ambient.
                    Log.d(TAG, "ambient cycle ended quietly ($err), re-arming")
                    rearm()
                } else {
                    consecutiveErrors++
                    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                        Log.w(TAG, "ambient stopping: $consecutiveErrors consecutive STT errors (last: $err) — offline model missing or recognizer unavailable?")
                        running = false
                    } else {
                        Log.d(TAG, "ambient cycle FAILED ($err), re-arming [$consecutiveErrors/$MAX_CONSECUTIVE_ERRORS]")
                        rearm()
                    }
                }
            },
        )
    }

    /** Errors that mean "nothing was said", not "the recognizer is broken". */
    private fun isBenign(err: String): Boolean {
        if (err == "Empty result") return true
        val code = err.substringAfterLast(' ').toIntOrNull() ?: return false
        return code in BENIGN_ERROR_CODES
    }

    private fun rearm() {
        if (!running) return
        mainHandler.postDelayed({ arm() }, RESTART_DELAY_MS)
    }

    fun stop() {
        // No early-return on !running: even if the error cutoff already fired, the
        // underlying SpeechRecognizer still holds a bound service connection that MUST
        // be released — otherwise every recording session leaks one (→ ERROR_BUSY after
        // a few sessions, killing both ambient and query STT).
        running = false
        recognizer.stopListening()
        mainHandler.postDelayed({ recognizer.destroy() }, DESTROY_GRACE_MS)
        Log.i(TAG, "ambient transcription stopped (recognizer release in ${DESTROY_GRACE_MS}ms)")
    }

    fun destroy() {
        running = false
        mainHandler.removeCallbacksAndMessages(null)
        recognizer.destroy()
    }
}
