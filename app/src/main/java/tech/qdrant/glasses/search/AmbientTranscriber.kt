package tech.qdrant.glasses.search

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * Continuously transcribes ambient speech during a recording session using Sherpa-ONNX
 * streaming recognition. ONE AudioRecord feeds PCM to a single long-lived recognizer
 * stream — no engine restarts, so no beep and no deaf windows. Endpoints segment the
 * stream; each finalized utterance is emitted as (text, tStartMs, tEndMs).
 *
 * Public contract is unchanged from the previous Google-ASR implementation:
 *   AmbientTranscriber(context, onSegment) ; start() / stop() / destroy()
 */
class AmbientTranscriber(
    private val context: Context,
    private val onSegment: (text: String, tStartMs: Long, tEndMs: Long) -> Unit,
) {
    companion object {
        private const val TAG = "AmbientTranscriber"
        private const val MIN_CHARS = 2
        private const val SAMPLE_RATE = SherpaStreamingAsr.SAMPLE_RATE  // 16000
        private const val CHUNK_SAMPLES = 1600  // ~100ms at 16kHz
        private const val MAX_WORDS = 12            // cut at ~1 sentence even without a pause (lecture)
        private const val MAX_SEGMENT_MS = 10_000L  // safety cap against runaway accumulation
        private const val OVERLAP_WORDS = 4         // carry the tail of a forced cut into the next chunk
    }

    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        if (running) { Log.w(TAG, "start skipped (already running)"); return }
        if (!SherpaStreamingAsr.ensureLoaded(context)) {
            Log.w(TAG, "ambient disabled — sherpa model unavailable")
            return
        }
        running = true
        Log.i(TAG, "ambient transcription started (sherpa streaming)")
        thread = Thread { runLoop() }.also { it.start() }
    }

    private fun runLoop() {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, CHUNK_SAMPLES * 2 * 4)
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized (mic busy / no permission?) — ambient disabled")
            rec.release()
            running = false
            return
        }
        var stream: com.k2fsa.sherpa.onnx.OnlineStream? = null
        try {
            stream = SherpaStreamingAsr.newStream()
            rec.startRecording()
            val s = stream  // non-null inside this try
            val pcm = ShortArray(CHUNK_SAMPLES)
            val floats = FloatArray(CHUNK_SAMPLES)
            var segStartMs = 0L
            var sawSpeech = false
            var pendingPrefix = ""   // overlap tail prepended to the next emitted chunk

            // Finalize the current chunk. byEndpoint=true = a real pause (clean boundary,
            // no overlap). false = a forced cap cut → carry an overlap tail so the word
            // on the seam isn't lost in either chunk.
            fun cut(text: String, byEndpoint: Boolean) {
                val full = (if (pendingPrefix.isEmpty()) text else "$pendingPrefix $text").trim()
                if (full.length >= MIN_CHARS) {
                    Log.i(TAG, "ambient segment (${if (byEndpoint) "endpoint" else "cap"}): \"${full.take(60)}\"")
                    onSegment(full, if (sawSpeech) segStartMs else System.currentTimeMillis(), System.currentTimeMillis())
                }
                pendingPrefix = if (byEndpoint) ""
                    else text.split(" ").filter { it.isNotEmpty() }.takeLast(OVERLAP_WORDS).joinToString(" ")
                SherpaStreamingAsr.reset(s)
                sawSpeech = false
            }

            while (running) {
                val n = rec.read(pcm, 0, pcm.size)
                if (n <= 0) continue
                for (i in 0 until n) floats[i] = pcm[i] / 32768f
                SherpaStreamingAsr.feed(s, if (n == floats.size) floats else floats.copyOf(n))
                val text = SherpaStreamingAsr.currentText(s)
                if (text.isNotEmpty() && !sawSpeech) { segStartMs = System.currentTimeMillis(); sawSpeech = true }
                val words = if (text.isEmpty()) 0 else text.split(" ").count { it.isNotEmpty() }
                val tooLong = sawSpeech && (System.currentTimeMillis() - segStartMs) >= MAX_SEGMENT_MS
                when {
                    SherpaStreamingAsr.isEndpoint(s) -> cut(text, byEndpoint = true)
                    words >= MAX_WORDS               -> cut(text, byEndpoint = false)
                    tooLong                          -> cut(text, byEndpoint = false)
                }
            }
            // Flush the final in-flight utterance before tearing down (with overlap prefix).
            SherpaStreamingAsr.finishStream(s)
            val rawTail = SherpaStreamingAsr.currentText(s)
            val tail = (if (pendingPrefix.isEmpty()) rawTail else "$pendingPrefix $rawTail").trim()
            if (tail.length >= MIN_CHARS) {
                Log.i(TAG, "ambient final segment: \"${tail.take(60)}\"")
                onSegment(tail, if (sawSpeech) segStartMs else System.currentTimeMillis(), System.currentTimeMillis())
            }
        } catch (e: Throwable) {
            Log.e(TAG, "ambient loop error", e)
        } finally {
            try { rec.stop() } catch (_: Throwable) {}
            rec.release()
            stream?.let { SherpaStreamingAsr.releaseStream(it) }
            Log.i(TAG, "ambient transcription stopped")
        }
    }

    fun stop() {
        if (!running) return
        running = false
        thread?.join(1500)
        thread = null
    }

    fun destroy() {
        running = false
        thread?.join(1500)
        thread = null
        // Recognizer is a process singleton kept warm across sessions; not released here.
    }
}
