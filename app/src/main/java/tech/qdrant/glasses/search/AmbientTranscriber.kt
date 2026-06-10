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
        val stream = SherpaStreamingAsr.newStream()
        rec.startRecording()
        val pcm = ShortArray(CHUNK_SAMPLES)
        val floats = FloatArray(CHUNK_SAMPLES)
        var segStartMs = 0L
        var sawSpeech = false
        try {
            while (running) {
                val n = rec.read(pcm, 0, pcm.size)
                if (n <= 0) continue
                for (i in 0 until n) floats[i] = pcm[i] / 32768f
                SherpaStreamingAsr.feed(stream, if (n == floats.size) floats else floats.copyOf(n))
                val text = SherpaStreamingAsr.currentText(stream)
                if (text.isNotEmpty() && !sawSpeech) { segStartMs = System.currentTimeMillis(); sawSpeech = true }
                if (SherpaStreamingAsr.isEndpoint(stream)) {
                    if (text.length >= MIN_CHARS) {
                        Log.i(TAG, "ambient segment: \"${text.take(60)}\"")
                        onSegment(text, if (sawSpeech) segStartMs else System.currentTimeMillis(), System.currentTimeMillis())
                    }
                    SherpaStreamingAsr.reset(stream)
                    sawSpeech = false
                }
            }
            // Flush the final in-flight utterance before tearing down.
            SherpaStreamingAsr.finishStream(stream)
            val tail = SherpaStreamingAsr.currentText(stream)
            if (tail.length >= MIN_CHARS) {
                Log.i(TAG, "ambient final segment: \"${tail.take(60)}\"")
                onSegment(tail, if (sawSpeech) segStartMs else System.currentTimeMillis(), System.currentTimeMillis())
            }
        } catch (e: Throwable) {
            Log.e(TAG, "ambient loop error", e)
        } finally {
            rec.stop(); rec.release()
            SherpaStreamingAsr.releaseStream(stream)
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
