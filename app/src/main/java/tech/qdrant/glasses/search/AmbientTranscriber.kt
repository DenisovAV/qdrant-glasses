package tech.qdrant.glasses.search

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * Continuously segments ambient audio with Silero VAD and transcribes each complete speech
 * segment with the offline Moonshine recognizer (via SherpaVadAsr). One AudioRecord feeds
 * the VAD; each finalized segment is emitted as (text, tStartMs, tEndMs). No streaming
 * endpoint, no decoder reset, no manual aggregation. Public contract unchanged:
 * AmbientTranscriber(context, onSegment); start()/stop()/destroy().
 */
class AmbientTranscriber(
    private val context: Context,
    private val onSegment: (text: String, tStartMs: Long, tEndMs: Long) -> Unit,
) {
    companion object {
        private const val TAG = "AmbientTranscriber"
        private const val MIN_CHARS = 2
        private const val SAMPLE_RATE = SherpaVadAsr.SAMPLE_RATE  // 16000
        private const val CHUNK_SAMPLES = 1600  // ~100ms at 16kHz
    }

    @Volatile private var running = false
    private var thread: Thread? = null
    private var recordingStartMs = 0L

    fun start() {
        if (running) { Log.w(TAG, "start skipped (already running)"); return }
        if (!SherpaVadAsr.ensureLoaded(context)) {
            Log.w(TAG, "ambient disabled — model unavailable")
            return
        }
        SherpaVadAsr.reset()
        recordingStartMs = System.currentTimeMillis()
        running = true
        Log.i(TAG, "ambient transcription started (VAD+offline)")
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
        try {
            rec.startRecording()
            val pcm = ShortArray(CHUNK_SAMPLES)
            val floats = FloatArray(CHUNK_SAMPLES)

            while (running) {
                val n = rec.read(pcm, 0, pcm.size)
                if (n <= 0) continue
                for (i in 0 until n) floats[i] = pcm[i] / 32768f
                SherpaVadAsr.acceptWaveform(if (n == floats.size) floats else floats.copyOf(n))

                // Drain all segments completed by this chunk.
                while (true) {
                    val seg = SherpaVadAsr.pollSegment() ?: break
                    handleSegment(seg)
                }
            }

            // Safety-net drain: a segment that completed on the last read must not be lost.
            while (true) {
                val seg = SherpaVadAsr.pollSegment() ?: break
                handleSegment(seg)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "ambient loop error", e)
        } finally {
            try { rec.stop() } catch (_: Throwable) {}
            rec.release()
            Log.i(TAG, "ambient transcription stopped")
        }
    }

    private fun handleSegment(seg: com.k2fsa.sherpa.onnx.SpeechSegment) {
        val tStart = recordingStartMs + seg.start * 1000L / SAMPLE_RATE
        val tEnd = tStart + seg.samples.size * 1000L / SAMPLE_RATE
        val text = try {
            SherpaVadAsr.transcribe(seg.samples).trim()
        } catch (e: Throwable) {
            Log.e(TAG, "transcribe failed, dropping segment", e)
            return
        }
        if (text.length >= MIN_CHARS) {
            Log.i(TAG, "segment (${tEnd - tStart}ms): \"${text.take(70)}\"")
            onSegment(text, tStart, tEnd)
        }
    }

    fun stop() {
        if (!running) return
        running = false
        thread?.join(1500)
        thread = null
        SherpaVadAsr.reset()
    }

    fun destroy() {
        running = false
        thread?.join(1500)
        thread = null
        // VAD and recognizer are process-level warm singletons; not released here.
    }
}
