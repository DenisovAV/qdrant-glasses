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
        // Aggregation chunking: small recognized PIECES are accumulated into a meaningful
        // WINDOW before storing — 2-word fragments are useless for retrieval (research:
        // recall tanks below ~5 sentences). A window is emitted at TARGET_WORDS or after
        // FLUSH_SILENCE_MS of quiet (a remark that ended). PIECE_WORDS forces a piece even
        // in pause-less speech so the buffer keeps filling.
        private const val PIECE_WORDS = 12          // finalize a piece this big even without a pause
        private const val TARGET_WORDS = 60         // emit a window once the buffer reaches this
        private const val OVERLAP_WORDS = 8         // seed the next window with this tail (seam safety)
        private const val FLUSH_SILENCE_MS = 4000L  // emit a non-empty buffer after this much quiet
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

            // Aggregation buffer: pieces accumulate here into a searchable window.
            var windowBuf = ""        // words gathered so far (includes overlap seed)
            var windowStartMs = 0L    // wall-clock of the first piece in this window
            var lastPieceMs = 0L      // wall-clock of the most recent finalized piece

            fun wordCount(t: String) = if (t.isBlank()) 0 else t.split(" ").count { it.isNotEmpty() }

            // Emit the accumulated window and seed the next one with an overlap tail.
            fun emitWindow() {
                val full = windowBuf.trim()
                if (full.length >= MIN_CHARS) {
                    Log.i(TAG, "ambient window (${wordCount(full)}w): \"${full.take(70)}\"")
                    onSegment(full, if (windowStartMs > 0) windowStartMs else System.currentTimeMillis(), System.currentTimeMillis())
                }
                windowBuf = full.split(" ").filter { it.isNotEmpty() }.takeLast(OVERLAP_WORDS).joinToString(" ")
                windowStartMs = 0L     // next real piece sets a fresh start
            }

            // Append a finalized recognition piece to the window buffer.
            fun addPiece(piece: String) {
                val p = piece.trim()
                if (p.isEmpty()) return
                if (windowStartMs == 0L) windowStartMs = System.currentTimeMillis()
                windowBuf = if (windowBuf.isEmpty()) p else "$windowBuf $p"
                lastPieceMs = System.currentTimeMillis()
                SherpaStreamingAsr.reset(s)
                if (wordCount(windowBuf) >= TARGET_WORDS) emitWindow()
            }

            while (running) {
                val n = rec.read(pcm, 0, pcm.size)
                if (n <= 0) continue
                for (i in 0 until n) floats[i] = pcm[i] / 32768f
                SherpaStreamingAsr.feed(s, if (n == floats.size) floats else floats.copyOf(n))
                val text = SherpaStreamingAsr.currentText(s)
                // A piece is "done" on a real pause (endpoint) or once it's big enough that
                // a pause-less stream still yields pieces to accumulate.
                if (SherpaStreamingAsr.isEndpoint(s) || wordCount(text) >= PIECE_WORDS) {
                    addPiece(text)
                }
                // Idle flush: a remark ended and nothing followed — store what we have.
                if (windowBuf.isNotBlank() && lastPieceMs > 0 &&
                    System.currentTimeMillis() - lastPieceMs >= FLUSH_SILENCE_MS) {
                    emitWindow()
                    windowBuf = ""        // idle flush is a clean boundary — drop the overlap seed
                    lastPieceMs = 0L
                }
            }
            // Drain the last in-flight piece, then flush whatever remains.
            SherpaStreamingAsr.finishStream(s)
            addPiece(SherpaStreamingAsr.currentText(s))
            if (windowBuf.isNotBlank()) emitWindow()
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
