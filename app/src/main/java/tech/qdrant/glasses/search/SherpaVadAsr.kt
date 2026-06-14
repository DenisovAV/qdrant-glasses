package tech.qdrant.glasses.search

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineMoonshineModelConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.SpeechSegment
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig

/**
 * Process-level warm singleton wrapping sherpa-onnx Silero VAD and Moonshine offline recognizer.
 *
 * The VAD and recognizer load ONCE via [ensureLoaded] and remain alive for the process lifetime —
 * they are intentionally never released (no [Vad.release] / [OfflineRecognizer.release] calls in
 * normal operation), mirroring the pattern in the streaming recognizer used in earlier iterations. This amortises the ~multi-
 * second load cost across all recording sessions in one app run.
 *
 * Threading contract:
 * - [ensureLoaded] is @Synchronized and may be called from any thread.
 * - [acceptWaveform], [pollSegment], [transcribe], and [reset] MUST be called from ONE dedicated
 *   audio thread (owned by AmbientTranscriber). No internal locking is provided for those methods;
 *   the caller is responsible for serialisation.
 */
object SherpaVadAsr {
    private const val TAG = "SherpaVadAsr"

    // VAD configuration constants
    const val SAMPLE_RATE = 16000
    private const val VAD_MODEL = "vad/silero_vad.onnx"
    private const val VAD_THRESHOLD = 0.5f
    private const val VAD_MIN_SILENCE_DURATION = 0.5f
    private const val VAD_MIN_SPEECH_DURATION = 0.25f
    private const val VAD_MAX_SPEECH_DURATION = 10f
    private const val VAD_WINDOW_SIZE = 512
    private const val NUM_THREADS = 2

    // Moonshine asset paths (relative to assets root)
    private const val MOONSHINE_PREPROCESSOR = "moonshine/preprocess.onnx"
    private const val MOONSHINE_ENCODER = "moonshine/encode.onnx"
    private const val MOONSHINE_UNCACHED_DECODER = "moonshine/uncached_decode.onnx"
    private const val MOONSHINE_CACHED_DECODER = "moonshine/cached_decode.onnx"
    private const val MOONSHINE_TOKENS = "moonshine/tokens.txt"

    @Volatile private var vad: Vad? = null
    @Volatile private var recognizer: OfflineRecognizer? = null

    /**
     * Load Silero VAD and Moonshine offline recognizer. Idempotent and thread-safe.
     * Returns true immediately if both are already loaded.
     * Returns false (and logs the error) on any failure — the app continues without ASR (soft-degrade).
     *
     * Each component is constructed only if its field is still null, so a retry after a half-load
     * reuses the already-built native object and only constructs the missing one (no orphaned handles).
     * Callers MUST receive true before invoking [acceptWaveform], [pollSegment], [transcribe], or
     * [reset] — see the class-level threading contract.
     */
    @Synchronized
    fun ensureLoaded(context: Context): Boolean {
        if (vad != null && recognizer != null) return true
        val t0 = System.currentTimeMillis()
        return try {
            if (vad == null) {
                val vadConfig = VadModelConfig(
                    sileroVadModelConfig = SileroVadModelConfig(
                        model = VAD_MODEL,
                        threshold = VAD_THRESHOLD,
                        minSilenceDuration = VAD_MIN_SILENCE_DURATION,
                        minSpeechDuration = VAD_MIN_SPEECH_DURATION,
                        windowSize = VAD_WINDOW_SIZE,
                        maxSpeechDuration = VAD_MAX_SPEECH_DURATION,
                    ),
                    sampleRate = SAMPLE_RATE,
                    numThreads = NUM_THREADS,
                )
                val t1 = System.currentTimeMillis()
                vad = Vad(context.assets, vadConfig)
                Log.i(TAG, "VAD loaded in ${System.currentTimeMillis() - t1}ms")
            }

            if (recognizer == null) {
                val recognizerConfig = OfflineRecognizerConfig(
                    featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80, dither = 0f),
                    modelConfig = OfflineModelConfig(
                        moonshine = OfflineMoonshineModelConfig(
                            preprocessor = MOONSHINE_PREPROCESSOR,
                            encoder = MOONSHINE_ENCODER,
                            uncachedDecoder = MOONSHINE_UNCACHED_DECODER,
                            cachedDecoder = MOONSHINE_CACHED_DECODER,
                            mergedDecoder = "",
                        ),
                        tokens = MOONSHINE_TOKENS,
                        numThreads = NUM_THREADS,
                        modelType = "moonshine",
                    ),
                )
                val t2 = System.currentTimeMillis()
                recognizer = OfflineRecognizer(context.assets, recognizerConfig)
                Log.i(TAG, "Moonshine recognizer loaded in ${System.currentTimeMillis() - t2}ms")
            }

            Log.i(TAG, "SherpaVadAsr fully loaded in ${System.currentTimeMillis() - t0}ms total")
            true
        } catch (e: Throwable) {
            Log.e(TAG, "SherpaVadAsr failed to load", e)
            false
        }
    }

    /**
     * Feed PCM floats in [-1, 1] to the VAD. No-op if not loaded.
     * Must be called from the dedicated audio thread.
     */
    fun acceptWaveform(samples: FloatArray) {
        vad?.acceptWaveform(samples)
    }

    /**
     * Pop the next completed speech segment from the VAD, or null if none is ready.
     * Must be called from the dedicated audio thread.
     */
    fun pollSegment(): SpeechSegment? {
        val v = vad ?: return null
        if (v.empty()) return null
        val seg = v.front()
        v.pop()
        return seg
    }

    /**
     * Transcribe one complete segment's audio with the offline Moonshine recognizer.
     * Creates a fresh [com.k2fsa.sherpa.onnx.OfflineStream] per call (one-shot, not reused).
     * Returns "" if the recognizer is not loaded.
     * Must be called from the dedicated audio thread.
     */
    fun transcribe(samples: FloatArray): String {
        val rec = recognizer ?: return ""
        val stream = rec.createStream()
        return try {
            stream.acceptWaveform(samples, SAMPLE_RATE)
            rec.decode(stream)
            rec.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    /**
     * Force any in-progress buffered speech into the segment queue without waiting for
     * trailing silence — call at session stop so a mid-utterance final segment is not lost.
     * Must be called from the dedicated audio thread.
     */
    fun flush() {
        vad?.flush()
    }

    /**
     * Clear any buffered VAD state at session start/stop.
     * Must be called from the dedicated audio thread.
     */
    fun reset() {
        vad?.reset()
    }
}
