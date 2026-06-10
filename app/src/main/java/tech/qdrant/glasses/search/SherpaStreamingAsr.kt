package tech.qdrant.glasses.search

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.EndpointConfig
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig

/**
 * Process-level warm singleton over sherpa-onnx streaming recognition. The ~180MB model
 * loads ONCE and is reused across recording sessions (GlassesViewModel builds a fresh
 * AmbientTranscriber each session, so the recognizer must NOT live in that instance).
 *
 * Assets live in assets/sherpa/ — sherpa reads them natively via AssetManager, no extraction.
 *
 * Contract: call [ensureLoaded] first; the stream methods must be called on a single
 * dedicated audio thread, with one [OnlineStream] per recording session.
 */
object SherpaStreamingAsr {
    private const val TAG = "SherpaAsr"
    private const val DIR = "sherpa"
    const val SAMPLE_RATE = 16000

    @Volatile private var recognizer: OnlineRecognizer? = null

    @Synchronized
    fun ensureLoaded(context: Context): Boolean {
        if (recognizer != null) return true
        return try {
            val t0 = System.currentTimeMillis()
            val config = OnlineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80, dither = 0f),
                modelConfig = OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = "$DIR/encoder.onnx",
                        decoder = "$DIR/decoder.onnx",
                        joiner = "$DIR/joiner.onnx",
                    ),
                    tokens = "$DIR/tokens.txt",
                    numThreads = 2,
                    modelType = "zipformer",
                ),
                endpointConfig = EndpointConfig(),
                enableEndpoint = true,
                decodingMethod = "greedy_search",
            )
            recognizer = OnlineRecognizer(context.assets, config)
            Log.i(TAG, "sherpa recognizer ready in ${System.currentTimeMillis() - t0}ms")
            true
        } catch (e: Throwable) {
            Log.e(TAG, "sherpa recognizer failed to load", e)
            false
        }
    }

    fun newStream(): OnlineStream {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        return rec.createStream("")
    }

    fun feed(stream: OnlineStream, samples: FloatArray) {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        stream.acceptWaveform(samples, SAMPLE_RATE)
        while (rec.isReady(stream)) rec.decode(stream)
    }

    /**
     * Flush the stream at session end: signal end-of-input and drain the decoder so the
     * final utterance (which may not have triggered an endpoint) is fully decoded before
     * the caller reads [currentText] and releases the stream.
     */
    fun finishStream(stream: OnlineStream) {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        stream.inputFinished()
        while (rec.isReady(stream)) rec.decode(stream)
    }

    fun currentText(stream: OnlineStream): String {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        return rec.getResult(stream).text.trim()
    }

    fun isEndpoint(stream: OnlineStream): Boolean {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        return rec.isEndpoint(stream)
    }

    fun reset(stream: OnlineStream) {
        val rec = checkNotNull(recognizer) { "SherpaStreamingAsr: call ensureLoaded() first" }
        rec.reset(stream)
    }

    fun releaseStream(stream: OnlineStream) = stream.release()
}
