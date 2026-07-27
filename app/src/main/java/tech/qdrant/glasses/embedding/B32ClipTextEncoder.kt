package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.LongBuffer

/**
 * CLIP ViT-B/32 TEXT tower — the query side, MUST match the [QnnClipVisionEncoder] (same model =
 * same 512-dim space, so a W8A16-quantized image vector still compares correctly against a text
 * vector, cosine 0.995 to float). Input `input_ids` [1,77] int64 → `text_embeds` [1,512] float.
 *
 * Text is the COLD path (once per voice query), so it doesn't share the HTP with the per-crop
 * vision encoder. We build TWO sessions and time both on every query to compare in one APK:
 *  - **GPU (float)** — Adreno via ORT's QNN GPU backend (libQnnGpu.so). The GPU has fp16 and handles
 *    the token-embedding Gather, both of which the AR1 HTP does NOT — so it runs the FLOAT tower,
 *    exact vectors, no quantization. May fail to compile on some ops → caught, CPU still serves.
 *  - **CPU (int8)** — dynamic-int8 ONNX on ORT CPU (like the shipped TinyCLIP text path).
 *
 * The returned vector is always the CPU-int8 one (always available); the GPU run is timed for
 * comparison and logged to `clip_text_latency.csv` (logcat is flooded by the RayNeo camera).
 */
class B32ClipTextEncoder(context: Context) : TextEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val tokenizer = TokenizerFactory.create(context)
    private val latencyFile = File(context.filesDir, "clip_text_latency.csv")

    private val cpuSession: OrtSession
    private val gpuSession: OrtSession?

    init {
        latencyFile.appendText("=== text session ${System.currentTimeMillis()} ViT-B32 GPU(float) vs CPU(int8) ===\n")
        cpuSession = env.createSession(extractAsset(context, INT8_ASSET).absolutePath, OrtSession.SessionOptions())
        gpuSession = try {
            val opts = OrtSession.SessionOptions().apply { addQnn(mapOf("backend_path" to "libQnnGpu.so")) }
            val s = env.createSession(extractAsset(context, FLOAT_ASSET).absolutePath, opts)
            Log.i(TAG, "B32-text GPU(float) session OK")
            latencyFile.appendText("# GPU session created OK\n")
            s
        } catch (e: Throwable) {
            Log.e(TAG, "B32-text GPU session failed — CPU only", e)
            latencyFile.appendText("# GPU session FAILED: ${e.message}\n")
            null
        }
    }

    override fun encode(text: String): FloatArray {
        val ids = tokenizer.encodeToIds(text)  // [SOT, tok…, EOT, 0-pad], length 77
        val maxLen = Tokenizer.MAX_LENGTH

        // GPU (float) — timed for comparison, result discarded.
        gpuSession?.let { g ->
            val buf = LongBuffer.allocate(maxLen).apply { for (i in 0 until maxLen) put(ids[i].toLong()); rewind() }
            val t = OnnxTensor.createTensor(env, buf, longArrayOf(1, maxLen.toLong()))
            try {
                val t0 = System.currentTimeMillis()
                t.use { g.run(mapOf("input_ids" to it)) }.use { }
                val ms = System.currentTimeMillis() - t0
                latencyFile.appendText("GPU $ms\n")
                Log.i(TAG, "B32-text GPU(float) run=${ms}ms")
            } catch (e: Throwable) {
                latencyFile.appendText("# GPU run failed: ${e.message}\n")
            }
        }

        // CPU (int8) — the served vector.
        val buf = LongBuffer.allocate(maxLen).apply { for (i in 0 until maxLen) put(ids[i].toLong()); rewind() }
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, maxLen.toLong()))
        val t0 = System.currentTimeMillis()
        val results = tensor.use { cpuSession.run(mapOf("input_ids" to it)) }
        val ms = System.currentTimeMillis() - t0
        latencyFile.appendText("CPU $ms\n")
        Log.i(TAG, "B32-text CPU(int8) run=${ms}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("text_embeds").get().value as Array<FloatArray>)[0]
        }
    }

    override fun close() {
        gpuSession?.close()
        cpuSession.close()
        env.close()
    }

    companion object {
        private const val TAG = "ClipEncoder"
        private const val FLOAT_ASSET = "clip-vitb32-text-float.onnx"
        private const val INT8_ASSET = "clip-vitb32-text-int8.onnx"
    }
}
