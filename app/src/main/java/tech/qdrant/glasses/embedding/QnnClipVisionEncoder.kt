package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * CLIP ViT-B/32 vision tower on the Hexagon NPU via ORT's QNN Execution Provider, compiling the
 * HTP context ON THIS DEVICE — same pattern as [tech.qdrant.glasses.detect.YoloQnnDetector].
 *
 * The model is a **W8A16** static-quant QDQ-ONNX: 8-bit weights, **16-bit activations**. The AR1's
 * HTP has NO fp16 (compiler rejects any fp16 op for soc_model 58), and plain int8 (W8A8) collapses
 * CLIP embeddings (cosine ~0.45 vs float). W8A16 is what keeps them correct — measured 0.995 vs the
 * float model, and ~28 ms/crop isolated on the AR1 NPU (vs ~200 ms for TinyCLIP on CPU).
 *
 * We ship the SoC-agnostic QDQ-ONNX and let ORT's QNN EP do the HTP "preparation" pass for OUR SoC
 * on first run (`ep.context_enable=1`), caching to `<filesDir>/clip_vitb32_ctx.onnx`; a precompiled
 * context binary is SoC-locked and won't load. Per-crop latency is written to a FILE
 * (`clip_npu_latency.csv`) because the RayNeo camera floods logcat and evicts our lines.
 *
 * Preprocessing matches CLIP: resize 224, /255, subtract mean / divide std, NCHW float32.
 * I/O: input "image" [1,3,224,224] float → output "image_embeds" [1,512] float.
 */
class QnnClipVisionEncoder(context: Context) : VisionEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val std  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    private val latencyFile = File(context.filesDir, "clip_npu_latency.csv")
    private var nInfer = 0
    private var sumInfer = 0L

    init {
        latencyFile.appendText("=== clip session ${System.currentTimeMillis()} backend=QNN_HTP ViT-B32 W8A16 ===\n")
        val ctxCache = File(context.filesDir, "clip_vitb32_ctx.onnx")
        val cached = ctxCache.exists()
        val modelPath = if (cached) ctxCache.absolutePath else extractAsset(context, ASSET).absolutePath
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
            if (!cached) {
                addConfigEntry("ep.context_enable", "1")
                addConfigEntry("ep.context_file_path", ctxCache.absolutePath)
            }
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(modelPath, opts)
        Log.i(TAG, "CLIP-QNN session created (cached=$cached) in ${System.currentTimeMillis() - t0}ms")
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG, IMG, true)
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, IMG.toLong(), IMG.toLong()))
        if (resized !== bitmap) resized.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("image" to it)) }
        val ms = System.currentTimeMillis() - t0
        nInfer++; sumInfer += ms
        latencyFile.appendText("$ms\n")
        if (nInfer % 10 == 0) latencyFile.appendText("# n=$nInfer mean=${sumInfer / nInfer}ms (last=$ms)\n")
        Log.i(TAG, "CLIP-QNN vision run=${ms}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("image_embeds").get().value as Array<FloatArray>)[0]
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(IMG * IMG).also { bitmap.getPixels(it, 0, IMG, 0, 0, IMG, IMG) }
        val buf = FloatBuffer.allocate(3 * IMG * IMG)
        val stride = IMG * IMG
        for (i in pixels.indices) {
            val px = pixels[i]
            buf.put(i,              ((px shr 16 and 0xFF) / 255f - mean[0]) / std[0])
            buf.put(i + stride,     ((px shr  8 and 0xFF) / 255f - mean[1]) / std[1])
            buf.put(i + stride * 2, ((px        and 0xFF) / 255f - mean[2]) / std[2])
        }
        return buf.apply { rewind() }
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        private const val TAG = "ClipEncoder"
        private const val ASSET = "clip-vitb32-w8a16.onnx"
        private const val IMG = 224
    }
}
