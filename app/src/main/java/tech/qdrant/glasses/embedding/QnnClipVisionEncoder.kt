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
 * CLIP ViT-B/32 vision tower on the Hexagon NPU via ORT's QNN Execution Provider — running the
 * **native, all-on-HTP** context binary through an **EPContext** wrapper.
 *
 * The model is W8A16 (8-bit weights, 16-bit activations): the AR1 HTP has NO fp16 and plain int8
 * collapses CLIP (cosine ~0.45), so W8A16 is what keeps vectors correct (0.995 to float).
 *
 * WHY the EPContext wrapper and not a plain QDQ-ONNX: if you hand ORT the QDQ-ONNX and let its QNN
 * EP partition on-device, it REJECTS every LayerNorm + GELU-Div + Gather (error 3110) and falls
 * them back to the CPU — dozens of HTP↔CPU round-trips, ~87ms/inference (worse under load). The
 * native `qnn-context-binary-generator` puts all 761 layers on the HTP in one graph; wrapping that
 * pre-built binary in a single **EPContext** node means ORT runs it whole, no re-partition:
 * **~24ms/inference measured on the live AR1, cosine 0.9949 to the QDQ path** (~6x faster).
 *
 * The wrapper (`clip-vitb32-epctx.onnx`) carries float32 I/O with Q/DQ nodes to the graph's uint16
 * activations (input scale 5.99e-05 / zp 29863, output scale 2.04e-04 / zp 51271). It is
 * **SoC-LOCKED** to the AR1 (HTP V73) — regenerate for another SoC (see the private
 * ar1-npu-findings.md recipe). No on-device compile step: the context is already built, so session
 * creation is fast (vs ~21s for the QDQ on-device "preparation" pass).
 *
 * Preprocessing matches CLIP: resize 224, /255, subtract mean / divide std, NCHW float32.
 * I/O: input "pixel_values" [1,3,224,224] float → output "embeds" [1,512] float.
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
        latencyFile.appendText("=== clip session ${System.currentTimeMillis()} backend=QNN_HTP ViT-B32 W8A16 EPContext ===\n")
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(extractAsset(context, ASSET).absolutePath, opts)
        Log.i(TAG, "CLIP-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms")
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG, IMG, true)
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, IMG.toLong(), IMG.toLong()))
        if (resized !== bitmap) resized.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("pixel_values" to it)) }
        val ms = System.currentTimeMillis() - t0
        nInfer++; sumInfer += ms
        latencyFile.appendText("$ms\n")
        if (nInfer % 10 == 0) latencyFile.appendText("# n=$nInfer mean=${sumInfer / nInfer}ms (last=$ms)\n")
        Log.i(TAG, "CLIP-QNN vision run=${ms}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("embeds").get().value as Array<FloatArray>)[0]
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
        private const val ASSET = "clip-vitb32-epctx.onnx"
        private const val IMG = 224
    }
}
