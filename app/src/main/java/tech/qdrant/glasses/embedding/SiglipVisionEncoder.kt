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
 * SigLIP2 vision tower on the Hexagon NPU via ORT's QNN Execution Provider — running a W8A16 HTP
 * context binary wrapped in an **EPContext** ONNX, same shape of deployment as
 * [QnnClipVisionEncoder] (see that file's KDoc for why EPContext and not a plain QDQ graph).
 *
 * Preprocessing DIFFERS from CLIP: SigLIP normalizes with mean = std = 0.5 on every channel
 * (`(v/255 - 0.5) / 0.5`), not the CLIP dataset mean/std. Resize 224, NCHW float32.
 * I/O: input "image" [1,3,224,224] float -> output "embeds" [1,768] float (wrapper external names; the EPContext node internally uses the binary's pixel_values/pooler_output).
 */
class SiglipVisionEncoder(context: Context) : VisionEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Per-channel lookup: (v/255 - 0.5)/0.5 for every byte value — same LUT trick as
    // QnnClipVisionEncoder, just with SigLIP's flat 0.5/0.5 mean/std instead of the CLIP stats.
    private val lut = Array(3) { FloatArray(256) { v -> (v / 255f - 0.5f) / 0.5f } }

    private val latencyFile = File(context.filesDir, "siglip_vision_latency.csv")
    private var nInfer = 0
    private var sumInfer = 0L

    init {
        latencyFile.appendText("=== siglip session ${System.currentTimeMillis()} backend=QNN_HTP SigLIP2 W8A16 EPContext ===\n")
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(extractAsset(context, ASSET).absolutePath, opts)
        Log.i(TAG, "SigLIP-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms")
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        // Skip the copy when the crop is already 224x224 (createScaledBitmap allocates + filters
        // even for a no-op resize).
        val resized = if (bitmap.width == IMG && bitmap.height == IMG) bitmap
                      else Bitmap.createScaledBitmap(bitmap, IMG, IMG, true)
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, IMG.toLong(), IMG.toLong()))
        if (resized !== bitmap) resized.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("image" to it)) }
        val ms = System.currentTimeMillis() - t0
        nInfer++; sumInfer += ms
        latencyFile.appendText("$ms\n")
        if (nInfer % 10 == 0) latencyFile.appendText("# n=$nInfer mean=${sumInfer / nInfer}ms (last=$ms)\n")
        Log.i(TAG, "SigLIP-QNN vision run=${ms}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("embeds").get().value as Array<FloatArray>)[0]
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(IMG * IMG).also { bitmap.getPixels(it, 0, IMG, 0, 0, IMG, IMG) }
        val arr = FloatArray(3 * IMG * IMG)
        val stride = IMG * IMG
        val lutR = lut[0]; val lutG = lut[1]; val lutB = lut[2]
        for (i in pixels.indices) {
            val px = pixels[i]
            arr[i]              = lutR[px shr 16 and 0xFF]
            arr[i + stride]     = lutG[px shr  8 and 0xFF]
            arr[i + stride * 2] = lutB[px        and 0xFF]
        }
        return FloatBuffer.wrap(arr)
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        private const val TAG = "SiglipEncoder"
        private const val ASSET = "siglip-vision-epctx.onnx"
        private const val IMG = 224
    }
}
