package tech.qdrant.glasses.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import tech.qdrant.glasses.embedding.extractAsset
import java.io.File
import java.nio.FloatBuffer

/**
 * DBNet text-region DETECTOR on the Hexagon NPU (ORT-QNN Execution Provider, EPContext-wrapped int8
 * HTP binary) — the live "is there text + where" gate for Stage 3's on-stream highlighting. Measured
 * ~10.6ms on the AR1 (see the DBNet-on-NPU spike / ocr-ondevice-spike memory), fast enough to run
 * per-frame alongside YOLO the same way [tech.qdrant.glasses.detect.YoloQnnDetector] does.
 *
 * Deployment mirrors [tech.qdrant.glasses.embedding.SiglipVisionEncoder] exactly: an EPContext ONNX
 * (`ocr/dbnet-det-epctx.onnx`, det_q.bin embedded, external I/O `image`→`probmap`) run whole on the
 * HTP in burst mode. FIXED 640x640 input (the compiled binary's shape) — great for PROMINENT text
 * (whiteboards, signs); tiny/far text (small screen fonts) is left to the CPU [OcrEngine]@1536
 * background recognition path. This detector only LOCATES text; recognition stays on OcrEngine.
 *
 * Output: text-line boxes as normalized [0,1] [RectF]s (fractions of the frame), ready to scale onto
 * whatever preview/overlay surface. Empty list ⇒ no text in view. Call OFF the main thread.
 */
class QnnDbnetDetector(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Per-channel ImageNet-normalize LUT (DBNet preproc): (v/255 - mean)/std — same LUT trick as the
    // QNN vision encoders, just with the detector's ImageNet stats.
    private val lut = Array(3) { c -> FloatArray(256) { v -> (v / 255f - MEAN[c]) / STD[c] } }

    private val latencyFile = File(context.filesDir, "dbnet_qnn_latency.csv")

    init {
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(extractAsset(context, ASSET).absolutePath, opts)
        Log.i(TAG, "DBNet-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms")
    }

    /** Text-region boxes in normalized [0,1] coords. Empty if no text detected. */
    fun detect(bitmap: Bitmap): List<RectF> {
        val resized = if (bitmap.width == S && bitmap.height == S) bitmap
                      else Bitmap.createScaledBitmap(bitmap, S, S, true)
        val tensor = OnnxTensor.createTensor(env, toTensor(resized), longArrayOf(1, 3, S.toLong(), S.toLong()))
        if (resized !== bitmap) resized.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("image" to it)) }
        val ms = System.currentTimeMillis() - t0
        latencyFile.appendText("$ms\n")
        val boxes = results.use { r ->
            @Suppress("UNCHECKED_CAST")
            val prob = (r.get("probmap").get().value as Array<Array<Array<FloatArray>>>)[0][0]  // [S][S]
            lineBoxes(prob)
        }
        Log.i(TAG, "DBNet-QNN run=${ms}ms -> ${boxes.size} text-boxes")
        return boxes
    }

    private fun toTensor(bmp: Bitmap): FloatBuffer {
        val px = IntArray(S * S).also { bmp.getPixels(it, 0, S, 0, 0, S, S) }
        val arr = FloatArray(3 * S * S); val stride = S * S
        val r = lut[0]; val g = lut[1]; val b = lut[2]
        for (i in px.indices) {
            val p = px[i]
            arr[i] = r[p shr 16 and 0xFF]; arr[i + stride] = g[p shr 8 and 0xFF]; arr[i + stride * 2] = b[p and 0xFF]
        }
        return FloatBuffer.wrap(arr)
    }

    // Horizontal-projection line segmentation → normalized [0,1] boxes (same convention as OcrEngine).
    private fun lineBoxes(prob: Array<FloatArray>): List<RectF> {
        val h = prob.size; val w = prob[0].size
        val minCols = maxOf(3, (w * 0.005f).toInt())
        val out = ArrayList<RectF>()
        var y = 0
        while (y < h) {
            var c = 0; for (x in 0 until w) if (prob[y][x] > BIN) c++
            if (c < minCols) { y++; continue }
            var y1 = y
            while (y1 < h) { var cc = 0; for (x in 0 until w) if (prob[y1][x] > BIN) cc++; if (cc < minCols) break; y1++ }
            if (y1 - y >= 6) {
                var x0 = w; var x1 = 0
                for (yy in y until y1) for (x in 0 until w) if (prob[yy][x] > BIN) { if (x < x0) x0 = x; if (x > x1) x1 = x }
                out.add(RectF(x0.toFloat() / w, y.toFloat() / h, x1.toFloat() / w, y1.toFloat() / h))
            }
            y = y1
        }
        return out
    }

    override fun close() { session.close() }

    companion object {
        private const val TAG = "DbnetQnn"
        private const val ASSET = "ocr/dbnet-det-epctx.onnx"
        private const val S = 640
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val BIN = 0.3f
    }
}
