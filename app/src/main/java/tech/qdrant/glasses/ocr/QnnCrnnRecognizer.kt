package tech.qdrant.glasses.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import tech.qdrant.glasses.embedding.extractAsset
import java.io.File
import java.nio.FloatBuffer

/**
 * CRNN text RECOGNIZER on the Hexagon NPU (ORT-QNN EPContext, W8A16 HTP binary). The companion to
 * [QnnDbnetDetector]: it turns one detected text-line crop into a string. Together they put the WHOLE
 * OCR on the NPU (~15-30ms vs the ~4s CPU [OcrEngine] path) — see the whole-OCR-on-NPU spike.
 *
 * KEY: this is a PP-OCRv2 CRNN (MobileNetV3 + 2xBiLSTM + CTC), NOT the v3 SVTR — the SVTR's attention
 * reshapes the sequence into the batch dim and won't convert to QNN, whereas the CRNN's LSTMs compile
 * WHOLE onto the HTP (QNN HTP does support LSTM). FIXED input 32x320 (the compiled binary's shape):
 * a line is resized to 32x320 → 80 CTC timesteps (W/4). External I/O: "image"[1,3,32,320]f32 ->
 * "logits"[1,80,6625]f32.  (T=80 is the context-binary's real output — an earlier T=25 in this
 * wrapper mis-declared the shape and scrambled the CTC read into confident garbage; see the wrapper's
 * QuantizeLinear input scale/zp, which must also match the binary's quantized input encoding exactly.)
 *
 * Call OFF the main thread. Same CTC decode + dict (ppocr_keys, index->char, 0=blank) as [OcrEngine].
 */
class QnnCrnnRecognizer(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val chars: List<String>
    // rec normalize (x/255 - 0.5)/0.5, per channel
    private val lut = FloatArray(256) { v -> (v / 255f - 0.5f) / 0.5f }
    private val latencyFile = File(context.filesDir, "crnn_qnn_latency.csv")

    init {
        chars = context.assets.open("ocr/ppocr_keys.txt").bufferedReader().use { it.readLines() }
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(extractAsset(context, ASSET).absolutePath, opts)
        Log.i(TAG, "CRNN-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms (dict=${chars.size})")
    }

    /** Recognize one text-line [crop] → (text, mean CTC conf). */
    fun recognize(crop: Bitmap): Pair<String, Float> {
        val rs = if (crop.width == W && crop.height == H) crop
                 else Bitmap.createScaledBitmap(crop, W, H, true)
        val tensor = OnnxTensor.createTensor(env, toTensor(rs), longArrayOf(1, 3, H.toLong(), W.toLong()))
        if (rs !== crop) rs.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("image" to it)) }
        val ms = System.currentTimeMillis() - t0
        latencyFile.appendText("$ms\n")
        val out = results.use { r ->
            @Suppress("UNCHECKED_CAST")
            val logits = (r.get("logits").get().value as Array<Array<FloatArray>>)[0]   // [T=80][6625]
            ctcDecode(logits)
        }
        Log.i(TAG, "CRNN-QNN run=${ms}ms -> \"${out.first}\"")
        return out
    }

    private fun toTensor(bmp: Bitmap): FloatBuffer {
        val px = IntArray(W * H).also { bmp.getPixels(it, 0, W, 0, 0, W, H) }
        val arr = FloatArray(3 * H * W); val stride = H * W
        for (i in px.indices) {
            val p = px[i]
            arr[i] = lut[p shr 16 and 0xFF]; arr[i + stride] = lut[p shr 8 and 0xFF]; arr[i + stride * 2] = lut[p and 0xFF]
        }
        return FloatBuffer.wrap(arr)
    }

    private fun ctcDecode(logits: Array<FloatArray>): Pair<String, Float> {
        val sb = StringBuilder(); var prev = -1; var cSum = 0f; var cN = 0
        for (t in logits.indices) {
            var best = 0; var bestV = logits[t][0]
            for (k in 1 until logits[t].size) if (logits[t][k] > bestV) { bestV = logits[t][k]; best = k }
            if (best != 0 && best != prev) { sb.append(chars.getOrElse(best) { "" }); cSum += bestV; cN++ }
            prev = best
        }
        return sb.toString() to if (cN > 0) cSum / cN else 0f
    }

    override fun close() { session.close() }

    companion object {
        private const val TAG = "CrnnQnn"
        private const val ASSET = "ocr/crnn-rec-epctx.onnx"
        private const val H = 32
        private const val W = 320
    }
}
