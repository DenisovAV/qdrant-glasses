package tech.qdrant.glasses.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import tech.qdrant.glasses.embedding.extractAsset
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * ViTSTR text RECOGNIZER on the Hexagon NPU (ORT-QNN EPContext, W8A16 HTP binary). The NON-recurrent
 * replacement for the CRNN whose LSTM cannot run on the AR1 HTP (unrolled-quant collapses, native-quant
 * hits the requant-gain limit, FP16 is unsupported — see the ocr-ondevice-spike memory). ViTSTR is a
 * plain ViT-Small encoder + a per-token linear head: the SAME attention op family as our SigLIP/CLIP
 * vision towers that already run W8A16 on this HTP, so it compiles WHOLE onto the Hexagon.
 *
 * Validated on-device (OcrSpikeTest#vitstrOnNpu): Mobile/Hello/MCP/2026 = 4/4 exact vs the fp32 model,
 * ~24ms/word. FIXED input [1,1,224,224] grayscale, normalized (x/255-0.5)/0.5 → [-1,1] (a text-line
 * crop is resized to the square — ViTSTR is trained that way). Output [1,25,96], greedy per-position
 * argmax: pos0 = [GO], stop at [s] = index 1, index ≥ 2 → [VOCAB] char. Call OFF the main thread.
 *
 * External I/O of the EPContext wrapper: "pixel"[1,1,224,224]f32 → "out"[1,25,96]f32.
 */
class QnnVitstrRecognizer(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val lut = FloatArray(256) { v -> (v / 255f - 0.5f) / 0.5f }

    init {
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(extractAsset(context, ASSET).absolutePath, opts)
        Log.i(TAG, "ViTSTR-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms")
    }

    /** Recognize one text-line [crop] → (text, mean winner softmax-prob [0..1]). */
    fun recognize(crop: Bitmap): Pair<String, Float> {
        val rs = if (crop.width == S && crop.height == S) crop
                 else Bitmap.createScaledBitmap(crop, S, S, true)
        val tensor = OnnxTensor.createTensor(env, toTensor(rs), longArrayOf(1, 1, S.toLong(), S.toLong()))
        if (rs !== crop) rs.recycle()
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("pixel" to it)) }
        val ms = System.currentTimeMillis() - t0
        val out = results.use { r ->
            @Suppress("UNCHECKED_CAST")
            val logits = (r.get("out").get().value as Array<Array<FloatArray>>)[0]   // [25][96]
            decode(logits)
        }
        Log.i(TAG, "ViTSTR-QNN run=${ms}ms -> \"${out.first}\"")
        return out
    }

    // grayscale (ITU-R 601 luma, matching PIL "L"), single-channel NCHW, (x/255-0.5)/0.5
    private fun toTensor(bmp: Bitmap): FloatBuffer {
        val px = IntArray(S * S).also { bmp.getPixels(it, 0, S, 0, 0, S, S) }
        val arr = FloatArray(S * S)
        for (i in px.indices) {
            val p = px[i]
            val y = ((p shr 16 and 0xFF) * 299 + (p shr 8 and 0xFF) * 587 + (p and 0xFF) * 114) / 1000
            arr[i] = lut[y]
        }
        return FloatBuffer.wrap(arr)
    }

    private fun decode(logits: Array<FloatArray>): Pair<String, Float> {
        val sb = StringBuilder(); var cSum = 0f; var cN = 0
        for (pos in 1 until logits.size) {                 // skip pos0 = [GO]
            val row = logits[pos]
            var best = 0; var bv = row[0]
            for (k in 1 until row.size) if (row[k] > bv) { bv = row[k]; best = k }
            if (best == 1) break                           // [s] end token
            if (best >= 2) {
                sb.append(VOCAB[best])
                var sum = 0f; for (v in row) sum += exp(v - bv)   // winner softmax prob = 1/Σexp(v-bv)
                cSum += 1f / sum; cN++
            }
        }
        return sb.toString() to if (cN > 0) cSum / cN else 0f
    }

    override fun close() { session.close() }

    companion object {
        private const val TAG = "VitstrQnn"
        private const val ASSET = "ocr/vitstr-epctx.onnx"
        private const val S = 224
        // ViTSTR vocab: idx0=[GO], idx1=[s], idx≥2 → printable char (94: digits, lower, upper, symbols)
        private val BASE = "0123456789abcdefghijklmnopqrstuvwxyz" +
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                           "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        private val VOCAB: List<String> = listOf("[GO]", "[s]") + BASE.map { it.toString() }
    }
}
