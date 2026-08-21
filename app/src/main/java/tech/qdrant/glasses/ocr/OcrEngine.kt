package tech.qdrant.glasses.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import java.nio.FloatBuffer

/** One recognized text line: the [text], its box in ORIGINAL-image pixel coords, and mean CTC conf. */
data class OcrLine(val text: String, val box: RectF, val conf: Float)

/**
 * On-device OCR "read" channel (Stage 3) — PP-OCRv3-mobile via ONNX Runtime, fully local (no relay,
 * no Play Services). Two stages, mirroring the proven [tech.qdrant.glasses.embedding] encoders:
 *  - DET (DBNet, `ocr/ch_PP-OCRv3_det_infer.onnx`): whole frame → per-pixel text-probability map.
 *    Horizontal-projection line segmentation turns that into line boxes. This stage is the "is there
 *    text?" gate — zero boxes ⇒ no recognition runs. (Feasible on the Hexagon HTP at ~10.6ms — see
 *    the DBNet-on-NPU spike — but this class runs it on the CPU; the NPU path is a later optimization.)
 *  - REC (CRNN, `ocr/ch_PP-OCRv3_rec_infer.onnx`): each line crop → per-timestep char logits →
 *    CTC-decoded against `ocr/ppocr_keys.txt` (6625 entries, index→char, 0=blank, last=space).
 *
 * Must be called OFF the main thread (ORT inference). Intended to run in the BACKGROUND on a stored
 * keyframe, not in the live capture hot loop. Reuses the calibrated preprocessing from OcrSpikeTest.
 */
class OcrEngine(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val det: OrtSession
    private val rec: OrtSession
    private val chars: List<String>

    init {
        val a = context.assets
        det = a.open("ocr/ch_PP-OCRv3_det_infer.onnx").use { env.createSession(it.readBytes(), OrtSession.SessionOptions()) }
        rec = a.open("ocr/ch_PP-OCRv3_rec_infer.onnx").use { env.createSession(it.readBytes(), OrtSession.SessionOptions()) }
        chars = a.open("ocr/ppocr_keys.txt").bufferedReader().use { it.readLines() }
        Log.i(TAG, "OCR engine ready (dict=${chars.size})")
    }

    /** Detect + recognize all text lines in [bmp]. Empty list if no text (the DET gate found none). */
    fun recognize(bmp: Bitmap): List<OcrLine> {
        val t0 = System.currentTimeMillis()
        val (prob, sx, sy) = runDet(bmp)
        val boxes = lineBoxes(prob)
        if (boxes.isEmpty()) return emptyList()
        val out = ArrayList<OcrLine>(boxes.size)
        for (b in boxes) {
            val ox0 = (b[0] / sx).toInt().coerceIn(0, bmp.width - 1)
            val oy0 = (b[1] / sy).toInt().coerceIn(0, bmp.height - 1)
            val ox1 = (b[2] / sx).toInt().coerceIn(ox0 + 1, bmp.width)
            val oy1 = (b[3] / sy).toInt().coerceIn(oy0 + 1, bmp.height)
            val crop = Bitmap.createBitmap(bmp, ox0, oy0, ox1 - ox0, oy1 - oy0)
            val (txt, conf) = runRec(crop)
            if (txt.isNotBlank()) out += OcrLine(txt, RectF(ox0.toFloat(), oy0.toFloat(), ox1.toFloat(), oy1.toFloat()), conf)
        }
        Log.i(TAG, "OCR ${System.currentTimeMillis() - t0}ms -> ${out.size} lines")
        return out
    }

    // ---- DET: resize /32 (≤960), ImageNet-norm NCHW -> prob map [H][W] + scale-to-original ----
    private fun runDet(bmp: Bitmap): Triple<Array<FloatArray>, Float, Float> {
        val scale = minOf(1f, DET_MAX_SIDE.toFloat() / maxOf(bmp.width, bmp.height))
        val w = (((bmp.width * scale).toInt() + 31) / 32 * 32).coerceAtLeast(32)
        val h = (((bmp.height * scale).toInt() + 31) / 32 * 32).coerceAtLeast(32)
        val rs = Bitmap.createScaledBitmap(bmp, w, h, true)
        val px = IntArray(w * h); rs.getPixels(px, 0, w, 0, 0, w, h)
        val buf = FloatBuffer.allocate(3 * h * w); val arr = buf.array()
        for (c in 0 until 3) for (i in 0 until h * w) {
            val p = px[i]; val v = when (c) { 0 -> (p shr 16) and 0xFF; 1 -> (p shr 8) and 0xFF; else -> p and 0xFF }
            arr[c * h * w + i] = (v / 255f - DET_MEAN[c]) / DET_STD[c]
        }
        val t = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong()))
        val r = t.use { det.run(mapOf(det.inputNames.first() to it)) }
        @Suppress("UNCHECKED_CAST")
        val map = (r.get(0).value as Array<Array<Array<FloatArray>>>)[0][0]
        r.close()
        return Triple(map, w.toFloat() / bmp.width, h.toFloat() / bmp.height)
    }

    // ---- horizontal-projection line boxes (upright text; whole line per box) ----
    private fun lineBoxes(prob: Array<FloatArray>): List<IntArray> {
        val h = prob.size; val w = prob[0].size
        val minCols = maxOf(3, (w * 0.005f).toInt())
        val rowHas = IntArray(h) { y -> var c = 0; for (x in 0 until w) if (prob[y][x] > BIN_THRESH) c++; c }
        val boxes = ArrayList<IntArray>()
        var y = 0
        while (y < h) {
            if (rowHas[y] < minCols) { y++; continue }
            var y1 = y; while (y1 < h && rowHas[y1] >= minCols) y1++
            if (y1 - y >= 6) {
                var x0 = w; var x1 = 0
                for (yy in y until y1) for (x in 0 until w) if (prob[yy][x] > BIN_THRESH) { if (x < x0) x0 = x; if (x > x1) x1 = x }
                boxes.add(intArrayOf((x0 - PAD).coerceAtLeast(0), (y - PAD).coerceAtLeast(0), (x1 + PAD).coerceAtMost(w), (y1 + PAD).coerceAtMost(h)))
            }
            y = y1
        }
        return boxes
    }

    // ---- REC: resize H=48, (x/255-0.5)/0.5, CRNN, CTC decode ----
    private fun runRec(crop: Bitmap): Pair<String, Float> {
        val w = (REC_H * crop.width.toFloat() / crop.height).toInt().coerceIn(16, 1600)
        val rs = Bitmap.createScaledBitmap(crop, w, REC_H, true)
        val px = IntArray(w * REC_H); rs.getPixels(px, 0, w, 0, 0, w, REC_H)
        val buf = FloatBuffer.allocate(3 * REC_H * w); val arr = buf.array()
        for (c in 0 until 3) for (i in 0 until REC_H * w) {
            val p = px[i]; val v = when (c) { 0 -> (p shr 16) and 0xFF; 1 -> (p shr 8) and 0xFF; else -> p and 0xFF }
            arr[c * REC_H * w + i] = (v / 255f - 0.5f) / 0.5f
        }
        val t = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, REC_H.toLong(), w.toLong()))
        val r = t.use { rec.run(mapOf(rec.inputNames.first() to it)) }
        @Suppress("UNCHECKED_CAST")
        val logits = (r.get(0).value as Array<Array<FloatArray>>)[0]   // [T][6625]
        r.close()
        val sb = StringBuilder(); var prev = -1; var cSum = 0f; var cN = 0
        for (tt in logits.indices) {
            var best = 0; var bestV = logits[tt][0]
            for (k in 1 until logits[tt].size) if (logits[tt][k] > bestV) { bestV = logits[tt][k]; best = k }
            if (best != 0 && best != prev) { sb.append(chars.getOrElse(best) { "" }); cSum += bestV; cN++ }
            prev = best
        }
        return sb.toString() to if (cN > 0) cSum / cN else 0f
    }

    override fun close() { det.close(); rec.close() }

    companion object {
        private const val TAG = "OcrEngine"
        private val DET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val DET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        // 1536 (was 960): the detector downscales the frame's long side to this before running. At
        // 960 small/far text (a laptop screen at arm's length — the screen is ~¼ of the frame, its
        // text a few px tall) shrank below DBNet's resolution and detected 0 lines. 1536 keeps ~2.5×
        // the pixels so small text survives; costs ~5s/frame on CPU (it's background-only, per-keyframe).
        private const val DET_MAX_SIDE = 1536
        private const val BIN_THRESH = 0.3f
        private const val REC_H = 48
        private const val PAD = 4
    }
}
