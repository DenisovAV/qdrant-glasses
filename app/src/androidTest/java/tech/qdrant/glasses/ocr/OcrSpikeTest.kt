package tech.qdrant.glasses.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.FloatBuffer

/**
 * OCR FEASIBILITY SPIKE (Stage 3, fully on-device, no relay) — runs PP-OCRv3-mobile (ONNX) via ORT
 * on the glasses: DBNet text-detector → horizontal-projection line boxes → CRNN recognizer → CTC
 * decode. Answers: does our arm64 ORT run PP-OCR on the AR1, does it READ text on-device, how fast,
 * how big. NOT production wiring — a one-file spike, mirroring CropEncoderAbTest.
 *
 * Assets (`app/src/androidTest/assets/ocr/`): det/rec/cls .onnx (13.7MB total), ppocr_keys.txt
 * (6625 lines, index→char; 0=blank, last=space), and two synthetic text images.
 * Read the result: `adb logcat -d -s OCR_SPIKE:I`.
 */
@RunWith(AndroidJUnit4::class)
class OcrSpikeTest {
    private val TAG = "OCR_SPIKE"
    private val env = OrtEnvironment.getEnvironment()

    // det ImageNet normalize; rec (x/255-0.5)/0.5
    private val detMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val detStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val DET_MAX_SIDE = 960
    private val BIN_THRESH = 0.3f
    private val REC_H = 48

    @Test fun ocrOnDevice() {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val chars = assets.open("ocr/ppocr_keys.txt").bufferedReader().readLines()
        Log.i(TAG, "dict loaded: ${chars.size} chars (0='${chars[0]}' last='${chars.last()}')")

        val t0 = System.currentTimeMillis()
        val det = env.createSession(assets.open("ocr/ch_PP-OCRv3_det_infer.onnx").readBytes(), OrtSession.SessionOptions())
        val rec = env.createSession(assets.open("ocr/ch_PP-OCRv3_rec_infer.onnx").readBytes(), OrtSession.SessionOptions())
        Log.i(TAG, "ORT sessions created in ${System.currentTimeMillis() - t0}ms  (det.in=${det.inputNames} rec.in=${rec.inputNames})")

        var anyText = false
        for (name in listOf("t_whiteboard.png", "t_label.png")) {
            val bmp = BitmapFactory.decodeStream(assets.open("ocr/$name"))
            Log.i(TAG, "==== $name  ${bmp.width}x${bmp.height} ====")

            val td = System.currentTimeMillis()
            val (prob, sx, sy) = runDet(det, bmp)          // prob[H][W] in det coords + scale to orig
            val boxes = lineBoxes(prob)                    // list of (x0,y0,x1,y1) in det coords
            val detMs = System.currentTimeMillis() - td
            Log.i(TAG, "  det ${detMs}ms -> ${boxes.size} line-boxes")

            var recMsTot = 0L
            for (b in boxes) {
                val ox0 = (b[0] / sx).toInt().coerceIn(0, bmp.width - 1)
                val oy0 = (b[1] / sy).toInt().coerceIn(0, bmp.height - 1)
                val ox1 = (b[2] / sx).toInt().coerceIn(ox0 + 1, bmp.width)
                val oy1 = (b[3] / sy).toInt().coerceIn(oy0 + 1, bmp.height)
                val crop = Bitmap.createBitmap(bmp, ox0, oy0, ox1 - ox0, oy1 - oy0)
                val tr = System.currentTimeMillis()
                val (txt, conf) = runRec(rec, crop, chars)
                recMsTot += System.currentTimeMillis() - tr
                if (txt.isNotBlank()) { anyText = true; Log.i(TAG, "  %.2f  %s".format(conf, txt)) }
            }
            Log.i(TAG, "  rec ${recMsTot}ms total  |  full-image ${detMs + recMsTot}ms")
        }
        det.close(); rec.close()
        assertTrue("OCR spike produced no text on-device", anyText)
    }

    // ---- detection: preprocess (resize /32, ImageNet-norm NCHW) -> prob map [H][W] ----
    private fun runDet(sess: OrtSession, bmp: Bitmap): Triple<Array<FloatArray>, Float, Float> {
        val scale = minOf(1f, DET_MAX_SIDE.toFloat() / maxOf(bmp.width, bmp.height))
        val w = (((bmp.width * scale).toInt() + 31) / 32 * 32).coerceAtLeast(32)
        val h = (((bmp.height * scale).toInt() + 31) / 32 * 32).coerceAtLeast(32)
        val rs = Bitmap.createScaledBitmap(bmp, w, h, true)
        val px = IntArray(w * h); rs.getPixels(px, 0, w, 0, 0, w, h)
        val buf = FloatBuffer.allocate(3 * h * w)
        val arr = buf.array()
        for (c in 0 until 3) for (i in 0 until h * w) {
            val p = px[i]; val v = when (c) { 0 -> (p shr 16) and 0xFF; 1 -> (p shr 8) and 0xFF; else -> p and 0xFF }
            arr[c * h * w + i] = (v / 255f - detMean[c]) / detStd[c]
        }
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong()))
        val out = tensor.use { sess.run(mapOf(sess.inputNames.first() to it)) }
        @Suppress("UNCHECKED_CAST")
        val map = (out.get(0).value as Array<Array<Array<FloatArray>>>)[0][0]  // [h][w]
        out.close()
        return Triple(map, w.toFloat() / bmp.width, h.toFloat() / bmp.height)
    }

    // ---- horizontal-projection line segmentation (upright text; whole line per box) ----
    private fun lineBoxes(prob: Array<FloatArray>): List<IntArray> {
        val h = prob.size; val w = prob[0].size
        val rowHas = IntArray(h)
        val minCols = maxOf(3, (w * 0.005f).toInt())
        for (y in 0 until h) { var c = 0; for (x in 0 until w) if (prob[y][x] > BIN_THRESH) c++; rowHas[y] = c }
        val boxes = ArrayList<IntArray>()
        var y = 0
        while (y < h) {
            if (rowHas[y] < minCols) { y++; continue }
            var y1 = y; while (y1 < h && rowHas[y1] >= minCols) y1++
            if (y1 - y >= 6) {                                   // skip specks
                var x0 = w; var x1 = 0
                for (yy in y until y1) for (x in 0 until w) if (prob[yy][x] > BIN_THRESH) { if (x < x0) x0 = x; if (x > x1) x1 = x }
                val pad = 4
                boxes.add(intArrayOf((x0 - pad).coerceAtLeast(0), (y - pad).coerceAtLeast(0),
                    (x1 + pad).coerceAtMost(w), (y1 + pad).coerceAtMost(h)))
            }
            y = y1
        }
        return boxes
    }

    // ---- recognition: resize to H=48, (x/255-0.5)/0.5, run, CTC decode ----
    private fun runRec(sess: OrtSession, crop: Bitmap, chars: List<String>): Pair<String, Float> {
        val w = (REC_H * crop.width.toFloat() / crop.height).toInt().coerceIn(16, 1600)
        val rs = Bitmap.createScaledBitmap(crop, w, REC_H, true)
        val px = IntArray(w * REC_H); rs.getPixels(px, 0, w, 0, 0, w, REC_H)
        val buf = FloatBuffer.allocate(3 * REC_H * w); val arr = buf.array()
        for (c in 0 until 3) for (i in 0 until REC_H * w) {
            val p = px[i]; val v = when (c) { 0 -> (p shr 16) and 0xFF; 1 -> (p shr 8) and 0xFF; else -> p and 0xFF }
            arr[c * REC_H * w + i] = (v / 255f - 0.5f) / 0.5f
        }
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, REC_H.toLong(), w.toLong()))
        val out = tensor.use { sess.run(mapOf(sess.inputNames.first() to it)) }
        @Suppress("UNCHECKED_CAST")
        val logits = (out.get(0).value as Array<Array<FloatArray>>)[0]         // [T][6625]
        out.close()
        val sb = StringBuilder(); var prev = -1; var confSum = 0f; var confN = 0
        for (t in logits.indices) {
            var best = 0; var bestV = logits[t][0]
            for (k in 1 until logits[t].size) if (logits[t][k] > bestV) { bestV = logits[t][k]; best = k }
            if (best != 0 && best != prev) { sb.append(chars.getOrElse(best) { "" }); confSum += bestV; confN++ }
            prev = best
        }
        return sb.toString() to if (confN > 0) confSum / confN else 0f
    }
}
