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
import tech.qdrant.glasses.embedding.extractAsset
import java.nio.ByteBuffer
import java.nio.ByteOrder
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

    /** WHOLE OCR on the NPU: DBNet detector (EPContext) + CRNN recognizer (EPContext), both on the
     *  Hexagon HTP. Reads text end-to-end on-device and times det+rec. Read: `adb logcat -d -s
     *  OCR_SPIKE:I CrnnQnn:I DbnetQnn:I`. */
    @Test fun wholeOcrOnNpu() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val a = InstrumentationRegistry.getInstrumentation().context.assets
        val dbnet = QnnDbnetDetector(ctx)
        val crnn = QnnCrnnRecognizer(ctx)
        var anyText = false
        for (name in listOf("t_whiteboard.png", "t_label.png")) {
            val bmp = a.open("ocr/$name").use { BitmapFactory.decodeStream(it) }
            val td = System.currentTimeMillis()
            val boxes = dbnet.detect(bmp)
            val detMs = System.currentTimeMillis() - td
            Log.i(TAG, "==== NPU-OCR $name: ${boxes.size} boxes, det ${detMs}ms ====")
            var recMs = 0L
            for (b in boxes) {
                val x0 = (b.left * bmp.width).toInt().coerceIn(0, bmp.width - 1)
                val y0 = (b.top * bmp.height).toInt().coerceIn(0, bmp.height - 1)
                val x1 = (b.right * bmp.width).toInt().coerceIn(x0 + 1, bmp.width)
                val y1 = (b.bottom * bmp.height).toInt().coerceIn(y0 + 1, bmp.height)
                val crop = Bitmap.createBitmap(bmp, x0, y0, x1 - x0, y1 - y0)
                val tr = System.currentTimeMillis()
                val (txt, conf) = crnn.recognize(crop)
                recMs += System.currentTimeMillis() - tr
                if (txt.isNotBlank()) { anyText = true; Log.i(TAG, "  %.2f  %s".format(conf, txt)) }
            }
            Log.i(TAG, "  NPU-OCR total: det ${detMs}ms + rec ${recMs}ms = ${detMs + recMs}ms")
        }
        dbnet.close(); crnn.close()
        assertTrue("NPU-OCR produced no text", anyText)
    }

    /** ISOLATION: feed the CRNN-NPU the EXACT fp32 input the host model reads correctly ("MCP..."),
     *  straight into the EPContext session — NO DBNet, NO Kotlin preprocessing. If the NPU read differs
     *  from the input-invariant `块煨块8:` garbage, the binary DOES respond to input → the bug was upstream
     *  (crops/preprocess), not the binary. If it's the same garbage → the quantized binary is the fault. */
    @Test fun crnnKnownGoodInput() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val a = InstrumentationRegistry.getInstrumentation().context.assets
        val chars = a.open("ocr/ppocr_keys.txt").bufferedReader().readLines()
        val bytes = a.open("ocr/c0_fp32.raw").use { it.readBytes() }
        val arr = FloatArray(3 * 32 * 320)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(arr)
        var mn = arr[0]; var mx = arr[0]; for (v in arr) { if (v < mn) mn = v; if (v > mx) mx = v }
        Log.i(TAG, "c0 loaded: ${arr.size} floats  min=$mn max=$mx")

        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "burst"))
        }
        val sess = env.createSession(extractAsset(ctx, "ocr/crnn-rec-epctx.onnx").absolutePath, opts)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), longArrayOf(1, 3, 32, 320))
        val res = tensor.use { sess.run(mapOf("image" to it)) }
        @Suppress("UNCHECKED_CAST")
        val logits = (res.get("logits").get().value as Array<Array<FloatArray>>)[0]
        res.close(); sess.close()

        val sb = StringBuilder(); var prev = -1; var cSum = 0f; var cN = 0
        for (t in logits.indices) {
            var best = 0; var bv = logits[t][0]
            for (k in 1 until logits[t].size) if (logits[t][k] > bv) { bv = logits[t][k]; best = k }
            if (best != 0 && best != prev) { sb.append(chars.getOrElse(best) { "" }); cSum += bv; cN++ }
            prev = best
        }
        Log.i(TAG, "NPU read of KNOWN-GOOD c0 (T=${logits.size}): \"$sb\"  conf=${if (cN>0) cSum/cN else 0f}")
    }

    /** ViTSTR (ViT + CTC-ish head) recognizer on the NPU — the NON-recurrent replacement for the CRNN
     *  whose LSTM can't run on the AR1 HTP. Feeds pre-preprocessed [1,1,224,224] text-line rasters
     *  (fp32, already resized+normalized to [-1,1], from the host) through the EPContext QNN session and
     *  greedy-decodes [1,25,96] (pos0=[GO], stop at [s]=1, index>=2 -> vocab char). Read: logcat VitstrQnn. */
    @Test fun vitstrOnNpu() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val a = InstrumentationRegistry.getInstrumentation().context.assets
        val vocab = a.open("ocr/vitstr_vocab.txt").bufferedReader().use { it.readLines() }
        val opts = OrtSession.SessionOptions().apply {
            addQnn(mapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "burst"))
        }
        val t0 = System.currentTimeMillis()
        val sess = env.createSession(extractAsset(ctx, "ocr/vitstr-epctx.onnx").absolutePath, opts)
        Log.i("VitstrQnn", "ViTSTR-QNN EPContext session created in ${System.currentTimeMillis() - t0}ms (vocab=${vocab.size})")

        val cases = listOf("vt_mobile.raw" to "Mobile", "vt_hello.raw" to "Hello",
                           "vt_mcp.raw" to "MCP", "vt_2026.raw" to "2026")
        var ok = 0
        for ((file, expect) in cases) {
            val bytes = a.open("ocr/$file").use { it.readBytes() }
            val arr = FloatArray(1 * 1 * 224 * 224)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(arr)
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), longArrayOf(1, 1, 224, 224))
            val tr = System.currentTimeMillis()
            val res = tensor.use { sess.run(mapOf("pixel" to it)) }
            val ms = System.currentTimeMillis() - tr
            @Suppress("UNCHECKED_CAST")
            val logits = (res.get("out").get().value as Array<Array<FloatArray>>)[0]   // [25][96]
            res.close()
            val sb = StringBuilder()
            for (pos in 1 until logits.size) {                   // skip pos0 = [GO]
                var best = 0; var bv = logits[pos][0]
                for (k in 1 until logits[pos].size) if (logits[pos][k] > bv) { bv = logits[pos][k]; best = k }
                if (best == 1) break                             // [s] end
                if (best >= 2) sb.append(vocab.getOrElse(best) { "" })
            }
            val hit = sb.toString() == expect
            if (hit) ok++
            Log.i("VitstrQnn", "NPU ${ms}ms  read=\"$sb\"  expect=\"$expect\"  ${if (hit) "OK" else "x"}")
        }
        sess.close()
        Log.i("VitstrQnn", "ViTSTR NPU accuracy: $ok/${cases.size} exact")
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
