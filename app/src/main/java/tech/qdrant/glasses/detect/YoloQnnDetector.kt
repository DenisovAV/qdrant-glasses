package tech.qdrant.glasses.detect

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import tech.qdrant.glasses.embedding.extractAsset
import java.io.File
import java.nio.ByteBuffer

/**
 * YOLOv8n int8 detector on the Hexagon NPU via ONNX Runtime's QNN Execution Provider, compiling
 * the HTP context ON THIS DEVICE.
 *
 * A pre-compiled qai-hub context binary is SoC-locked (built for 8 Gen 1, it fails to load on the
 * AR1 with QNN error 5005). Instead we ship the SoC-agnostic QDQ-ONNX (int8 w8a8) and let ORT's
 * QNN EP do the HTP "preparation" pass for OUR SoC on first run (`ep.context_enable=1`), caching
 * the result to `<filesDir>/yolov8_det_ctx.onnx`. First launch pays a one-time compile (seconds);
 * later launches load the cached context instantly. This is the LiteRT-delegate alternative that
 * keeps the whole graph in one HTP context (no 97-partition split).
 *
 * I/O is uint8-quantized: input is raw 0..255 pixels NHWC (the graph bakes /255); outputs are
 * dequantized with the metadata scale/zero_point before decoding.
 */
class YoloQnnDetector(context: Context) : ObjectDetector {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val decoder = YoloDecoder()
    private val classInt = IntArray(N)

    init {
        val onnx = extractAsset(context, "$DIR/$ONNX")
        // The QDQ .onnx references its weights as ONNX external data (yolov8_det.data) by a path
        // relative to the .onnx, so it must be extracted alongside it.
        extractAsset(context, "$DIR/$DATA")
        val ctxCache = File(context.filesDir, "yolov8_det_ctx.onnx")
        val opts = OrtSession.SessionOptions().apply {
            // QNN HTP backend. backend_path resolves libQnnHtp.so from the app's native lib dir
            // (qnn-runtime 2.45, incl. the V73 HTP skel for the AR1).
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
            // Compile the HTP context on-device (SoC-correct) and cache it next to the model, so
            // only the first launch pays the graph-preparation cost.
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_file_path", ctxCache.absolutePath)
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(onnx.absolutePath, opts)
        Log.i(TAG, "YOLO-QNN session created (ORT QNN EP, on-device HTP compile) in ${System.currentTimeMillis() - t0}ms")
    }

    override fun detect(bitmap: Bitmap): List<Detection> {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        // NHWC uint8: raw RGB bytes, 0..255 (graph normalizes internally).
        val px = IntArray(SIZE * SIZE).also { resized.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
        val buf = ByteBuffer.allocateDirect(SIZE * SIZE * 3)
        for (p in px) {
            buf.put((p shr 16 and 0xFF).toByte()); buf.put((p shr 8 and 0xFF).toByte()); buf.put((p and 0xFF).toByte())
        }
        buf.rewind()
        val input = OnnxTensor.createTensor(env, buf, longArrayOf(1, SIZE.toLong(), SIZE.toLong(), 3), OnnxJavaType.UINT8)
        val t0 = System.currentTimeMillis()
        val out = input.use { session.run(mapOf("image" to it)) }
        Log.i(TAG, "YOLO-QNN inference=${System.currentTimeMillis() - t0}ms")

        return out.use { r ->
            // Dequantize uint8 outputs: real = (u8 - zero_point) * scale.
            val boxesU8 = (r.get("boxes").get().value as Array<Array<ByteArray>>)[0]      // [8400][4]
            val scoresU8 = (r.get("scores").get().value as Array<ByteArray>)[0]           // [8400]
            val clsU8 = (r.get("class_idx").get().value as Array<ByteArray>)[0]           // [8400]
            val boxes = Array(1) { Array(N) { i -> FloatArray(4) { j -> ((boxesU8[i][j].toInt() and 0xFF) - BOX_ZP) * BOX_SCALE } } }
            val scores = Array(1) { FloatArray(N) { i -> (scoresU8[i].toInt() and 0xFF) * SCORE_SCALE } }
            for (i in 0 until N) classInt[i] = clsU8[i].toInt() and 0xFF
            decoder.decode(boxes, scores, classInt, modelSize = SIZE, frameW = bitmap.width, frameH = bitmap.height)
        }
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        private const val TAG = "YoloQnnDetector"
        private const val DIR = "detect/qnn"
        private const val ONNX = "yolov8_det.onnx"   // QDQ-ONNX (int8, SoC-agnostic; compiled on-device)
        private const val DATA = "yolov8_det.data"   // ONNX external weights referenced by the .onnx
        private const val SIZE = 640
        private const val N = 8400
        // Quantization params from the qai-hub metadata.json (w8a8 export).
        private const val BOX_SCALE = 3.1009655f
        private const val BOX_ZP = 25
        private const val SCORE_SCALE = 0.00390625f
    }
}
