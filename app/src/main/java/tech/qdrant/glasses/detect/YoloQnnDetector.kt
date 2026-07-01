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
    // Pre-allocated dequant outputs — reused every frame instead of allocating 8400 FloatArray(4)
    // objects per call (that was heavy GC churn at ~9 detect/sec). Mirrors YoloDetector's approach.
    private val boxesF = Array(1) { Array(N) { FloatArray(4) } }
    private val scoresF = Array(1) { FloatArray(N) }
    // Per-inference latency goes to a FILE, not logcat: the RayNeo camera floods logcat
    // (IS_ALGO) and evicts our lines before they can be read. The file survives that, the
    // recording hang, and buffer rotation — just `adb pull` / `run-as cat` it afterwards.
    private val latencyFile = File(context.filesDir, "npu_latency.csv")
    private var nInfer = 0
    private var sumInfer = 0L

    init {
        latencyFile.appendText("=== session ${System.currentTimeMillis()} backend=QNN_HTP ===\n")
        val ctxCache = File(context.filesDir, "yolov8_det_ctx.onnx")
        val cached = ctxCache.exists()
        val modelPath = if (cached) {
            // Second+ launch: load the already-compiled EPContext model directly (it carries the
            // SoC-correct HTP context; ORT errors if you ask it to RE-generate over an existing one).
            ctxCache.absolutePath
        } else {
            // First launch: extract the SoC-agnostic QDQ model (+ its external .data weights) and
            // let ORT compile the HTP context for THIS SoC, caching it to ctxCache.
            extractAsset(context, "$DIR/$DATA")
            extractAsset(context, "$DIR/$ONNX").absolutePath
        }
        val opts = OrtSession.SessionOptions().apply {
            // QNN HTP backend. backend_path resolves libQnnHtp.so from the app's native lib dir
            // (qnn-runtime 2.45, incl. the V73 HTP skel for the AR1).
            addQnn(mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
            ))
            if (!cached) {
                // Generate + cache the on-device context only when we don't have one yet.
                addConfigEntry("ep.context_enable", "1")
                addConfigEntry("ep.context_file_path", ctxCache.absolutePath)
            }
        }
        val t0 = System.currentTimeMillis()
        session = env.createSession(modelPath, opts)
        Log.i(TAG, "YOLO-QNN session created (cached=$cached) in ${System.currentTimeMillis() - t0}ms")
    }

    override fun detect(bitmap: Bitmap): List<Detection> {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        // NCHW uint8: the qai-hub QDQ graph expects [1,3,640,640] (channel-planar), raw 0..255
        // (the graph bakes /255). Fill three contiguous planes R,G,B — NOT interleaved RGB.
        val px = IntArray(SIZE * SIZE).also { resized.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
        val plane = SIZE * SIZE
        val buf = ByteBuffer.allocateDirect(3 * plane)
        for (p in px) buf.put((p shr 16 and 0xFF).toByte())   // R plane
        for (p in px) buf.put((p shr 8 and 0xFF).toByte())    // G plane
        for (p in px) buf.put((p and 0xFF).toByte())          // B plane
        buf.rewind()
        val input = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()), OnnxJavaType.UINT8)
        val t0 = System.currentTimeMillis()
        val out = input.use { session.run(mapOf("image" to it)) }
        val ms = System.currentTimeMillis() - t0
        // Persist to file (logcat-proof) with a running summary every 10 frames.
        nInfer++; sumInfer += ms
        latencyFile.appendText("$ms\n")
        if (nInfer % 10 == 0) latencyFile.appendText("# n=$nInfer mean=${sumInfer / nInfer}ms (last=$ms)\n")
        Log.i(TAG, "YOLO-QNN inference=${ms}ms")

        return out.use { r ->
            // Dequantize uint8 outputs: real = (u8 - zero_point) * scale.
            val boxesU8 = (r.get("boxes").get().value as Array<Array<ByteArray>>)[0]      // [8400][4]
            val scoresU8 = (r.get("scores").get().value as Array<ByteArray>)[0]           // [8400]
            val clsU8 = (r.get("class_idx").get().value as Array<ByteArray>)[0]           // [8400]
            // Fill pre-allocated arrays (no per-frame heap churn).
            for (i in 0 until N) {
                val b = boxesU8[i]
                val bf = boxesF[0][i]
                bf[0] = ((b[0].toInt() and 0xFF) - BOX_ZP) * BOX_SCALE
                bf[1] = ((b[1].toInt() and 0xFF) - BOX_ZP) * BOX_SCALE
                bf[2] = ((b[2].toInt() and 0xFF) - BOX_ZP) * BOX_SCALE
                bf[3] = ((b[3].toInt() and 0xFF) - BOX_ZP) * BOX_SCALE
                scoresF[0][i] = (scoresU8[i].toInt() and 0xFF) * SCORE_SCALE
                classInt[i] = clsU8[i].toInt() and 0xFF
            }
            decoder.decode(boxesF, scoresF, classInt, modelSize = SIZE, frameW = bitmap.width, frameH = bitmap.height)
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
