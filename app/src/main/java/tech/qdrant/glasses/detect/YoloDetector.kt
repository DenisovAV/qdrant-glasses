package tech.qdrant.glasses.detect

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import tech.qdrant.glasses.embedding.Accelerator
import tech.qdrant.glasses.embedding.LiteRtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLOv8n object detector on the Adreno GPU via LiteRtSession. (The NPU was benchmarked and
 * does NOT accelerate this model on the AR1 — the firmware's QNN driver fragments the graph,
 * op-validation 3110; GPU @113ms is the fastest working path and beats EfficientDet-GPU @149ms.)
 * Input [1,640,640,3] f32 NHWC /255. Outputs: boxes[1,8400,4] f32 (xyxy 640-grid),
 * scores[1,8400] f32, class_idx[1,8400] u8. Decoded by YoloDecoder.
 */
class YoloDetector(
    context: Context,
    asset: String = ASSET,
    accelerator: Accelerator = Accelerator.GPU,
    npuQuantized: Boolean = false,
) : ObjectDetector {
    private val session = LiteRtSession(context, asset, accelerator, npuQuantized = npuQuantized)
        .also { Log.i(TAG, "YOLO detector: asset=$asset accelerator=$accelerator quantized=$npuQuantized") }
    private val decoder = YoloDecoder()
    private val inputBuf = ByteBuffer.allocateDirect(1 * SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder())
    private val boxes = Array(1) { Array(N) { FloatArray(4) } }
    private val scores = Array(1) { FloatArray(N) }
    private val clsIdx = Array(1) { ByteArray(N) }              // uint8 output
    private val classInt = IntArray(N)

    override fun detect(bitmap: Bitmap): List<Detection> {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        val px = IntArray(SIZE * SIZE).also { resized.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
        if (resized !== bitmap) resized.recycle()   // pixels copied into `px`; free the scaled copy
        val fb = inputBuf.asFloatBuffer()
        for (i in px.indices) { val p = px[i]; val o = i * 3
            fb.put(o, (p shr 16 and 0xFF) / 255f); fb.put(o + 1, (p shr 8 and 0xFF) / 255f); fb.put(o + 2, (p and 0xFF) / 255f) }
        inputBuf.rewind()
        val outs = HashMap<Int, Any>().apply { put(0, boxes); put(1, scores); put(2, clsIdx) }
        session.interpreter.runForMultipleInputsOutputs(arrayOf(inputBuf), outs)
        // uint8 class idx → Int (0..79); Kotlin Byte is signed, so & 0xFF.
        for (i in 0 until N) classInt[i] = clsIdx[0][i].toInt() and 0xFF
        return decoder.decode(boxes, scores, classInt, modelSize = SIZE, frameW = bitmap.width, frameH = bitmap.height)
    }

    override fun close() = session.close()

    companion object {
        private const val TAG = "YoloDetector"
        private const val ASSET = "detect/yolov8n.tflite"
        private const val SIZE = 640
        private const val N = 8400
    }
}
