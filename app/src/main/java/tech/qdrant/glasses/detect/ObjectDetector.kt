package tech.qdrant.glasses.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import tech.qdrant.glasses.embedding.Accelerator

/** One detected object in the analyzed bitmap. bbox is in that bitmap's PIXEL coordinates. */
data class Detection(val bbox: RectF, val label: String, val score: Float)

/** Detects objects in a single frame. */
interface ObjectDetector : AutoCloseable {
    fun detect(bitmap: Bitmap): List<Detection>
}

/**
 * Selects the detector backend (mirrors EncoderFactory).
 *  - MEDIAPIPE: EfficientDet-Lite (COCO-80), Adreno GPU. Default.
 *  - YOLO:      YOLOv8n float, Adreno GPU (~113ms, beats EfficientDet-GPU).
 *  - YOLO_NPU:  YOLOv8n int8 (w8a8, qai-hub) on the Hexagon NPU — the experiment to beat GPU.
 *               int8 quantization made the qai-hub profile place ALL 282 ops on NPU @1.59ms (S22);
 *               whether it fits the AR1's smaller 1.5MB TCM is what this backend measures on-device.
 */
object DetectorFactory {
    enum class Backend { MEDIAPIPE, YOLO, YOLO_NPU }
    val backend = Backend.MEDIAPIPE        // default MediaPipe-GPU; YOLO=GPU; YOLO_NPU=int8/HTP (97-partition split via LiteRt delegate — needs ORT QNN EP for a single context, WIP)
    fun create(context: Context): ObjectDetector = when (backend) {
        Backend.MEDIAPIPE -> MediaPipeDetector(context)
        Backend.YOLO -> YoloDetector(context)
        Backend.YOLO_NPU -> YoloDetector(context, asset = "detect/yolov8n_int8.tflite", accelerator = Accelerator.NPU, npuQuantized = true)
    }
}
