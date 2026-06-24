package tech.qdrant.glasses.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF

/** One detected object in the analyzed bitmap. bbox is in that bitmap's PIXEL coordinates. */
data class Detection(val bbox: RectF, val label: String, val score: Float)

/** Detects objects in a single frame. */
interface ObjectDetector : AutoCloseable {
    fun detect(bitmap: Bitmap): List<Detection>
}

/**
 * Selects the detector backend (mirrors EncoderFactory). MEDIAPIPE = EfficientDet-Lite
 * (COCO-80) on-device. Later: MLKIT, YOLO_QNN.
 */
object DetectorFactory {
    enum class Backend { MEDIAPIPE, YOLO }
    val backend = Backend.MEDIAPIPE        // default stays MediaPipe-GPU; flip to YOLO to use the YOLOv8n-GPU detector
    fun create(context: Context): ObjectDetector = when (backend) {
        Backend.MEDIAPIPE -> MediaPipeDetector(context)
        Backend.YOLO -> YoloDetector(context)
    }
}
