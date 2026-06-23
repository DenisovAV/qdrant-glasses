package tech.qdrant.glasses.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector as MpObjectDetector

/**
 * EfficientDet-Lite0 (COCO-80) via MediaPipe Tasks. The Tasks API does decoding + NMS
 * internally, so detect() returns final boxes. Synchronous IMAGE mode — we feed it the
 * same bitmap the analysis loop already has.
 */
class MediaPipeDetector(context: Context) : ObjectDetector {
    private val detector: MpObjectDetector

    init {
        val base = BaseOptions.builder()
            .setModelAssetPath("detect/efficientdet_lite0.tflite")
            .build()
        val options = MpObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setScoreThreshold(0.4f)
            .setMaxResults(10)
            .build()
        detector = MpObjectDetector.createFromOptions(context, options)
        Log.i(TAG, "MediaPipe detector ready")
    }

    override fun detect(bitmap: Bitmap): List<Detection> =
        // MPImage is Closeable; close it per frame so native buffers don't accumulate
        // (detect() is called once per camera frame).
        BitmapImageBuilder(bitmap).build().use { mpImage ->
            detector.detect(mpImage).detections().mapNotNull { d ->
                val cat = d.categories().firstOrNull() ?: return@mapNotNull null
                val b = d.boundingBox()
                Detection(
                    bbox = RectF(b.left, b.top, b.right, b.bottom),  // already pixel coords
                    label = cat.categoryName(),
                    score = cat.score(),
                )
            }
        }

    override fun close() = detector.close()

    companion object { private const val TAG = "MediaPipeDetector" }
}
