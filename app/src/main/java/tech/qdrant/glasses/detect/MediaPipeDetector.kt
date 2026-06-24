package tech.qdrant.glasses.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
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
        detector = try {
            val gpuBase = BaseOptions.builder()
                .setModelAssetPath("detect/efficientdet_lite0.tflite")
                .setDelegate(Delegate.GPU)
                .build()
            val opts = MpObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(gpuBase).setRunningMode(RunningMode.IMAGE)
                .setScoreThreshold(0.4f).setMaxResults(10).build()
            MpObjectDetector.createFromOptions(context, opts).also { Log.i(TAG, "detector ready [GPU]") }
        } catch (e: Throwable) {
            Log.w(TAG, "GPU delegate unavailable, falling back to CPU: ${e.message}")
            val cpuBase = BaseOptions.builder()
                .setModelAssetPath("detect/efficientdet_lite0.tflite")
                .setDelegate(Delegate.CPU)
                .build()
            val opts = MpObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(cpuBase).setRunningMode(RunningMode.IMAGE)
                .setScoreThreshold(0.4f).setMaxResults(10).build()
            MpObjectDetector.createFromOptions(context, opts).also { Log.i(TAG, "detector ready [CPU]") }
        }
    }

    override fun detect(bitmap: Bitmap): List<Detection> {
        // BitmapImageBuilder wraps the bitmap by reference and MPImage.close() calls
        // bitmap.recycle() on it. We must NOT hand MediaPipe our caller's bitmap, or it
        // will be recycled before drawBoxes/cropFrom can use it. Pass a dedicated mutable
        // copy that MediaPipe is free to recycle; the caller's bitmap is left untouched.
        val mpBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        return BitmapImageBuilder(mpBitmap).build().use { mpImage ->
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
    }

    override fun close() = detector.close()

    companion object { private const val TAG = "MediaPipeDetector" }
}
