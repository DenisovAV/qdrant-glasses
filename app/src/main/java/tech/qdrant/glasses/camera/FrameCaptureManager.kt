package tech.qdrant.glasses.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executors
import kotlin.math.abs

class FrameCaptureManager(
    private val context: Context,
    private val onFrame: (Bitmap) -> Unit
) {
    companion object { private const val TAG = "FrameCapture" }

    private val executor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var lastFrameTimeMs = 0L
    private val frameIntervalMs = 1000L
    private val forceSaveIntervalMs = 2000L

    private var lastFramePixels: IntArray? = null
    private val ssimSize = 32
    private val similarityThreshold = 0.85f

    private var framesAnalyzed = 0
    private var framesSent = 0
    private var framesSkippedTime = 0
    private var framesSkippedSimilar = 0

    fun start(lifecycleOwner: LifecycleOwner) {
        Log.i(TAG, "start: binding camera")
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(executor) { proxy -> analyzeFrame(proxy) }
            cameraProvider?.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            Log.i(TAG, "start: camera bound OK")
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        Log.i(TAG, "stop: analyzed=$framesAnalyzed sent=$framesSent skippedTime=$framesSkippedTime skippedSimilar=$framesSkippedSimilar")
        cameraProvider?.unbindAll()
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        try {
            framesAnalyzed++
            val nowMs = System.currentTimeMillis()
            if (nowMs - lastFrameTimeMs < frameIntervalMs) {
                framesSkippedTime++
                return
            }
            val raw = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            if (framesAnalyzed == 1) Log.i(TAG, "camera: raw=${raw.width}x${raw.height} rotation=$rotation")
            val bitmap = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
            } else raw
            val similarity = computeSimilarity(bitmap)
            val forced = (nowMs - lastFrameTimeMs) >= forceSaveIntervalMs
            if (similarity > similarityThreshold && !forced) {
                framesSkippedSimilar++
                Log.v(TAG, "frame skipped: similarity=%.2f".format(similarity))
                return
            }
            lastFrameTimeMs = nowMs
            framesSent++
            Log.d(TAG, "frame accepted: similarity=%.2f forced=$forced sent=$framesSent".format(similarity))
            onFrame(bitmap)
        } finally {
            proxy.close()
        }
    }

    private fun computeSimilarity(bitmap: Bitmap): Float {
        val small = Bitmap.createScaledBitmap(bitmap, ssimSize, ssimSize, false)
        val pixels = IntArray(ssimSize * ssimSize).also {
            small.getPixels(it, 0, ssimSize, 0, 0, ssimSize, ssimSize)
        }
        val prev = lastFramePixels
        lastFramePixels = pixels
        if (prev == null) return 0f

        var diffSum = 0L
        for (i in pixels.indices) {
            val lum1 = luminance(prev[i])
            val lum2 = luminance(pixels[i])
            diffSum += abs(lum1 - lum2)
        }
        return 1f - (diffSum.toFloat() / pixels.size / 255f)
    }

    private fun luminance(argb: Int): Int {
        val r = argb shr 16 and 0xFF
        val g = argb shr  8 and 0xFF
        val b = argb        and 0xFF
        return (0.299 * r + 0.587 * g + 0.114 * b).toInt()
    }
}
