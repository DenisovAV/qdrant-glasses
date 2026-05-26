package tech.qdrant.glasses.camera

import android.content.Context
import android.graphics.Bitmap
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
    private val executor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var lastFrameTimeMs = 0L
    private val frameIntervalMs = 1000L

    private var lastFramePixels: IntArray? = null
    private val ssimSize = 32
    private val similarityThreshold = 0.85f

    fun start(lifecycleOwner: LifecycleOwner) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            cameraProvider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(executor) { proxy -> analyzeFrame(proxy) }
            cameraProvider?.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        try {
            val nowMs = System.currentTimeMillis()
            if (nowMs - lastFrameTimeMs < frameIntervalMs) return
            val bitmap = proxy.toBitmap()
            if (isTooSimilar(bitmap)) return
            lastFrameTimeMs = nowMs
            onFrame(bitmap)
        } finally {
            proxy.close()
        }
    }

    private fun isTooSimilar(bitmap: Bitmap): Boolean {
        val small = Bitmap.createScaledBitmap(bitmap, ssimSize, ssimSize, false)
        val pixels = IntArray(ssimSize * ssimSize).also {
            small.getPixels(it, 0, ssimSize, 0, 0, ssimSize, ssimSize)
        }
        val prev = lastFramePixels
        lastFramePixels = pixels
        if (prev == null) return false

        var diffSum = 0L
        for (i in pixels.indices) {
            val lum1 = luminance(prev[i])
            val lum2 = luminance(pixels[i])
            diffSum += abs(lum1 - lum2)
        }
        val similarity = 1f - (diffSum.toFloat() / pixels.size / 255f)
        return similarity > similarityThreshold
    }

    private fun luminance(argb: Int): Int {
        val r = argb shr 16 and 0xFF
        val g = argb shr  8 and 0xFF
        val b = argb        and 0xFF
        return (0.299 * r + 0.587 * g + 0.114 * b).toInt()
    }
}
