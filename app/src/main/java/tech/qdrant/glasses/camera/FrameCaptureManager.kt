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
    // Object mode: deliver every analyzed frame (still ~2/sec via analyzeIntervalMs), skipping
    // the 3s interval gate and the SSIM dedup — object tracking needs frame continuity, and
    // near-identical frames are exactly what keeps a track alive between distinct objects.
    private val passthrough: Boolean = false,
    private val onFrame: (Bitmap) -> Unit
) {
    companion object {
        private const val TAG = "FrameCapture"
        // Max edge of the delivered frame. RayNeo ignores setTargetResolution and sends the full
        // sensor (~1728x2304); downscale to this so detection/stream/crops stay fast. ~960 keeps
        // enough detail for detection + a crisp browser stream while the JPEG stays tens of KB.
        private const val MAX_EDGE = 960
    }

    private val executor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private var lastFrameTimeMs = 0L
    private val frameIntervalMs = 3000L
    private val forceSaveIntervalMs = 12000L
    // Once past the interval gate, don't pay toBitmap+scale at full camera fps while
    // Sampling cap. Was 150ms (~6 FPS) when detection cost ~149ms on the GPU. With the NPU
    // detector at ~8ms, detect+draw+compress is ~30ms, so 50ms (~20 FPS) gives a smooth stream
    // without starving the inferLane.
    private var lastAnalysisMs = 0L
    private val analyzeIntervalMs = 50L

    // Baseline for dedupe = the last ACCEPTED frame. Comparing against the last
    // ANALYZED frame degenerates to frame-vs-33ms-ago (always similar), so a gradual
    // scene change would never trigger an accept until the force interval.
    private var lastAcceptedPixels: IntArray? = null
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

    /**
     * Start a fresh recording: drop the dedup baseline and timers so the FIRST frame of the
     * session is always accepted. The camera runs continuously from app launch, so without
     * this the first recorded frame is deduped against a frame captured BEFORE recording (the
     * same static scene), leaving the opening seconds — and any speech then — with no frame.
     */
    fun resetForNewSession() {
        lastAcceptedPixels = null   // next frame can't be "similar to last" → forced accept
        lastFrameTimeMs = 0L        // no interval gate against a pre-recording frame
        lastAnalysisMs = 0L
        Log.i(TAG, "resetForNewSession: dedup baseline cleared")
    }

    private fun analyzeFrame(proxy: ImageProxy) {
        try {
            framesAnalyzed++
            val nowMs = System.currentTimeMillis()
            // The 3s interval gate is for the legacy CLIP path; passthrough delivers far more
            // frequently (capped only by analyzeIntervalMs below) so tracking stays continuous.
            if (!passthrough && nowMs - lastFrameTimeMs < frameIntervalMs) {
                framesSkippedTime++
                return
            }
            if (nowMs - lastAnalysisMs < analyzeIntervalMs) {
                framesSkippedTime++
                return
            }
            lastAnalysisMs = nowMs
            val raw = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            if (framesAnalyzed == 1) Log.i(TAG, "camera: raw=${raw.width}x${raw.height} rotation=$rotation passthrough=$passthrough")
            // rotate creates a NEW bitmap; recycle the source raw once we have the rotated copy.
            val rotated = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true).also { raw.recycle() }
            } else raw
            // setTargetResolution(640x480) is only a HINT — RayNeo's camera ignores it and
            // delivers the full sensor (~1728x2304). That huge frame makes detection slow
            // (resize cost), the MJPEG stream choke (a ~MB JPEG 2x/sec overflows the pipe →
            // blank browser), and crops oversized. Downscale to a max edge of MAX_EDGE here so
            // the whole pipeline below works at a sane size while keeping the aspect ratio.
            // scale creates a NEW bitmap; recycle the (intermediate) rotated once scaled. These
            // intermediates would otherwise leak ~31MB/frame at full sensor res → OOM in ~1 min.
            val maxEdge = maxOf(rotated.width, rotated.height)
            val bitmap = if (maxEdge > MAX_EDGE) {
                val s = MAX_EDGE.toFloat() / maxEdge
                Bitmap.createScaledBitmap(rotated, (rotated.width * s).toInt(), (rotated.height * s).toInt(), true)
                    .also { if (it !== rotated) rotated.recycle() }
            } else rotated
            if (!passthrough) {
                val pixels = downscalePixels(bitmap)
                val similarity = lastAcceptedPixels?.let { similarityBetween(it, pixels) } ?: 0f
                val forced = (nowMs - lastFrameTimeMs) >= forceSaveIntervalMs
                if (similarity > similarityThreshold && !forced) {
                    framesSkippedSimilar++
                    Log.v(TAG, "frame skipped: similarity=%.2f".format(similarity))
                    return
                }
                lastAcceptedPixels = pixels  // baseline moves ONLY on accept
                Log.d(TAG, "frame accepted: similarity=%.2f forced=$forced sent=${framesSent + 1}".format(similarity))
            }
            lastFrameTimeMs = nowMs
            framesSent++
            onFrame(bitmap)
        } finally {
            proxy.close()
        }
    }

    private fun downscalePixels(bitmap: Bitmap): IntArray {
        val small = Bitmap.createScaledBitmap(bitmap, ssimSize, ssimSize, false)
        return IntArray(ssimSize * ssimSize).also {
            small.getPixels(it, 0, ssimSize, 0, 0, ssimSize, ssimSize)
        }
    }

    private fun similarityBetween(prev: IntArray, pixels: IntArray): Float {
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
