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
    // Invoked if the camera fails to bind (future.get()/bindToLifecycle throw on the main
    // executor). Default no-op; MainActivity wires it to a fatal error state.
    private val onError: (Throwable) -> Unit = {},
    private val onFrame: (Bitmap) -> Unit
) {
    companion object {
        private const val TAG = "FrameCapture"
        // Max edge of the delivered frame. RayNeo ignores setTargetResolution and sends the full
        // sensor (~1728x2304); downscale to this so detection/stream/crops stay fast. ~960 keeps
        // enough detail for detection + a crisp browser stream while the JPEG stays tens of KB.
        private const val MAX_EDGE = 960
        // Rotation applied to the sensor frame. The sensor is 640x480 landscape; the demo wants
        // landscape on the projector, so we do NOT honor the sensor's 90° rotation. 0 = native
        // landscape; set 180 if the image is upside-down.
        private const val STREAM_ROTATION = 90
    }

    private val executor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    // Kept so we can unbind/rebind the SAME analysis use case (see setActive).
    private var lifecycleOwner: LifecycleOwner? = null
    private var analysis: ImageAnalysis? = null
    private val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    private var lastFrameTimeMs = 0L
    private val frameIntervalMs = 3000L
    private val forceSaveIntervalMs = 12000L
    // Once past the interval gate, don't pay toBitmap+scale at full camera fps while
    // Sampling cap. Was 150ms (~6 FPS) when detection cost ~149ms on the GPU. With the NPU
    // detector at ~8ms, detect+draw+compress is ~30ms, so 33ms (~30 FPS) gives a smooth stream
    // without starving the inferLane.
    private var lastAnalysisMs = 0L
    private val analyzeIntervalMs = 33L

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
        this.lifecycleOwner = lifecycleOwner
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
          try {
            cameraProvider = future.get()
            // Ask the camera for a SMALL resolution via the modern ResolutionSelector (the
            // deprecated setTargetResolution was ignored by RayNeo → full 4MP sensor, whose
            // toBitmap+rotate+scale is the pipeline bottleneck). ResolutionStrategy with
            // FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER lets camera2 pick the nearest supported size
            // ≤640×480 if it honors it; if RayNeo still forces 4MP, analyzeFrame's downscale still
            // handles it (just slower). Measured via the `camera: raw=` log.
            val resSelector = androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                .setResolutionStrategy(
                    androidx.camera.core.resolutionselector.ResolutionStrategy(
                        Size(640, 480),
                        androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                ).build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(executor) { proxy -> analyzeFrame(proxy) }
            this.analysis = analysis
            cameraProvider?.bindToLifecycle(lifecycleOwner, cameraSelector, analysis)
            Log.i(TAG, "start: camera bound OK")
          } catch (e: Throwable) {
            // future.get()/bindToLifecycle failure otherwise propagates uncaught on the main
            // executor — a crash or a silently dead camera. Surface it as a fatal error state.
            Log.e(TAG, "start: camera bind FAILED", e); onError(e)
          }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        Log.i(TAG, "stop: analyzed=$framesAnalyzed sent=$framesSent skippedTime=$framesSkippedTime skippedSimilar=$framesSkippedSimilar")
        cameraProvider?.unbindAll()
    }

    /**
     * Turn the camera stream off/on by UNBINDING the analysis use case — not just skipping our
     * frame processing. On the AR1 the camera HAL (`vendor.qti.camera.provider`) burns ~110% of a
     * core on its own while bound, on top of our ~100% for toBitmap+rotate+scale; together that's
     * ~2 of the 4 cores, which is why a query starved the CPU text encoder (~1.2s). Unbinding stops
     * the HAL too. Off during a voice query (Listening/Processing/Results), on for Idle + Recording.
     * Rebinding on resume costs ~0.5s before frames flow again — fine outside recording.
     * Idempotent; runs the bind/unbind on the main thread (CameraX requirement).
     */
    fun setActive(active: Boolean) {
        val provider = cameraProvider ?: return
        val a = analysis ?: return
        val owner = lifecycleOwner ?: return
        ContextCompat.getMainExecutor(context).execute {
            try {
                if (active && !provider.isBound(a)) {
                    provider.bindToLifecycle(owner, cameraSelector, a)
                    Log.i(TAG, "camera resumed")
                } else if (!active && provider.isBound(a)) {
                    provider.unbind(a)
                    Log.i(TAG, "camera stopped for query (frees ~2 cores)")
                }
            } catch (e: Throwable) {
                Log.e(TAG, "setActive($active) failed", e)
            }
        }
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
            // The sensor delivers 640x480 (landscape). CameraX's imageInfo.rotationDegrees=90 would
            // rotate it to 480x640 PORTRAIT — but the demo needs LANDSCAPE (wide) for the projector.
            // So we force STREAM_ROTATION instead of honoring the sensor rotation. 0 = keep the
            // native landscape frame; flip to 180 if it comes out upside-down. (The AR glasses aren't
            // a phone, so the sensor's "up" is arbitrary — pick what looks right in the browser.)
            val rotation = STREAM_ROTATION
            if (framesAnalyzed == 1) Log.i(TAG, "camera: raw=${raw.width}x${raw.height} sensorRot=${proxy.imageInfo.rotationDegrees} usedRot=$rotation passthrough=$passthrough")
            // ResolutionSelector now yields a small 640x480 frame, so rotate+scale is cheap. Fold
            // rotate + any downscale into ONE Matrix pass (was two full-frame passes on the 4MP sensor),
            // which capped the whole pipeline at ~2 FPS. Fold rotate + downscale into ONE Matrix so a
            // single createBitmap produces the small, upright frame directly — one pass, no 4MP
            // intermediate. That's the real FPS win (the sensor-size processing was the bottleneck,
            // above detect/stream). Scale factor keeps aspect ratio; source `raw` is recycled after.
            val maxEdge = maxOf(raw.width, raw.height)
            val s = if (maxEdge > MAX_EDGE) MAX_EDGE.toFloat() / maxEdge else 1f
            val bitmap = if (rotation != 0 || s != 1f) {
                val matrix = Matrix().apply {
                    if (rotation != 0) postRotate(rotation.toFloat())
                    if (s != 1f) postScale(s, s)
                }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true).also { raw.recycle() }
            } else raw
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
