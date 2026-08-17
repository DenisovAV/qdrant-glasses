package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import tech.qdrant.glasses.detect.ObjectDetector
import tech.qdrant.glasses.detect.ObjectTracker
import tech.qdrant.glasses.stream.HudPublisher

/**
 * Object-mode per-frame perception: detect → track → stream boxes → hand the frame off to the
 * memory path. Moved VERBATIM out of [tech.qdrant.glasses.GlassesViewModel.onObjectFrame] (Task 7
 * of the God-object decomposition) — do NOT change the hot-path logic here; every recycle,
 * backpressure CAS gate, and inferLane marshal is preserved exactly as verified on-device.
 *
 * **Task 2.4 (episodic-memory plan, Spec §2/§9 Stage 2):** the crop-embed → semantic-dedup →
 * `store.upsert(ObjectPayload)` block that used to close this class's detect lane is RETIRED.
 * [MomentCapture]'s whole-frame + CLIP-verified-region memory is now the ONLY write path for
 * OBJECTS-mode memory; [detector]/[tracker] here are boxes for the live stream overlay plus
 * [latestConfirmedRegions] — the region-CANDIDATE source [momentCapture] verifies against a CLIP
 * text embedding before attaching a tag (Spec §2 "CLIP-verify-the-label"). YOLO never gates
 * memory admission (Spec §3: "vector search over moment embeddings, always").
 *
 * Threading:
 *  - [onFrame] runs on the camera analyzer thread (FrameCaptureManager's single-thread executor),
 *    NOT inferLane. The caller retains + recycles `bitmap`; this class snapshots it synchronously
 *    and never touches it after [onFrame] returns.
 *  - [tracker] is NOT thread-safe: [ObjectTracker.update]/[ObjectTracker.confirmed] must both run
 *    on [inferLane] (its only legal lane).
 *  - [onFrame] additionally hands `bitmap` to [momentCapture] as a THIRD, independent branch:
 *    scene-change-gated keyframe capture ([tech.qdrant.glasses.Config.MOMENT_MEMORY], default ON —
 *    `debug.qdrant.memory=0` disables it for A/B; [momentCapture] is null only then, making the
 *    call a no-op). It shares no gate/copy with the stream/detect branches below —
 *    [tech.qdrant.glasses.pipeline.MomentCapture.onFrame] owns its OWN check-before-copy +
 *    backpressure gate and never touches `bitmap` after it returns (see its KDoc), so no extra
 *    copy is needed at this layer; it runs unconditionally, BEFORE the detect lane's busy-gate
 *    `return`, so a busy detect lane never silently skips a moment evaluation.
 *  - EVERY launch uses the injected [scope] (= viewModelScope) so the VM's onCleared drain joins
 *    in-flight work before the FFI/interpreter is closed.
 */
class PerceptionPipeline(
    private val scope: CoroutineScope,
    private val inferLane: CoroutineDispatcher,
    private val detector: ObjectDetector,
    private val tracker: ObjectTracker,
    private val hud: HudPublisher,
    private val isRecording: () -> Boolean,
    // The whole-frame keyframe memory path (Config.MOMENT_MEMORY, default ON — Task 2.4). Null
    // only when the sysprop explicitly disables it, making onFrame's call to it a no-op; this
    // class carries NO memory-write path of its own anymore (Task 2.4 retired the crop store).
    private val momentCapture: MomentCapture?,
) {
    companion object {
        private const val TAG = "GlassesVM"
        // Browser stream is downscaled from the ~960px detection frame to keep JPEG encode cheap
        // (~30-40ms → smooth ~25 FPS). Height is derived from the frame's aspect ratio at runtime.
        private const val STREAM_WIDTH = 640
        private const val STREAM_QUALITY = 60
    }

    // Stream lane: JPEG-compress + offerFrame ONLY, decoupled from detection so the video stream
    // runs at its own (fast) rate instead of being serialized behind the ~110ms detect pipeline on
    // inferLane. CPU-bound (Default, not IO), single-thread to keep MJPEG frames in order.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val streamLane = Dispatchers.Default.limitedParallelism(1)
    // Latest detections, published by inferLane after trk.update, read by streamLane to overlay
    // boxes. Immutable List + @Volatile ref = lock-free. Boxes lag the video by ~1 detect cycle
    // (~110ms); the tracker smooths positions so the lag is imperceptible.
    @Volatile private var latestDetections: List<tech.qdrant.glasses.detect.Detection> = emptyList()
    // Region-candidate snapshot for MomentCapture's region layer (Task 2.2) — same lock-free
    // publish pattern as [latestDetections] just above, read from a DIFFERENT lane (embedLane, via
    // MomentCapture's regionsProvider) instead of streamLane. Read-only: MomentCapture never mutates
    // the tracker through this, and this class never dedups/filters it beyond `confirmed()`'s own
    // sightings gate — see [RegionCandidate]'s KDoc for why confirmation here is a TAG-QUALITY gate,
    // not a memory-admission one.
    @Volatile var latestConfirmedRegions: List<RegionCandidate> = emptyList()
        private set
    // Backpressure: the camera pushes ~30 FPS but streamLane/inferLane are slower. Without a gate,
    // every camera frame launches a coroutine and they QUEUE UP unboundedly → the browser sees
    // frames seconds old (latency creeps to ~1s) and detection lags. These flags drop a new frame
    // for a lane that's still busy with the previous one, so each lane always works the FRESHEST
    // frame and the queue never builds. AtomicBoolean = the camera thread sets, the lane clears.
    private val streamBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val inferBusy = java.util.concurrent.atomic.AtomicBoolean(false)

    /**
     * Object-mode per-frame processing: detect → track → stream boxes → hand the frame to the
     * memory path.
     *
     * Called from the camera analyzer thread (FrameCaptureManager's single-thread executor),
     * NOT inferLane. ObjectTracker is NOT thread-safe, so the whole detect/track/publish body is
     * dispatched onto inferLane (the tracker's only legal thread).
     */
    fun onFrame(bitmap: Bitmap) {
        if (!isRecording()) return
        // onFrame recycles `bitmap` the instant this returns, so take TWO independent snapshots
        // synchronously: one owned by the stream lane, one by the detect lane. Each lane recycles
        // its own — no bitmap is shared across threads. streamCopy is mutable (true) so we can draw
        // boxes onto the downscaled copy in place.
        // Only snapshot for the stream if a HUD is actually attached. This copy is a full
        // ARGB_8888 frame (~1.2MB at 640x480); allocating it unconditionally and recycling it a few
        // lines later churned ~36MB/s at 30fps for nothing whenever no HUD was connected.
        val streamCopy = if (!hud.hasClient) null else
            try { bitmap.copy(Bitmap.Config.ARGB_8888, true) } catch (e: Throwable) {
                // Don't abandon detection just because the HUD copy failed — the HUD is cosmetic.
                Log.w(TAG, "streamCopy failed: ${e.message}"); null
            }
        val frame = try { bitmap.copy(Bitmap.Config.ARGB_8888, false) } catch (e: Throwable) {
            streamCopy?.recycle(); Log.w(TAG, "frame snapshot failed: ${e.message}"); return
        }

        // STREAM LANE: encode THIS frame with the LAST-known boxes, right now, independent of the
        // detect pipeline. Downscale to STREAM_WIDTH + JPEG Q60 keeps encode ~30-40ms → ~25 FPS.
        // Backpressure: if the previous frame is still encoding, DROP this one (recycle + skip) so
        // frames don't queue and the stream stays real-time instead of drifting seconds behind.
        // CAS the gate only AFTER both copies exist, so the early-return above can never strand it.
        if (streamCopy != null && streamBusy.compareAndSet(false, true)) {
            val dets = latestDetections   // volatile read — at most ~1 detect cycle stale
            scope.launch(streamLane) {
                try {
                    val aspect = streamCopy.height.toFloat() / streamCopy.width
                    val sh = maxOf(1, (STREAM_WIDTH * aspect).toInt())
                    // createScaledBitmap(...,true) filter=true gives a smooth, MUTABLE bitmap we own.
                    val scaled = Bitmap.createScaledBitmap(streamCopy, STREAM_WIDTH, sh, true)
                    try {
                        // Boxes are in the ORIGINAL frame's pixel space; scale them to the stream size.
                        val sx = STREAM_WIDTH.toFloat() / streamCopy.width
                        val sy = sh.toFloat() / streamCopy.height
                        val scaledDets = if (dets.isEmpty()) dets else dets.map {
                            it.copy(bbox = android.graphics.RectF(
                                it.bbox.left * sx, it.bbox.top * sy, it.bbox.right * sx, it.bbox.bottom * sy))
                        }
                        tech.qdrant.glasses.stream.drawBoxesInPlace(scaled, scaledDets)
                        val baos = java.io.ByteArrayOutputStream()
                        scaled.compress(Bitmap.CompressFormat.JPEG, STREAM_QUALITY, baos)
                        hud.offerFrame(baos.toByteArray())
                    } finally { if (scaled !== streamCopy) scaled.recycle() }
                } catch (e: Throwable) {
                    Log.w(TAG, "stream frame failed: ${e.message}")
                } finally { streamCopy.recycle(); streamBusy.set(false) }
            }
        } else {
            streamCopy?.recycle()   // lane busy → drop this frame's copy (null = no HUD, none taken)
        }

        // MOMENT LANE (Config.MOMENT_MEMORY, default ON — Task 2.4 made this the ONLY memory write
        // path): purely additive scene-change-gated keyframe capture, running independently of the
        // stream/detect branches above/below. Deliberately BEFORE the detect lane's busy-gate
        // `return` right below — that gate governs only the detect branch, and a busy detect lane
        // must never silently skip a moment evaluation.
        // `momentCapture` is null when the sysprop disables it, making this a single null-check per
        // frame. When non-null, `bitmap` (the untouched original — streamCopy/frame above are
        // independent COPIES of it) is handed straight to MomentCapture.onFrame: that call owns
        // its own check-before-copy + backpressure gate and never touches `bitmap` after it
        // returns (see its KDoc), so no extra copy/gate belongs at this layer, and the caller
        // (GlassesViewModel.onFrame) only recycles `bitmap` once this whole method returns.
        momentCapture?.onFrame(bitmap)

        // DETECT LANE: full detect → track → publish boxes, at its own (slower) rate. Backpressure
        // (same as stream): if detection is still busy, drop this frame so the tracker always sees
        // the freshest frame and no backlog builds up behind a slow detect.
        if (!inferBusy.compareAndSet(false, true)) { frame.recycle(); return }
        scope.launch(inferLane) {
            try {
                // Re-check state HERE, not just at onObjectFrame entry. When the user stops
                // recording, a frame already dispatched here would keep detecting in Idle. Bail
                // before touching the tracker (finally still recycles + releases).
                if (!isRecording()) return@launch
                val t0 = System.currentTimeMillis()
                val detections = try { detector.detect(frame) } catch (e: Throwable) {
                    Log.e(TAG, "detect failed", e); return@launch
                }
                val detMs = System.currentTimeMillis() - t0
                val tracks = tracker.update(detections)
                latestDetections = detections   // publish for the stream lane (volatile write)
                // Publish for MomentCapture's region layer (Task 2.2) — confirmed() reads tracker
                // state without touching `embedded`, so this is safe to build here on inferLane
                // (the tracker's only legal thread) and hand off as an immutable List (volatile
                // write, lock-free, same pattern as latestDetections above).
                latestConfirmedRegions = tracker.confirmed().map {
                    RegionCandidate(
                        label = it.label,
                        left = (it.bbox.left / frame.width).coerceIn(0f, 1f),
                        top = (it.bbox.top / frame.height).coerceIn(0f, 1f),
                        right = (it.bbox.right / frame.width).coerceIn(0f, 1f),
                        bottom = (it.bbox.bottom / frame.height).coerceIn(0f, 1f),
                        conf = it.conf,
                    )
                }
                Log.d(TAG, "object frame: detect=${detMs}ms detections=${detections.size}")
            } finally {
                frame.recycle()  // our snapshot
                inferBusy.set(false)  // release the lane so the next camera frame can be detected
            }
        }
    }
}

/**
 * One CONFIRMED tracker box, snapshotted by [PerceptionPipeline] after each `tracker.update(...)`
 * for [MomentCapture]'s region layer (plan Task 2.2, Spec §2 "CLIP-verify-the-label") to read
 * WITHOUT touching [tech.qdrant.glasses.detect.ObjectTracker] directly — the tracker is
 * inferLane-confined and NOT thread-safe (see its KDoc), while MomentCapture's region embeds run
 * on embedLane. Built from [tech.qdrant.glasses.detect.ObjectTracker.confirmed] — NOT
 * `confirmedUnembedded` — so reading this snapshot never touches the `embedded` flag ObjectTracker
 * also tracks (a bookkeeping detail the now-retired crop-store path used exclusively — Task 2.4 —
 * and nothing in this class touches anymore); confirmation here gates TAG QUALITY (a box seen
 * enough times to trust its label), never memory admission (Spec §2: regions are additive vector
 * points, search never gates on YOLO).
 *
 * `left`/`top`/`right`/`bottom` are normalized ([0,1] fractions of the DETECT frame's width/height,
 * same convention [PerceptionPipeline.onFrame]'s (retired) crop-store `bboxStr` used) rather than a
 * pixel [android.graphics.RectF] — a region candidate read at detect-frame time N is applied by
 * [MomentCapture] against whichever LATER frame its sharpness-selection window picked as the
 * keyframe (Spec §4's window is ~800ms; boxes drift only slightly over that span), so a
 * resolution-independent bbox is what actually survives the hop between the two frames.
 */
data class RegionCandidate(
    val label: String,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val conf: Float,
)
