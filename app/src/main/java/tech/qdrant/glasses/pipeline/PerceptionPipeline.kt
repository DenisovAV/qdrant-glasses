package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import tech.qdrant.glasses.detect.ObjectDetector
import tech.qdrant.glasses.detect.ObjectTracker
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.storage.ObjectStore
import tech.qdrant.glasses.stream.HudPublisher
import java.io.File
import java.io.FileOutputStream

/**
 * Object-mode per-frame perception: detect → track → stream boxes → embed confirmed crops.
 * Moved VERBATIM out of [tech.qdrant.glasses.GlassesViewModel.onObjectFrame] (Task 7 of the
 * God-object decomposition) — do NOT change the hot-path logic here; every recycle, backpressure
 * CAS gate, and inferLane marshal is preserved exactly as verified on-device.
 *
 * Threading:
 *  - [onFrame] runs on the camera analyzer thread (FrameCaptureManager's single-thread executor),
 *    NOT inferLane. The caller retains + recycles `bitmap`; this class snapshots it synchronously
 *    and never touches it after [onFrame] returns.
 *  - [tracker] is NOT thread-safe: update/confirmedUnembedded/markEmbedded/unmarkEmbedded all run
 *    on [inferLane] (its only legal lane). The crop embed (a network call) runs on a SEPARATE
 *    [cropLane] so it can't block detection; on success it marshals the counter bump / on failure
 *    the unmark BACK to [inferLane].
 *  - EVERY launch/withContext uses the injected [scope] (= viewModelScope) so the VM's onCleared
 *    drain joins in-flight work before the FFI/interpreter is closed.
 */
class PerceptionPipeline(
    private val scope: CoroutineScope,
    private val inferLane: CoroutineDispatcher,
    private val detector: ObjectDetector,
    private val tracker: ObjectTracker,
    private val cropEncoder: CropEncoder,
    private val store: ObjectStore,
    private val hud: HudPublisher,
    private val isRecording: () -> Boolean,
    private val onMemoryIndexed: () -> Unit,
    private val objectThumbsDir: File,
) {
    companion object {
        private const val TAG = "GlassesVM"
        // Semantic-dedup threshold: a new crop whose nearest stored neighbor has cosine ≥ this is
        // treated as a duplicate and not saved. SigLIP2 crops of the SAME object across frames
        // (slightly different angle/bbox) land around 0.88–0.93, not 0.97+, so 0.95 let visible
        // duplicates (two cups, the same person) slip through. 0.90 catches near-identical views
        // while still keeping genuinely different objects apart. The dedup-check log prints the
        // real nearest-neighbor cosine per object so this can be tuned on data at rehearsal.
        private const val DEDUP_COSINE = 0.90f
        // Browser stream is downscaled from the ~960px detection frame to keep JPEG encode cheap
        // (~30-40ms → smooth ~25 FPS). Height is derived from the frame's aspect ratio at runtime.
        private const val STREAM_WIDTH = 640
        private const val STREAM_QUALITY = 60
        // Fraction of the bbox size to add as context padding on EACH side when cropping an object
        // (0.20 = grow the box 20% left/right/top/bottom). Enough context to disambiguate the object
        // and give the embedder scene cues, without letting the background dominate the crop.
        private const val CROP_PADDING = 0.20f
        // The THUMBNAIL gets much wider context than the embed crop: the memory card should show
        // WHERE the object is (the cup on that corner of the desk), not a tight cutout. Kept
        // separate from CROP_PADDING so the search-score calibration (gates, dedup) is unaffected.
        private const val THUMB_PADDING = 1.20f
    }

    // Crop embedding is a network call (Mac endpoint). It runs on its OWN single-thread lane
    // so a slow embed never blocks detection on inferLane. markEmbedded is marshalled BACK to
    // inferLane (the tracker's only legal thread) after a successful embed+upsert.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val cropLane = Dispatchers.IO.limitedParallelism(1)
    // Stream lane: JPEG-compress + offerFrame ONLY, decoupled from detection so the video stream
    // runs at its own (fast) rate instead of being serialized behind the ~110ms detect pipeline on
    // inferLane. CPU-bound (Default, not IO), single-thread to keep MJPEG frames in order.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val streamLane = Dispatchers.Default.limitedParallelism(1)
    // Latest detections, published by inferLane after trk.update, read by streamLane to overlay
    // boxes. Immutable List + @Volatile ref = lock-free. Boxes lag the video by ~1 detect cycle
    // (~110ms); the tracker smooths positions so the lag is imperceptible.
    @Volatile private var latestDetections: List<tech.qdrant.glasses.detect.Detection> = emptyList()
    // Backpressure: the camera pushes ~30 FPS but streamLane/inferLane are slower. Without a gate,
    // every camera frame launches a coroutine and they QUEUE UP unboundedly → the browser sees
    // frames seconds old (latency creeps to ~1s) and detection lags. These flags drop a new frame
    // for a lane that's still busy with the previous one, so each lane always works the FRESHEST
    // frame and the queue never builds. AtomicBoolean = the camera thread sets, the lane clears.
    private val streamBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private val inferBusy = java.util.concurrent.atomic.AtomicBoolean(false)

    /**
     * Object-mode per-frame processing: detect → track → stream boxes → embed confirmed crops.
     *
     * Called from the camera analyzer thread (FrameCaptureManager's single-thread executor),
     * NOT inferLane. ObjectTracker is NOT thread-safe, so the whole detect/track/confirm body
     * is dispatched onto inferLane (the tracker's only legal thread). The crop embed (a network
     * call to the Mac endpoint) runs on a SEPARATE cropLane so it can't block detection; on
     * success it marshals markEmbedded BACK to inferLane — the dedup flag is part of tracker
     * state and may only be mutated there.
     */
    fun onFrame(bitmap: Bitmap) {
        if (!isRecording()) return
        // onFrame recycles `bitmap` the instant this returns, so take TWO independent snapshots
        // synchronously: one owned by the stream lane, one by the detect lane. Each lane recycles
        // its own — no bitmap is shared across threads. streamCopy is mutable (true) so we can draw
        // boxes onto the downscaled copy in place.
        val streamCopy = try { bitmap.copy(Bitmap.Config.ARGB_8888, true) } catch (e: Throwable) {
            Log.w(TAG, "streamCopy failed: ${e.message}"); return
        }
        val frame = try { bitmap.copy(Bitmap.Config.ARGB_8888, false) } catch (e: Throwable) {
            streamCopy.recycle(); Log.w(TAG, "frame snapshot failed: ${e.message}"); return
        }

        // STREAM LANE: encode THIS frame with the LAST-known boxes, right now, independent of the
        // detect pipeline. Downscale to STREAM_WIDTH + JPEG Q60 keeps encode ~30-40ms → ~25 FPS.
        // Backpressure: if the previous frame is still encoding, DROP this one (recycle + skip) so
        // frames don't queue and the stream stays real-time instead of drifting seconds behind.
        val streamHandled = hud.hasClient && streamBusy.compareAndSet(false, true)
        if (streamHandled) {
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
            streamCopy.recycle()   // no HUD client, or lane busy → drop this frame's copy
        }

        // DETECT LANE: full detect → track → publish boxes → embed, at its own (slower) rate.
        // Backpressure (same as stream): if detection is still busy, drop this frame so the tracker
        // always sees the freshest frame and no backlog builds up behind a slow detect.
        if (!inferBusy.compareAndSet(false, true)) { frame.recycle(); return }
        scope.launch(inferLane) {
            try {
                // Re-check state HERE, not just at onObjectFrame entry. When the user stops
                // recording, a frame already dispatched here would keep detecting and STORING objects
                // in Idle. Bail before touching the tracker/store (finally still recycles + releases).
                if (!isRecording()) return@launch
                val t0 = System.currentTimeMillis()
                val detections = try { detector.detect(frame) } catch (e: Throwable) {
                    Log.e(TAG, "detect failed", e); return@launch
                }
                val detMs = System.currentTimeMillis() - t0
                val tracks = tracker.update(detections)
                latestDetections = detections   // publish for the stream lane (volatile write)
                Log.d(TAG, "object frame: detect=${detMs}ms detections=${detections.size}")

                // Embed newly-confirmed objects on cropLane (network — must not block detection).
                //    confirmedUnembedded() reads tracker state, so it stays on inferLane here.
                //    cropFrom() copies pixels out of `frame` synchronously, so the snapshot can be
                //    safely recycled below even while these cropLane coroutines are still running.
                for (track in tracker.confirmedUnembedded()) {
                    val crop = cropFrom(frame, track.bbox) ?: continue
                    // Separate, wider crop for the visible thumbnail (see THUMB_PADDING). Copied
                    // out of `frame` synchronously, same as `crop`.
                    val thumbCrop = cropFrom(frame, track.bbox, THUMB_PADDING) ?: crop
                    // Thumb is written LATER (in cropLane, only if this isn't a semantic duplicate),
                    // so a deduped object never leaves a stray JPEG on disk.
                    val thumbFile = File(objectThumbsDir, "obj_${track.trackId}_${System.currentTimeMillis()}.jpg")
                    val bboxStr = "%.3f,%.3f,%.3f,%.3f".format(
                        track.bbox.left / frame.width, track.bbox.top / frame.height,
                        track.bbox.width() / frame.width, track.bbox.height() / frame.height)
                    val tid = track.trackId; val label = track.label
                    // Mark embedded NOW (we're on inferLane). The embed is async (~hundreds of ms);
                    // if we waited to mark until it returned, confirmedUnembedded() would re-emit this
                    // same track on every frame in the meantime and launch a duplicate embed per frame
                    // (~10 dupes per object). Mark up-front to claim it; the cropLane coroutine rolls it
                    // back via unmarkEmbedded on failure so a failed embed is retried on a later sighting.
                    tracker.markEmbedded(tid)
                    scope.launch(cropLane) {
                        val embedT0 = System.currentTimeMillis()
                        try {
                            val vec = cropEncoder.encode(crop)
                            val embedMs = System.currentTimeMillis() - embedT0

                            // Semantic dedup: if a near-identical crop is already stored, skip it.
                            // Track-ID dedup only stops repeats within one continuous sighting; this
                            // catches the object leaving and re-entering frame (new track) and a second
                            // pass over the same scene. High threshold so only an almost-identical view
                            // counts as a dupe — two genuinely different objects (even same class) stay.
                            // We DON'T unmark the track here: this is "handled, just not stored", not a
                            // failure, so we must not re-run this search every frame for the same track.
                            val dedupT0 = System.currentTimeMillis()
                            val nearest = store.search(vec, topK = 1).firstOrNull()
                            val dedupSearchMs = System.currentTimeMillis() - dedupT0
                            // DIAGNOSTIC: log the nearest-neighbor cosine for EVERY new object (not
                            // just skips) so we can see the real distribution on a live scene and
                            // tune DEDUP_COSINE on data instead of guessing.
                            Log.i(TAG, "dedup-check: $label (track $tid) nearest cos=%.3f (\"%s\") threshold=%.2f"
                                .format(nearest?.score ?: -1f, nearest?.label ?: "—", DEDUP_COSINE))
                            if (nearest != null && nearest.score >= DEDUP_COSINE) {
                                Log.i(TAG, "dedup: skip $label (track $tid) — cos=%.3f matches \"%s\""
                                    .format(nearest.score, nearest.label))
                                return@launch
                            }

                            // Not a dupe → now write the thumb (wide-context crop) and store.
                            // If the write fails, log it (don't swallow) and persist an EMPTY
                            // thumb_path so the stored object doesn't point at a file that was never
                            // written (which showed as a permanently broken rail card with no trace).
                            val thumbOk = try {
                                FileOutputStream(thumbFile).use { thumbCrop.compress(Bitmap.CompressFormat.JPEG, 85, it) }
                                true
                            } catch (e: Throwable) {
                                Log.w(TAG, "thumb write failed for $label (track $tid): ${e.message}"); false
                            }
                            val storeT0 = System.currentTimeMillis()
                            store.upsert(vec, label, bboxStr, System.currentTimeMillis(), tid,
                                if (thumbOk) thumbFile.absolutePath else "")
                            val storeMs = System.currentTimeMillis() - storeT0
                            // Bump the session counter on inferLane (the only lane that touches it),
                            // only while still Recording (a late embed after stopRecording must not
                            // bump a dead session) — guarded inside the atomic update.
                            withContext(inferLane) {
                                onMemoryIndexed()
                            }
                            val count = store.count()
                            val key = thumbFile.nameWithoutExtension
                            hud.registerThumb(key, thumbFile.absolutePath)
                            hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.storedEvent(key, label, count))
                            hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.tickEvent(detMs, embedMs, storeMs, count))
                            Log.i(TAG, "object stored: $label (track $tid), total=$count (embed=${embedMs}ms qsearch=${dedupSearchMs}ms upsert=${storeMs}ms)")
                        } catch (e: Throwable) {
                            Log.w(TAG, "embed failed for $label (track $tid), will retry: ${e.message}")
                            // roll back the up-front mark so this track is retried on a later sighting
                            withContext(inferLane) { tracker.unmarkEmbedded(tid) }
                        } finally {
                            crop.recycle()  // crop is our own pixels; release once encoded/stored
                            // thumbCrop is a separate wide-context copy that leaked on both the store
                            // and dedup-skip paths. It aliases `crop` when the wider cropFrom returned
                            // null (the `?: crop` fallback), so guard against a double free.
                            if (thumbCrop !== crop) thumbCrop.recycle()
                        }
                    }
                }
            } finally {
                frame.recycle()  // our snapshot; crops were already copied out above
                inferBusy.set(false)  // release the lane so the next camera frame can be detected
            }
        }
    }

    private fun cropFrom(frame: Bitmap, box: android.graphics.RectF, padding: Float = CROP_PADDING): Bitmap? =
        // Grow the box by `padding` of its own size on each side so the crop carries some
        // surrounding CONTEXT (a cup on a table, not a cup in a void). Context helps both the
        // SigLIP/CLIP embedding (richer scene semantics → better search) and the rail thumbnail
        // (more recognizable). Clamped to the frame so the padding never runs off the edge.
        // The try/catch is DEFENSIVE (matches the pre-refactor cropFrom): createBitmap ALLOCATES
        // and can throw OutOfMemoryError even with valid coords, and this codebase has a history
        // of OOM crashes — swallow it → null → the caller's `?: continue` skips this track.
        paddedCropRect(box, padding, frame.width, frame.height)?.let { r ->
            try { Bitmap.createBitmap(frame, r.left, r.top, r.width(), r.height()) } catch (_: Throwable) { null }
        }
}
