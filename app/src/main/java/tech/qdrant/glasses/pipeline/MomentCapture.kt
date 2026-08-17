package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * Scene-change-gated keyframe capture (plan Task 1.4, Spec §4). Two independent pieces live in
 * this one file, per the task split:
 *
 *  - [decide] (+ [Decision]) — the PURE capture-trigger decision (Spec §4 steps 2 + 5). No
 *    Bitmap/coroutine/store touched, so it is fully JVM unit-tested (`MomentCaptureGateTest`).
 *  - [MomentCapture] — the stateful wiring around it: the armed sharpness-selection window
 *    (Spec §4 step 3), the semantic confirm (step 4), and the store + thumb write. This half is
 *    verified on-device at the Stage 1 gate, not by assertion — Spec §4's thresholds are STARTING
 *    values pending an on-device calibration pass (Global Constraints), and the window/confirm
 *    logic needs a real camera feed to exercise meaningfully.
 *
 * [MomentCapture] mirrors [PerceptionPipeline]'s lane/snapshot/recycle/backpressure discipline
 * (its KDoc) — NOT its store-search dedup: a moment's "dedup" IS the pre-gate + confirm below,
 * there is no global vector search on the capture path (Spec §4: "this REPLACES the current
 * global dedup").
 */

/** Outcome of one gate evaluation against the last STORED keyframe (not the last analyzed frame —
 *  Spec §4). CAPTURE and HEARTBEAT both lead to the same downstream action (arm the sharpness
 *  window → confirm → store); the distinction exists so a rehearsal pass can tell "the scene
 *  changed" captures apart from "nothing changed, but time-window queries need a keyframe anyway"
 *  keep-alives in the logs. */
enum class Decision { SKIP, CAPTURE, HEARTBEAT }

// Pixel pre-gate (Spec §4 step 2): a candidate whose 32x32 luma grid is at least this similar to
// the LAST STORED keyframe's grid is "no visible change" -> no capture, unless a heartbeat is due.
// Same value FrameCaptureManager used for its (now-superseded) whole-pipeline gate.
private const val PREGATE_SIMILARITY = 0.85f

// Minimum time between two STORES, regardless of scene change (Spec §4 step 5) — a hard flood cap
// under heavy motion. Deliberately shorter than HEARTBEAT_MS so a heartbeat is never itself
// throttled by the cooldown it just satisfied (see [decide]'s ordering).
private const val CAPTURE_COOLDOWN_MS = 8000L

// Force a store even when the scene never visibly changes, so "what did I see N minutes ago"
// stays answerable for a wearer standing still (Spec §4 step 5).
private const val HEARTBEAT_MS = 45_000L

/**
 * The pure capture-trigger decision (Spec §4 steps 2 + 5). Takes the pre-gate grids and clock
 * directly — deterministic, Bitmap/coroutine-free — see `MomentCaptureGateTest`.
 *
 * - `prevGrid == null` (nothing stored yet this session) → [Decision.CAPTURE] unconditionally —
 *   there is no baseline to diff against, and the first frame of a session must always be kept.
 * - Within [CAPTURE_COOLDOWN_MS] of the last store → [Decision.SKIP], no matter how different the
 *   scene looks. The cooldown is a hard cap, checked BEFORE the pixel diff, so it never gets
 *   bypassed by a big enough scene change (Spec §4 step 5: "caps the flood under heavy motion").
 * - Past cooldown and the scene changed (`similarity(prevGrid, candGrid) < [PREGATE_SIMILARITY]`)
 *   → [Decision.CAPTURE].
 * - Past cooldown, scene unchanged, but [HEARTBEAT_MS] has elapsed since the last store →
 *   [Decision.HEARTBEAT].
 * - Otherwise (past cooldown, scene unchanged, heartbeat not yet due) → [Decision.SKIP].
 */
fun decide(prevGrid: FloatArray?, candGrid: FloatArray, lastStoreMs: Long, nowMs: Long): Decision {
    if (prevGrid == null) return Decision.CAPTURE
    if (nowMs - lastStoreMs < CAPTURE_COOLDOWN_MS) return Decision.SKIP
    val sim = similarity(prevGrid, candGrid)
    return when {
        sim < PREGATE_SIMILARITY -> Decision.CAPTURE
        nowMs - lastStoreMs >= HEARTBEAT_MS -> Decision.HEARTBEAT
        else -> Decision.SKIP
    }
}

/**
 * Owns one recording session's keyframe capture: cheap pre-gate → armed sharpness-selection
 * window → semantic confirm → [MomentStore.storeMoment] + one JPEG thumb.
 *
 * Threading (mirrors [PerceptionPipeline]'s KDoc):
 *  - [onFrame] runs on the camera analyzer thread, NOT [embedLane]. The caller retains + recycles
 *    `bitmap`; this class takes its OWN snapshot synchronously and never touches the caller's
 *    bitmap after [onFrame] returns.
 *  - Every candidate frame this class keeps (the pre-gate check, each sharpness-window sample, the
 *    NPU embed, the store + thumb write) runs on [embedLane] — a single-thread dispatcher, injected
 *    rather than owned, because Task 1.5 shares ONE lane between this class and
 *    [PerceptionPipeline]'s crop embeds so `OrtSession.run` calls from both paths are still
 *    serialized (Spec §8.2). All capture STATE (window/grid/vector bookkeeping) is therefore only
 *    ever read or written from that one lane and needs no locking of its own.
 *  - [onFrame] itself only reads a handful of `@Volatile` bookkeeping fields (cheap, no bitmap
 *    work) to decide whether this frame is even worth copying — see the "due" check below. That is
 *    a soft, racy check by design (the fields can flip on [embedLane] between the read and the
 *    dispatch): worst case a frame is processed one camera tick late or an extra one is dropped,
 *    never a correctness issue for this ladder of second-to-tens-of-second thresholds.
 *  - [busy] is the hard backpressure gate (mirrors `inferBusy`/`streamBusy`): if [embedLane] is
 *    still finishing a PREVIOUS eligible frame (most likely the embed+store at the end of a
 *    window), a new eligible frame is dropped rather than queued, so a slow store can never make
 *    the camera thread block or candidates pile up out of order.
 */
class MomentCapture(
    private val scope: CoroutineScope,
    private val embedLane: CoroutineDispatcher,
    private val cropEncoder: CropEncoder,
    private val store: MomentStore,
    private val momentThumbsDir: File,
    private val isRecording: () -> Boolean,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
    // Spec §6: a frame's `episode_id` = the recording session's start timestamp. MomentCapture has
    // no session concept of its own, so the caller (Task 1.5) passes the session start; absent
    // that, default to construction time so a standalone instance still stamps SOMETHING sane.
    private val episodeId: Long = nowMs(),
    // Fired after a successful storeMoment(), on embedLane. Deliberately NOT a HudPublisher
    // reference — Task 1.6 wires the HUD timeline event through this callback so MomentCapture
    // stays unaware of the HUD (same seam style as ObjectSearcher not knowing about HudPublisher
    // internals).
    var onMoment: ((MomentHit) -> Unit)? = null,
) {
    companion object {
        private const val TAG = "MomentCapture"
        // Cadence check (Spec §4 step 1): at most one gate EVALUATION per this interval while no
        // window is armed. Independent of CAPTURE_COOLDOWN_MS, which caps STORES, not evaluations.
        private const val MOMENT_CHECK_MS = 3000L
        // Sharpness-selection window (Spec §4 step 3): once the pre-gate fires (or a heartbeat is
        // due), keep sampling candidates for this long...
        private const val SELECT_WINDOW_MS = 800L
        // ...at most this often, so a 30fps camera doesn't turn "sample the window" into "score
        // every single frame" (~4 samples over the 800ms window, matching the spec).
        private const val SELECT_SAMPLE_MS = 200L
        // Semantic confirm (Spec §4 step 4): the chosen frame's CLIP cosine vs the last STORED
        // keyframe vector must fall below this to count as a genuinely new scene. Bypassed for a
        // HEARTBEAT decision (see confirmAndStore) — that path exists specifically to store an
        // UNCHANGED scene, so re-gating it on cosine would defeat the heartbeat entirely.
        private const val CONFIRM_COSINE = 0.85f
        // Grid side for BOTH the pixel pre-gate and the sharpness score — same value
        // FrameCaptureManager used for its ssimSize, and SceneDiff.downscaleLuma's default `out`.
        private const val GRID_SIDE = 32
        // Intermediate scaled-bitmap side the grid is computed FROM, matching Spec §4 step 3's
        // "~160 px gray copy" — sharpness needs more source detail than the 32x32 grid alone to
        // meaningfully separate a sharp frame from a motion-blurred one; a straight 960px->32px
        // downscale would already have blurred that detail away.
        private const val PIXEL_SCALE_SIDE = 160
        private const val THUMB_QUALITY = 85
    }

    private val busy = AtomicBoolean(false)

    // ---- Cross-thread bookkeeping: read on the camera thread by the "due" check in onFrame,
    // written only from embedLane. @Volatile for visibility across that thread hop (see class KDoc).
    @Volatile private var windowArmed = false
    @Volatile private var lastCheckMs = 0L
    @Volatile private var lastSampleMs = 0L
    @Volatile private var windowDeadlineMs = 0L

    // ---- embedLane-confined state: only ever touched from embedLane (single-threaded), so plain
    // vars are safe — no synchronization needed, same discipline PerceptionPipeline uses for
    // ObjectTracker state confined to inferLane.
    private var bestBitmap: Bitmap? = null
    private var bestGrid: FloatArray? = null
    private var bestSharpness: Float = Float.NEGATIVE_INFINITY
    private var pendingDecision: Decision = Decision.CAPTURE
    private var lastStoredGrid: FloatArray? = null
    private var lastStoredVec: FloatArray? = null
    private var lastStoreMs: Long = 0L

    /**
     * Camera-thread entry point. Cheap eligibility check first (no bitmap work at all on the
     * overwhelming common case — no window armed, cadence not yet due), THEN the hard [busy]
     * backpressure gate, and only once both pass does this take its own snapshot and dispatch to
     * [embedLane]. That ordering — check-before-copy — is the one deliberate deviation from
     * [PerceptionPipeline]'s copy-then-gate order: PerceptionPipeline must inspect every frame
     * (detection needs frame continuity), MomentCapture's "is this frame even interesting" answer
     * is "no" for all but a handful of frames per minute, so paying for `bitmap.copy` on those is
     * pure waste.
     */
    fun onFrame(bitmap: Bitmap) {
        if (!isRecording()) return
        val now = nowMs()
        val due = if (windowArmed) {
            now - lastSampleMs >= SELECT_SAMPLE_MS || now >= windowDeadlineMs
        } else {
            now - lastCheckMs >= MOMENT_CHECK_MS
        }
        if (!due) return
        if (!busy.compareAndSet(false, true)) return   // embedLane still finishing a prior frame

        val frame = try {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } catch (e: Throwable) {
            Log.w(TAG, "moment frame snapshot failed: ${e.message}")
            busy.set(false)
            return
        }
        scope.launch(embedLane) {
            try {
                process(frame, now)
            } finally {
                busy.set(false)
            }
        }
    }

    /** Runs entirely on [embedLane]. Owns `frame` from here on: every path below either recycles
     *  it, hands it to [bestBitmap] to be recycled later, or passes it into [confirmAndStore]
     *  (which recycles it when done). */
    private fun process(frame: Bitmap, now: Long) {
        // Re-check HERE, not just at onFrame entry — a stop-recording that races a dispatched
        // frame must not arm a new window or store into a dead session (mirrors PerceptionPipeline).
        if (!isRecording()) {
            frame.recycle()
            abortWindow()
            return
        }

        if (windowArmed) {
            val grid = gridOf(frame)
            val sharp = sharpness(grid, GRID_SIDE)
            if (sharp > bestSharpness) {
                bestBitmap?.recycle()
                bestBitmap = frame
                bestGrid = grid
                bestSharpness = sharp
            } else {
                frame.recycle()
            }
            lastSampleMs = now
            if (now >= windowDeadlineMs) {
                windowArmed = false
                val chosen = bestBitmap
                val chosenGrid = bestGrid
                val decision = pendingDecision
                bestBitmap = null
                bestGrid = null
                bestSharpness = Float.NEGATIVE_INFINITY
                if (chosen != null && chosenGrid != null) {
                    confirmAndStore(chosen, chosenGrid, decision)
                } else {
                    Log.w(TAG, "moment window closed with no candidate (unexpected)")
                }
            }
            return
        }

        lastCheckMs = now
        val grid = gridOf(frame)
        val decision = decide(lastStoredGrid, grid, lastStoreMs, now)
        Log.d(TAG, "moment gate: decision=$decision")
        when (decision) {
            Decision.SKIP -> frame.recycle()
            Decision.CAPTURE, Decision.HEARTBEAT -> {
                windowArmed = true
                windowDeadlineMs = now + SELECT_WINDOW_MS
                lastSampleMs = now
                pendingDecision = decision
                bestBitmap = frame
                bestGrid = grid
                bestSharpness = sharpness(grid, GRID_SIDE)
            }
        }
    }

    /** Abandons an in-flight sharpness window (recording stopped mid-window) without leaking the
     *  held candidate bitmap or leaving stale state armed for the next session. */
    private fun abortWindow() {
        if (!windowArmed) return
        windowArmed = false
        bestBitmap?.recycle()
        bestBitmap = null
        bestGrid = null
        bestSharpness = Float.NEGATIVE_INFINITY
    }

    /**
     * Semantic confirm (Spec §4 step 4) + store (step 4 continued) for the sharpest frame the
     * window collected. Always runs on [embedLane]. Recycles `bitmap` unconditionally before
     * returning — every exit path below funnels through the `finally`.
     */
    private fun confirmAndStore(bitmap: Bitmap, grid: FloatArray, decision: Decision) {
        try {
            val embedT0 = System.currentTimeMillis()
            val vec = try {
                cropEncoder.encode(bitmap)
            } catch (e: Throwable) {
                Log.w(TAG, "moment embed failed, will retry on the next gate fire: ${e.message}")
                return
            }
            val embedMs = System.currentTimeMillis() - embedT0

            val prevVec = lastStoredVec
            val cos = if (prevVec != null) cosine(prevVec, vec) else -1f
            // A HEARTBEAT decision exists specifically to force a store on an UNCHANGED scene
            // (Spec §4 step 5) — re-gating it on cosine here would reject it right back out and
            // defeat the entire point, so only CAPTURE goes through the confirm check.
            if (decision == Decision.CAPTURE && prevVec != null && cos >= CONFIRM_COSINE) {
                Log.i(TAG, "moment confirm: not a new scene (cos=%.3f >= %.2f) — skip store"
                    .format(cos, CONFIRM_COSINE))
                return
            }

            val ts = nowMs()
            val thumbFile = File(momentThumbsDir, "moment_$ts.jpg")
            // Bitmap.compress returns false on an ENCODING failure without throwing — the try/catch
            // alone doesn't see that case. A false return used to be treated as success anyway,
            // which persisted a thumbPath pointing at an empty/invalid file (a broken HUD card with
            // no trace of what went wrong). Fold the Boolean return into thumbOk too, and delete the
            // file in both the false-return and the exception case: compress() can leave a
            // zero-byte/partial file behind before either reports failure.
            val thumbOk = try {
                val compressed = FileOutputStream(thumbFile).use {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, THUMB_QUALITY, it)
                }
                if (!compressed) Log.w(TAG, "moment thumb compress returned false: $thumbFile")
                compressed
            } catch (e: Throwable) {
                Log.w(TAG, "moment thumb write failed: ${e.message}")
                false
            }
            if (!thumbOk) thumbFile.delete()

            val storeT0 = System.currentTimeMillis()
            try {
                // type/momentId are placeholders — storeMoment stamps type="frame" and momentId=the
                // new point's own id itself (Spec §6 invariant), same convention QdrantEdgeMomentStore
                // documents on storeMoment(). The gate embedding above IS the stored vector: it is
                // never re-embedded here or anywhere else on this path.
                val id = store.storeMoment(vec, MomentPayload(
                    type = "frame",
                    momentId = "",
                    episodeId = episodeId,
                    timestampMs = ts,
                    tEndMs = ts,
                    thumbPath = if (thumbOk) thumbFile.absolutePath else "",
                    bbox = "",
                    label = "",
                    yoloConf = 0f,
                    verifyCos = 0f,
                    text = "",
                ))
                val storeMs = System.currentTimeMillis() - storeT0

                // Advance the baseline ONLY on a successful store — a failed embed/store above already
                // returned early, leaving lastStoredGrid/lastStoredVec/lastStoreMs untouched so the
                // NEXT gate fire retries against the same last-known-good keyframe, not a phantom one.
                lastStoredGrid = grid
                lastStoredVec = vec
                lastStoreMs = ts

                val count = store.count()
                Log.i(TAG, "moment stored: id=$id decision=$decision cos=%.3f (embed=${embedMs}ms store=${storeMs}ms) total=$count"
                    .format(cos))
                onMoment?.invoke(MomentHit(
                    id = id, score = 0f, type = "frame", momentId = id, timestampMs = ts,
                    thumbPath = if (thumbOk) thumbFile.absolutePath else "", label = "", bbox = "",
                ))
            } catch (e: Throwable) {
                // storeMoment (native shard/flush) or onMoment (Task 1.6's HUD forward) can throw.
                // Left unguarded this escaped scope.launch(embedLane) UNCAUGHT (mirrors the crop-embed
                // catch in PerceptionPipeline's onFrame) and orphaned the thumb just written above.
                // Log + delete the orphan and DO NOT rethrow: lastStoredGrid/Vec/Ms are untouched
                // (the advance above never ran), so the next gate fire retries cleanly against the
                // same last-known-good keyframe instead of a phantom one.
                Log.w(TAG, "moment store failed, will retry on the next gate fire: ${e.message}")
                if (thumbOk) thumbFile.delete()
            }
        } finally {
            bitmap.recycle()
        }
    }

    /** [PIXEL_SCALE_SIDE]-square gray copy → [GRID_SIDE]x[GRID_SIDE] luma grid (Spec §4 step 3's
     *  "~160 px gray copy"), used for both the pixel pre-gate similarity and the sharpness score so
     *  a candidate frame is only downscaled once. `createScaledBitmap`'s "may return the source
     *  itself" contract (see [PerceptionPipeline.cropFrom]'s KDoc for the same gotcha) can't apply
     *  here since `bitmap` is always far larger than [PIXEL_SCALE_SIDE], but the guard costs
     *  nothing and matches the defensive idiom used everywhere else this call appears. */
    private fun gridOf(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, PIXEL_SCALE_SIDE, PIXEL_SCALE_SIDE, false)
        try {
            val pixels = IntArray(PIXEL_SCALE_SIDE * PIXEL_SCALE_SIDE)
            scaled.getPixels(pixels, 0, PIXEL_SCALE_SIDE, 0, 0, PIXEL_SCALE_SIDE, PIXEL_SCALE_SIDE)
            return downscaleLuma(pixels, PIXEL_SCALE_SIDE, PIXEL_SCALE_SIDE, GRID_SIDE)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }
}

/** Plain cosine similarity between two CLIP vectors. 0f if either is degenerate (all-zero) rather
 *  than dividing by zero — an all-zero embedding should never happen, but "no confirm match" is a
 *  safer failure than a crash or a NaN silently poisoning the comparison. */
private fun cosine(a: FloatArray, b: FloatArray): Float {
    var dot = 0f
    var na = 0f
    var nb = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    if (na <= 0f || nb <= 0f) return 0f
    return (dot / (sqrt(na.toDouble()) * sqrt(nb.toDouble()))).toFloat()
}
