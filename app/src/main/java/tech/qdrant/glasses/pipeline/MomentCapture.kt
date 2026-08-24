package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import org.json.JSONObject
import tech.qdrant.glasses.embedding.BgeTextEncoder
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.embedding.LabelVectorCache
import tech.qdrant.glasses.fleet.UploadQueue
import tech.qdrant.glasses.ocr.OcrEngine
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.storage.MomentType
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.sqrt

/**
 * Scene-change-gated keyframe capture (plan Task 1.4, Spec §4). Two independent pieces live in
 * this one file, per the task split:
 *
 *  - [decide] (+ [Decision]) — the PURE capture-trigger decision (Spec §4 steps 2 + 5). No
 *    Bitmap/coroutine/store touched, so it is fully JVM unit-tested (`MomentCaptureGateTest`).
 *  - [MomentCapture] — the stateful wiring around it: the armed sharpness-selection window
 *    (Spec §4 step 3), the semantic confirm (step 4), the store + thumb write, and (Task 2.2) the
 *    CLIP-verified YOLO region layer stored alongside each successful frame keyframe. This half is
 *    verified on-device at the Stage gates, not by assertion — Spec §4's thresholds (and Task
 *    2.2's `VERIFY_COS`) are STARTING values pending an on-device calibration pass (Global
 *    Constraints), and the window/confirm/region logic needs a real camera feed to exercise
 *    meaningfully. Doc-rot note: Spec §4 step 3 also specifies a `SHARPNESS_MIN` floor (reject a
 *    window's sharpest sample if it's still below an absolute blur threshold) — that floor is NOT
 *    implemented below; the window unconditionally keeps whichever sample scored highest, however
 *    blurry, so `SHARPNESS_MIN` remains a Spec §8 unknown-#7 item with no code to calibrate yet.
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
// 0.98 (was 0.85): `sim < this -> CAPTURE`, so RAISING it lets MORE frames through, not fewer. The
// pixel luma-grid vs the last STORED keyframe barely moves in a visually-uniform room — a whole-room
// pan measured pregateSim 0.91–0.97, all >0.85, so every view was SKIPped before the embedding ran
// (~1 keyframe stored). 0.98 passes anything with any real pixel change to the semantic CONFIRM
// (sceneDedupCosine 0.90) stage — the actual scene discriminator; only a near-static frame (~0.99)
// is pre-skipped (the 45s heartbeat still covers it). Trades a per-3s embedding for real coverage.
private const val PREGATE_SIMILARITY = 0.98f

// Minimum time between two STORES, regardless of scene change (Spec §4 step 5) — a hard flood cap
// under heavy motion. Deliberately shorter than HEARTBEAT_MS so a heartbeat is never itself
// throttled by the cooldown it just satisfied (see [decide]'s ordering).
// 3000 (was 8000): the cooldown is a HARD cap checked before every other gate, so at 8s an active
// look-around the room stored only ~1 moment / 8s (a whole-room pan → ~3 keyframes). 3s gives a
// richer memory during deliberate recording while still throttling the NPU/store on a static scene.
private const val CAPTURE_COOLDOWN_MS = 3000L

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
    // CLIP-verify-the-label cache (plan Task 2.1/2.2, Spec §2) — null disables the region layer
    // entirely (no verification possible without it), leaving this class's frame-store behavior
    // identical to before Task 2.2. GlassesComponents always builds one alongside a non-null
    // regionsProvider when Config.MOMENT_MEMORY is on, so in practice this is non-null whenever
    // regions matter; the nullability exists so a bare MomentCapture (unit tests, a hypothetical
    // frame-only deployment) needs no cache just to store keyframes.
    private val labelCache: LabelVectorCache? = null,
    // Stage 3 "OCR read channel" (additive, like [labelCache]/the region layer): both null disables
    // OCR entirely — no assets, no BGE model load required. GlassesComponents only builds a non-null
    // pair when `ocr/` assets provisioned cleanly, so a device missing them still runs frame+region
    // capture exactly as before Stage 3 (same nullable-optional-feature contract as [labelCache]).
    private val ocrEngine: OcrEngine? = null,
    private val bgeEncoder: BgeTextEncoder? = null,
    // Fleet-sync Task 10 (Spec §4 UP flow / §5 dual-write, decision C): when non-null, every
    // successfully stored frame moment ADDITIONALLY enqueues a COPY of itself (id, clip vector,
    // payload + sync_ts + thumb_b64 — see [buildUploadPayloadJson]) onto this queue, for
    // [tech.qdrant.glasses.fleet.FleetSync.pushDrain] (Task 11) to batch-upsert into the private
    // Qdrant's `fleet_inbox` next time it's online. The LOCAL point [store] just persisted is
    // NEVER deleted because of this — decision C makes the local mutable moment the user's
    // PERMANENT memory; a fleet upload is a side-channel copy, not a move, and this class has no
    // "delete one point" call to make even if it wanted to ([MomentStore] only exposes
    // [MomentStore.deleteAll], the demo-wipe gesture). Null (the default) whenever
    // `Config.FLEET_URL` is unset or a build wires no fleet tier — same nullable-optional-feature
    // contract as [labelCache]/[ocrEngine] above; GlassesComponents only builds a non-null queue
    // when the fleet tier is actually configured.
    private val uploadQueue: UploadQueue? = null,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
    // Spec §6: a frame's `episode_id` = the recording session's start timestamp. Defaults to
    // construction time so a standalone instance still stamps SOMETHING sane; the real value for
    // a live recording is set by [startSession] at the START of each session (a `var`, not a
    // `val`, precisely because this class is constructed ONCE at `GlassesComponents.load()` but
    // must stamp a FRESH episode on every stop→start in the same process — see [startSession]).
    private var episodeId: Long = nowMs(),
    // Fired after a successful storeMoment(), on embedLane. Deliberately NOT a HudPublisher
    // reference — Task 1.6 wires the HUD timeline event through this callback so MomentCapture
    // stays unaware of the HUD (same seam style as ObjectSearcher not knowing about HudPublisher
    // internals). @Volatile for the same reason as [regionsProvider] just below: GlassesViewModel
    // reassigns this field AFTER GlassesComponents.load() has already constructed MomentCapture (to
    // wrap the HUD forward in its own session.onMemoryIndexed() counter update — see
    // GlassesViewModel.init), and a plain var write has no happens-before with embedLane's read of
    // this field, so a camera-dispatched frame right after that reassignment could still observe
    // the OLD callback. GlassesViewModel also orders the reassignment before `perception` itself is
    // published, same belt-and-suspenders discipline regionsProvider's wiring documents.
    @Volatile var onMoment: ((MomentHit) -> Unit)? = null,
    // Region source (Task 2.2, Spec §2 "CLIP-verify-the-label") — the tracker's CONFIRMED boxes at
    // the time confirmAndStore runs. Defaults to no regions so a bare MomentCapture (unit tests,
    // regions disabled) behaves exactly as before Task 2.2. A `var`, not a constructor-injected
    // `val` set once, for the same reason [onMoment] is: MomentCapture is built inside
    // GlassesComponents.load(), strictly BEFORE PerceptionPipeline exists —
    // GlassesViewModel constructs PerceptionPipeline with `c.momentCapture` as one of ITS OWN
    // constructor args, so the two classes have a genuine circular dependency at wiring time. The
    // real provider ({ perception.latestConfirmedRegions }) is wired by GlassesViewModel right
    // after PerceptionPipeline is constructed — see its init block. @Volatile (Codex P2 fix): a
    // plain var write has no happens-before with [embedLane]'s read of this field, so a camera
    // frame dispatched right after the assignment could still observe the default empty provider.
    // GlassesViewModel also orders that assignment BEFORE `perception` itself is published, so the
    // camera path can't even start moment capture until the real provider is already visible here.
    @Volatile var regionsProvider: () -> List<RegionCandidate> = { emptyList() },
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
        // Semantic confirm (Spec §4 step 4): the chosen frame's whole-frame cosine vs the last STORED
        // keyframe must fall below the gate to count as a genuinely new scene. The gate is PER-BACKEND
        // — see CropEncoderFactory.sceneDedupCosine — because the whole-frame image↔image cosine scale
        // differs by encoder (SigLIP's is compressed: different scenes 0.77–0.92, so it needs 0.90 vs
        // CLIP's 0.85). Bypassed for a HEARTBEAT decision (that path exists to store an UNCHANGED scene).
        // Grid side for BOTH the pixel pre-gate and the sharpness score — same value
        // FrameCaptureManager used for its ssimSize, and SceneDiff.downscaleLuma's default `out`.
        private const val GRID_SIDE = 32
        // Intermediate scaled-bitmap side the grid is computed FROM, matching Spec §4 step 3's
        // "~160 px gray copy" — sharpness needs more source detail than the 32x32 grid alone to
        // meaningfully separate a sharp frame from a motion-blurred one; a straight 960px->32px
        // downscale would already have blurred that detail away.
        private const val PIXEL_SCALE_SIDE = 160
        private const val THUMB_QUALITY = 85
        // Region layer (Task 2.2, Spec §2): at most this many confirmed tracker boxes get a region
        // embedding per stored moment, highest yolo_conf first — a bound on the extra NPU/store work
        // one keyframe can trigger, not a quality signal (a scene with more objects just loses the
        // weakest-confidence ones). UNCALIBRATED — tuned at the Stage 2 gate. (Doc-rot fix: NOT one
        // of Spec §8 unknown #7's listed calibration values — that list is CONFIRM_COSINE,
        // SHARPNESS_MIN, VERIFY_COS, the frame-channel search gate, and TAG_BOOST_LAMBDA; this is an
        // engineering bound on region work, not a threshold #7 is tracking. No §-citation, unlike
        // MomentSearcher.kt's/MomentFusion.kt's "(Spec §7/§8 unknown #7)" constants, which ARE on
        // that list.)
        const val REGIONS_MAX_PER_MOMENT = 6
        // Region label-verify threshold (Spec §2/§7): a region embedding's cosine against its YOLO
        // label's TEXT vector must clear the gate to keep the label as a display tag. This gate is
        // now PER-BACKEND — see CropEncoderFactory.verifyGate — because it lives on the same
        // text→image modality-gap scale as searchGate: a fixed CLIP-scale 0.20 silently dropped
        // EVERY SigLIP region's label (SigLIP present-cosines ~0.11, not ~0.26). Below the gate the
        // region vector is still stored (a valid recall signal); only the label is dropped, for
        // FUTURE captures only — points already stored keep their own capture-time verify_cos.
    }

    private val busy = AtomicBoolean(false)

    // Per-backend label-verify gate for the active crop encoder (read once; see the companion note
    // and CropEncoderFactory.verifyGate). SigLIP's compressed scale needs ~0.10, CLIP ~0.20.
    private val verifyGate: Float = CropEncoderFactory.verifyGate

    // Per-backend whole-frame scene-dedup cosine (read once; see CropEncoderFactory.sceneDedupCosine).
    // SigLIP whole-frames barely separate scenes → 0.90 (vs CLIP 0.85) or genuinely-new views get deduped.
    private val sceneDedupCosine: Float = CropEncoderFactory.sceneDedupCosine

    // DEDICATED lane for background OCR (Stage 3). OCR is ~4s/keyframe on the CPU (DBNet@1536 +
    // per-line CRNN); running it on [embedLane] would block the NEXT keyframe's embed for those ~4s
    // and throttle capture. Its OWN single-thread dispatcher runs it IN PARALLEL with embedLane (the
    // OcrEngine's ORT sessions are still serialized to one thread, since limitedParallelism(1)).
    @OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
    private val ocrLane = kotlinx.coroutines.Dispatchers.Default.limitedParallelism(1)

    // Session-generation counter: bumped SYNCHRONOUSLY (on the CALLING thread, not embedLane) at
    // the very top of every startSession(), before its reset is even posted. onFrame() captures
    // the generation at dispatch time and threads it through to process(), which discards
    // (recycle, no state touched) any frame whose captured generation no longer matches the
    // CURRENT one. This is what actually closes the cross-session race startSession()'s own KDoc
    // describes: the reset itself still runs async on embedLane, but the generation bump ahead of
    // it is synchronous, so a frame already queued on embedLane — whether still ahead of that
    // reset in the queue, or dispatched in the gap between beginRecording() and this call — always
    // carries the OLD generation and gets dropped rather than attributed to the wrong session
    // (wrong episodeId, or gated against a baseline the new session never produced).
    private val sessionGen = AtomicInteger(0)

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
     * Resets this capture for a NEW recording session. [MomentCapture] is constructed ONCE at
     * `GlassesComponents.load()` and lives for the whole process, but nothing reset it when a
     * NEW recording started: after a stop→start in the same process, the first frames of the new
     * session were gated against the PRIOR session's [lastStoredGrid]/[lastStoredVec]/
     * [lastStoreMs] (so a scene that only changed since the OLD session's last keyframe could look
     * "unchanged" and get skipped) and every stored frame kept stamping the app-init [episodeId]
     * instead of the new session's own start (Spec §6: a frame's `episode_id` = ITS session's
     * start). Mirrors the intent of
     * [tech.qdrant.glasses.camera.FrameCaptureManager.resetForNewSession] for this class's own
     * baseline.
     *
     * Aborts any in-flight sharpness window (a window armed by the OLD session must not be
     * confirmed/stored under the NEW one's episodeId), clears the stored-keyframe baseline so the
     * very first candidate of the new session is unconditionally a [Decision.CAPTURE] (mirrors
     * [decide]'s `prevGrid == null` case), resets the cadence clock so the new session doesn't
     * inherit a stale [MOMENT_CHECK_MS] wait from whenever the old session's last gate fired, and
     * stamps a fresh [episodeId].
     *
     * Posted onto [embedLane], not applied inline: every field touched here — the armed window,
     * `lastStoredGrid`/`lastStoredVec`/`lastStoreMs`, `episodeId` — is embedLane-confined state
     * per the class KDoc's threading discipline, and the caller
     * ([tech.qdrant.glasses.GlassesViewModel.startRecording]) runs on the main thread. The reset
     * body is still fire-and-forget (the caller doesn't wait for it), but the [sessionGen] bump
     * right below is NOT: it happens SYNCHRONOUSLY on the calling thread, before this reset is
     * even posted, so any frame [onFrame] already dispatched to [embedLane] — still queued ahead
     * of this reset, or dispatched in the gap between `beginRecording()` and this call — carries
     * the OLD generation and gets discarded by [process]'s generation check instead of running
     * against a baseline/episodeId this reset is about to replace out from under it. That closes a
     * real correctness gap (wrong `episode_id`, or a false "unchanged scene" against a baseline
     * the new session never produced) — NOT just the "gated one frame late" soft race [onFrame]'s
     * KDoc documents for its own cheap cross-thread reads.
     */
    fun startSession(sessionStartMs: Long = nowMs()) {
        val gen = sessionGen.incrementAndGet()
        scope.launch(embedLane) {
            abortWindow()
            lastStoredGrid = null
            lastStoredVec = null
            lastStoreMs = 0L
            lastCheckMs = 0L
            episodeId = sessionStartMs
            Log.i(TAG, "startSession: episodeId=$sessionStartMs gen=$gen (baseline + window reset)")
        }
    }

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
        // Captured HERE, at dispatch time — not read again until process() runs on embedLane, so
        // it reflects whichever session was current the instant this frame was queued. See
        // [sessionGen] and [process]'s check.
        val gen = sessionGen.get()
        scope.launch(embedLane) {
            try {
                process(frame, now, gen)
            } finally {
                busy.set(false)
            }
        }
    }

    /** Runs entirely on [embedLane]. `gen` is the session generation [onFrame] captured at
     *  dispatch time (see [sessionGen]); if it no longer matches the CURRENT generation, this
     *  frame belongs to a session [startSession] has already reset past and is discarded before
     *  touching any window/baseline state (the reset queued right behind it, per [startSession]'s
     *  KDoc, is what actually cleans that state up). Otherwise owns `frame` from here on: every
     *  path below either recycles it, hands it to [bestBitmap] to be recycled later, or passes it
     *  into [confirmAndStore] (which recycles it when done). */
    private fun process(frame: Bitmap, now: Long, gen: Int) {
        if (gen != sessionGen.get()) {
            frame.recycle()
            return
        }
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
        val pregateSim = lastStoredGrid?.let { similarity(it, grid) } ?: -1f
        Log.d(TAG, "moment gate: decision=$decision pregateSim=%.3f".format(pregateSim))
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
            if (decision == Decision.CAPTURE && prevVec != null && cos >= sceneDedupCosine) {
                Log.i(TAG, "moment confirm: not a new scene (cos=%.3f >= %.2f) — skip store"
                    .format(cos, sceneDedupCosine))
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
            // storeMoment (native shard/flush) is the ONLY failure that may still delete thumbFile:
            // nothing durable references it yet, so an orphan here is a real leak. Once storeMoment
            // RETURNS, the persisted payload's thumbPath points at thumbFile — from that point on
            // deleting it would break a durable timeline card instead of cleaning up after a failed
            // one, so count()/onMoment below get their OWN try/catch that only logs (Codex P2 fix:
            // this used to be one broad catch around all three calls, so a count()/onMoment failure
            // AFTER a successful store deleted the thumb the just-persisted payload still points at,
            // with the baseline already advanced — a durable card pointing at a missing file, no
            // retry). Rethrows nothing either way: the outer `finally` below still recycles `bitmap`.
            // type/momentId are placeholders — storeMoment stamps type="frame" and momentId=the new
            // point's own id itself (Spec §6 invariant), same convention QdrantEdgeMomentStore
            // documents on storeMoment(). The gate embedding above IS the stored vector: it is never
            // re-embedded here or anywhere else on this path. Held in a local (not inlined into the
            // storeMoment call below) so the fleet-upload enqueue further down can reuse the EXACT
            // same payload via [MomentPayload.toJson] rather than re-deriving it (Task 10).
            val framePayload = MomentPayload(
                type = MomentType.FRAME,
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
            )
            val id = try {
                store.storeMoment(vec, framePayload)
            } catch (e: Throwable) {
                // Nothing persisted — thumbFile is a genuine orphan. lastStoredGrid/Vec/Ms are
                // untouched, so the next gate fire retries cleanly against the same last-known-good
                // keyframe instead of a phantom one.
                Log.w(TAG, "moment store failed, will retry on the next gate fire: ${e.message}")
                if (thumbOk) thumbFile.delete()
                return
            }
            val storeMs = System.currentTimeMillis() - storeT0

            // Advance the baseline ONLY on a successful store — a failed embed/store above already
            // returned early, leaving lastStoredGrid/lastStoredVec/lastStoreMs untouched so the
            // NEXT gate fire retries against the same last-known-good keyframe, not a phantom one.
            lastStoredGrid = grid
            lastStoredVec = vec
            lastStoreMs = ts

            // Fleet-sync Task 10 (Spec §4 UP flow / §5 dual-write, decision C): enqueue a COPY of
            // the just-stored moment for upload. `uploadQueue` is null unless a fleet tier is
            // actually wired (Config.FLEET_URL set), in which case this whole block is skipped —
            // byte-for-byte today's local-only behavior (Global Constraint). Same "already durable,
            // only log on failure" discipline as count()/onMoment right below: storeMoment already
            // succeeded, so an enqueue failure here must never look like a store failure, and
            // [UploadQueue.enqueue] itself is already fail-soft (never throws) — the try/catch here
            // only guards [thumbFile.readBytes] / [buildUploadPayloadJson] ahead of it.
            val queue = uploadQueue
            if (queue != null) {
                try {
                    val thumbB64 = if (thumbOk) {
                        Base64.encodeToString(thumbFile.readBytes(), Base64.NO_WRAP)
                    } else ""
                    queue.enqueue(id, vec, buildUploadPayloadJson(framePayload.toJson(), ts, thumbB64))
                } catch (e: Throwable) {
                    Log.w(TAG, "moment stored (id=$id) but fleet upload enqueue failed (non-fatal): ${e.message}")
                }
            }

            // storeMoment succeeded: id/thumbPath are now durable. A count() or onMoment (Task 1.6's
            // HUD forward) failure past this point is logged and swallowed, never rethrown and never
            // a reason to touch thumbFile — the persisted payload already references it.
            val count = try {
                store.count()
            } catch (e: Throwable) {
                Log.w(TAG, "moment stored (id=$id) but count() failed: ${e.message}")
                null
            }
            Log.i(TAG, "moment stored: id=$id decision=$decision cos=%.3f (embed=${embedMs}ms store=${storeMs}ms) total=${count ?: "?"}"
                .format(cos))
            try {
                onMoment?.invoke(MomentHit(
                    id = id, score = 0f, type = MomentType.FRAME, momentId = id, timestampMs = ts,
                    thumbPath = if (thumbOk) thumbFile.absolutePath else "", label = "", bbox = "",
                ))
            } catch (e: Throwable) {
                Log.w(TAG, "moment stored (id=$id) but onMoment callback failed: ${e.message}")
            }

            // OCR "read channel" (Stage 3, additive): recognize any text lines in the SAME keyframe
            // pixels just stored above, embed each line's text with BGE into the 384-dim `text`
            // named-vector space, and store as `type=ocr` points sharing this moment's id. The frame
            // moment is ALREADY durable at this point (id/thumbPath persisted, baseline advanced
            // above), so — same contract as the region layer right below — a failure here is only
            // ever logged, never allowed to look like a frame-store failure. Takes its OWN bitmap
            // copy (not the region layer's `crop`s) because `bitmap` is unconditionally recycled by
            // the outer `finally` the instant this whole method returns, and OCR runs as a
            // SEPARATELY launched coroutine on the dedicated [ocrLane] (NOT embedLane — a ~4s OCR must
            // not block the next keyframe's embed) so it never extends the critical path that just
            // stored the frame, and runs in parallel with capture. id/ts/
            // episodeId are captured into locals before the launch — not read again once the
            // coroutine actually runs — so a startSession() reset racing in behind this launch can't
            // attribute the OCR points to the wrong session.
            val ocr = ocrEngine
            val bge = bgeEncoder
            if (ocr != null && bge != null) {
                val ocrMomentId = id
                val ocrTs = ts
                val ocrEpisodeId = episodeId
                val ocrBitmap = try {
                    bitmap.copy(Bitmap.Config.ARGB_8888, false)
                } catch (e: Throwable) {
                    Log.w(TAG, "moment $id: ocr bitmap copy failed: ${e.message}")
                    null
                }
                if (ocrBitmap != null) {
                    val w = bitmap.width.toFloat()
                    val h = bitmap.height.toFloat()
                    scope.launch(ocrLane) {
                        try {
                            val lines = ocr.recognize(ocrBitmap)
                            var stored = 0
                            for (line in lines) {
                                if (line.text.isBlank()) continue
                                val vec = bge.encode(line.text)
                                store.storeOcr(vec, MomentPayload(
                                    type = MomentType.OCR,
                                    momentId = ocrMomentId,
                                    episodeId = ocrEpisodeId,
                                    timestampMs = ocrTs,
                                    tEndMs = ocrTs,
                                    // No thumb of its own — an OCR hit is folded into an existing
                                    // frame/region hit by momentId at search time (MomentSearcher),
                                    // never displayed as its own HUD card, so it needs no thumbPath.
                                    thumbPath = "",
                                    // Same normalized-fraction convention as the region layer's bbox
                                    // just below (OcrLine.box is in ORIGINAL-image pixel coords).
                                    bbox = "%.3f,%.3f,%.3f,%.3f".format(
                                        line.box.left / w, line.box.top / h,
                                        line.box.right / w, line.box.bottom / h,
                                    ),
                                    label = "",
                                    yoloConf = 0f,
                                    verifyCos = 0f,
                                    text = line.text,
                                ))
                                stored++
                            }
                            Log.i(TAG, "ocr: momentId=$ocrMomentId lines=${lines.size} stored=$stored")
                        } catch (e: Throwable) {
                            Log.w(TAG, "moment $ocrMomentId: ocr failed: ${e.message}")
                        } finally {
                            ocrBitmap.recycle()
                        }
                    }
                }
            }

            // Region layer (Task 2.2, Spec §2 "CLIP-verify-the-label"): CLIP-verified YOLO
            // regions sharing THIS moment's id, layered ADDITIVELY on top of the frame keyframe
            // just stored above. Still on embedLane, still holding `bitmap` (the chosen keyframe,
            // recycled only in the outer `finally`), so the crop below is against the EXACT pixels
            // that were embedded/stored as the frame vector. One try/catch around the whole block
            // (not per-region): the frame moment is ALREADY durable at this point (id/thumbPath
            // persisted, baseline advanced above) — a region failure here is diagnostic-worthy but
            // must never be allowed to look like a frame-store failure.
            try {
                val cache = labelCache
                if (cache == null) {
                    Log.d(TAG, "moment $id: no LabelVectorCache — region layer skipped")
                } else {
                    val regions = regionsProvider()
                        .sortedByDescending { it.conf }
                        .take(REGIONS_MAX_PER_MOMENT)
                    for (region in regions) {
                        val box = RectF(
                            region.left * bitmap.width, region.top * bitmap.height,
                            region.right * bitmap.width, region.bottom * bitmap.height,
                        )
                        val crop = cropFrom(bitmap, box) ?: continue
                        try {
                            val regionVec = cropEncoder.encode(crop)
                            val verifyCos = cache.verify(regionVec, region.label)
                            val verified = verifyCos >= verifyGate
                            // dedup-check-style diagnostic line (PerceptionPipeline's convention) so
                            // the per-backend verifyGate can be calibrated on real data at Stage 2.
                            Log.i(TAG, "region: label=${region.label} verifyCos=%.3f yoloConf=%.3f -> %s"
                                .format(verifyCos, region.conf, if (verified) "stored" else "label-dropped"))
                            store.storeRegion(regionVec, MomentPayload(
                                type = MomentType.REGION,
                                momentId = id,
                                episodeId = episodeId,
                                timestampMs = ts,
                                tEndMs = ts,
                                thumbPath = if (thumbOk) thumbFile.absolutePath else "",
                                bbox = "%.3f,%.3f,%.3f,%.3f".format(region.left, region.top, region.right, region.bottom),
                                // Verified → keep the label as a display tag; unverified → keep the
                                // vector as a recall signal but drop the (unreliable) label (Spec §2).
                                label = if (verified) region.label else "",
                                yoloConf = region.conf,
                                verifyCos = verifyCos,
                                text = "",
                            ))
                        } finally {
                            crop.recycle()
                        }
                    }
                }
            } catch (e: Throwable) {
                Log.w(TAG, "moment stored (id=$id) but region layer failed: ${e.message}")
            }
        } finally {
            bitmap.recycle()
        }
    }

    /** [PIXEL_SCALE_SIDE]-square gray copy → [GRID_SIDE]x[GRID_SIDE] luma grid (Spec §4 step 3's
     *  "~160 px gray copy"), used for both the pixel pre-gate similarity and the sharpness score so
     *  a candidate frame is only downscaled once. `createScaledBitmap`'s "may return the source
     *  itself" contract (see [cropFrom]'s KDoc, `CropGeometry.kt`, for the same gotcha) can't apply
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

/**
 * Builds the fleet-upload payload JSON for one just-stored moment (plan Task 10, Spec §4 UP flow
 * / §6 schema) from the SAME [MomentPayload] JSON [MomentCapture.confirmAndStore] just persisted
 * locally, adding the two upload-only fields the fleet hub needs: [syncTs] (Spec §5's
 * last-writer-wins conflict timestamp) and [thumbB64] (the thumbnail bytes, base64-encoded — the
 * device's own `thumb_path` is a device-local file path, meaningless off-device, so the pixels
 * themselves have to travel for another node to render this moment, Spec §6). [framePayloadJson]
 * is [MomentPayload.toJson]'s output, unmodified; [thumbB64] "" (never null) when the thumbnail
 * write failed (mirrors [MomentPayload.thumbPath]'s own "" convention on that same failure).
 *
 * Pure/no I/O — pulled out of [MomentCapture] (same "extract the pure part" move as this file's
 * own [decide]) so the payload-shaping half of the enqueue-on-store behavior can be reasoned about
 * on its own; `MomentCaptureUploadTest` still drives the real [MomentCapture] end to end (fake
 * [CropEncoder]/[MomentStore], real [UploadQueue]) to cover the wiring this function alone can't.
 */
fun buildUploadPayloadJson(framePayloadJson: String, syncTs: Long, thumbB64: String): String =
    JSONObject(framePayloadJson).put("sync_ts", syncTs).put("thumb_b64", thumbB64).toString()
