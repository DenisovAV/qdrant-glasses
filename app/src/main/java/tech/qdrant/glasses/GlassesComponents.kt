package tech.qdrant.glasses

import android.app.Application
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import tech.qdrant.glasses.detect.DetectorFactory
import tech.qdrant.glasses.detect.ObjectDetector
import tech.qdrant.glasses.detect.ObjectTracker
import tech.qdrant.glasses.embedding.BgeTextEncoder
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.embedding.EncoderFactory
import tech.qdrant.glasses.embedding.LabelVectorCache
import tech.qdrant.glasses.embedding.TextEncoder
import tech.qdrant.glasses.embedding.VisionEncoder
import tech.qdrant.glasses.ocr.OcrEngine
import tech.qdrant.glasses.pipeline.MomentCapture
import tech.qdrant.glasses.search.MomentRetriever
import tech.qdrant.glasses.search.SherpaVadAsr
import tech.qdrant.glasses.storage.DbBenchRunner
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.storage.QdrantEdgeMomentStore
import tech.qdrant.glasses.storage.VisionMemoryStore
import tech.qdrant.glasses.stream.HudEvents
import tech.qdrant.glasses.stream.HudPublisher
import java.io.File

/**
 * Bundles every model/store/detector GlassesViewModel depends on, plus their boot sequence
 * ([load]) and teardown order ([close]). Extracted from GlassesViewModel's `init{}` /
 * `onCleared` (Task 5 of the God-object decomposition) — behavior-preserving: identical
 * loading order, identical per-mode nullability (LEGACY vs OBJECTS), identical close order.
 *
 * [load] THROWS on any failure (missing asset, native init error, …) — the caller (VM `init`)
 * owns the try/catch that maps a load failure onto [AppState.Error].
 */
class GlassesComponents(
    val store: VisionMemoryStore,
    val visionEncoder: VisionEncoder?,
    val textEncoder: TextEncoder?,
    val bgeEncoder: BgeTextEncoder,
    val detector: ObjectDetector?,
    val tracker: ObjectTracker?,
    val cropEncoder: CropEncoder?,
    val retriever: MomentRetriever?,
    // The memory path (Task 2.4: the ONLY OBJECTS-mode memory path, default ON via
    // Config.MOMENT_MEMORY). Both null when mode != OBJECTS, or when the sysprop explicitly
    // disables them (`debug.qdrant.memory=0`, a kill switch for A/B/regression) — "nullable by
    // mode/opt-out, never by timing", same rule [retriever] above already follows.
    val momentStore: MomentStore?,
    val momentCapture: MomentCapture?,
    // Stage 3 "OCR read channel" — null whenever momentCapture is (never active without the moment
    // pipeline; see [load]'s MOMENT_MEMORY gate), OR when `ocr/` assets are missing/fail to load
    // (guarded independently — a missing OCR model must not fail the whole app boot).
    val ocrEngine: OcrEngine?,
    // Stage 3 live text-region highlighting: DBNet on the NPU, run per detect-frame by
    // PerceptionPipeline (guarded independently — a missing EPContext model must not fail boot).
    val dbnetDetector: tech.qdrant.glasses.ocr.QnnDbnetDetector?,
    // Stage 3 NPU recognizer: ViTSTR (W8A16), used by [ocrEngine]'s NPU path. Owned here (not by
    // OcrEngine) and closed in [close]; null → OcrEngine reads on the CPU. Guarded independently.
    val vitstrRec: tech.qdrant.glasses.ocr.QnnVitstrRecognizer?,
    // Fleet-sync Task 6 (Spec §3/§7/§9 P1): the pulled fleet corpus, or null when Config.FLEET_URL
    // is blank OR the pull failed (FleetSync.pull already logs+swallows every Throwable) — same
    // nullable-optional-feature contract as momentStore/momentCapture above. GlassesViewModel passes
    // this straight through as MomentSearcher's `fleet` argument; a null here just means "no fleet
    // tier this session", byte-for-byte today's local-only search (Global Constraint). Owned here so
    // [close] can release its native EdgeShard alongside [momentStore]'s.
    val fleetStore: tech.qdrant.glasses.fleet.FleetShardStore?,
) : AutoCloseable {

    companion object {
        private const val TAG = "GlassesComponents"

        /**
         * Runs on: IO (called from GlassesViewModel's init, inside viewModelScope.launch(Dispatchers.IO)).
         *
         * [scope]/[embedLane]/[isRecording] are needed ONLY to build [MomentCapture] — it is
         * constructed here, alongside its [QdrantEdgeMomentStore], so the OBJECTS branch owns the
         * whole memory path in one place, the same way it already owns [ObjectTracker]/[CropEncoder].
         * [embedLane] is [MomentCapture]'s OWN single-thread dispatcher for every `OrtSession.run`
         * call it makes (frame confirms + region embeds, Spec §8.2) — Task 2.4 retired
         * [tech.qdrant.glasses.pipeline.PerceptionPipeline]'s crop embed, so nothing else shares
         * this lane anymore.
         *
         * [hud] is needed ONLY to forward [MomentCapture.onMoment] to the HUD timeline (Task 1.6) —
         * same instance the VM constructs independently of [GlassesComponents] and later hands to
         * `PerceptionPipeline`/`MomentSearcher` directly (see the "moment-timeline backfill" comment
         * below for why `hud` itself isn't owned here).
         */
        suspend fun load(
            app: Application,
            mode: AppMode,
            scope: CoroutineScope,
            embedLane: CoroutineDispatcher,
            isRecording: () -> Boolean,
            hud: HudPublisher,
        ): GlassesComponents {
            Log.d(TAG, "load: opening VisionMemoryStore")
            val store = VisionMemoryStore(app)
            Log.d(TAG, "load: VisionMemoryStore OK, stored frames=${store.count()}")
            store.dumpAll()  // DIAG: log the whole base at startup
            // retriever is created below (OBJECTS mode) with the encoder's own vision gate;
            // LEGACY mode falls back to the default gate.

            // The whole-frame CLIP encoders (~945MB of on-device weights) are LEGACY-only:
            // in OBJECTS mode crop embedding runs on the Mac (SigLIP2), so these models are
            // never used — and are excluded from the APK via androidResources.ignoreAssetsPattern.
            // Loading must therefore be gated by mode too: touching a missing asset here would
            // throw and the init try/catch would never reach Idle.
            var visionEncoder: VisionEncoder? = null
            var textEncoder: TextEncoder? = null
            if (mode == AppMode.LEGACY) {
                Log.d(TAG, "load: loading vision encoder [${EncoderFactory.backend}]")
                visionEncoder = EncoderFactory.createVision(app)
                Log.d(TAG, "load: vision encoder OK")

                Log.d(TAG, "load: loading text encoder [${EncoderFactory.backend}]")
                textEncoder = EncoderFactory.createText(app)
                Log.d(TAG, "load: text encoder OK")
            }

            val bgeEncoder = BgeTextEncoder(app)
            Log.d(TAG, "load: bge encoder OK")

            var detector: ObjectDetector? = null
            var tracker: ObjectTracker? = null
            var cropEncoder: CropEncoder? = null
            var retriever: MomentRetriever? = null
            var momentStore: MomentStore? = null
            var momentCapture: MomentCapture? = null
            var ocrEngine: OcrEngine? = null
            var dbnetDetector: tech.qdrant.glasses.ocr.QnnDbnetDetector? = null
            var vitstrRec: tech.qdrant.glasses.ocr.QnnVitstrRecognizer? = null
            var fleetStore: tech.qdrant.glasses.fleet.FleetShardStore? = null
            if (mode == AppMode.OBJECTS) {
                detector = DetectorFactory.create(app)
                // Stage 3 live text detector (NPU) — guarded like ocrEngine: a missing
                // `assets/ocr/dbnet-det-epctx.onnx` just leaves text-highlighting off, no boot failure.
                dbnetDetector = try {
                    tech.qdrant.glasses.ocr.QnnDbnetDetector(app).also { Log.d(TAG, "load: DBNet-QNN OK") }
                } catch (e: Throwable) {
                    Log.w(TAG, "load: DBNet-QNN unavailable, text-highlight off: ${e.message}"); null
                }
                tracker = ObjectTracker(confirmSightings = 3)
                cropEncoder = CropEncoderFactory.create(app)
                // Build the retriever with THIS encoder's calibrated vision gate (SigLIP2 and
                // TinyCLIP have different cosine scales, so an absent query returns nothing).
                retriever = MomentRetriever(store, visionMinScore = cropEncoder.visionMinScore)
                Log.i(TAG, "object mode ready (backend=${CropEncoderFactory.backend}, dim=${cropEncoder.dim})")
                if (Config.MOMENT_MEMORY) {
                    // The whole-frame + CLIP-verified-region keyframe memory path (Task 2.4: the
                    // ONLY OBJECTS-mode memory write path — the crop-embed-and-store block it used
                    // to run alongside is retired). Namespace matches the crop encoder, same
                    // convention every on-disk collection in this app uses, so switching
                    // CropEncoderFactory.backend still needs no manual data wipe.
                    val thumbsDir = File(app.filesDir, "moment_thumbs").also { it.mkdirs() }
                    // dim = cropEncoder.dim (Codex P2 fix): the store defaulted to a hard-coded
                    // 512, which matches QNN_B32/ON_DEVICE but fails storeMoment's dim `require`
                    // the moment CropEncoderFactory.backend is MAC_ENDPOINT (768-dim SigLIP2).
                    val ms = QdrantEdgeMomentStore(app, namespace = CropEncoderFactory.namespace, dim = cropEncoder.dim)
                    momentStore = ms
                    // Label-text vector cache (Task 2.1/2.2, Spec §2 "CLIP-verify-the-label") — one
                    // encodeText call per distinct YOLO label, ever, memoized + persisted to disk.
                    // Namespaced by CropEncoderFactory.namespace (Task 2.1 concern #2): a text vector
                    // from one crop-encoder backend's space is meaningless cosine'd against a region
                    // vector from another, so switching CropEncoderFactory.backend must not reuse a
                    // stale file — same convention momentStore already uses for its own on-disk
                    // collection.
                    val labelCache = LabelVectorCache(
                        encodeText = { cropEncoder.encodeText(it) },
                        // Codex P1 fix: lets loadPersisted drop a truncated persisted line (a torn
                        // append) instead of caching a short vector that would later be indexed
                        // past its end against a full-length region vector.
                        dim = cropEncoder.dim,
                        persistFile = File(app.filesDir, "label_vectors_${CropEncoderFactory.namespace}.tsv"),
                    )
                    // Stage 3 "OCR read channel": constructed once, alongside momentCapture, so it is
                    // reused (not per-frame) exactly like cropEncoder/labelCache above. Guarded on
                    // its OWN try/catch — a device missing `assets/ocr/` (or any other load failure)
                    // must not fail the whole app boot the way a missing crop-encoder asset would;
                    // it just means Stage 3 stays off (MomentCapture treats a null ocrEngine as
                    // "OCR disabled", same nullable-optional-feature contract as labelCache).
                    // Stage 3 recognizer on the NPU: ViTSTR (non-recurrent ViT, W8A16) — the CRNN's LSTM
                    // can't run on the AR1 HTP (unrolled-quant collapses, native-quant hits the requant
                    // limit, FP16 unsupported — see ocr-ondevice-spike). Guarded like dbnetDetector: a
                    // missing `assets/ocr/vitstr-epctx.onnx` just makes OcrEngine fall back to its CPU
                    // PP-OCR path. Paired with dbnetDetector (NPU det) it puts the WHOLE read channel on
                    // the HTP (~10ms det + ~24ms/line rec vs the CPU path's ~4s). dbnetDetector is shared
                    // with PerceptionPipeline's live highlighter — ORT run() is thread-safe.
                    vitstrRec = if (!Config.OCR_NPU) {
                        Log.i(TAG, "OCR recognizer = CPU CRNN (proven path; set debug.qdrant.ocr_npu=1 for ViTSTR-NPU)")
                        null
                    } else try {
                        tech.qdrant.glasses.ocr.QnnVitstrRecognizer(app).also { Log.d(TAG, "load: ViTSTR-QNN OK → NPU OCR rec") }
                    } catch (e: Throwable) {
                        Log.w(TAG, "load: ViTSTR-QNN unavailable, OCR rec stays on CPU: ${e.message}"); null
                    }
                    ocrEngine = try {
                        // Read-channel OCR: CPU DBNet det@1536 (full coverage — catches small/far text the
                        // 640² NPU DBNet drops) + ViTSTR-NPU rec (~24ms/line). dbnetDetector stays for
                        // PerceptionPipeline's live highlighter only.
                        OcrEngine(app, vitstr = vitstrRec).also { Log.d(TAG, "load: ocr engine OK") }
                    } catch (e: Throwable) {
                        Log.w(TAG, "load: ocr engine unavailable, Stage 3 disabled: ${e.message}")
                        null
                    }
                    momentCapture = MomentCapture(
                        scope = scope,
                        embedLane = embedLane,
                        cropEncoder = cropEncoder,
                        store = ms,
                        momentThumbsDir = thumbsDir,
                        isRecording = isRecording,
                        labelCache = labelCache,
                        ocrEngine = ocrEngine,
                        bgeEncoder = bgeEncoder,
                    ).also { mc ->
                        // Task 1.6: forward each stored keyframe to the HUD timeline — register the
                        // thumb for /thumb/<key> BEFORE pushing the event. On the WIRED path
                        // (MjpegServer) that order is a real guarantee: registerThumb() there is a
                        // synchronous in-memory map write, so it's always visible before the event
                        // pushed right after it. On the WIRELESS path (MjpegPusher) it is NOT —
                        // Codex review (P2, deferred): registerThumb/pushEvent are independent
                        // fire-and-forget async OkHttp POSTs there (push_thumb / push_event) with no
                        // completion ordering between them, and neither HudPublisher nor MjpegPusher
                        // exposes a hook to make one wait on the other. Calling registerThumb first
                        // only narrows the race, it doesn't close it: a client can still see the
                        // `moment` event before the relay has the thumb and 404 once on
                        // /thumb/<key> — a non-fatal class of failure (one blank HUD thumbnail),
                        // never a reason to add a blocking wait onto embedLane here.
                        mc.onMoment = { hit ->
                            val key = File(hit.thumbPath).nameWithoutExtension
                            hud.registerThumb(key, hit.thumbPath)
                            // frameCount(), not count() (whole-branch review fix, F1): count() is
                            // every point across BOTH channels (frame + region), so a keyframe with
                            // its verified regions was reporting as up to ~7 "moments" per store.
                            hud.pushEvent(HudEvents.momentEvent(key, hit.timestampMs, ms.frameCount()))
                        }
                    }
                    Log.i(TAG, "moment mode ready (namespace=${CropEncoderFactory.namespace}), moments=${ms.count()}")
                    // Fleet-sync Task 6 (Spec §3/§7/§9 P1): OPTIONAL cloud tier, gated purely on
                    // Config.FLEET_URL (Global Constraint — blank URL means byte-for-byte today's
                    // offline behavior, so this whole block is skipped, not just no-op'd). Runs on
                    // THIS coroutine (load() is suspend precisely so this pull() can happen here,
                    // off-main like every other blocking call in this function — see GlassesViewModel's
                    // `viewModelScope.launch(Dispatchers.IO) { GlassesComponents.load(...) }`).
                    // FleetSync.pull() itself never throws (wraps create-snapshot/download/unpack/load
                    // in one try/catch, Spec §7) — a null result here just means no fleet this session.
                    if (Config.FLEET_URL.isNotBlank()) {
                        val fleetSync = tech.qdrant.glasses.fleet.FleetSync(
                            tech.qdrant.glasses.fleet.FleetQdrantClient(Config.FLEET_URL),
                            app.filesDir, cropEncoder.dim,
                        )
                        fleetStore = fleetSync.pull()
                        Log.i(TAG, "load: fleet pull ${if (fleetStore != null) "OK" else "unavailable (local-only)"}")
                    }
                }
                // Optional in-app vector-DB benchmark, gated + off the main thread. This file
                // compiles into both flavors, so the actual sysprop-check + launch is indirected
                // through a flavor seam: a no-op in the demo flavor, the real thing in benchmark
                // (see DbBenchRunner's KDoc, and its two flavor copies, for why).
                DbBenchRunner.runIfEnabled(app)
                // NOTE: the moment-timeline backfill for an already-connected HUD does NOT happen
                // here — it needs `hud`, which the VM constructs independently of `components`.
                // The VM's init pushes momentStore.timeline() itself right after load() returns.
            }

            // Pre-warm the ambient ASR model (~290MB) off the main thread so the first
            // recording doesn't block the UI loading it. ensureLoaded is idempotent +
            // @Synchronized, so AmbientTranscriber.start() becomes a warm cache hit. Only
            // LEGACY uses the heard channel — in OBJECTS mode ambient segments are dropped (no
            // textEncoder), so loading the model there is ~290MB of wasted RAM. Gate to LEGACY.
            if (mode == AppMode.LEGACY) {
                SherpaVadAsr.ensureLoaded(app)
            }

            return GlassesComponents(
                store = store,
                visionEncoder = visionEncoder,
                textEncoder = textEncoder,
                bgeEncoder = bgeEncoder,
                detector = detector,
                tracker = tracker,
                cropEncoder = cropEncoder,
                retriever = retriever,
                momentStore = momentStore,
                momentCapture = momentCapture,
                ocrEngine = ocrEngine,
                dbnetDetector = dbnetDetector,
                vitstrRec = vitstrRec,
                fleetStore = fleetStore,
            )
        }
    }

    /**
     * Runs on: main (called from GlassesViewModel.onCleared).
     *
     * Exact close order preserved from the former `onCleared` cascade.
     */
    override fun close() {
        visionEncoder?.close()
        textEncoder?.close()
        bgeEncoder.close()
        store.close()
        detector?.close()
        dbnetDetector?.close()
        cropEncoder?.close()
        momentStore?.close()
        fleetStore?.close()
        ocrEngine?.close()          // NPU-mode OcrEngine doesn't own dbnet/vitstr (shared) — close them after
        vitstrRec?.close()
    }
}
