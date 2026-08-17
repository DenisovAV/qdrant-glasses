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
import tech.qdrant.glasses.pipeline.MomentCapture
import tech.qdrant.glasses.search.MomentRetriever
import tech.qdrant.glasses.search.SherpaVadAsr
import tech.qdrant.glasses.storage.DbBenchRunner
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.storage.QdrantEdgeMomentStore
import tech.qdrant.glasses.storage.VectorStore
import tech.qdrant.glasses.storage.VectorStoreFactory
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
    val objectStore: VectorStore?,
    val retriever: MomentRetriever?,
    // Opt-in whole-frame keyframe memory (Task 1.5, Config.MOMENT_MEMORY). Both null unless the
    // sysprop is on AND mode == OBJECTS — "nullable by mode/opt-in, never by timing", same rule
    // as objectStore/retriever above.
    val momentStore: MomentStore?,
    val momentCapture: MomentCapture?,
) : AutoCloseable {

    companion object {
        private const val TAG = "GlassesComponents"

        /**
         * Runs on: IO (called from GlassesViewModel's init, inside viewModelScope.launch(Dispatchers.IO)).
         *
         * [scope]/[embedLane]/[isRecording] are needed ONLY to build [MomentCapture] (Task 1.5) —
         * it is constructed here, alongside its [QdrantEdgeMomentStore], so the OBJECTS branch
         * owns the whole opt-in moment path in one place, the same way it already owns
         * objectStore/retriever. [embedLane] must be the SAME dispatcher instance the caller later
         * hands to [tech.qdrant.glasses.pipeline.PerceptionPipeline] (Spec §8.2 — one lane for
         * every `OrtSession.run` call, crop or moment).
         *
         * [hud] is needed ONLY to forward [MomentCapture.onMoment] to the HUD timeline (Task 1.6) —
         * same instance the VM constructs independently of [GlassesComponents] and later hands to
         * `PerceptionPipeline`/`ObjectSearcher` directly (see the "fill any already-connected HUDs'
         * rails" comment below for why `hud` itself isn't owned here).
         */
        fun load(
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
            var objectStore: VectorStore? = null
            var retriever: MomentRetriever? = null
            var momentStore: MomentStore? = null
            var momentCapture: MomentCapture? = null
            if (mode == AppMode.OBJECTS) {
                detector = DetectorFactory.create(app)
                tracker = ObjectTracker(confirmSightings = 3)
                cropEncoder = CropEncoderFactory.create(app)
                // The vector engine is the single build-time switch (VectorStoreFactory.backend);
                // QDRANT_EDGE is the default → identical behavior to the former direct ObjectStore.
                // Namespace stays per-crop-encoder so each variant keeps its own on-disk collection.
                objectStore = VectorStoreFactory.create(
                    app,
                    dim = cropEncoder.dim,
                    namespace = CropEncoderFactory.namespace,
                )
                // Build the retriever with THIS encoder's calibrated vision gate (SigLIP2 and
                // TinyCLIP have different cosine scales, so an absent query returns nothing).
                retriever = MomentRetriever(store, visionMinScore = cropEncoder.visionMinScore)
                Log.i(TAG, "object mode ready (store=${objectStore.name}, backend=${CropEncoderFactory.backend}, dim=${cropEncoder.dim}), objects=${objectStore.count()}")
                if (Config.MOMENT_MEMORY) {
                    // Opt-in whole-frame keyframe path (Task 1.5, episodic-memory plan Stage 1),
                    // running IN PARALLEL with the crop-store path above so the two can be A/B'd
                    // on one walk-through (Config.MOMENT_MEMORY / debug.qdrant.memory). Namespace
                    // matches the crop encoder, same convention objectStore uses, so switching
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
                    // stale file — same convention objectStore/momentStore already use for their
                    // on-disk collections.
                    val labelCache = LabelVectorCache(
                        encodeText = { cropEncoder.encodeText(it) },
                        // Codex P1 fix: lets loadPersisted drop a truncated persisted line (a torn
                        // append) instead of caching a short vector that would later be indexed
                        // past its end against a full-length region vector.
                        dim = cropEncoder.dim,
                        persistFile = File(app.filesDir, "label_vectors_${CropEncoderFactory.namespace}.tsv"),
                    )
                    momentCapture = MomentCapture(
                        scope = scope,
                        embedLane = embedLane,
                        cropEncoder = cropEncoder,
                        store = ms,
                        momentThumbsDir = thumbsDir,
                        isRecording = isRecording,
                        labelCache = labelCache,
                    ).also { mc ->
                        // Task 1.6: forward each stored keyframe to the HUD timeline — register the
                        // thumb for /thumb/<key> BEFORE pushing the event, same call ORDER
                        // PerceptionPipeline uses for the object rail (its ~line 287-288). On the
                        // WIRED path (MjpegServer) that order is a real guarantee: registerThumb()
                        // there is a synchronous in-memory map write, so it's always visible before
                        // the event pushed right after it. On the WIRELESS path (MjpegPusher) it is
                        // NOT — Codex review (P2, deferred): registerThumb/pushEvent are independent
                        // fire-and-forget async OkHttp POSTs there (push_thumb / push_event) with no
                        // completion ordering between them, and neither HudPublisher nor MjpegPusher
                        // exposes a hook to make one wait on the other. Calling registerThumb first
                        // only narrows the race, it doesn't close it: a client can still see the
                        // `moment` event before the relay has the thumb and 404 once on
                        // /thumb/<key>. PerceptionPipeline's object path has the SAME best-effort
                        // ordering for the SAME reason, so this matches rather than fixes it — a
                        // raced 404 just leaves one HUD thumbnail blank, the same non-fatal class of
                        // failure MjpegPusher.registerThumb's own KDoc already accepts for a lost
                        // thumb, never a reason to add a blocking wait onto embedLane here.
                        mc.onMoment = { hit ->
                            val key = File(hit.thumbPath).nameWithoutExtension
                            hud.registerThumb(key, hit.thumbPath)
                            hud.pushEvent(HudEvents.momentEvent(key, hit.timestampMs, ms.count()))
                        }
                    }
                    Log.i(TAG, "moment mode ready (namespace=${CropEncoderFactory.namespace}), moments=${ms.count()}")
                }
                // Optional in-app vector-DB benchmark, gated + off the main thread. This file
                // compiles into both flavors, so the actual sysprop-check + launch is indirected
                // through a flavor seam: a no-op in the demo flavor, the real thing in benchmark
                // (see DbBenchRunner's KDoc, and its two flavor copies, for why).
                DbBenchRunner.runIfEnabled(app)
                // NOTE: the "fill any already-connected HUDs' rails" broadcast does NOT happen
                // here — it needs `hud`, which the VM constructs independently of `components`.
                // The VM's init calls hud.broadcastRailSnapshot() itself right after load() returns.
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
                objectStore = objectStore,
                retriever = retriever,
                momentStore = momentStore,
                momentCapture = momentCapture,
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
        cropEncoder?.close()
        objectStore?.close()
        momentStore?.close()
    }
}
