package tech.qdrant.glasses

import android.app.Application
import android.util.Log
import tech.qdrant.glasses.detect.DetectorFactory
import tech.qdrant.glasses.detect.ObjectDetector
import tech.qdrant.glasses.detect.ObjectTracker
import tech.qdrant.glasses.embedding.BgeTextEncoder
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.embedding.EncoderFactory
import tech.qdrant.glasses.embedding.TextEncoder
import tech.qdrant.glasses.embedding.VisionEncoder
import tech.qdrant.glasses.search.MomentRetriever
import tech.qdrant.glasses.search.SherpaVadAsr
import tech.qdrant.glasses.storage.ObjectStore
import tech.qdrant.glasses.storage.VisionMemoryStore

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
    val objectStore: ObjectStore?,
    val retriever: MomentRetriever?,
) : AutoCloseable {

    companion object {
        private const val TAG = "GlassesComponents"

        fun load(app: Application, mode: AppMode): GlassesComponents {
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
            var objectStore: ObjectStore? = null
            var retriever: MomentRetriever? = null
            if (mode == AppMode.OBJECTS) {
                detector = DetectorFactory.create(app)
                tracker = ObjectTracker(confirmSightings = 3)
                cropEncoder = CropEncoderFactory.create(app)
                objectStore = ObjectStore(
                    app,
                    dim = cropEncoder.dim,
                    namespace = CropEncoderFactory.namespace,
                )
                // Build the retriever with THIS encoder's calibrated vision gate (SigLIP2 and
                // TinyCLIP have different cosine scales, so an absent query returns nothing).
                retriever = MomentRetriever(store, visionMinScore = cropEncoder.visionMinScore)
                Log.i(TAG, "object mode ready (backend=${CropEncoderFactory.backend}, dim=${cropEncoder.dim}), objects=${objectStore.count()}")
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
            )
        }
    }

    /** Exact close order preserved from the former `onCleared` cascade. */
    override fun close() {
        visionEncoder?.close()
        textEncoder?.close()
        bgeEncoder.close()
        store.close()
        detector?.close()
        cropEncoder?.close()
        objectStore?.close()
    }
}
