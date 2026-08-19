package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap

/**
 * On-device SigLIP2-base crop + text encoder — a fully-on-device alternative to [QnnB32CropEncoder]
 * (CLIP ViT-B/32). Vision (SigLIP2 ViT-B/16) runs **W8A16 on the Hexagon NPU** (~177ms encode via
 * [SiglipVisionEncoder]); text runs **int8 on the CPU** (~254–468ms once per query via
 * [SiglipTextEncoder], with SigLIP's Gemma/SentencePiece tokenizer). Both towers share ONE 768-dim
 * space, so a crop vector and a text-query vector compare directly.
 *
 * Why SigLIP over the current CLIP: measured on the live AR1 it out-retrieves the deployed CLIP —
 * top-1 0.667 vs 0.500, and absent-object rejection floor 0.069 vs 0.221 (the modality-gap
 * discriminator) — even though SigLIP's vision quantizes worse (0.82 vs 0.99 cosine-to-fp32), because
 * its fp32 lead absorbs the bigger quant loss. Full measurements:
 * `qdrant_glasses_private/a4-siglip-eval/FINDINGS.md`.
 *
 * DIFFERENT vector space + dim from every CLIP index → own namespace ("siglipnpu"), 768-dim, so a
 * SigLIP crop vector is never searched against a 512-dim CLIP-built collection (see [CropEncoderFactory]).
 *
 * Must be called OFF the main thread (ORT inference), like every [CropEncoder].
 */
class SiglipCropEncoder(context: Context) : CropEncoder {
    override val dim: Int = 768

    // SigLIP2's text→image cosine band on-device is compressed and shifted vs CLIP's — measured from
    // the 14-crop A4 rehearsal: a query naming a PRESENT object scores ~0.12, an ABSENT object floors
    // at ~0.069. 0.09 is the midpoint of that gap; recalibrate against a live on-device rehearsal
    // (integration Task 5) before trusting search precision — this is a starting value, not final.
    override val visionMinScore: Float = 0.09f

    // Vision loads eagerly — it is the per-crop capture path, needed as soon as recording starts.
    private val vision = SiglipVisionEncoder(context)

    // Text is DEFERRED. Constructing it at the SAME instant as the vision DSP-context load + the camera
    // startup inrush spikes DSP+CPU+power together and reboots the AR1 — even though the STEADY-STATE of
    // both encoders is only ~668 MB and fits the ~2.4 GB budget fine (measured, SiglipMemoryTest). It is
    // the simultaneous LOADING transient at startup that kills it, not resident RAM. So build text lazily
    // and warm it on a background thread a few seconds after startup, once vision + the camera have
    // settled — it is ready before the first voice query without ever coinciding with the startup peak,
    // and text is CPU-only so it never adds to the DSP load the vision context + detector already carry.
    private val textLazy = lazy { SiglipTextEncoder(context) }

    init {
        Thread {
            try {
                Thread.sleep(WARM_DELAY_MS)
                textLazy.value.encode("warm up")   // force the lazy load off the startup path
            } catch (_: Throwable) { /* app closing / interrupted — encodeText will lazy-init on demand */ }
        }.apply { isDaemon = true; name = "siglip-text-warm" }.start()
    }

    override fun encode(crop: Bitmap): FloatArray = vision.encode(crop)

    // If a query arrives before the background warm finished, this lazy-inits text here (one-time stall);
    // in practice the ~8 s warm completes well before the first search.
    override fun encodeText(query: String): FloatArray = textLazy.value.encode(query)

    override fun close() {
        vision.close()
        if (textLazy.isInitialized()) textLazy.value.close()
    }

    companion object {
        // Long enough for the vision DSP-context load (~1.7 s) + camera startup inrush to clear before
        // the text load's CPU/memory spike lands. Tunable if the reboot recurs.
        private const val WARM_DELAY_MS = 8000L
    }
}
