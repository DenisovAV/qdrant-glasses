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

    // BOTH towers load EAGERLY here (in the ctor). Loading them at the SAME instant as the camera
    // startup inrush + the detector spikes DSP+CPU+power together and reboots the AR1 — even though the
    // encoders' STEADY-STATE is only ~668 MB and fits the ~2.4 GB budget fine (measured, SiglipMemoryTest).
    // The fix is SEQUENCING, done at the app level: GlassesComponents.load (which constructs this encoder)
    // runs to completion — loading both towers — BEFORE MainActivity starts the camera (it gates
    // cameraManager.start() on the app leaving AppState.Loading). So both loads finish while the camera is
    // still OFF; the two peaks never overlap. Do NOT reintroduce a lazy/deferred text load here — that put
    // the text spike back in overlap with the running camera and still rebooted.
    private val vision = SiglipVisionEncoder(context)
    private val text = SiglipTextEncoder(context)

    override fun encode(crop: Bitmap): FloatArray = vision.encode(crop)

    override fun encodeText(query: String): FloatArray = text.encode(query)

    override fun close() {
        vision.close()
        text.close()
    }
}
