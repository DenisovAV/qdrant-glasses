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

    // Search-hit gate (feeds MomentRetriever via GlassesComponents). SigLIP2's text→image cosine band
    // is compressed vs CLIP's. CALIBRATED on-device (CropEncoderAbTest, Task 5): present-query top-1
    // cosines 0.064–0.119, absent distractors 0.049–0.080. 0.085 rejects both distractors and keeps the
    // stronger 7/12 present; the bands OVERLAP so this trades some recall for precision. Kept in sync
    // with CropEncoderFactory.searchGate (SIGLIP_NPU). A bigger distractor set would firm it.
    override val visionMinScore: Float = 0.085f

    // BOTH towers load EAGERLY here (in the ctor) — and that is fine. Measured on a cleanly-rebooted
    // AR1: switching the live backend to SIGLIP_NPU loads both towers (text ~10s: DJL tokenizer ~7s +
    // ORT session ~3s), reaches AppState.Idle, runs steady-state (vision ~82–113ms/crop NPU, text
    // ~600–780ms CPU) with the camera attached, and lowmemorykiller reports "watermarks ok" — ZERO
    // memory pressure. Steady-state is ~668 MB, well under the ~2.4 GB budget. The 269 MB text tower
    // does NOT OOM. An earlier note here claimed the simultaneous load "reboots the AR1" and required
    // gating the camera on AppState.Loading — that was a MISDIAGNOSIS: the AR1 reboots spontaneously on
    // its own (seen on a pure-CLIP build at 96% battery), and after ANY reboot the CameraX HAL wedges
    // (blank preview), which read as a SigLIP crash. No eager-load sequencing is needed; the camera-gate
    // was reverted (0ffa860). Evidence: qdrant_glasses_private/a4-siglip-eval/FINDINGS.md Part 8-retracted.
    private val vision = SiglipVisionEncoder(context)
    private val text = SiglipTextEncoder(context)

    override fun encode(crop: Bitmap): FloatArray = vision.encode(crop)

    override fun encodeText(query: String): FloatArray = text.encode(query)

    override fun close() {
        vision.close()
        text.close()
    }
}
