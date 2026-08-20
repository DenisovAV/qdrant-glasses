package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap

/** Embeds an object crop and a text query into the SAME vector space. */
interface CropEncoder : AutoCloseable {
    fun encode(crop: Bitmap): FloatArray       // image → vector
    fun encodeText(query: String): FloatArray  // text  → vector
    val dim: Int
    /**
     * Absolute cosine below which a text→image match is treated as "no result" (nothing found).
     * This is a PER-MODEL property: the text/image cosine scale differs by encoder (TinyCLIP's
     * modality gap sits ~0.15–0.40; SigLIP2's is different), so a single hard-coded gate in the
     * retriever would be right for one model and wrong for the other. Each encoder declares its own.
     */
    val visionMinScore: Float
    override fun close() {}
}

/**
 * Selects the crop-embedding backend — the single switch for the two demo variants:
 *  - MAC_ENDPOINT: SigLIP2-base over USB to the Mac (768-dim). Higher fine-grained quality;
 *    the embedding step is NOT on the glasses.
 *  - ON_DEVICE:    TinyCLIP-40M on the glasses (512-dim). "Everything on the glasses" is then
 *    literally true; weaker on small objects / fine attributes.
 *
 * The two backends use DIFFERENT dims and DIFFERENT vector spaces, so switching requires a
 * fresh index (clear the Qdrant Edge collection, e.g. `pm clear`) — never search vectors from
 * one backend against an index built by the other.
 *
 * CLOUD is reserved (not built).
 */
object CropEncoderFactory {
    enum class Backend { MAC_ENDPOINT, CLOUD, ON_DEVICE, QNN_B32, SIGLIP_NPU }
    val backend = Backend.SIGLIP_NPU
    fun create(context: Context): CropEncoder = when (backend) {
        // Wireless: reach the Mac on its LAN IP; wired: localhost via adb reverse. One constant.
        Backend.MAC_ENDPOINT -> MacEndpointEncoder(baseUrl = tech.qdrant.glasses.Config.MAC_BASE_URL)
        Backend.ON_DEVICE -> OnDeviceCropEncoder(context)
        // CLIP ViT-B/32 W8A16 on the Hexagon NPU (crop) + ViT-B/32 text — one 512-dim space,
        // ~28ms/crop isolated vs TinyCLIP's ~200ms CPU. Its own namespace (different space).
        Backend.QNN_B32 -> QnnB32CropEncoder(context)
        // SigLIP2-base: vision W8A16 on the NPU + text int8 on CPU, one 768-dim space. Fully
        // on-device. Out-retrieves QNN_B32 on the AR1 (see SiglipCropEncoder / A4 FINDINGS.md).
        Backend.SIGLIP_NPU -> SiglipCropEncoder(context)
        // Fail fast rather than silently serving wrong-space data if this is selected.
        Backend.CLOUD -> TODO("CLOUD crop encoder not implemented in v1")
    }

    /**
     * Stable per-backend name for the ObjectStore collection directory, so each variant keeps
     * its own index on disk and switching backends needs no data wipe.
     */
    val namespace: String get() = when (backend) {
        Backend.MAC_ENDPOINT -> "mac"
        Backend.ON_DEVICE -> "ondevice"
        Backend.QNN_B32 -> "qnnb32"
        Backend.SIGLIP_NPU -> "siglipnpu"
        Backend.CLOUD -> "cloud"
    }

    /**
     * Minimum cosine score for a voice-search hit to be shown. Text→image cosine distributions
     * differ per encoder (modality gap), so the gate is per-backend. SigLIP2 calibration from
     * rehearsal: absent objects → 0.084 ("keys") … 0.099 ("cup"); present → 0.118 ("what is
     * laptop") … 0.128 ("laptop"). The scale is compressed and the bands nearly touch — 0.11 is
     * the midpoint of the observed gap. Phrasing shifts scores (~0.01): rehearse the exact stage
     * queries. Below the gate = "nothing found", not junk cards.
     */
    val searchGate: Float get() = when (backend) {
        Backend.MAC_ENDPOINT -> 0.08f
        Backend.ON_DEVICE -> 0.25f
        // ViT-B/32 W8A16 — uncalibrated on real crops; reuse the ON_DEVICE midpoint as a start.
        Backend.QNN_B32 -> 0.25f
        // SigLIP2 on-device — CALIBRATED (CropEncoderAbTest, Task 5): present-query top-1 cosines
        // 0.064–0.119, absent distractors 0.049 (dog) / 0.080 (laptop). 0.09 dropped 7/12 present;
        // 0.085 keeps the stronger 7/12 while rejecting both distractors. Bands OVERLAP (SigLIP's
        // compressed scale) and N is small — a bigger distractor set would firm this.
        Backend.SIGLIP_NPU -> 0.085f
        Backend.CLOUD -> 0.12f
    }

    /**
     * Minimum cosine for a region's crop-embedding against its YOLO label's TEXT vector to KEEP the
     * label as a display tag (MomentCapture's capture-time verify). Same text→image modality gap as
     * [searchGate], so it is per-backend on the SAME compressed/uncompressed scale — a CLIP-scale
     * gate silently drops EVERY SigLIP region's label (SigLIP present-object cosines are ~0.11, not
     * ~0.26). Below this the region vector is still stored (a valid recall signal); only the label
     * is dropped. CALIBRATED on-device (CropEncoderAbTest, 14 labeled crops, Task 5):
     *  - QNN_B32 / ON_DEVICE (CLIP-scale): 0.20 — correct-label cosines 0.208–0.267 (min 0.208), so
     *    0.20 keeps them; CLIP's own value, unchanged.
     *  - SIGLIP_NPU / MAC_ENDPOINT (SigLIP2, compressed scale): 0.06. SigLIP correct-label cosines are
     *    only 0.059–0.098 (mean 0.084) — an earlier 0.10 guess dropped EVERY real label (all < 0.10),
     *    which is exactly why on-device regions came back label-less. 0.06 keeps 13/14 correct labels.
     *    NOTE: SigLIP's label-verify barely discriminates here (correct mean 0.084 ≈ wrong-label max-cos
     *    mean 0.082) — a coarse "on-topic" filter, not a fine one; label quality really comes from YOLO.
     */
    val verifyGate: Float get() = when (backend) {
        Backend.MAC_ENDPOINT -> 0.06f
        Backend.ON_DEVICE -> 0.20f
        Backend.QNN_B32 -> 0.20f
        Backend.SIGLIP_NPU -> 0.06f
        Backend.CLOUD -> 0.20f
    }

    /**
     * Whole-frame scene-dedup cosine for MomentCapture's semantic confirm — a candidate keyframe with
     * cosine >= this to the last stored keyframe is "not a new scene" and is SKIPPED. Per-backend
     * because the whole-frame image↔image cosine scale differs by encoder. CALIBRATED on-device
     * (CropEncoderAbTest.calibrateSceneDedup, 6 real stored keyframes):
     *  - QNN_B32 / ON_DEVICE (CLIP-scale): 0.85 — CLIP's original value (different scenes sit well
     *    below, same-scene above; kept).
     *  - SIGLIP_NPU / MAC_ENDPOINT: 0.90. SigLIP whole-frame embeddings barely separate scenes —
     *    DIFFERENT real scenes measured 0.774–0.924 (mean 0.834) while same-scene (live logs) is
     *    0.85–0.94, so the bands OVERLAP and no threshold cleans them. 0.85 sat inside the different-
     *    scene band and deduped genuinely-new views (too few moments captured). 0.90 lets almost all
     *    different scenes store (only the strongest >0.90 dedup), trading a few same-scene near-dupes
     *    for far better capture coverage. A weak discriminator — the real scene signal is region tags.
     */
    val sceneDedupCosine: Float get() = when (backend) {
        Backend.MAC_ENDPOINT -> 0.90f
        Backend.ON_DEVICE -> 0.85f
        Backend.QNN_B32 -> 0.85f
        Backend.SIGLIP_NPU -> 0.90f
        Backend.CLOUD -> 0.85f
    }
}
