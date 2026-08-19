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
    val backend = Backend.QNN_B32
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
        // SigLIP2 on-device: present ~0.12 / absent-floor ~0.069 (A4 rehearsal) → 0.09 midpoint;
        // recalibrate from a live on-device rehearsal (integration Task 5).
        Backend.SIGLIP_NPU -> 0.09f
        Backend.CLOUD -> 0.12f
    }
}
