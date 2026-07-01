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
    enum class Backend { MAC_ENDPOINT, CLOUD, ON_DEVICE }
    val backend = Backend.MAC_ENDPOINT
    fun create(context: Context): CropEncoder = when (backend) {
        // Wireless: reach the Mac on its LAN IP; wired: localhost via adb reverse. One constant.
        Backend.MAC_ENDPOINT -> MacEndpointEncoder(baseUrl = tech.qdrant.glasses.Config.MAC_BASE_URL)
        Backend.ON_DEVICE -> OnDeviceCropEncoder(context)
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
        Backend.CLOUD -> "cloud"
    }
}
