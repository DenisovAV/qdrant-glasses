package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap

/** Embeds an object crop and a text query into the SAME vector space. */
interface CropEncoder : AutoCloseable {
    fun encode(crop: Bitmap): FloatArray       // image → vector
    fun encodeText(query: String): FloatArray  // text  → vector
    val dim: Int
    override fun close() {}
}

/**
 * Selects the crop-embedding backend. MAC_ENDPOINT = SigLIP2 on the Mac over USB.
 * CLOUD/ON_DEVICE reserved (not built in v1).
 */
object CropEncoderFactory {
    enum class Backend { MAC_ENDPOINT, CLOUD, ON_DEVICE }
    val backend = Backend.MAC_ENDPOINT
    fun create(context: Context): CropEncoder = when (backend) {
        Backend.MAC_ENDPOINT -> MacEndpointEncoder()
        // Fail fast rather than silently serving MAC_ENDPOINT data if these are selected.
        Backend.CLOUD -> TODO("CLOUD crop encoder not implemented in v1")
        Backend.ON_DEVICE -> TODO("ON_DEVICE crop encoder not implemented in v1")
    }
}
