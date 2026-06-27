package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap

/**
 * On-device crop/text embedding — the fully-on-glasses alternative to [MacEndpointEncoder].
 *
 * Reuses the existing TinyCLIP towers ([TinyClipVisionEncoder] / [TinyClipTextEncoder]):
 * one shared 512-dim CLIP-family space, image and text aligned (a text query and a matching
 * crop land near each other), so the same encode-once / search-by-text flow works with NO
 * network, NO Mac, NO USB. This is the variant that makes "everything on the glasses" literally
 * true — at the cost of TinyCLIP-40M's weaker fine-grained quality vs SigLIP2-base.
 *
 * Trade vs MAC_ENDPOINT (SigLIP2, 768-dim, on the Mac):
 *  - dim is 512 here vs 768 there → DIFFERENT vector space → switching backends needs a fresh
 *    index (clear the Qdrant Edge collection / `pm clear`), exactly like switching EncoderFactory.
 *  - crop attributes (e.g. "red jacket" vs "blue jacket", small distant objects) are less reliable
 *    than SigLIP2; pick contrast-y demo objects to stay within TinyCLIP's strength.
 *  - latency is local (~450ms vision measured on RayNeo X3 Pro) instead of a USB round-trip.
 *
 * Must be called OFF the main thread (ONNX Runtime inference); the ViewModel crop-embed lane
 * already runs off-main, matching [MacEndpointEncoder]'s contract.
 */
class OnDeviceCropEncoder(context: Context) : CropEncoder {
    override val dim: Int = 512

    private val vision = TinyClipVisionEncoder(context)
    private val text = TinyClipTextEncoder(context)

    override fun encode(crop: Bitmap): FloatArray = vision.encode(crop)

    override fun encodeText(query: String): FloatArray = text.encode(query)

    override fun close() {
        vision.close()
        text.close()
    }
}
