package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap

/** Produces a 512-dim CLIP-family embedding for an image. */
interface VisionEncoder : AutoCloseable {
    fun encode(bitmap: Bitmap): FloatArray
}

/** Produces a 512-dim CLIP-family embedding for a text query. */
interface TextEncoder : AutoCloseable {
    fun encode(text: String): FloatArray
}

/**
 * Selects the inference backend. Flip [backend] and rebuild to switch between:
 *  - ONNX: ONNX Runtime (CPU/NNAPI), CLIP ViT-B/32 (~150M, ~690ms/frame measured)
 *  - TINYCLIP: ONNX Runtime, TinyCLIP-ViT-8M-16 int8 (~8M params, 24MB — speed probe)
 *  - LITERT_GPU: LiteRT on Adreno GPU, MobileCLIP2-S0
 *  - LITERT_NPU: LiteRT on Hexagon NPU (QNN HTP fp16), MobileCLIP2-S0
 * Every backend pair is one model's two towers, but DIFFERENT backends are DIFFERENT
 * models / embedding spaces — switching needs re-indexing (pm clear). Only
 * LITERT_GPU/LITERT_NPU share a model (index-compatible between them).
 */
object EncoderFactory {
    enum class Backend { ONNX, TINYCLIP, LITERT_CPU, LITERT_GPU, LITERT_NPU }

    val backend = Backend.TINYCLIP

    fun createVision(context: Context): VisionEncoder = when (backend) {
        Backend.ONNX -> OnnxClipVisionEncoder(context)
        Backend.TINYCLIP -> TinyClipVisionEncoder(context)
        Backend.LITERT_CPU -> LiteRtClipVisionEncoder(context, Accelerator.CPU)
        Backend.LITERT_GPU -> LiteRtClipVisionEncoder(context, Accelerator.GPU)
        Backend.LITERT_NPU -> LiteRtClipVisionEncoder(context, Accelerator.NPU)
    }

    fun createText(context: Context): TextEncoder = when (backend) {
        Backend.ONNX -> OnnxClipTextEncoder(context)
        Backend.TINYCLIP -> TinyClipTextEncoder(context)
        Backend.LITERT_CPU -> LiteRtClipTextEncoder(context, Accelerator.CPU)
        Backend.LITERT_GPU -> LiteRtClipTextEncoder(context, Accelerator.GPU)
        Backend.LITERT_NPU -> LiteRtClipTextEncoder(context, Accelerator.NPU)
    }
}
