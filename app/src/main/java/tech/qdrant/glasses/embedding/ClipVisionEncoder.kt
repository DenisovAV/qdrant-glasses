package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

class ClipVisionEncoder(context: Context) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // CLIP ViT-B/32 normalization — NOT standard ImageNet means/stds
    private val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val std  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    init {
        val modelBytes = context.assets.open("clip-vision-int8.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputName = session.inputNames.iterator().next() // "pixel_values"
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, 224, 224))
        val results = tensor.use { session.run(mapOf(inputName to it)) }
        return results.use { (it[0].value as Array<FloatArray>)[0] }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(224 * 224).also { bitmap.getPixels(it, 0, 224, 0, 0, 224, 224) }
        val buf = FloatBuffer.allocate(3 * 224 * 224)
        val stride = 224 * 224
        for (i in pixels.indices) {
            val px = pixels[i]
            buf.put(i,              ((px shr 16 and 0xFF) / 255f - mean[0]) / std[0])
            buf.put(i + stride,     ((px shr  8 and 0xFF) / 255f - mean[1]) / std[1])
            buf.put(i + stride * 2, ((px        and 0xFF) / 255f - mean[2]) / std[2])
        }
        return buf.apply { rewind() }
    }

    override fun close() {
        session.close()
        env.close()
    }
}
