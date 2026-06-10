package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MobileCLIP2-S0 image tower via LiteRT (Adreno GPU).
 * Input is NCHW [1,3,256,256] float32, range [0,1] (no CLIP mean/std — the model
 * was trained with mean=0/std=1, confirmed from open_clip preprocess). Output [1,512].
 */
class LiteRtClipVisionEncoder(context: Context, accelerator: Accelerator) : VisionEncoder {

    private val session = LiteRtSession(context, "clip-vision.tflite", accelerator)

    override fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        val pixels = IntArray(SIZE * SIZE).also { resized.getPixels(it, 0, SIZE, 0, 0, SIZE, SIZE) }
        // NCHW: all R, then all G, then all B (planar), each pixel / 255.
        val input = ByteBuffer.allocateDirect(1 * 3 * SIZE * SIZE * 4).order(ByteOrder.nativeOrder())
        val fb = input.asFloatBuffer()
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            fb.put(i,             (px shr 16 and 0xFF) / 255f)  // R
            fb.put(i + plane,     (px shr  8 and 0xFF) / 255f)  // G
            fb.put(i + plane * 2, (px        and 0xFF) / 255f)  // B
        }
        val output = Array(1) { FloatArray(512) }
        val t1 = System.currentTimeMillis()
        session.interpreter.run(input, output)
        Log.i("LiteRtEncoder", "vision gpuRun=${System.currentTimeMillis() - t1}ms")
        return output[0]
    }

    override fun close() = session.close()

    companion object { private const val SIZE = 256 }
}
