package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

class OnnxClipVisionEncoder(context: Context) : VisionEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val std  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    init {
        val modelFile = extractAsset(context, "clip-vision-int8.onnx")
        session = createAcceleratedSession(env, modelFile.absolutePath)
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, 224, 224))
        val t1 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf(inputName to it)) }
        Log.i("ClipEncoder", "ONNX vision run=${System.currentTimeMillis() - t1}ms")
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

internal fun extractAsset(context: Context, name: String): File {
    val dest = File(context.filesDir, name)
    dest.parentFile?.mkdirs()
    // Validate by SIZE, not mere existence: a previous extraction can be left truncated
    // if the process was killed mid-copy (RayNeo's BackgroundAppManager does exactly this
    // for a backgrounded app), and `dest.exists()` would then happily reuse the fragment
    // → ORT_INVALID_PROTOBUF. Copy to a temp file and rename atomically so a partial
    // copy never appears at the final path.
    val expectedSize = context.assets.openFd(name).use { it.length }
    if (dest.exists() && dest.length() == expectedSize) return dest
    val tmp = File(context.filesDir, "$name.tmp")
    context.assets.open(name).use { input -> tmp.outputStream().use { input.copyTo(it) } }
    if (tmp.length() != expectedSize) {
        tmp.delete()
        error("extractAsset: $name copied ${tmp.length()} of $expectedSize bytes")
    }
    if (!tmp.renameTo(dest)) { tmp.copyTo(dest, overwrite = true); tmp.delete() }
    return dest
}

/**
 * Creates an ORT session that tries to offload to the device's NPU/GPU via NNAPI.
 * On the RayNeo X3 Pro (Snapdragon AR1 Gen1) this routes supported ops to the
 * Hexagon NPU / Adreno 621 instead of the CPU. ORT partitions the graph: ops the
 * NNAPI driver can't run stay on CPU (we do NOT set CPU_DISABLED, so unsupported
 * nodes fall back gracefully). If NNAPI can't initialise at all, we fall back to a
 * plain CPU session — same behaviour as before, never worse.
 */
internal fun createAcceleratedSession(env: OrtEnvironment, modelPath: String): OrtSession {
    return try {
        val opts = OrtSession.SessionOptions().apply {
            // Empty flag set = let NNAPI use NPU/GPU with CPU fallback enabled.
            addNnapi()
        }
        env.createSession(modelPath, opts).also {
            Log.i("ClipEncoder", "ORT session created with NNAPI execution provider")
        }
    } catch (e: Throwable) {
        Log.w("ClipEncoder", "NNAPI unavailable, falling back to CPU: ${e.message}")
        env.createSession(modelPath)
    }
}
