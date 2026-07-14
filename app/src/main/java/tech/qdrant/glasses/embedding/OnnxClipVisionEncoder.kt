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
 * Creates the ORT session for the CLIP towers. **In practice this always runs on the CPU**, and
 * that is not an oversight — it is the fastest thing this device can do with a ViT. Read this
 * before "fixing" it:
 *
 * ## The NNAPI attempt below ALWAYS fails. It is a probe, not a fast path.
 * The shipped `onnxruntime-android-qnn` AAR contains **no NNAPI execution provider**: it exports
 * only `OrtSessionOptionsAppendExecutionProvider_CPU` and imports **zero** `ANeuralNetworks*`
 * symbols. So `addNnapi()` throws `ORT_INVALID_ARGUMENT: This binary was not compiled with NNAPI
 * support` on every launch and we land in the CPU branch. It is kept as a probe so the day an ORT
 * build ships with NNAPI, we pick it up for free — and so the log says out loud which EP we got.
 *
 * Do NOT trust `ai.onnxruntime.OrtProvider` to tell you otherwise: that enum lists all 17
 * providers in *every* build, whether or not they are linked in. It is a menu, not an inventory.
 * An earlier version of this comment claimed NNAPI "routes supported ops to the Hexagon NPU /
 * Adreno 621" — it never did, and that fiction cost real debugging time.
 *
 * ## Why CPU is the right answer anyway (measured on the RayNeo X3 Pro, under the live pipeline)
 * Every accelerator this SoC exposes was tried against a CLIP-class ViT, and every one lost to the
 * CPU. The decisive variable is **int8 vs fp32**, not CPU vs accelerator:
 *
 *  | model / route                                  | vision p50 |
 *  |------------------------------------------------|-----------|
 *  | TinyCLIP-40M **int8**, ORT **CPU** (this path)  | **203ms**  |
 *  | MobileCLIP2-S0 fp32, LiteRT Adreno GPU          |   929ms   |
 *  | MobileCLIP2-S0 fp32, LiteRT CPU (XNNPACK)       |  ~3.5s    |
 *  | MobileCLIP2-S0 fp32, LiteRT Hexagon HTP (fp16)  |  slower still |
 *
 * Latency on this part is LOAD-dependent, so always compare like with like. The AR1 is a **4-core**
 * SoC (2xA78 + 2xA55) and the encoder shares those cores with the camera's YUV conversion, the
 * detector, the store — and the HUD's per-frame JPEG encode. Closing the HUD alone takes this path
 * from ~870ms to ~203ms (see Config.HUD_STREAM). The rows above are all measured HUD-off, one build,
 * one session. An old "~450-490ms" note was an idle-device number and is not comparable to any of
 * them.
 *
 * The Hexagon HTP takes an int8 YOLOv8n **CNN** at ~8ms — but of a 490-node ViT it accepts only
 * **44 nodes** and shreds the graph into **50 partitions** (the Adreno GPU delegate manages 198).
 * Every partition boundary is a round trip back to the CPU, which costs more than the accelerator
 * saves. Meanwhile int8 matmuls hit ARM dot-product instructions and run several times faster than
 * fp32 — which is the entire reason this CPU path wins.
 *
 * The unlock is not another delegate; it is a **quantized ViT that Hexagon will actually take**.
 */
internal fun createAcceleratedSession(env: OrtEnvironment, modelPath: String): OrtSession {
    return try {
        val opts = OrtSession.SessionOptions().apply { addNnapi() }
        env.createSession(modelPath, opts).also {
            Log.i("ClipEncoder", "ORT session created with NNAPI execution provider")
        }
    } catch (e: Throwable) {
        // Expected on every launch with the current AAR — see the KDoc. CPU is the fast path here.
        Log.i("ClipEncoder", "no NNAPI EP in this ORT build (expected) — running CLIP on CPU: ${e.message}")
        env.createSession(modelPath)
    }
}
