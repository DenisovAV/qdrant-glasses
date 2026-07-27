package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.LongBuffer

/**
 * CLIP ViT-B/32 TEXT tower — the query side, MUST match the [QnnClipVisionEncoder] (same model =
 * same 512-dim space, so a W8A16-quantized image vector still compares correctly against a text
 * vector, cosine 0.995 to float). Input `input_ids` [1,77] int64 → `text_embeds` [1,512] float.
 *
 * Runs **int8 on the ORT CPU EP**. We measured the alternatives on the live AR1 and the CPU wins
 * outright: text CPU-int8 ~177ms vs the same tower as float on the Adreno GPU (ORT QNN GPU backend)
 * ~1100ms — ~6x slower. The GPU path was dropped (it also cost 254MB of float weights in the APK).
 * That matches the SoC's pattern for CLIP-class ViTs: the accelerators lose to the CPU here.
 *
 * Text is the COLD path — once per voice query, not per crop — so ~177ms is off the hot loop.
 */
class B32ClipTextEncoder(context: Context) : TextEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val tokenizer = TokenizerFactory.create(context)
    private val latencyFile = File(context.filesDir, "clip_text_latency.csv")
    private val session: OrtSession

    init {
        latencyFile.appendText("=== text session ${System.currentTimeMillis()} ViT-B32 CPU(int8) ===\n")
        session = env.createSession(extractAsset(context, INT8_ASSET).absolutePath, OrtSession.SessionOptions())
    }

    override fun encode(text: String): FloatArray {
        val ids = tokenizer.encodeToIds(text)  // [SOT, tok…, EOT, 0-pad], length 77
        val maxLen = Tokenizer.MAX_LENGTH
        val buf = LongBuffer.allocate(maxLen).apply { for (i in 0 until maxLen) put(ids[i].toLong()); rewind() }
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, maxLen.toLong()))
        val t0 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("input_ids" to it)) }
        val ms = System.currentTimeMillis() - t0
        latencyFile.appendText("$ms\n")
        Log.i(TAG, "B32-text CPU(int8) run=${ms}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("text_embeds").get().value as Array<FloatArray>)[0]
        }
    }

    override fun close() {
        session.close()
        env.close()
    }

    companion object {
        private const val TAG = "ClipEncoder"
        private const val INT8_ASSET = "clip-vitb32-text-int8.onnx"
    }
}
