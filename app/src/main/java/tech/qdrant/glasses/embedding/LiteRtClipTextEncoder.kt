package tech.qdrant.glasses.embedding

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * MobileCLIP2-S0 text tower via LiteRT (Adreno GPU).
 * Input is int32 token ids [1,77] (model converted with int32 so TFLite could
 * legalize the EOT arg_max). Reuses the standard CLIP BPE tokenizer. Output [1,512].
 */
class LiteRtClipTextEncoder(context: Context, accelerator: Accelerator) : TextEncoder {

    private val session = LiteRtSession(context, "clip-text.tflite", accelerator)
    private val tokenizer = TokenizerFactory.create(context)

    override fun encode(text: String): FloatArray {
        val ids = tokenizer.encodeToIds(text)  // IntArray(77): [SOT, tok…, EOT, 0-pad]
        val input = ByteBuffer.allocateDirect(Tokenizer.MAX_LENGTH * 4).order(ByteOrder.nativeOrder())
        val ib = input.asIntBuffer()
        for (i in ids.indices) ib.put(i, ids[i])
        val output = Array(1) { FloatArray(512) }
        val t1 = System.currentTimeMillis()
        session.interpreter.run(input, output)
        Log.i("LiteRtEncoder", "text gpuRun=${System.currentTimeMillis() - t1}ms")
        return output[0]
    }

    override fun close() = session.close()
}
