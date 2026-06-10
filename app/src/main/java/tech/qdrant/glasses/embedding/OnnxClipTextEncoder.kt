package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.LongBuffer

class OnnxClipTextEncoder(context: Context) : TextEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer = TokenizerFactory.create(context)

    init {
        val modelFile = extractAsset(context, "clip-text-int8.onnx")
        session = createAcceleratedSession(env, modelFile.absolutePath)
    }

    override fun encode(text: String): FloatArray {
        val ids = tokenizer.encodeToIds(text)  // [SOT, tok…, EOT, 0-pad], length 77
        val maxLen = Tokenizer.MAX_LENGTH
        val inputIds = LongBuffer.allocate(maxLen)
        val attnMask = LongBuffer.allocate(maxLen)
        for (i in 0 until maxLen) {
            val id = ids[i]
            inputIds.put(id.toLong())
            // mask = 1 for SOT/tokens/EOT (non-zero, plus index 0=SOT), 0 for pad
            attnMask.put(if (id != 0 || i == 0) 1L else 0L)
        }
        inputIds.rewind(); attnMask.rewind()

        val shape = longArrayOf(1, maxLen.toLong())
        val idsTensor = OnnxTensor.createTensor(env, inputIds, shape)
        val maskTensor = OnnxTensor.createTensor(env, attnMask, shape)
        val results = idsTensor.use { i -> maskTensor.use { m ->
            session.run(mapOf("input_ids" to i, "attention_mask" to m))
        }}
        return results.use { (it[0].value as Array<FloatArray>)[0] }
    }

    override fun close() {
        session.close()
        env.close()
    }
}
