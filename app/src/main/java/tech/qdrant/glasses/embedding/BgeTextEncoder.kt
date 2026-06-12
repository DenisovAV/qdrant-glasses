package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * bge-small-en-v1.5 (int8 ONNX): sentence embeddings for the "heard" channel.
 * CLS-pool of last_hidden_state, L2-normalized → FloatArray(384). Lives in its OWN
 * 384-dim space (named vector field "text") — never compared with CLIP vectors.
 */
class BgeTextEncoder(context: Context) : AutoCloseable {
    companion object { const val DIM = 384; private const val TAG = "BgeEncoder" }

    private val env = OrtEnvironment.getEnvironment()
    private val tokenizer = WordPieceTokenizer(context)
    private val session = createAcceleratedSession(
        env, extractAsset(context, "bge/model.onnx").absolutePath
    )

    fun encode(text: String): FloatArray {
        val (ids, len) = tokenizer.encode(text)
        val maxLen = WordPieceTokenizer.MAX_LEN
        val mask = LongArray(maxLen) { if (it < len) 1L else 0L }
        val types = LongArray(maxLen)  // all zeros
        val shape = longArrayOf(1, maxLen.toLong())
        val t0 = System.currentTimeMillis()
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { i ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape).use { m ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(types), shape).use { t ->
                    session.run(mapOf("input_ids" to i, "attention_mask" to m, "token_type_ids" to t)).use { r ->
                        @Suppress("UNCHECKED_CAST")
                        val hidden = r.get(0).value as Array<Array<FloatArray>> // [1, seq, 384]
                        val cls = hidden[0][0]
                        Log.i(TAG, "bge encode=${System.currentTimeMillis() - t0}ms")
                        return l2(cls)
                    }
                }
            }
        }
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0f; for (x in v) s += x * x
        val n = sqrt(s).coerceAtLeast(1e-12f)
        return FloatArray(v.size) { v[it] / n }
    }

    override fun close() { session.close() }
}
