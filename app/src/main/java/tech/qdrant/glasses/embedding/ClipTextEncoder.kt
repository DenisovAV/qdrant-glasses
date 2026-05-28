package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.LongBuffer
import tech.qdrant.glasses.embedding.extractAsset

class ClipTextEncoder(context: Context) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: Map<String, Int>
    private val merges: List<Pair<String, String>>

    companion object {
        private const val MAX_LENGTH = 77
        private const val SOT_TOKEN = 49406  // <|startoftext|>
        private const val EOT_TOKEN = 49407  // <|endoftext|>
    }

    init {
        val modelFile = extractAsset(context, "clip-text-int8.onnx")
        session = env.createSession(modelFile.absolutePath)

        val root = JSONObject(context.assets.open("clip-tokenizer.json").bufferedReader().readText())
        val model = root.getJSONObject("model")

        val vocabObj = model.getJSONObject("vocab")
        vocab = buildMap { vocabObj.keys().forEach { key -> put(key, vocabObj.getInt(key)) } }

        val mergesArray = model.getJSONArray("merges")
        merges = buildList {
            for (i in 0 until mergesArray.length()) {
                val parts = mergesArray.getString(i).split(" ")
                if (parts.size == 2) add(parts[0] to parts[1])
            }
        }
    }

    fun encode(text: String): FloatArray {
        val tokenIds = tokenize(text.lowercase().trim())
        val inputIds = LongBuffer.allocate(MAX_LENGTH)
        val attnMask = LongBuffer.allocate(MAX_LENGTH)

        inputIds.put(SOT_TOKEN.toLong()); attnMask.put(1L)
        for (i in tokenIds.indices) {
            if (i >= MAX_LENGTH - 2) break
            inputIds.put(tokenIds[i].toLong()); attnMask.put(1L)
        }
        inputIds.put(EOT_TOKEN.toLong()); attnMask.put(1L)
        while (inputIds.position() < MAX_LENGTH) { inputIds.put(0L); attnMask.put(0L) }
        inputIds.rewind(); attnMask.rewind()

        val shape = longArrayOf(1, MAX_LENGTH.toLong())
        val idsTensor   = OnnxTensor.createTensor(env, inputIds, shape)
        val maskTensor  = OnnxTensor.createTensor(env, attnMask, shape)
        val results = idsTensor.use { ids -> maskTensor.use { mask ->
            session.run(mapOf("input_ids" to ids, "attention_mask" to mask))
        }}
        return results.use { (it[0].value as Array<FloatArray>)[0] }
    }

    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        for (word in text.split(" ").filter { it.isNotEmpty() }) {
            var chars = word.map { it.toString() }.toMutableList()
            var changed = true
            while (changed && chars.size > 1) {
                changed = false
                for ((first, second) in merges) {
                    val idx = (0 until chars.size - 1).firstOrNull {
                        chars[it] == first && chars[it + 1] == second
                    } ?: continue
                    chars[idx] = first + second
                    chars.removeAt(idx + 1)
                    changed = true
                    break
                }
            }
            chars.forEach { tok ->
                vocab[tok + "</w>"]?.let { tokens.add(it) } ?: vocab[tok]?.let { tokens.add(it) }
            }
        }
        return tokens
    }

    override fun close() {
        session.close()
        env.close()
    }
}
