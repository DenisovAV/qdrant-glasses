package tech.qdrant.glasses.embedding

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * Original CLIP BPE tokenizer: brute-force scan of ALL merge rules per word.
 * O(words × iterations × merges) — kept as the baseline to benchmark against.
 */
class NaiveBpeTokenizer(context: Context) : Tokenizer {
    private val vocab: Map<String, Int>
    private val merges: List<Pair<String, String>>

    init {
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

    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        for (word in text.lowercase().trim().split(" ").filter { it.isNotEmpty() }) {
            val chars = word.map { it.toString() }.toMutableList()
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

    override fun encodeToIds(text: String): IntArray {
        val t0 = System.currentTimeMillis()
        val ids = IntArray(Tokenizer.MAX_LENGTH)
        var pos = 0
        ids[pos++] = Tokenizer.SOT_TOKEN
        for (t in tokenize(text)) {
            if (pos >= Tokenizer.MAX_LENGTH - 1) break
            ids[pos++] = t
        }
        ids[pos] = Tokenizer.EOT_TOKEN
        Log.i("Tokenizer", "NAIVE tokenize=${System.currentTimeMillis() - t0}ms")
        return ids
    }
}
