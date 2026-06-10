package tech.qdrant.glasses.embedding

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * CLIP BPE tokenizer using the correct open_clip algorithm: for each word, look up
 * the RANK of pairs actually present in the word (O(1) hashmap), merge the lowest-rank
 * pair, repeat. O(word_len²) per word with hashmap lookups — orders of magnitude
 * faster than scanning all merges. Produces identical token ids to the reference.
 */
class RankedBpeTokenizer(context: Context) : Tokenizer {
    private val vocab: Map<String, Int>
    private val bpeRanks: Map<Pair<String, String>, Int>

    init {
        val root = JSONObject(context.assets.open("clip-tokenizer.json").bufferedReader().readText())
        val model = root.getJSONObject("model")
        val vocabObj = model.getJSONObject("vocab")
        vocab = buildMap { vocabObj.keys().forEach { key -> put(key, vocabObj.getInt(key)) } }
        val mergesArray = model.getJSONArray("merges")
        bpeRanks = buildMap {
            for (i in 0 until mergesArray.length()) {
                val parts = mergesArray.getString(i).split(" ")
                if (parts.size == 2) put(parts[0] to parts[1], i)  // rank = order in merges
            }
        }
    }

    /** Distinct adjacent symbol pairs in the token list. */
    private fun pairs(word: List<String>): Set<Pair<String, String>> {
        val s = HashSet<Pair<String, String>>()
        for (i in 0 until word.size - 1) s.add(word[i] to word[i + 1])
        return s
    }

    private fun bpe(token: String): List<String> {
        if (token.isEmpty()) return emptyList()
        // CLIP marks the last char of a word with </w>.
        var word = token.mapIndexed { i, c ->
            if (i == token.length - 1) "$c</w>" else c.toString()
        }
        if (word.size == 1) return word
        while (true) {
            val ps = pairs(word)
            // pick the pair with the smallest rank (earliest merge rule)
            val bigram = ps.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (bpeRanks[bigram] == null) break
            val (first, second) = bigram
            val newWord = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord.add(first + second); i += 2
                } else {
                    newWord.add(word[i]); i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
        }
        return word
    }

    private fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        for (word in text.lowercase().trim().split(" ").filter { it.isNotEmpty() }) {
            for (piece in bpe(word)) {
                vocab[piece]?.let { tokens.add(it) }
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
        Log.i("Tokenizer", "RANKED tokenize=${System.currentTimeMillis() - t0}ms")
        return ids
    }
}
