package tech.qdrant.glasses.embedding

import android.content.Context

/**
 * BERT-uncased WordPiece for bge-small-en-v1.5: lowercase → split on
 * whitespace/punctuation → greedy longest-match with "##" continuations.
 * NOT interchangeable with the CLIP BPE tokenizer (different algorithm and vocab).
 */
class WordPieceTokenizer(context: Context, vocabAsset: String = "bge/vocab.txt") {
    companion object {
        const val CLS = 101
        const val SEP = 102
        const val UNK = 100
        const val MAX_LEN = 64  // our transcripts/queries are short utterances
    }

    private val vocab: Map<String, Int> =
        context.assets.open(vocabAsset).bufferedReader().readLines()
            .withIndex().associate { (i, tok) -> tok to i }

    /** Returns ids WITH [CLS]…[SEP], zero-padded to MAX_LEN; second = attention length. */
    fun encode(text: String): Pair<LongArray, Int> {
        val ids = ArrayList<Int>(MAX_LEN).apply { add(CLS) }
        for (word in basicTokenize(text)) {
            if (ids.size >= MAX_LEN - 1) break
            ids.addAll(wordPiece(word))
        }
        while (ids.size > MAX_LEN - 1) ids.removeAt(ids.size - 1)
        ids.add(SEP)
        val len = ids.size
        val out = LongArray(MAX_LEN)
        for (i in 0 until len) out[i] = ids[i].toLong()
        return out to len
    }

    private fun basicTokenize(text: String): List<String> {
        val sb = StringBuilder()
        for (c in text.lowercase()) {
            when {
                c.isLetterOrDigit() -> sb.append(c)
                else -> sb.append(' ').append(if (c.isWhitespace()) ' ' else c).append(' ')
            }
        }
        return sb.split(' ').filter { it.isNotBlank() }
    }

    private fun wordPiece(word: String): List<Int> {
        if (word.length > 100) return listOf(UNK)
        val out = ArrayList<Int>(4)
        var start = 0
        while (start < word.length) {
            var end = word.length
            var id = -1
            while (start < end) {
                val piece = (if (start > 0) "##" else "") + word.substring(start, end)
                val v = vocab[piece]
                if (v != null) { id = v; break }
                end--
            }
            if (id == -1) return listOf(UNK)  // whole word unknown
            out.add(id)
            start = end
        }
        return out
    }
}
