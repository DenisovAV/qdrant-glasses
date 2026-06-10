package tech.qdrant.glasses.embedding

import android.content.Context

/** CLIP BPE tokenizer: text -> fixed-length token id array [SOT, toks…, EOT, 0-pad]. */
interface Tokenizer {
    fun encodeToIds(text: String): IntArray

    companion object {
        const val MAX_LENGTH = 77
        const val SOT_TOKEN = 49406  // <|startoftext|>
        const val EOT_TOKEN = 49407  // <|endoftext|>
    }
}

/**
 * Selects the BPE tokenizer implementation. Flip [backend] and rebuild to compare:
 * NAIVE (original brute-force over all merges), RANKED (open_clip-style rank lookup),
 * DJL (HuggingFace tokenizers via JNI). All must produce identical token ids.
 */
object TokenizerFactory {
    enum class Backend { NAIVE, RANKED, DJL }

    val backend = Backend.RANKED

    fun create(context: Context): Tokenizer = when (backend) {
        Backend.NAIVE -> NaiveBpeTokenizer(context)
        Backend.RANKED -> RankedBpeTokenizer(context)
        Backend.DJL -> DjlBpeTokenizer(context)
    }
}
