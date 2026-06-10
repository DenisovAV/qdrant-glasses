package tech.qdrant.glasses.embedding

import android.content.Context

/**
 * Placeholder for a HuggingFace-tokenizers (DJL, ai.djl.huggingface:tokenizers)
 * implementation. Not wired yet because the DJL JNI + Rust .so adds tens of MB to
 * the APK for tokenizing a 77-token query — only worth it if NAIVE and RANKED both
 * prove too slow. To enable: add the gradle dependency, load clip-tokenizer.json via
 * HuggingFaceTokenizer, map encodeToIds here, then set TokenizerFactory.backend = DJL.
 */
class DjlBpeTokenizer(@Suppress("UNUSED_PARAMETER") context: Context) : Tokenizer {
    override fun encodeToIds(text: String): IntArray {
        throw NotImplementedError(
            "DJL tokenizer not wired — add ai.djl.huggingface:tokenizers dependency first"
        )
    }
}
