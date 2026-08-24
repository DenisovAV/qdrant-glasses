package tech.qdrant.glasses.embedding

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.LongBuffer

/**
 * SigLIP2 TEXT tower — the query side, MUST land in the SAME 768-dim space as
 * [SiglipVisionEncoder] (same model, same reasons as [B32ClipTextEncoder]/[QnnClipVisionEncoder]).
 * Input `input_ids` [1,64] int64 -> output `pooler_output` [1,768] float (the model also exposes
 * `last_hidden_state`, unused here — see the model's graph.output for both names).
 *
 * Tokenizer is the SigLIP2/Gemma SentencePiece-BPE model (256k vocab, byte-fallback) shipped as
 * `siglip-tokenizer.json` — NOT the CLIP BPE family elsewhere in this package
 * ([RankedBpeTokenizer]/[NaiveBpeTokenizer]/[DjlBpeTokenizer]'s own placeholder target): those are
 * a different algorithm over a different, much smaller vocab and would silently produce wrong ids
 * against this one. Loaded via DJL's real HuggingFace `tokenizers` JNI binding (the Rust crate,
 * not a Kotlin reimplementation) — verified byte-for-byte against the tokenizer.json's own
 * baked-in padding config (Fixed length 64, pad_id 0) offline on a JVM before ever touching the
 * device: `HuggingFaceTokenizer.newInstance(path, options)` does NOT inherit that config by
 * default (its Java wrapper unconditionally re-applies ITS OWN default padding=LONGEST/
 * truncation=LONGEST_FIRST over whatever the JSON specifies — see HuggingFaceTokenizer's
 * updateTruncationAndPadding()), so `padding`/`maxLength` are passed explicitly below to
 * reproduce the same fixed-64/pad-0 behavior the JSON declares.
 *
 * Runs int8 on the ORT CPU EP — same call as [B32ClipTextEncoder], and for the same reason: this
 * was measured against the NPU (QNN HTP, int16/W8A16) and CPU wins on ACCURACY here (cosine 0.998
 * to the fp32 reference vs 0.87 on the HTP — the text tower's LayerNorms lose more to int16 than
 * vision's convs do), and text is the cold path (once per voice query, not per frame), so the
 * extra CPU ms are off the hot loop.
 */
class SiglipTextEncoder(context: Context) : TextEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val tokenizer: HuggingFaceTokenizer
    private val latencyFile = File(context.filesDir, "siglip_text_latency.csv")
    private val session: OrtSession

    init {
        // DJL's HuggingFaceTokenizer opt-in-by-default "call home" telemetry
        // (Ec2Utils.callHome, fired from the FIRST newInstance()) tries an HTTP hit against the
        // AWS link-local metadata IP (169.254.169.254) — unroutable on ordinary WiFi, so it just
        // burns its connect timeout for no reason on every cold start. Not on this SoC's path.
        System.setProperty("ai.djl.offline", "true")

        val tokenizerFile = extractAsset(context, TOKENIZER_ASSET)
        val t0 = System.currentTimeMillis()
        tokenizer = HuggingFaceTokenizer.newInstance(
            tokenizerFile.toPath(),
            mapOf("padding" to "MAX_LENGTH", "maxLength" to MAX_LENGTH.toString())
        )
        Log.i(TAG, "DJL HuggingFaceTokenizer loaded in ${System.currentTimeMillis() - t0}ms")

        latencyFile.appendText("=== siglip text session ${System.currentTimeMillis()} CPU(int8) ===\n")
        val t1 = System.currentTimeMillis()
        // Default ORT session options (CPU arena on, mem-pattern on, weight pre-packing on). An earlier
        // version turned all three OFF to cut peak RAM, believing the full SIGLIP_NPU app OOM/rebooted the
        // AR1 at startup — that was a MISDIAGNOSIS (measured on a clean device: lowmemorykiller reports
        // "watermarks ok", zero memory pressure; the reboots were the AR1's own spontaneous reboot + a
        // post-reboot camera-HAL wedge, see qdrant_glasses_private/a4-siglip-eval/FINDINGS.md Part
        // 8-retracted). Pre-packing matters here: it speeds this model's int8 matmuls, and text runs on
        // the once-per-query search path — and there is RAM headroom (steady-state ~668MB of ~2.4GB).
        val opts = OrtSession.SessionOptions()
        session = env.createSession(extractAsset(context, MODEL_ASSET).absolutePath, opts)
        Log.i(TAG, "SigLIP-text ORT CPU session created in ${System.currentTimeMillis() - t1}ms")
    }

    override fun encode(text: String): FloatArray {
        val t0 = System.currentTimeMillis()
        val ids = tokenizer.encode(text.lowercase()).ids  // already padded/truncated to MAX_LENGTH, pad id 0
        val tokMs = System.currentTimeMillis() - t0
        val tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, MAX_LENGTH.toLong()))
        val t1 = System.currentTimeMillis()
        val results = tensor.use { session.run(mapOf("input_ids" to it)) }
        val infMs = System.currentTimeMillis() - t1
        latencyFile.appendText("$infMs\n")
        Log.i(TAG, "SigLIP-text tokenize=${tokMs}ms CPU(int8) run=${infMs}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("pooler_output").get().value as Array<FloatArray>)[0]
        }
    }

    override fun close() {
        tokenizer.close()
        session.close()
        env.close()
    }

    companion object {
        private const val TAG = "SiglipEncoder"
        private const val TOKENIZER_ASSET = "siglip-tokenizer.json"
        private const val MODEL_ASSET = "siglip-text-int8.onnx"
        private const val MAX_LENGTH = 64
    }
}
