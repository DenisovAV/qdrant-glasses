package tech.qdrant.glasses.embedding

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * On-device checks for [SiglipTextEncoder] — the two things that can silently produce garbage
 * without ever throwing:
 *
 * 1. TOKENIZER PARITY: the DJL `HuggingFaceTokenizer` (real Rust `tokenizers` crate over JNI) must
 *    produce the EXACT same ids as a reference run (`siglip_tok_expected.json`, generated off the
 *    same `siglip-tokenizer.json` on a JVM — see SiglipTextEncoder's KDoc for why the padding
 *    options are passed explicitly). This is the critical guard on the SentencePiece/BPE path: a
 *    wrong id here still runs through ORT and returns a finite-looking vector, just a WRONG one.
 * 2. EMBED: `encode()` end-to-end lands in the same space as a host fp32 reference, à la
 *    [SiglipVisionEncoderTest].
 */
@RunWith(AndroidJUnit4::class)
class SiglipTextEncoderTest {

    private val TAG = "SiglipEncoder"

    @Test fun tokenizerMatchesReferenceIds() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testAssets = instrumentation.context.assets
        val targetCtx = instrumentation.targetContext

        val expected: Map<String, IntArray> = testAssets.open("siglip_tok_expected.json").use { stream ->
            val root = JSONObject(stream.bufferedReader().readText())
            buildMap {
                root.keys().forEach { q ->
                    val arr = root.getJSONArray(q)
                    put(q, IntArray(arr.length()) { arr.getInt(it) })
                }
            }
        }
        assertTrue("expected fixture must be non-empty", expected.isNotEmpty())

        // Copies straight from the target app's assets (not this test APK's own assets) — the
        // asset under test lives in app/src/main/assets. Deliberately NOT [extractAsset] (that's
        // `internal` to the main compilation module, not visible from this androidTest module).
        val tokenizerFile = java.io.File(targetCtx.filesDir, "siglip-tokenizer.json").apply {
            val expectedSize = targetCtx.assets.openFd("siglip-tokenizer.json").use { it.length }
            if (!exists() || length() != expectedSize) {
                targetCtx.assets.open("siglip-tokenizer.json").use { input ->
                    outputStream().use { input.copyTo(it) }
                }
            }
        }
        val tokenizer = HuggingFaceTokenizer.newInstance(
            tokenizerFile.toPath(),
            mapOf("padding" to "MAX_LENGTH", "maxLength" to "64")
        )
        try {
            for ((query, expectedIds) in expected) {
                assertEquals("$query: reference ids must be 64-long", 64, expectedIds.size)
                val ids = tokenizer.encode(query.lowercase()).ids.map { it.toInt() }.toIntArray()
                Log.i(TAG, "tok \"$query\" -> ${ids.take(8)}...")
                assertEquals("$query: id count", expectedIds.size, ids.size)
                assertTrue(
                    "$query: DJL ids ${ids.toList()} != expected ${expectedIds.toList()}",
                    ids.contentEquals(expectedIds)
                )
            }
        } finally {
            tokenizer.close()
        }
        Log.i(TAG, "tokenizer parity OK for ${expected.size} queries")
    }

    @Test fun encodeMatchesFp32Reference() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testAssets = instrumentation.context.assets
        val targetCtx = instrumentation.targetContext

        val ref = testAssets.open("siglip_text_ref_plant.raw").use { stream ->
            val bytes = stream.readBytes()
            val floats = FloatArray(bytes.size / 4)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
            floats
        }
        assertEquals("reference vector must be 768-d", 768, ref.size)

        val encoder = SiglipTextEncoder(targetCtx)
        val embedding: FloatArray
        val ms: Long
        try {
            val t0 = System.currentTimeMillis()
            embedding = encoder.encode("a potted plant")
            ms = System.currentTimeMillis() - t0
        } finally {
            encoder.close()
        }

        assertEquals("embedding must be 768-d", 768, embedding.size)
        assertTrue("embedding must be all finite", embedding.all { it.isFinite() })

        val cos = cosine(l2(embedding), l2(ref))
        Log.i(TAG, "text encode latency=${ms}ms")
        Log.i(TAG, "cosine(on-device text, fp32-ref)=$cos")
        assertTrue("cosine to fp32 reference must be > 0.90 (got $cos)", cos > 0.90f)
    }

    private fun l2(v: FloatArray): FloatArray {
        var sumSq = 0.0
        for (x in v) sumSq += x.toDouble() * x.toDouble()
        val norm = sqrt(sumSq).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(v.size) { v[it] / norm }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }
}
