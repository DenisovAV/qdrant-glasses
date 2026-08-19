package tech.qdrant.glasses.embedding

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * On-device sanity check for [SiglipVisionEncoder]: encode `ab/o0001.jpg` on the Hexagon NPU and
 * compare (cosine, after L2-normalizing both) against a host-computed fp32 reference vector
 * (`siglip_ref_o0001.raw`, 768 little-endian float32). 0.70 is the pass bar — the measured
 * on-NPU-vs-fp32 cosine is ~0.82; 0.70 tolerates the Android `createScaledBitmap` vs PIL resize
 * difference while still catching a broken graph or preprocessing bug.
 */
@RunWith(AndroidJUnit4::class)
class SiglipVisionEncoderTest {

    private val TAG = "SiglipEncoder"

    @Test fun encodeMatchesFp32Reference() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testAssets = instrumentation.context.assets
        val targetCtx = instrumentation.targetContext

        val bitmap = testAssets.open("ab/o0001.jpg").use { BitmapFactory.decodeStream(it) }
        val ref = testAssets.open("siglip_ref_o0001.raw").use { stream ->
            val bytes = stream.readBytes()
            val floats = FloatArray(bytes.size / 4)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
            floats
        }
        assertEquals("reference vector must be 768-d", 768, ref.size)

        val encoder = SiglipVisionEncoder(targetCtx)
        val embedding: FloatArray
        val ms: Long
        try {
            val t0 = System.currentTimeMillis()
            embedding = encoder.encode(bitmap)
            ms = System.currentTimeMillis() - t0
        } finally {
            encoder.close()
        }

        assertEquals("embedding must be 768-d", 768, embedding.size)
        assertTrue("embedding must be all finite", embedding.all { it.isFinite() })

        val cos = cosine(l2(embedding), l2(ref))
        Log.i(TAG, "encode latency=${ms}ms")
        Log.i(TAG, "cosine(on-device, fp32-ref)=$cos")
        assertTrue("cosine to fp32 reference must be > 0.70 (got $cos)", cos > 0.70f)
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
