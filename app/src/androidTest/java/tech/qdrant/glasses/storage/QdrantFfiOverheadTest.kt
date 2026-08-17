package tech.qdrant.glasses.storage

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Random
import kotlin.math.sqrt

/**
 * Verifies WHY Qdrant-scan's small-N search latency is what it is — replicates Qdrant's own
 * `ffiOverheadDiagnostic` (UniFFI→JNA per-call cost) through OUR [QdrantEdgeStore]:
 *  - count() does trivial engine work but the same JNA round-trip as search(). If count ≈ search at
 *    tiny N, the cost is the binding, not the engine.
 *  - search() also pays our adapter's cost (vector.toList() boxes 512 floats + QueryRequest build),
 *    so (search − count) isolates our wrapper overhead.
 *  - search median across N=100/1k/10k shows the floor is N-independent until the scan cost bites.
 * Logged under tag "FFIDIAG" — capture with: adb logcat -s FFIDIAG:I
 */
@RunWith(AndroidJUnit4::class)
class QdrantFfiOverheadTest {

    private val TAG = "FFIDIAG"

    @Test fun qdrant_ffi_overhead() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store = QdrantEdgeStore(ctx, 512, "ffidiag")
        try {
            val rnd = Random(1)
            val q = unit(rnd)

            // ---- count() vs search() at 100 pts: isolate binding vs engine ----
            store.deleteAll()
            store.upsertBatch((0 until 100).map { unit(rnd) to payload() })
            repeat(20) { store.search(q, 5); store.count() }         // warm
            val search100 = medianMs(50) { store.search(q, 5) }
            val count100 = medianMs(50) { store.count() }
            Log.i(TAG, "at 100 pts: search=%.3fms  count=%.3fms  (search-count)=%.3fms".format(
                search100, count100, search100 - count100))

            // ---- search median across N: is the floor N-independent? ----
            for (n in intArrayOf(100, 1_000, 10_000)) {
                store.deleteAll()
                store.upsertBatch((0 until n).map { unit(rnd) to payload() })
                repeat(10) { store.search(q, 5) }
                val s = medianMs(30) { store.search(q, 5) }
                Log.i(TAG, "search @ N=%d: %.3fms".format(n, s))
            }
        } finally {
            store.deleteAll(); store.close()
        }
    }

    private inline fun medianMs(runs: Int, op: () -> Unit): Double {
        val ms = DoubleArray(runs)
        for (i in 0 until runs) { val t = System.nanoTime(); op(); ms[i] = (System.nanoTime() - t) / 1e6 }
        ms.sort(); return ms[runs / 2]
    }

    private fun payload() = ObjectPayload("obj", "", 0L, 0, "", "")
    private fun unit(rnd: Random): FloatArray {
        val v = FloatArray(512) { rnd.nextGaussian().toFloat() }
        var n = 0.0; for (x in v) n += (x * x).toDouble()
        val inv = if (n > 0) (1.0 / sqrt(n)).toFloat() else 0f
        for (i in v.indices) v[i] *= inv
        return v
    }
}
