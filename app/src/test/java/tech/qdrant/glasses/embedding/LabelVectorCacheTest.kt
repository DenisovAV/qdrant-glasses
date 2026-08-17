package tech.qdrant.glasses.embedding

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

/**
 * Drives [LabelVectorCache] with a COUNTING fake `encodeText` (plan Task 2.1) — asserts the
 * memoization contract (encode once per NORMALIZED label, never again) without a real CLIP text
 * tower. Robolectric only because the production file logs via `android.util.Log` (mirrors
 * [tech.qdrant.glasses.storage.ObjectPayloadTest]'s reason for the same runner on an otherwise
 * plain-JVM-testable class).
 */
@RunWith(RobolectricTestRunner::class)
class LabelVectorCacheTest {

    /** Counts calls and returns a deterministic, label-dependent vector so both the "encoded
     *  exactly once" and the cosine-value assertions are checkable without a real model. */
    private class CountingEncoder : (String) -> FloatArray {
        var calls = 0
        override fun invoke(label: String): FloatArray {
            calls++
            val seed = label.hashCode()
            return FloatArray(8) { i -> ((seed + i * 31) % 97).toFloat() }
        }
    }

    @Test fun get_firstCallEncodesOnce() {
        val encoder = CountingEncoder()
        val cache = LabelVectorCache(encoder)
        cache.get("cup")
        assertEquals(1, encoder.calls)
    }

    @Test fun get_secondCallForSameLabelDoesNotReEncode() {
        val encoder = CountingEncoder()
        val cache = LabelVectorCache(encoder)
        cache.get("cup")
        cache.get("cup")
        assertEquals(1, encoder.calls)
    }

    @Test fun get_normalizesLabelKeyTrimAndCase() {
        val encoder = CountingEncoder()
        val cache = LabelVectorCache(encoder)
        cache.get("Cup")
        cache.get(" cup ")
        cache.get("CUP")
        assertEquals(1, encoder.calls)
    }

    @Test fun get_differentLabelsEncodeSeparately() {
        val encoder = CountingEncoder()
        val cache = LabelVectorCache(encoder)
        cache.get("cup")
        cache.get("laptop")
        assertEquals(2, encoder.calls)
    }

    @Test fun verify_identicalVectorsCosineNearOne() {
        val v = floatArrayOf(1f, 2f, 3f, 4f)
        val cache = LabelVectorCache({ v.copyOf() })
        assertEquals(1.0f, cache.verify(v, "cup"), 1e-4f)
    }

    @Test fun verify_orthogonalVectorsCosineNearZero() {
        val labelVec = floatArrayOf(1f, 0f, 0f, 0f)
        val cache = LabelVectorCache({ labelVec })
        val regionVec = floatArrayOf(0f, 1f, 0f, 0f)
        assertEquals(0.0f, cache.verify(regionVec, "cup"), 1e-4f)
    }

    @Test fun verify_returnsCosineInValidRange() {
        val cache = LabelVectorCache(CountingEncoder())
        val region = FloatArray(8) { i -> (i * 3 - 5).toFloat() }
        val cos = cache.verify(region, "cup")
        assertTrue(cos in -1.0f..1.0f)
    }

    @Test fun verify_zeroNormRegionVectorIsSafe() {
        val cache = LabelVectorCache(CountingEncoder())
        val cos = cache.verify(FloatArray(8), "cup")
        assertEquals(0.0f, cos, 1e-6f)
    }

    @Test fun verify_zeroNormLabelVectorIsSafe() {
        val cache = LabelVectorCache({ FloatArray(8) })
        val region = FloatArray(8) { 1f }
        assertEquals(0.0f, cache.verify(region, "cup"), 1e-6f)
    }

    @Test fun persistence_survivesAcrossInstances() {
        val file = File.createTempFile("label-vec-cache", ".tsv")
        file.delete() // start from "doesn't exist yet" like a fresh install
        try {
            val encoder1 = CountingEncoder()
            val v1 = LabelVectorCache(encoder1, file).get("cup")
            assertEquals(1, encoder1.calls)

            // A fresh instance (new process, new encoder) loads "cup" from disk instead of
            // re-embedding it — the whole point of persistence (Spec §7: pay the ~180ms once).
            val encoder2 = CountingEncoder()
            val v2 = LabelVectorCache(encoder2, file).get("cup")
            assertEquals(0, encoder2.calls)
            assertArrayEquals(v1, v2, 1e-4f)
        } finally {
            file.delete()
        }
    }

    @Test fun persistence_newEntryIsAppendedForNextInstance() {
        val file = File.createTempFile("label-vec-cache-append", ".tsv")
        file.delete()
        try {
            LabelVectorCache(CountingEncoder(), file).get("cup")
            val encoder2 = CountingEncoder()
            val cache2 = LabelVectorCache(encoder2, file)
            cache2.get("cup")      // persisted by the first instance
            cache2.get("laptop")   // new to the cache/file
            assertEquals(1, encoder2.calls) // only "laptop" was a miss

            val encoder3 = CountingEncoder()
            val cache3 = LabelVectorCache(encoder3, file)
            cache3.get("cup")
            cache3.get("laptop")
            assertEquals(0, encoder3.calls) // both now persisted
        } finally {
            file.delete()
        }
    }

    @Test fun persistence_malformedLinesAreSkippedNotFatal() {
        val file = File.createTempFile("label-vec-cache-bad", ".tsv")
        try {
            file.writeText("cup\t1.0,2.0,3.0\nbroken-line-no-tab\nlaptop\t1.0,notanumber,3.0\n")
            val encoder = CountingEncoder()
            val cache = LabelVectorCache(encoder, file)
            cache.get("cup")    // well-formed persisted line -> no encode call
            assertEquals(0, encoder.calls)
            cache.get("laptop") // malformed persisted line -> falls back to a fresh encode
            assertEquals(1, encoder.calls)
        } finally {
            file.delete()
        }
    }

    @Test fun persistence_nullFileIsInMemoryOnly() {
        val encoder = CountingEncoder()
        val cache = LabelVectorCache(encoder, persistFile = null)
        cache.get("cup")
        cache.get("cup")
        assertEquals(1, encoder.calls)
    }
}
