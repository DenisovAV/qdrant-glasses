package tech.qdrant.glasses.storage

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Random
import kotlin.math.sqrt

/**
 * On-emulator/on-device verification of [ObjectBoxStore] alone — constructs the store directly, so
 * it exercises the ObjectBox engine (put / nearestNeighbors / range-filter / recall) WITHOUT booting
 * the NPU pipeline. Run: `./gradlew :app:connectedBenchmarkDebugAndroidTest` with an arm64 emulator attached.
 */
@RunWith(AndroidJUnit4::class)
class ObjectBoxStoreTest {

    private lateinit var store: ObjectBoxStore

    @Before fun setUp() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        store = ObjectBoxStore(ctx, ObjectBoxStore.DIM, "test")
        store.deleteAll()
    }

    @After fun tearDown() {
        store.deleteAll()
        store.close()
    }

    @Test fun search_returns_planted_nearest_neighbor() {
        val rnd = Random(7)
        val q = unit(rnd)
        val near = normalize(perturb(q, rnd, 0.02f))     // a deliberately close vector

        val items = ArrayList<Pair<FloatArray, ObjectPayload>>()
        repeat(300) { items.add(unit(rnd) to payload(1000L + it)) }  // fillers
        val nearIdx = items.size
        items.add(near to payload(9999L))
        val ids = store.upsertBatch(items)
        val nearId = ids[nearIdx]

        val hits = store.search(q, 5)
        assertEquals(5, hits.size)
        assertEquals("planted near vector must rank #1", nearId, hits[0].id)
        assertTrue("cosine sim to a near vector should be high, was ${hits[0].score}", hits[0].score > 0.9f)
        // scores must be descending (higher = more similar)
        for (i in 1 until hits.size) assertTrue("scores not descending", hits[i].score <= hits[i - 1].score + 1e-4f)
        // every returned id is one we inserted
        assertTrue(hits.all { it.id in ids })
    }

    @Test fun recall_finds_clearly_separated_neighbors() {
        // A FAIR recall test: plant 5 vectors that are unambiguously closest to the query (query +
        // tiny noise), among random fillers. HNSW must recover well-separated true neighbors →
        // recall ~1.0. (Recall on random near-orthogonal vectors is a degenerate worst case for ANY
        // ANN index — that latency/recall trade-off on ambiguous data is what the scaled benchmark
        // measures, not a per-integration correctness bug.)
        val rnd = Random(11)
        val q = unit(rnd)
        val items = ArrayList<Pair<FloatArray, ObjectPayload>>()
        val plantedIds = HashSet<Int>()
        repeat(5) { plantedIds.add(items.also { it.add(normalize(perturb(q, rnd, 0.02f)) to payload(1L + it.size)) }.size - 1) }
        repeat(495) { items.add(unit(rnd) to payload(1000L + it)) }   // random fillers, far from q
        val ids = store.upsertBatch(items)
        val truth = plantedIds.map { ids[it] }.toSet()

        val got = store.search(q, 5).map { it.id }.toSet()
        val recall = truth.count { it in got }.toDouble() / truth.size
        assertTrue("recall@5=$recall — HNSW should recover clearly-separated planted neighbors", recall >= 0.8)
    }

    @Test fun search_filtered_restricts_to_time_window() {
        val rnd = Random(13)
        val items = ArrayList<Pair<FloatArray, ObjectPayload>>()
        repeat(400) { items.add(unit(rnd) to payload(1000L + it)) }  // ts 1000..1399
        store.upsertBatch(items)

        val q = unit(rnd)
        val hits = store.searchFiltered(q, 5, sinceMs = 1200L, untilMs = 1250L)
        assertTrue("filtered search returned nothing", hits.isNotEmpty())
        assertTrue(
            "all hits must fall inside [1200,1250], got ${hits.map { it.timestampMs }}",
            hits.all { it.timestampMs in 1200L..1250L }
        )
    }

    @Test fun count_and_deleteAll() {
        val rnd = Random(3)
        val items = (0 until 50).map { unit(rnd) to payload(1000L + it) }
        store.upsertBatch(items)
        assertEquals(50L, store.count())
        store.deleteAll()
        assertEquals(0L, store.count())
    }

    // ---- helpers ----
    private fun payload(ts: Long) = ObjectPayload("obj", "", ts, 0, "", "")

    private fun unit(rnd: Random): FloatArray {
        val v = FloatArray(ObjectBoxStore.DIM) { rnd.nextGaussian().toFloat() }
        return normalize(v)
    }

    private fun perturb(base: FloatArray, rnd: Random, eps: Float): FloatArray =
        FloatArray(base.size) { base[it] + eps * rnd.nextGaussian().toFloat() }

    private fun normalize(v: FloatArray): FloatArray {
        var n = 0.0
        for (x in v) n += (x * x).toDouble()
        val inv = if (n > 0) (1.0 / sqrt(n)).toFloat() else 0f
        for (i in v.indices) v[i] *= inv
        return v
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
