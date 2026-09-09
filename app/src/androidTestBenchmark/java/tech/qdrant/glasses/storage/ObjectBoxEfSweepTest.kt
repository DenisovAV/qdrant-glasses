package tech.qdrant.glasses.storage

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.kotlin.query
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.PriorityQueue
import java.util.Random
import kotlin.math.sqrt

/**
 * FAIRNESS CHECK for the scaled vector-DB benchmark's ObjectBox (HNSW) engine.
 *
 * The scaled bench queried ObjectBox with `nearestNeighbors(vec, topK)` — i.e. maxResultCount = topK,
 * which ObjectBox documents as the HNSW *ef* search parameter and explicitly warns is the
 * lowest-quality setting ("use maxResultCount 100 with a query limit of 10 … better quality than
 * just passing in 10"). This test isolates ObjectBox and sweeps ef on the SAME clustered synthetic
 * workload the scaled bench uses (unit cluster centers + per-dim Gaussian noise, std=0.05), scoring
 * recall@5 against an exact brute-force ground truth. recall is hardware-independent, so the emulator
 * measures it validly (unlike latency/build-time, which are the glasses' ARM+flash story).
 *
 * If recall climbs from ~ef=topK up to ~1.0 as ef grows, the benchmark's low HNSW recall was an
 * under-ef artifact, not a property of HNSW. Run:
 *   ./gradlew :app:connectedBenchmarkDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=tech.qdrant.glasses.storage.ObjectBoxEfSweepTest
 */
@RunWith(AndroidJUnit4::class)
class ObjectBoxEfSweepTest {

    private companion object {
        const val TAG = "efsweep"
        const val DIM = ObjectBoxStore.DIM      // 512
        const val N = 20_000                    // fillers; clusters scale with N (≈100 pts/cluster)
        const val CLUSTER_PTS = 100
        const val CLUSTER_STD = 0.05f
        const val N_QUERIES = 30
        const val TOPK = 5
        val EFS = intArrayOf(5, 10, 20, 50, 100, 200)
        const val SEED = 1234L
        const val CENTER_SEED = 4321L
        const val QUERY_SEED = 99L
    }

    @Test fun ef_sweep_recall_on_clustered_data() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store: BoxStore = MyObjectBox.builder()
            .androidContext(ctx.applicationContext).name("objectbox_efsweep").build()
        val box: Box<ObjectBoxMemory> = store.boxFor(ObjectBoxMemory::class.java)
        try {
            box.removeAll()

            val centers = buildCenters(N)
            val filler = Random(SEED)
            val vectors = Array(N) { clustered(filler, centers) }
            val entities = vectors.map { v -> ObjectBoxMemory().apply { embedding = v; timestampMs = 0 } }
            val t0 = System.nanoTime()
            box.put(entities)                    // incremental HNSW build folded in here
            val buildMs = (System.nanoTime() - t0) / 1e6
            val ids = LongArray(N) { entities[it].id }
            Log.i(TAG, "loaded N=$N dim=$DIM in ${buildMs.toInt()}ms; count=${box.count()}")
            assertTrue("expected $N stored, got ${box.count()}", box.count() == N.toLong())

            // Queries drawn near cluster centers (like the scaled bench) so each has real neighbours.
            val qRnd = Random(QUERY_SEED)
            val queries = Array(N_QUERIES) { clustered(qRnd, centers) }

            // Exact top-k ground truth (brute-force cosine over all N; vectors are unit-norm so dot=cos).
            val truth = Array(N_QUERIES) { qi ->
                val h = PriorityQueue<Pair<Long, Float>>(TOPK + 1, compareBy { it.second })
                for (i in 0 until N) {
                    val s = dot(queries[qi], vectors[i])
                    if (h.size < TOPK) h.add(ids[i] to s)
                    else if (s > h.peek()!!.second) { h.poll(); h.add(ids[i] to s) }
                }
                h.map { it.first }.toSet()
            }

            // Reference: exact brute-force scan latency (what Qdrant Edge does — a flat cosine scan).
            // Kotlin loop, so a PESSIMISTIC proxy for Qdrant's native SIMD scan (Qdrant would be
            // faster) — but it anchors "HNSW query vs a flat scan" on the real 4-core wearable.
            run {
                repeat(2) { bruteTopK(queries[0], vectors, ids) }   // warmup
                val ms = DoubleArray(N_QUERIES) { qi ->
                    val t0 = System.nanoTime(); bruteTopK(queries[qi], vectors, ids); (System.nanoTime() - t0) / 1e6
                }
                Log.i(TAG, "brute-force exact scan (Kotlin, pessimistic): qMed=%.2fms".format(ms.sorted()[ms.size / 2]))
            }

            Log.i(TAG, "=== ObjectBox recall@$TOPK + query latency vs ef (N=$N, ${centers.size} clusters, std=$CLUSTER_STD) ===")
            val recallByEf = LinkedHashMap<Int, Double>()
            repeat(3) { box.query(ObjectBoxMemory_.embedding.nearestNeighbors(queries[0], 64)).build().use { it.findWithScores() } } // warmup
            for (ef in EFS) {
                var sum = 0.0
                var minReturned = Int.MAX_VALUE
                val ms = DoubleArray(N_QUERIES)
                for (qi in 0 until N_QUERIES) {
                    val t0 = System.nanoTime()
                    val got = box.query(ObjectBoxMemory_.embedding.nearestNeighbors(queries[qi], ef)).build()
                        .use { q -> q.findWithScores().take(TOPK).map { it.get().id }.toSet() }
                    ms[qi] = (System.nanoTime() - t0) / 1e6
                    if (got.size < minReturned) minReturned = got.size
                    sum += truth[qi].count { it in got }.toDouble() / TOPK
                }
                val recall = sum / N_QUERIES
                recallByEf[ef] = recall
                Log.i(TAG, "ef=%3d  recall@%d=%.3f  qMed=%.2fms  (minReturned=%d)"
                    .format(ef, TOPK, recall, ms.sorted()[ms.size / 2], minReturned))
            }
            // Machine-readable one-liner for easy grep off logcat.
            Log.i(TAG, "RESULT " + recallByEf.entries.joinToString(" ") { "ef${it.key}=%.3f".format(it.value) })

            // The point of the check: a fair ef must recover materially more than ef=topK.
            val low = recallByEf[EFS.first()]!!
            val high = recallByEf[EFS.last()]!!
            assertTrue(
                "recall did not improve with ef (ef=${EFS.first()}→$low, ef=${EFS.last()}→$high) — " +
                    "expected the scaled bench's low HNSW recall to be an under-ef artifact",
                high >= low,
            )
        } finally {
            runCatching { box.removeAll() }
            store.close()
        }
    }

    /**
     * FILTERED search probe: does ObjectBox apply the scalar range filter DURING the ANN traversal
     * (filter-aware / pre-filter) or AFTER it retrieves maxResultCount candidates (post-filter)? We
     * can't tell from the docs, so measure it: for a "last W points" time window (selectivity W/N),
     * score filtered-recall@5 vs the EXACT in-window top-5 at several ef. Our scaled bench used
     * ef = max(topK*8, 64) = 64 for the filtered path. If recall is low at ef=64 on a selective
     * window and only climbs as ef → N, ObjectBox is POST-filtering (the ANN doesn't know about the
     * window), and no reasonable over-fetch rescues a selective filter — an architectural limit, not
     * an under-ef artifact. If it's ~1.0 at ef=64 regardless of window, it's pre-filter (bench fair).
     */
    @Test fun filtered_search_recall_vs_ef_and_window() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store: BoxStore = MyObjectBox.builder()
            .androidContext(ctx.applicationContext).name("objectbox_efsweep_filt").build()
        val box: Box<ObjectBoxMemory> = store.boxFor(ObjectBoxMemory::class.java)
        try {
            box.removeAll()
            val centers = buildCenters(N)
            val filler = Random(SEED)
            val vectors = Array(N) { clustered(filler, centers) }
            // timestampMs = insertion index i, so "the last W points" is a clean selectivity window.
            val entities = vectors.mapIndexed { i, v -> ObjectBoxMemory().apply { embedding = v; timestampMs = i.toLong() } }
            box.put(entities)
            val ids = LongArray(N) { entities[it].id }
            assertTrue("expected $N stored, got ${box.count()}", box.count() == N.toLong())

            val qRnd = Random(QUERY_SEED)
            val queries = Array(N_QUERIES) { clustered(qRnd, centers) }

            Log.i(TAG, "=== ObjectBox FILTERED recall@$TOPK vs ef × window (N=$N) — post-filter probe ===")
            for (w in intArrayOf(2000, 400, 100)) {
                val from = (N - w).toLong(); val to = (N - 1).toLong()
                // exact in-window top-5 (brute force over the window's indices only)
                val truth = Array(N_QUERIES) { qi ->
                    val h = PriorityQueue<Pair<Long, Float>>(TOPK + 1, compareBy { it.second })
                    for (i in (N - w) until N) {
                        val s = dot(queries[qi], vectors[i])
                        if (h.size < TOPK) h.add(ids[i] to s)
                        else if (s > h.peek()!!.second) { h.poll(); h.add(ids[i] to s) }
                    }
                    h.map { it.first }.toSet()
                }
                for (ef in intArrayOf(64, 256, 1024, 4096)) {
                    var sumRecall = 0.0; var sumRet = 0.0
                    for (qi in 0 until N_QUERIES) {
                        val got = box.query(
                            ObjectBoxMemory_.embedding.nearestNeighbors(queries[qi], ef)
                                .and(ObjectBoxMemory_.timestampMs.between(from, to))
                        ).build().use { q -> q.findWithScores().take(TOPK).map { it.get().id }.toSet() }
                        sumRet += got.size
                        sumRecall += truth[qi].count { it in got }.toDouble() / TOPK
                    }
                    Log.i(TAG, "window=%5d (%.2f%%) ef=%4d  filtered-recall@%d=%.3f  avgReturned=%.1f/%d"
                        .format(w, 100.0 * w / N, ef, TOPK, sumRecall / N_QUERIES, sumRet / N_QUERIES, TOPK))
                }
            }
        } finally {
            runCatching { box.removeAll() }
            store.close()
        }
    }

    // ---- clustered synthetic workload (mirrors VectorStoreBenchmark) ----
    private fun buildCenters(n: Int): Array<FloatArray> {
        val k = maxOf(TOPK + 1, n / CLUSTER_PTS)
        val r = Random(CENTER_SEED)
        return Array(k) { unit(r) }
    }

    private fun clustered(rnd: Random, centers: Array<FloatArray>): FloatArray {
        val c = centers[rnd.nextInt(centers.size)]
        val v = FloatArray(DIM) { c[it] + CLUSTER_STD * rnd.nextGaussian().toFloat() }
        return normalize(v)
    }

    private fun unit(rnd: Random): FloatArray = normalize(FloatArray(DIM) { rnd.nextGaussian().toFloat() })

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

    /** Exact top-k by a flat cosine scan over all [vectors] (unit-norm → dot = cosine). */
    private fun bruteTopK(q: FloatArray, vectors: Array<FloatArray>, ids: LongArray): List<Long> {
        val h = PriorityQueue<Pair<Long, Float>>(TOPK + 1, compareBy { it.second })
        for (i in vectors.indices) {
            val s = dot(q, vectors[i])
            if (h.size < TOPK) h.add(ids[i] to s)
            else if (s > h.peek()!!.second) { h.poll(); h.add(ids[i] to s) }
        }
        return h.map { it.first }
    }
}
