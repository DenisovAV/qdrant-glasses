package tech.qdrant.glasses.storage

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Head-to-head on-device benchmark of every working engine in ONE process — a clean comparison with
 * no NPU/camera contention (unlike the sysprop dbbench, which boots the full app). Reuses the
 * reviewed [VectorStoreBenchmark] matrix per engine; each engine gets a UNIQUE namespace so their
 * on-disk artifacts never collide. Every engine writes `db_bench_<name>.md` + `db_bench_<name>_<scale>.csv`
 * to filesDir; pull them and assemble the comparison table.
 *
 * Run on the glasses (or any arm64 device) — this test only exists in the `benchmark` flavor
 * (see `build.gradle.kts`'s `engines` dimension), so the task names carry that flavor:
 *   ./gradlew :app:assembleBenchmarkDebug :app:assembleBenchmarkDebugAndroidTest
 *   adb install -r <benchmarkDebug.apk> ; adb install -r <benchmarkDebugAndroidTest.apk>
 *   adb shell am instrument -w -e class tech.qdrant.glasses.storage.ComparisonBenchmarkTest \
 *     -e maxScale 100000 tech.qdrant.glasses.test/androidx.test.runner.AndroidJUnitRunner
 * or, in one step (build + install + run):
 *   ./gradlew :app:connectedBenchmarkDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.maxScale=100000
 *
 * `maxScale` (instrumentation arg, default 10000) caps the scale sweep (1k→1M) so a quick first pass
 * stays short and a full run is one flag away.
 */
@RunWith(AndroidJUnit4::class)
class ComparisonBenchmarkTest {

    @Test fun benchmark_all_engines_headtohead() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val max = InstrumentationRegistry.getArguments().getString("maxScale")?.toLongOrNull() ?: 10_000L
        val bench = VectorStoreBenchmark(ctx)
        val dim = 512

        // Both index strategies, IDENTICAL ops per engine:
        //   brute-force: qdrant-edge (scan), sqlite-vec   ·   HNSW: qdrant-hnsw, objectbox, chroma
        // An engine that can't ingest a scale in budget records DNF (never a silent cap).
        val qdrantEdge = bench.benchmark("qdrant-edge", max, "cmpq") { ns -> QdrantEdgeStore(ctx, dim, ns) }
        // Qdrant in HNSW mode: UNCAPPED — runs the full SCALES sweep (1k→1M) same `max` ceiling as
        // every other engine (was capped at 100k pending a re-measurement of the graph build
        // (optimize()) cost after cc2062d switched maxIndexingThreads 1→0/auto). That re-measurement
        // has NOT happened yet — the old ~5.5min@100k / ~1h46m@1M figures are still the stale,
        // pre-cc2062d upper bound (see QdrantEdgeStore.buildIndex's own caveat). A 1M HNSW build is
        // now INTENTIONALLY allowed to take that long: buildIndex()'s optimize() is a single
        // non-interruptible native call that VectorStoreBenchmark.LOAD_BUDGET_MS does NOT cover (only
        // the insert loop does — see its own "Known limitations" note), so this can genuinely run
        // past 30 min uninterrupted at 1M. Not a bug; that is the intentional trade for finally
        // getting a real number instead of a stale estimate.
        val qdrantHnsw = bench.benchmark("qdrant-hnsw", max, "cmpqh") { ns -> QdrantEdgeStore(ctx, dim, ns, hnsw = true) }
        val objectBox = bench.benchmark("objectbox", max, "cmpob") { ns -> ObjectBoxStore(ctx, dim, ns) }
        val sqliteVec = bench.benchmark("sqlite-vec", max, "cmpsv") { ns -> SqliteVecStore(ctx, dim, ns) }
        val chroma = bench.benchmark("chroma", max, "cmpch") { ns -> ChromaStore(ctx, dim, ns) }

        // The run must be able to FAIL — assert the deterministic, device-independent facts only
        // (never exact latency/RAM numbers, which vary with load; see VectorStoreBenchmark's
        // "Known limitations" note). SCALES[0] (smallest scale) is always attempted here since
        // maxScale defaults to 10_000 >= 1_000.
        val smallest = qdrantEdge.first()
        assertFalse(
            "qdrant-edge (baseline) skipped its smallest scale — maxScale too low?",
            smallest.skipped,
        )
        assertFalse(
            "qdrant-edge (baseline) DNF'd/failed at its smallest scale: $smallest",
            smallest.dnf || smallest.failed,
        )
        // qdrant-edge defaults to brute-force (exact) — recall@5 vs the Kotlin ground truth must be
        // ~1.0. Not asserted at hard 1.0: native cosine vs this test's Kotlin Double ground truth can
        // disagree on a boundary tie (same reasoning as SqliteVecStoreTest's exact-recall check).
        assertTrue(
            "exact engine (qdrant-edge, brute-force) recall@5=${smallest.recallAtK} at the smallest scale, expected ~1.0",
            smallest.recallAtK >= 0.99,
        )

        // The other three engines must at least COMPLETE the smallest scale — this is what stops
        // "all_engines" going green while 3 of 4 engines are fully broken (they used to be discarded
        // unasserted). DNF is a legitimate outcome at LARGER scales (e.g. ObjectBox's HNSW build
        // cost) but every engine here completes SCALES.first() (1000), so skipped/DNF/failed at the
        // smallest scale is always a real bug, never a budget limit.
        for ((name, summaries) in listOf("qdrant-hnsw" to qdrantHnsw, "objectbox" to objectBox, "sqlite-vec" to sqliteVec, "chroma" to chroma)) {
            val s = summaries.first()
            assertFalse("$name skipped its smallest scale — maxScale too low?", s.skipped)
            assertFalse("$name DNF'd/failed at its smallest scale: $s", s.dnf || s.failed)
        }
        // sqlite-vec is exact brute-force too (like qdrant-edge) — same recall bar applies.
        val sqliteSmallest = sqliteVec.first()
        assertTrue(
            "exact engine (sqlite-vec, brute-force) recall@5=${sqliteSmallest.recallAtK} at the smallest scale, expected ~1.0",
            sqliteSmallest.recallAtK >= 0.99,
        )
    }
}
