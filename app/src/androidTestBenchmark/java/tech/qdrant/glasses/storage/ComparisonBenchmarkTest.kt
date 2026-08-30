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
        // Realistic CLUSTERED workload by default; pass `-e workload random` for the adversarial column.
        val wl = if (InstrumentationRegistry.getArguments().getString("workload").equals("random", ignoreCase = true))
            VectorStoreBenchmark.Workload.RANDOM else VectorStoreBenchmark.Workload.CLUSTERED
        val bench = VectorStoreBenchmark(ctx)
        val dim = 512

        // Both index strategies, IDENTICAL ops per engine:
        //   brute-force: qdrant-edge (scan), sqlite-vec   ·   HNSW: qdrant-hnsw, objectbox, chroma
        // An engine that can't ingest a scale in budget records DNF (never a silent cap).
        val qdrantEdge = bench.benchmark("qdrant-edge", max, "cmpq", wl) { ns -> QdrantEdgeStore(ctx, dim, ns) }
        val objectBox = bench.benchmark("objectbox", max, "cmpob", wl) { ns -> ObjectBoxStore(ctx, dim, ns) }
        val sqliteVec = bench.benchmark("sqlite-vec", max, "cmpsv", wl) { ns -> SqliteVecStore(ctx, dim, ns) }
        val chroma = bench.benchmark("chroma", max, "cmpch", wl) { ns -> ChromaStore(ctx, dim, ns) }
        // Qdrant in HNSW mode: UNCAPPED — runs the full SCALES sweep (1k→1M) same `max` ceiling as
        // every other engine (was capped at 100k). buildIndex()'s optimize() is a single
        // non-interruptible native call that VectorStoreBenchmark.LOAD_BUDGET_MS does NOT cover (only
        // the insert loop does), so a 1M build can genuinely run ~1h46m uninterrupted — intentionally
        // allowed now to get a real number instead of a stale estimate.
        // Run LAST on purpose: it is the long/risky pole (that ~1h46m non-interruptible 1M build), so a
        // UVLO reboot mid-build doesn't cost the other four engines' results, already written+pulled by then.
        val qdrantHnsw = bench.benchmark("qdrant-hnsw", max, "cmpqh", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, hnsw = true) }

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

    /**
     * CLEAN single-engine run: one engine, selected by `-e engine <name>`, in THIS instrumentation
     * process. Run once per engine (a fresh `am instrument` each) with a filesDir wipe + force-stop
     * between — that gives an uncontaminated per-engine RAM (PSS) baseline (the head-to-head shares
     * ONE process, so its RAM column is polluted by earlier engines) and NO app/camera/NPU contention
     * (unlike the sysprop path, which boots the full app).
     *
     *   adb shell am instrument -w -e class \
     *     tech.qdrant.glasses.storage.ComparisonBenchmarkTest#benchmark_one \
     *     -e engine qdrant-binary -e maxScale 1000000 -e workload clustered \
     *     tech.qdrant.glasses.test/androidx.test.runner.AndroidJUnitRunner
     */
    @Test fun benchmark_one() {
        val args = InstrumentationRegistry.getArguments()
        val engine = args.getString("engine")
            ?: error("pass -e engine <qdrant-edge|qdrant-idx|qdrant-binary|qdrant-hnsw|sqlite-vec|sqlite-binary|chroma|objectbox>")
        val max = args.getString("maxScale")?.toLongOrNull() ?: 10_000L
        val wl = if (args.getString("workload").equals("random", ignoreCase = true))
            VectorStoreBenchmark.Workload.RANDOM else VectorStoreBenchmark.Workload.CLUSTERED
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val bench = VectorStoreBenchmark(ctx)
        val dim = 512
        val summaries = when (engine) {
            "qdrant-edge"   -> bench.benchmark("qdrant-edge", max, "oneqe", wl) { ns -> QdrantEdgeStore(ctx, dim, ns) }
            "qdrant-f16"    -> bench.benchmark("qdrant-f16", max, "oneqf16", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, storageDatatype = io.qdrant.edge.VectorStorageDatatype.FLOAT16) }
            "qdrant-uint8"  -> bench.benchmark("qdrant-uint8", max, "onequ8", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, storageDatatype = io.qdrant.edge.VectorStorageDatatype.UINT8) }
            "qdrant-opt"    -> bench.benchmark("qdrant-opt", max, "oneqopt", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, compact = true) }
            "qdrant-idx"    -> bench.benchmark("qdrant-idx", max, "oneqidx", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, payloadIndex = true) }
            "qdrant-f16-opt" -> bench.benchmark("qdrant-f16-opt", max, "oneqf16o", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, storageDatatype = io.qdrant.edge.VectorStorageDatatype.FLOAT16, compact = true) }
            "qdrant-binary" -> bench.benchmark("qdrant-binary", max, "oneqb", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, binary = true) }
            "qdrant-hnsw"   -> bench.benchmark("qdrant-hnsw", max, "oneqh", wl) { ns -> QdrantEdgeStore(ctx, dim, ns, hnsw = true) }
            "sqlite-vec"    -> bench.benchmark("sqlite-vec", max, "onesv", wl) { ns -> SqliteVecStore(ctx, dim, ns) }
            "sqlite-binary" -> bench.benchmark("sqlite-binary", max, "onesb", wl) { ns -> SqliteVecStore(ctx, dim, ns, binary = true) }
            "chroma"        -> bench.benchmark("chroma", max, "onech", wl) { ns -> ChromaStore(ctx, dim, ns) }
            "objectbox"     -> bench.benchmark("objectbox", max, "oneob", wl) { ns -> ObjectBoxStore(ctx, dim, ns) }
            else            -> error("unknown engine: $engine")
        }
        assertFalse("$engine skipped its smallest scale — maxScale too low?", summaries.first().skipped)
    }
}
