package tech.qdrant.glasses.storage

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Head-to-head on-device benchmark of every working engine in ONE process — a clean comparison with
 * no NPU/camera contention (unlike the sysprop dbbench, which boots the full app). Reuses the
 * reviewed [VectorStoreBenchmark] matrix per engine; each engine gets a UNIQUE namespace so their
 * on-disk artifacts never collide. Every engine writes `db_bench_<name>.md` + `db_bench_<name>_<scale>.csv`
 * to filesDir; pull them and assemble the comparison table.
 *
 * Run on the glasses (or any arm64 device):
 *   ./gradlew :app:assembleDebug :app:assembleDebugAndroidTest
 *   adb install -r <app.apk> ; adb install -r <androidTest.apk>
 *   adb shell am instrument -w -e class tech.qdrant.glasses.storage.ComparisonBenchmarkTest \
 *     -e maxScale 100000 tech.qdrant.glasses.test/androidx.test.runner.AndroidJUnitRunner
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
        //   brute-force: qdrant-edge (scan), sqlite-vec   ·   HNSW: qdrant-hnsw, objectbox
        // An engine that can't ingest a scale in budget records DNF (never a silent cap).
        bench.benchmark("qdrant-edge", max, "cmpq") { ns -> QdrantEdgeStore(ctx, dim, ns) }
        // Qdrant in HNSW mode: capped at 100k — its graph build (optimize()) is a single
        // non-interruptible native pass (~5.5min@100k, ~1h46m@1M in the edge-scale bench), so 500k/1M
        // are impractical on-device (shown as skipped, same honest limit ObjectBox hits via DNF).
        bench.benchmark("qdrant-hnsw", minOf(max, 100_000L), "cmpqh") { ns -> QdrantEdgeStore(ctx, dim, ns, hnsw = true) }
        bench.benchmark("objectbox", max, "cmpob") { ns -> ObjectBoxStore(ctx, dim, ns) }
        bench.benchmark("sqlite-vec", max, "cmpsv") { ns -> SqliteVecStore(ctx, dim, ns) }
    }
}
