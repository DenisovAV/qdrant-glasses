package tech.qdrant.glasses.storage

import android.content.Context
import android.os.Debug
import android.util.Log
import tech.qdrant.glasses.Config
import java.io.File
import java.util.PriorityQueue
import java.util.Random

/**
 * Sysprop-gated, in-app vector-DB benchmark — GENERIC over [VectorStore], so it measures whatever
 * engine this build compiled ([VectorStoreFactory.backend]) with no change here. Mirrors the
 * `clipbench` pattern (`QnnB32CropEncoder.runBenchmark`): triggered by a sysprop, runs off the main
 * thread, discards warmup, reports median/max/min, writes CSVs to `filesDir`.
 *
 *   adb shell setprop debug.qdrant.dbbench 1        # run the full matrix at next launch
 *   adb shell setprop debug.qdrant.dbbench.max 1000000   # raise the scale cap (default 100k)
 *
 * It runs against a DEDICATED namespace ([BENCH_NAMESPACE]) and wipes it as the last op of every
 * scale, so it never touches the demo's real object memory. Synthetic data is generated + inserted
 * IN CHUNKS ([CHUNK]) and each chunk is discarded — the full N is NEVER materialized in the JVM
 * heap (1M×512 f32 ≈ 2 GB → OOM). Vectors are unit-normalized f32, dim=[DIM], from a seeded
 * [Random] (reproducible). recall@k is scored against an EXACT brute-force top-k maintained
 * incrementally as each chunk is generated (so no need to hold all N to know the ground truth).
 *
 * Output:
 *  - `db_bench_<backend>_<scale>.csv` — raw per-op timings (one row per timed run).
 *  - `db_bench_<backend>.md` — one ready-to-paste summary row per scale (median/max/… + recall,
 *     RAM, disk, and the post-load/reopen point counts).
 */
// Known limitations / methodology caveats:
// - Single-process PSS baseline (ramMb) is SHARED/contaminated across all 4 engines in the
//   head-to-head run (ComparisonBenchmarkTest) — every earlier engine's live heap/native alloc is
//   still resident when the next engine's PSS delta is sampled.
// - Engine order is fixed (qdrant-edge, qdrant-hnsw, objectbox, sqlite-vec), never varied/rotated,
//   so a position-dependent effect (warm page cache, GC pressure) can't be told apart from a real
//   engine difference.
// - coldMs is a SINGLE un-warmed sample (no repeat/median) — one slow scheduling tick and it's noise.
// - recall@k uses only N_QUERIES=10 fixed queries — ±10% resolution (one query flipping = ±0.1 recall).
// - Ground truth is computed with Kotlin `Float`/`Double` dot products; each engine scores with its
//   own native cosine — a boundary tie between the two can rank differently.
// - forceGc() (System.gc() + a fixed sleep) is best-effort — the JVM is not obligated to actually
//   collect, so a PSS sample can occasionally include garbage that hasn't been reclaimed yet.
// - LOAD_BUDGET_MS (the 30-min safety cap) covers the insert loop only, NOT buildIndex(): an HNSW
//   optimize() at 1M is a single non-interruptible native call and can run unbounded past it.
class VectorStoreBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "dbbench"

        private const val DIM = 512               // current ViT-B/32 space
        private const val SEED = 1234L            // filler-vector stream (reproducible)
        private const val QUERY_SEED = 99L        // the fixed query vectors (independent of fillers)
        private const val CHUNK = 5000            // generate+insert this many at a time; never all N
        private const val N_QUERIES = 10          // fixed query set for search / recall
        private const val TOPK = 5                // headline kNN + recall@k
        private const val WARMUP = 3              // discarded before every timed op
        private const val TIMED = 10              // timed runs per search-style op (keeps 1M timing bounded)
        private const val SINGLE_INSERTS = 100    // flush-per-op upserts measured on top of the load
        private const val FILTER_WINDOW = 2000L   // width (in points) of the time filter window
        private const val TS_BASE = 1_000_000_000L // synthetic epoch; point i gets ts = TS_BASE + i
        private const val BENCH_NAMESPACE = "dbbench"
        private const val DEFAULT_MAX = 100_000L
        // DNF an ingest whose RATE collapses (pathological, e.g. ObjectBox HNSW build cost) — by rate,
        // not a fixed time cap, so linear-ingest engines still reach 1M (a fixed cap would wrongly
        // DNF a slow-but-healthy 500k/1M load). Below the floor for 2 consecutive chunks → DNF that
        // scale + all larger ones (never a silent cap). LOAD_BUDGET_MS is only a runaway safety net.
        private const val MIN_INGEST_RATE = 150.0     // pts/s
        private const val LOAD_BUDGET_MS = 1_800_000.0 // 30 min hard safety cap per scale

        // 1k → 10k → 100k → 500k → 1M. Anything above the sysprop cap is LOGGED as skipped, never
        // silently dropped.
        private val SCALES = longArrayOf(1_000, 10_000, 100_000, 500_000, 1_000_000)
    }

    /** Start the whole matrix on a dedicated daemon thread (never the main thread). */
    fun launch() {
        Thread({
            runCatching { run() }.onFailure { Log.e(TAG, "benchmark crashed", it) }
        }, "dbbench").apply { isDaemon = true; start() }
    }

    private fun run() {
        val backend = VectorStoreFactory.backend.name.lowercase()
        val maxScale = Config.sysprop("qdrant.dbbench.max").toLongOrNull() ?: DEFAULT_MAX
        benchmark(backend, maxScale, BENCH_NAMESPACE) { ns -> VectorStoreFactory.create(context, DIM, ns) }
    }

    /**
     * Minimal per-scale outcome exposed to callers (e.g. [ComparisonBenchmarkTest]) that want to
     * assert on a run without re-parsing the CSV/MD output. Deliberately small — the full detail
     * lives in the written files; this is just enough to catch a total breakage in code.
     */
    data class ScaleSummary(val scale: Long, val skipped: Boolean, val failed: Boolean, val dnf: Boolean, val recallAtK: Double)

    /**
     * Run the full matrix against [makeStore] (invoked with the bench namespace), write the raw CSV
     * + one-row-per-scale MD, log each scale. Reusable so an instrumented comparison test can bench
     * several engines back-to-back in ONE process — a clean head-to-head with no NPU/camera contention.
     * Returns one [ScaleSummary] per entry in [SCALES], in order (regardless of maxScale/DNF/failure).
     */
    fun benchmark(backendName: String, maxScale: Long, namespace: String, makeStore: (String) -> VectorStore): List<ScaleSummary> {
        Log.i(TAG, "START backend=$backendName dim=$DIM maxScale=$maxScale scales=${SCALES.toList()}")
        val rows = ArrayList<ScaleResult>()
        var gaveUp = false   // once a scale DNFs on the ingest budget, larger scales can't do better
        for (scale in SCALES) {
            if (scale > maxScale) {
                Log.i(TAG, "SKIP scale=$scale (> max=$maxScale) — logged, not dropped")
                rows.add(ScaleResult.skipped(scale))
                continue
            }
            if (gaveUp) {
                rows.add(ScaleResult.dnf(scale, "not attempted — a smaller scale exceeded the ingest budget"))
                continue
            }
            Log.i(TAG, "scale=$scale: begin")
            val r = try {
                runScale(scale, namespace, makeStore)
            } catch (e: LoadBudgetExceeded) {
                gaveUp = true
                Log.w(TAG, "scale=$scale DNF: ${e.message}")
                ScaleResult.dnf(scale, e.message ?: "ingest budget exceeded")
            } catch (t: Throwable) {
                Log.e(TAG, "scale=$scale failed", t)
                ScaleResult.failed(scale, t.message ?: t.javaClass.simpleName)
            }
            rows.add(r)
            writeCsv(backendName, r)
            writeMarkdown(backendName, rows)   // rewrite after every scale so partial results survive a mid-run device death
            Log.i(TAG, "scale=$scale: ${r.summaryLine()}")
        }
        writeMarkdown(backendName, rows)
        Log.i(TAG, "DONE ($backendName) — results in ${context.filesDir}")
        return rows.map { ScaleSummary(it.scale, it.skipped, it.failed, it.dnf, it.recallAtK) }
    }

    // ---- one scale --------------------------------------------------------------------------

    private fun runScale(n: Long, namespace: String, makeStore: (String) -> VectorStore): ScaleResult {
        val store = makeStore(namespace)
        // Tracks whichever store is CURRENTLY open, so the finally closes exactly once: `store`
        // gets an intentional close() at the cold-load step below (then this goes null), and
        // `reopened` takes its place. Without this the finally would double-close `store` — a
        // native use-after-free in the Rust shard, uncatchable by runCatching.
        var open: VectorStore? = store
        try {
            store.deleteAll() // start from empty regardless of a previous run's leftovers

            val queries = buildQueries()
            val groundTruth = Array(N_QUERIES) { TopK(TOPK) }
            val rnd = Random(SEED)

            // Baseline RAM (PSS) BEFORE loading, so the delta isolates the store's growth.
            forceGc()
            val pssBeforeKb = totalPssKb()

            // ---- LOAD (insert-batch throughput): chunked, ground truth updated per chunk ----
            var loadMs = 0.0
            var inserted = 0L
            var slowChunks = 0
            while (inserted < n) {
                val c = minOf(CHUNK.toLong(), n - inserted).toInt()
                val vecs = Array(c) { randomUnitVector(rnd) }
                val items = ArrayList<Pair<FloatArray, ObjectPayload>>(c)
                for (i in 0 until c) {
                    items.add(vecs[i] to payloadFor(TS_BASE + inserted + i))
                }
                val t0 = System.nanoTime()
                val ids = store.upsertBatch(items)
                val chunkMs = (System.nanoTime() - t0) / 1e6
                loadMs += chunkMs
                // Fold this chunk into the exact top-k for each query BEFORE discarding it.
                for (qi in 0 until N_QUERIES) {
                    val q = queries[qi]
                    for (i in 0 until c) groundTruth[qi].offer(ids[i], dot(q, vecs[i]))
                }
                inserted += c
                // DNF a collapsing ingest fast, by RATE (not a fixed time cap): 2 consecutive chunks
                // below the floor → pathological (HNSW build), bail. Skip the first (cold) chunk.
                val chunkRate = c / (chunkMs / 1000.0)
                if (inserted > CHUNK && chunkRate < MIN_INGEST_RATE) {
                    if (++slowChunks >= 2) throw LoadBudgetExceeded(inserted, loadMs)
                } else {
                    slowChunks = 0
                }
                if (loadMs > LOAD_BUDGET_MS) throw LoadBudgetExceeded(inserted, loadMs) // runaway guard
            }
            // Finalize the ANN index (HNSW engines build the graph here; no-op for brute-force).
            // Count it in ingest so HNSW's build cost shows up in pts/s — comparable to ObjectBox's
            // incremental build already folded into its insert time.
            val buildT0 = System.nanoTime()
            store.buildIndex()
            loadMs += (System.nanoTime() - buildT0) / 1e6

            // ---- integrity gate #1: the store must hold exactly what we inserted. A silently
            // short/over count would make every downstream number (recall, RAM, disk) meaningless,
            // so fail LOUDLY here instead of emitting a normal row on top of corrupted data.
            val storedCount = store.count()
            if (storedCount != inserted) {
                val msg = "post-load count mismatch: store.count()=$storedCount, inserted=$inserted"
                Log.e(TAG, "scale=$n INTEGRITY FAIL: $msg")
                return ScaleResult.failed(n, msg, storedCount = storedCount)
            }

            forceGc()
            val pssAfterKb = totalPssKb()
            val ramMb = (pssAfterKb - pssBeforeKb) / 1024.0
            val diskMb = dirSizeMb(namespace)

            // ---- search-kNN (topK=5) ----
            var minSearchResults = Int.MAX_VALUE
            val searchMs = timed { i ->
                val got = store.search(queries[i % N_QUERIES], TOPK)
                if (got.size < minSearchResults) minSearchResults = got.size
            }
            // A non-empty store (n >= TOPK for every entry in SCALES) must return a full topK on an
            // unfiltered search — fewer is a suspect result set, not "fast search on nothing found".
            if (minSearchResults < TOPK) {
                val msg = "search returned as few as $minSearchResults/$TOPK results on a non-empty " +
                    "store (n=$n) — suspect result set, latency/recall not trustworthy"
                Log.e(TAG, "scale=$n INTEGRITY FAIL: $msg")
                return ScaleResult.failed(n, msg, storedCount = storedCount)
            }

            // ---- recall@k vs the exact ground truth ----
            var recallSum = 0.0
            var shortRecallSearches = 0
            for (qi in 0 until N_QUERIES) {
                val got = store.search(queries[qi], TOPK)
                if (got.size < TOPK) shortRecallSearches++
                val gotIds = got.map { it.id }.toSet()
                val truth = groundTruth[qi].ids()
                if (truth.isNotEmpty()) recallSum += truth.count { it in gotIds }.toDouble() / truth.size
            }
            if (shortRecallSearches > 0) {
                val msg = "search returned < topK=$TOPK for $shortRecallSearches/$N_QUERIES recall " +
                    "queries on a non-empty store (n=$n) — suspect result set, recall not trustworthy"
                Log.e(TAG, "scale=$n INTEGRITY FAIL: $msg")
                return ScaleResult.failed(n, msg, storedCount = storedCount)
            }
            val recall = recallSum / N_QUERIES

            // ---- search-filtered (time window covering the last FILTER_WINDOW points) ----
            val untilMs = TS_BASE + n
            val sinceMs = TS_BASE + maxOf(0L, n - FILTER_WINDOW)
            val filteredReturned = store.searchFiltered(queries[0], TOPK, sinceMs, untilMs).size
            // Unlike the unfiltered search above, < topK is EXPECTED here (the window can legitimately
            // hold fewer than topK points) — only zero is suspect: the window always covers at least
            // one point once n > 0, so an empty result means the filter (or the index under it) is broken.
            if (filteredReturned == 0) {
                val msg = "searchFiltered returned 0 results for window [sinceMs=$sinceMs, " +
                    "untilMs=$untilMs] on a non-empty store (n=$n) — suspect (filter or index bug)"
                Log.e(TAG, "scale=$n INTEGRITY FAIL: $msg")
                return ScaleResult.failed(n, msg, storedCount = storedCount)
            }
            val filteredMs = timed { i -> store.searchFiltered(queries[i % N_QUERIES], TOPK, sinceMs, untilMs) }

            // ---- insert-single (flush-per-op) measured on top of the loaded store ----
            val singleMs = DoubleArray(SINGLE_INSERTS)
            for (i in 0 until SINGLE_INSERTS) {
                val v = randomUnitVector(rnd)
                val t0 = System.nanoTime()
                store.upsert(v, payloadFor(TS_BASE + n + i))
                singleMs[i] = (System.nanoTime() - t0) / 1e6
            }

            // ---- open / cold-load: close the populated shard, reopen it, time the open ----
            // NOTE: this is a WARM-CACHE open (we do NOT drop the OS page cache from the app; that
            // needs `echo 3 > /proc/sys/vm/drop_caches` with root between load and reopen).
            store.close()
            open = null
            val coldT0 = System.nanoTime()
            val reopened = makeStore(namespace)
            open = reopened
            val coldMs = (System.nanoTime() - coldT0) / 1e6
            val reopenedCount = reopened.count()

            // ---- integrity gate #2: the reopened shard must hold every point that was on disk —
            // the n loaded points PLUS the SINGLE_INSERTS single-inserts done above (also persisted,
            // BEFORE this close+reopen) — a partial/corrupted reload would otherwise sail through as
            // a normal, silently-wrong row.
            val wantReopened = n + SINGLE_INSERTS
            if (reopenedCount != wantReopened) {
                val msg = "reopen count mismatch: reopened.count()=$reopenedCount, want=$wantReopened " +
                    "(n=$n + SINGLE_INSERTS=$SINGLE_INSERTS)"
                Log.e(TAG, "scale=$n INTEGRITY FAIL: $msg")
                return ScaleResult.failed(n, msg, storedCount = storedCount, reopenedCount = reopenedCount)
            }

            // ---- deleteAll (clear time) ----
            val delT0 = System.nanoTime()
            reopened.deleteAll()
            val deleteMs = (System.nanoTime() - delT0) / 1e6
            reopened.close()
            open = null

            return ScaleResult(
                scale = n,
                loadMs = loadMs,
                insertBatchPtsPerSec = if (loadMs > 0) n / (loadMs / 1000.0) else 0.0,
                insertSingle = Stats.of(singleMs),
                search = Stats.of(searchMs),
                recallAtK = recall,
                filtered = Stats.of(filteredMs),
                filteredReturned = filteredReturned,
                deleteMs = deleteMs,
                coldLoadMs = coldMs,
                storedCount = storedCount,
                reopenedCount = reopenedCount,
                ramMb = ramMb,
                diskMb = diskMb,
            )
        } finally {
            // Best-effort: close whichever store is still open (the original on an early failure,
            // the reopened one on a late failure) — never the already-closed original.
            runCatching { open?.close() }
        }
    }

    // ---- synthetic data ---------------------------------------------------------------------

    private fun payloadFor(ts: Long) = ObjectPayload(
        label = "obj", bbox = "", timestampMs = ts, trackId = 0, thumbPath = "", caption = "",
    )

    /** A unit-normalized f32 vector (cosine == dot for these), from the given seeded stream. */
    private fun randomUnitVector(rnd: Random): FloatArray {
        val v = FloatArray(DIM)
        var norm = 0.0
        for (i in 0 until DIM) {
            val g = rnd.nextGaussian().toFloat()
            v[i] = g
            norm += (g * g).toDouble()
        }
        val inv = if (norm > 0) (1.0 / Math.sqrt(norm)).toFloat() else 0f
        for (i in 0 until DIM) v[i] *= inv
        return v
    }

    /** The fixed query set — its OWN seed so it's identical across scales and independent of fillers. */
    private fun buildQueries(): Array<FloatArray> {
        val rnd = Random(QUERY_SEED)
        return Array(N_QUERIES) { randomUnitVector(rnd) }
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    // ---- timing + stats ---------------------------------------------------------------------

    /** Warmup discarded, then [TIMED] timed runs; returns the raw per-run milliseconds. */
    private inline fun timed(op: (Int) -> Unit): DoubleArray {
        repeat(WARMUP) { op(it) }
        val ms = DoubleArray(TIMED)
        for (i in 0 until TIMED) {
            val t0 = System.nanoTime()
            op(i)
            ms[i] = (System.nanoTime() - t0) / 1e6
        }
        return ms
    }

    // ---- RAM / disk -------------------------------------------------------------------------

    /** Total PSS in KB — PSS, NOT native-heap-alloc (the latter misses mmap). */
    private fun totalPssKb(): Long {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        return mi.totalPss.toLong()
    }

    private fun forceGc() {
        // Best-effort settle before a PSS sample; a bench thread, so a brief pause is fine.
        System.gc()
        runCatching { Thread.sleep(150) }
    }

    /**
     * Disk footprint of the engine's on-disk artifacts for this bench. Each engine names its store
     * after the namespace (`objects_shard_<ns>` for Qdrant, `objectbox_<ns>` for ObjectBox,
     * `sqlitevec_<ns>.db[-wal/-shm]` for sqlite-vec), so summing every filesDir child whose name
     * carries the namespace is engine-agnostic — as long as the namespace is UNIQUE per engine in a
     * comparison run (which the comparison test ensures), there is no cross-engine double counting.
     */
    private fun dirSizeMb(namespace: String): Double {
        // Walk the WHOLE filesDir and sum any file whose path carries the namespace as a whole TOKEN
        // — catches the flat artifacts (Qdrant `objects_shard_<ns>`, sqlite-vec `sqlitevec_<ns>.db`)
        // AND ObjectBox's NESTED layout (`objectbox/objectbox_<ns>/...`). A plain `.contains(namespace)`
        // is NOT enough: the comparison test's namespaces include "cmpq" (qdrant-edge) and "cmpqh"
        // (qdrant-hnsw), and "cmpq" is a literal prefix of "cmpqh" — a substring match would fold
        // qdrant-hnsw's shard into qdrant-edge's disk total. Requiring a non-alphanumeric (or
        // string-boundary) character on both sides of the match rules that out; it holds for every
        // engine's actual naming since <ns> is always delimited by `_`/`.`/end-of-string there.
        val token = Regex("(?<![A-Za-z0-9])${Regex.escape(namespace)}(?![A-Za-z0-9])")
        val bytes = context.filesDir.walkTopDown()
            .filter { it.isFile && token.containsMatchIn(it.absolutePath) }
            .sumOf { it.length() }
        return bytes / (1024.0 * 1024.0)
    }

    // ---- output -----------------------------------------------------------------------------

    /** Write [sb] to [f], or throw loudly — for a benchmark, the output file IS the deliverable, so
     *  a write failure must abort the run (never a `Log.w` while the caller still logs "DONE"). */
    private fun writeOrThrow(f: File, sb: StringBuilder) {
        runCatching { f.writeText(sb.toString()) }
            .onFailure {
                Log.e(TAG, "WRITE FAILED for ${f.name} — benchmark output is incomplete", it)
                throw IllegalStateException("dbbench: failed to write ${f.name}", it)
            }
    }

    private fun writeCsv(backend: String, r: ScaleResult) {
        if (r.skipped) return   // never attempted — nothing to write
        val f = File(context.filesDir, "db_bench_${backend}_${r.scale}.csv")
        val sb = StringBuilder()
        sb.append("op,run,ms\n")
        if (r.failed || r.dnf) {
            // No reliable per-op timings for a failed/DNF scale, but if it was the count-integrity
            // gate that tripped, the observed counts ARE useful — surface them here too, not just
            // in the one logcat line.
            sb.append("# ${if (r.dnf) "DNF" else "failed"}: ${r.failReason}\n")
            if (r.storedCount >= 0) sb.append("stored_count,0,%.0f\n".format(r.storedCount.toDouble()))
            if (r.reopenedCount >= 0) sb.append("reopened_count,0,%.0f\n".format(r.reopenedCount.toDouble()))
            writeOrThrow(f, sb)
            return
        }
        r.search.raw.forEachIndexed { i, v -> sb.append("search,$i,%.4f\n".format(v)) }
        r.filtered.raw.forEachIndexed { i, v -> sb.append("search_filtered,$i,%.4f\n".format(v)) }
        r.insertSingle.raw.forEachIndexed { i, v -> sb.append("insert_single,$i,%.4f\n".format(v)) }
        // single-shot ops (one measurement each): run index 0
        sb.append("insert_batch_load,0,%.4f\n".format(r.loadMs))
        sb.append("cold_load,0,%.4f\n".format(r.coldLoadMs))
        sb.append("delete_all,0,%.4f\n".format(r.deleteMs))
        sb.append("stored_count,0,%.0f\n".format(r.storedCount.toDouble()))
        sb.append("reopened_count,0,%.0f\n".format(r.reopenedCount.toDouble()))
        writeOrThrow(f, sb)
    }

    private fun writeMarkdown(backend: String, rows: List<ScaleResult>) {
        val f = File(context.filesDir, "db_bench_$backend.md")
        val sb = StringBuilder()
        sb.append("# Vector-DB benchmark — `$backend`\n\n")
        sb.append("dim=$DIM · topK=$TOPK · queries=$N_QUERIES · timed=$TIMED (warmup=$WARMUP discarded) · ")
        sb.append("chunk=$CHUNK · seed=$SEED. Latency in ms. \"max\" is the slowest of $TIMED timed runs, ")
        sb.append("NOT a real percentile (too few samples for one). RAM = PSS delta after load. ")
        sb.append("Disk = shard-dir size. Cold-load is a WARM-cache open (page cache not dropped). ")
        sb.append("stored/reopened = post-load store.count() (vs target n) / post-reopen count() (vs ")
        sb.append("n+SINGLE_INSERTS, since the single-insert measurement below also persists) — ")
        sb.append("a mismatch fails the scale (see the row) rather than hiding in a normal result.\n\n")
        sb.append("| scale | insert-batch (pts/s) | insert-single med/max | search med/max | recall@$TOPK | ")
        sb.append("filtered med/max (n=ret) | deleteAll | cold-load | RAM MB | disk MB | stored/reopened |\n")
        sb.append("|------:|---------------------:|:---------------------:|:--------------:|:----------:|")
        sb.append(":------------------------:|---------:|---------:|-------:|--------:|:---------------:|\n")
        for (r in rows) {
            val counts = countsCell(r)
            if (r.skipped) {
                sb.append("| ${r.scale} | _skipped (> max)_ |  |  |  |  |  |  |  |  | $counts |\n")
                continue
            }
            if (r.failed) {
                sb.append("| ${r.scale} | _failed: ${r.failReason}_ |  |  |  |  |  |  |  |  | $counts |\n")
                continue
            }
            if (r.dnf) {
                sb.append("| ${r.scale} | _DNF: ${r.failReason}_ |  |  |  |  |  |  |  |  | $counts |\n")
                continue
            }
            sb.append(
                "| ${r.scale} | %.0f | %.2f/%.2f | %.3f/%.3f | %.3f | %.3f/%.3f (n=${r.filteredReturned}) | %.2f | %.2f | %.1f | %.2f | $counts |\n"
                    .format(
                        r.insertBatchPtsPerSec,
                        r.insertSingle.median, r.insertSingle.max,
                        r.search.median, r.search.max,
                        r.recallAtK,
                        r.filtered.median, r.filtered.max,
                        r.deleteMs, r.coldLoadMs, r.ramMb, r.diskMb,
                    )
            )
        }
        writeOrThrow(f, sb)
    }

    /** "storedCount/reopenedCount", blank where a scale never got far enough to know either (-1). */
    private fun countsCell(r: ScaleResult): String {
        if (r.storedCount < 0 && r.reopenedCount < 0) return ""
        val stored = if (r.storedCount >= 0) r.storedCount.toString() else "-"
        val reopened = if (r.reopenedCount >= 0) r.reopenedCount.toString() else "-"
        return "$stored/$reopened"
    }

    // ---- helper types -----------------------------------------------------------------------

    /** A bounded exact top-k (min-heap by score) — the recall@k ground truth, built incrementally. */
    private class TopK(private val k: Int) {
        private val heap = PriorityQueue<Pair<String, Float>>(k + 1, compareBy { it.second })
        fun offer(id: String, score: Float) {
            if (heap.size < k) { heap.add(id to score); return }
            val min = heap.peek()
            if (min != null && score > min.second) { heap.poll(); heap.add(id to score) }
        }
        fun ids(): List<String> = heap.map { it.first }
    }

    private class Stats(
        val raw: DoubleArray,
        val median: Double, val min: Double, val max: Double, val mean: Double,
    ) {
        companion object {
            /**
             * [TIMED] samples is too few for a real percentile: nearest-rank p90 over 10 samples
             * lands on index 9 — literally the max, just mislabeled as a percentile. So this reports
             * median/min/max/mean only, HONESTLY — no p90 field to imply a resolution we don't have.
             */
            fun of(ms: DoubleArray): Stats {
                if (ms.isEmpty()) return Stats(ms, 0.0, 0.0, 0.0, 0.0)
                val s = ms.sorted()
                return Stats(raw = ms, median = s[s.size / 2], min = s.first(), max = s.last(), mean = ms.average())
            }
        }
    }

    private class ScaleResult(
        val scale: Long,
        val loadMs: Double = 0.0,
        val insertBatchPtsPerSec: Double = 0.0,
        val insertSingle: Stats = Stats.of(DoubleArray(0)),
        val search: Stats = Stats.of(DoubleArray(0)),
        val recallAtK: Double = 0.0,
        val filtered: Stats = Stats.of(DoubleArray(0)),
        val filteredReturned: Int = 0,
        val deleteMs: Double = 0.0,
        val coldLoadMs: Double = 0.0,
        // -1 = "not reached this stage" (distinct from a genuine 0-point store). Set on success by
        // runScale; failed()/dnf() may also carry them when the count-integrity gate is what tripped,
        // so a mismatch is visible in the CSV/MD row, not buried in one logcat line.
        val storedCount: Long = -1,
        val reopenedCount: Long = -1,
        val ramMb: Double = 0.0,
        val diskMb: Double = 0.0,
        val skipped: Boolean = false,
        val failed: Boolean = false,
        val dnf: Boolean = false,
        val failReason: String = "",
    ) {
        fun summaryLine(): String =
            if (skipped) "skipped"
            else if (dnf) "DNF: $failReason"
            else if (failed) "FAILED: $failReason (stored=$storedCount reopened=$reopenedCount)"
            else "load=%.0fms (%.0f pts/s) search med=%.3fms recall@$TOPK=%.3f filtered med=%.3fms (n=%d) single med=%.2fms delete=%.2fms cold=%.2fms ram=%.1fMB disk=%.2fMB stored=%d reopened=%d"
                .format(loadMs, insertBatchPtsPerSec, search.median, recallAtK, filtered.median,
                    filteredReturned, insertSingle.median, deleteMs, coldLoadMs, ramMb, diskMb, storedCount, reopenedCount)

        companion object {
            fun skipped(scale: Long) = ScaleResult(scale = scale, skipped = true)
            fun failed(scale: Long, reason: String, storedCount: Long = -1, reopenedCount: Long = -1) =
                ScaleResult(scale = scale, failed = true, failReason = reason, storedCount = storedCount, reopenedCount = reopenedCount)
            fun dnf(scale: Long, reason: String) = ScaleResult(scale = scale, dnf = true, failReason = reason)
        }
    }

    /** Thrown when a scale's ingest RATE collapses (or the safety cap trips) — recorded as DNF. */
    private class LoadBudgetExceeded(inserted: Long, ms: Double) :
        Exception("ingest rate collapsed — ${inserted}pts in ${(ms / 1000).toInt()}s (< ${MIN_INGEST_RATE.toInt()} pts/s)")
}
