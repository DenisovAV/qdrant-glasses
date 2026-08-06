package tech.qdrant.glasses.storage

import android.content.Context
import android.os.Debug
import android.util.Log
import tech.qdrant.glasses.Config
import java.io.File
import java.util.PriorityQueue
import java.util.Random

/**
 * Sysprop-gated, in-app vector-DB benchmark (design §3) — GENERIC over [VectorStore], so it
 * measures whatever engine this build compiled ([VectorStoreFactory.backend]) with no change here.
 * Mirrors the `clipbench` pattern (`QnnB32CropEncoder.runBenchmark`): triggered by a sysprop, runs
 * off the main thread, discards warmup, reports median/p90/min/max, writes CSVs to `filesDir`.
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
 * Output (design §3.4):
 *  - `db_bench_<backend>_<scale>.csv` — raw per-op timings (one row per timed run).
 *  - `db_bench_<backend>.md` — one ready-to-paste summary row per scale (median/p90/… + recall,
 *     RAM, disk).
 */
class VectorStoreBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "dbbench"

        private const val DIM = 512               // current ViT-B/32 space (design §1)
        private const val SEED = 1234L            // filler-vector stream (reproducible)
        private const val QUERY_SEED = 99L        // the fixed query vectors (independent of fillers)
        private const val CHUNK = 5000            // generate+insert this many at a time; never all N
        private const val N_QUERIES = 10          // fixed query set for search / recall
        private const val TOPK = 5                // headline kNN + recall@k
        private const val WARMUP = 3              // discarded before every timed op
        private const val TIMED = 20              // timed runs per search-style op
        private const val SINGLE_INSERTS = 100    // flush-per-op upserts measured on top of the load
        private const val FILTER_WINDOW = 2000L   // width (in points) of the time filter window
        private const val TS_BASE = 1_000_000_000L // synthetic epoch; point i gets ts = TS_BASE + i
        private const val BENCH_NAMESPACE = "dbbench"
        private const val DEFAULT_MAX = 100_000L

        // 1k → 10k → 100k → 500k → 1M (design §3.2). Anything above the sysprop cap is LOGGED as
        // skipped, never silently dropped.
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
        Log.i(TAG, "START backend=$backend dim=$DIM maxScale=$maxScale scales=${SCALES.toList()}")

        val rows = ArrayList<ScaleResult>()
        for (scale in SCALES) {
            if (scale > maxScale) {
                Log.i(TAG, "SKIP scale=$scale (> max=$maxScale) — logged, not dropped")
                rows.add(ScaleResult.skipped(scale))
                continue
            }
            Log.i(TAG, "scale=$scale: begin")
            val r = runCatching { runScale(scale) }.getOrElse {
                Log.e(TAG, "scale=$scale failed", it)
                ScaleResult.failed(scale, it.message ?: it.javaClass.simpleName)
            }
            rows.add(r)
            writeCsv(backend, r)
            Log.i(TAG, "scale=$scale: ${r.summaryLine()}")
        }
        writeMarkdown(backend, rows)
        Log.i(TAG, "DONE — results in ${context.filesDir}")
    }

    // ---- one scale --------------------------------------------------------------------------

    private fun runScale(n: Long): ScaleResult {
        val store = VectorStoreFactory.create(context, DIM, BENCH_NAMESPACE)
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
            while (inserted < n) {
                val c = minOf(CHUNK.toLong(), n - inserted).toInt()
                val vecs = Array(c) { randomUnitVector(rnd) }
                val items = ArrayList<Pair<FloatArray, ObjectPayload>>(c)
                for (i in 0 until c) {
                    items.add(vecs[i] to payloadFor(TS_BASE + inserted + i))
                }
                val t0 = System.nanoTime()
                val ids = store.upsertBatch(items)
                loadMs += (System.nanoTime() - t0) / 1e6
                // Fold this chunk into the exact top-k for each query BEFORE discarding it.
                for (qi in 0 until N_QUERIES) {
                    val q = queries[qi]
                    for (i in 0 until c) groundTruth[qi].offer(ids[i], dot(q, vecs[i]))
                }
                inserted += c
            }

            forceGc()
            val pssAfterKb = totalPssKb()
            val ramMb = (pssAfterKb - pssBeforeKb) / 1024.0
            val diskMb = dirSizeMb(BENCH_NAMESPACE)

            // ---- search-kNN (topK=5) ----
            val searchMs = timed { i -> store.search(queries[i % N_QUERIES], TOPK) }

            // ---- recall@k vs the exact ground truth ----
            var recallSum = 0.0
            for (qi in 0 until N_QUERIES) {
                val got = store.search(queries[qi], TOPK).map { it.id }.toSet()
                val truth = groundTruth[qi].ids()
                if (truth.isNotEmpty()) recallSum += truth.count { it in got }.toDouble() / truth.size
            }
            val recall = recallSum / N_QUERIES

            // ---- search-filtered (time window covering the last FILTER_WINDOW points) ----
            val untilMs = TS_BASE + n
            val sinceMs = TS_BASE + maxOf(0L, n - FILTER_WINDOW)
            val filteredReturned = store.searchFiltered(queries[0], TOPK, sinceMs, untilMs).size
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
            val reopened = VectorStoreFactory.create(context, DIM, BENCH_NAMESPACE)
            open = reopened
            val coldMs = (System.nanoTime() - coldT0) / 1e6
            val reopenedCount = reopened.count()

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

    /** Total PSS in KB (design §3: PSS, NOT native-heap-alloc — the latter misses mmap). */
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
     * Disk footprint of the engine's on-disk collection. For Qdrant Edge the directory is
     * `objects_shard_<namespace>` (its [QdrantEdgeStore] convention) — a future engine with a
     * different layout adjusts this when its phase lands.
     */
    private fun dirSizeMb(namespace: String): Double {
        val dir = File(context.filesDir, "objects_shard_$namespace")
        if (!dir.exists()) return 0.0
        val bytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        return bytes / (1024.0 * 1024.0)
    }

    // ---- output -----------------------------------------------------------------------------

    private fun writeCsv(backend: String, r: ScaleResult) {
        if (r.skipped || r.failed) return
        val f = File(context.filesDir, "db_bench_${backend}_${r.scale}.csv")
        val sb = StringBuilder()
        sb.append("op,run,ms\n")
        r.search.raw.forEachIndexed { i, v -> sb.append("search,$i,%.4f\n".format(v)) }
        r.filtered.raw.forEachIndexed { i, v -> sb.append("search_filtered,$i,%.4f\n".format(v)) }
        r.insertSingle.raw.forEachIndexed { i, v -> sb.append("insert_single,$i,%.4f\n".format(v)) }
        // single-shot ops (one measurement each): run index 0
        sb.append("insert_batch_load,0,%.4f\n".format(r.loadMs))
        sb.append("cold_load,0,%.4f\n".format(r.coldLoadMs))
        sb.append("delete_all,0,%.4f\n".format(r.deleteMs))
        runCatching { f.writeText(sb.toString()) }
            .onFailure { Log.w(TAG, "csv write failed: ${it.message}") }
    }

    private fun writeMarkdown(backend: String, rows: List<ScaleResult>) {
        val f = File(context.filesDir, "db_bench_$backend.md")
        val sb = StringBuilder()
        sb.append("# Vector-DB benchmark — `$backend`\n\n")
        sb.append("dim=$DIM · topK=$TOPK · queries=$N_QUERIES · timed=$TIMED (warmup=$WARMUP discarded) · ")
        sb.append("chunk=$CHUNK · seed=$SEED. Latency in ms. RAM = PSS delta after load. ")
        sb.append("Disk = shard-dir size. Cold-load is a WARM-cache open (page cache not dropped).\n\n")
        sb.append("| scale | insert-batch (pts/s) | insert-single med/p90 | search med/p90 | recall@$TOPK | ")
        sb.append("filtered med/p90 (n=ret) | deleteAll | cold-load | RAM MB | disk MB |\n")
        sb.append("|------:|---------------------:|:---------------------:|:--------------:|:----------:|")
        sb.append(":------------------------:|---------:|---------:|-------:|--------:|\n")
        for (r in rows) {
            if (r.skipped) {
                sb.append("| ${r.scale} | _skipped (> max)_ |  |  |  |  |  |  |  |  |\n")
                continue
            }
            if (r.failed) {
                sb.append("| ${r.scale} | _failed: ${r.failReason}_ |  |  |  |  |  |  |  |  |\n")
                continue
            }
            sb.append(
                "| ${r.scale} | %.0f | %.2f/%.2f | %.3f/%.3f | %.3f | %.3f/%.3f (n=${r.filteredReturned}) | %.2f | %.2f | %.1f | %.2f |\n"
                    .format(
                        r.insertBatchPtsPerSec,
                        r.insertSingle.median, r.insertSingle.p90,
                        r.search.median, r.search.p90,
                        r.recallAtK,
                        r.filtered.median, r.filtered.p90,
                        r.deleteMs, r.coldLoadMs, r.ramMb, r.diskMb,
                    )
            )
        }
        runCatching { f.writeText(sb.toString()) }
            .onFailure { Log.w(TAG, "md write failed: ${it.message}") }
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
        val median: Double, val p90: Double, val min: Double, val max: Double, val mean: Double,
    ) {
        companion object {
            fun of(ms: DoubleArray): Stats {
                if (ms.isEmpty()) return Stats(ms, 0.0, 0.0, 0.0, 0.0, 0.0)
                val s = ms.sorted()
                return Stats(
                    raw = ms,
                    median = s[s.size / 2],
                    p90 = s[minOf(s.size - 1, (s.size * 9) / 10)],
                    min = s.first(), max = s.last(),
                    mean = ms.average(),
                )
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
        val reopenedCount: Long = 0,
        val ramMb: Double = 0.0,
        val diskMb: Double = 0.0,
        val skipped: Boolean = false,
        val failed: Boolean = false,
        val failReason: String = "",
    ) {
        fun summaryLine(): String =
            if (skipped) "skipped"
            else if (failed) "FAILED: $failReason"
            else "load=%.0fms (%.0f pts/s) search med=%.3fms recall@$TOPK=%.3f filtered med=%.3fms (n=%d) single med=%.2fms delete=%.2fms cold=%.2fms ram=%.1fMB disk=%.2fMB reopened=%d"
                .format(loadMs, insertBatchPtsPerSec, search.median, recallAtK, filtered.median,
                    filteredReturned, insertSingle.median, deleteMs, coldLoadMs, ramMb, diskMb, reopenedCount)

        companion object {
            fun skipped(scale: Long) = ScaleResult(scale = scale, skipped = true)
            fun failed(scale: Long, reason: String) = ScaleResult(scale = scale, failed = true, failReason = reason)
        }
    }
}
