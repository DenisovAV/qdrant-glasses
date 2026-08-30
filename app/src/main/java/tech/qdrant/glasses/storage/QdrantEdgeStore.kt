package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import io.qdrant.edge.CountRequest
import io.qdrant.edge.Distance
import io.qdrant.edge.EdgeConfig
import io.qdrant.edge.EdgeShard
import io.qdrant.edge.Point
import io.qdrant.edge.UpdateOperation
import io.qdrant.edge.VectorDataConfig
import io.qdrant.edge.VectorStorageDatatype
import io.qdrant.edge.Condition
import io.qdrant.edge.FieldCondition
import io.qdrant.edge.Filter
import io.qdrant.edge.HnswIndexConfig
import io.qdrant.edge.IntegerIndexParams
import io.qdrant.edge.PayloadIndexParams
import io.qdrant.edge.QuantizationConfig
import io.qdrant.edge.BinaryQuantizationParams
import io.qdrant.edge.BinaryQuantizationEncoding
import io.qdrant.edge.BinaryQuantizationQueryEncoding
import io.qdrant.edge.Memory
import io.qdrant.edge.NamedVector
import io.qdrant.edge.PointId
import io.qdrant.edge.Prefetch
import io.qdrant.edge.Query
import io.qdrant.edge.QueryRequest
import io.qdrant.edge.SearchParams
import io.qdrant.edge.RangeFloat
import io.qdrant.edge.ScoredPoint
import io.qdrant.edge.ScoringQuery
import io.qdrant.edge.ScrollRequest
import io.qdrant.edge.Vector
import io.qdrant.edge.WithPayload
import io.qdrant.edge.WithVector
import java.io.File
import java.util.UUID

/**
 * Object memory: one Qdrant Edge collection of object crops (dense cosine vectors).
 *
 * The former `ObjectStore` verbatim, now the [VectorStore] implementation for the
 * [VectorStoreFactory.Backend.QDRANT_EDGE] backend — logic UNCHANGED (same single monitor lock,
 * same cosine distance, same flush-per-`upsert`, same no-filter scroll in [all]). The new surface
 * ([upsertBatch], [searchFiltered], [deleteAll]) is additive; the demo path ([upsert]/[search]/
 * [all]/[count]) behaves exactly as before.
 *
 * [namespace] picks the on-disk shard directory, so different crop-encoder backends keep
 * SEPARATE collections (e.g. "mac" = SigLIP2/768-dim, "qnnb32" = ViT-B/32/512-dim). This lets you
 * flip [tech.qdrant.glasses.embedding.CropEncoderFactory.backend] and rebuild without clearing
 * data — each variant indexes into its own collection. Vectors of one dim never meet a collection
 * built for another.
 */
class QdrantEdgeStore(
    context: Context,
    private val dim: Int = OBJECT_DIM,
    namespace: String = "default",
    // When true, build an HNSW graph (approximate ANN) instead of the default exact brute-force scan
    // — the graph is finalized in [buildIndex] (`optimize()`). Lets us benchmark BOTH index strategies
    // in the SAME engine (design: brute-force {Qdrant-scan, sqlite-vec} vs HNSW {Qdrant-HNSW, ObjectBox}).
    private val hnsw: Boolean = false,
    // When true, binary-quantize the stored vectors (1 bit/dim → 512 bits = 64 B/vector) and search
    // over the codes (Hamming), i.e. "binary brute-force" — the edge-scale winner we want to measure
    // in-table, not cite from a stale run. Independent of [hnsw]; our benchmark uses binary+brute-force.
    private val binary: Boolean = false,
    // Vector STORAGE precision (independent of [binary]/[hnsw]). null = FLOAT32 (default, resident f32).
    // FLOAT16 halves resident RAM, UINT8 quarters it — a memory↔precision lever sqlite-vec has no
    // equivalent for. Measured against sqlite's streaming footprint for an honest on-device comparison.
    private val storageDatatype: VectorStorageDatatype? = null,
    // When true, call optimize() after the bulk load even in brute-force mode — compacts segments and
    // TRUNCATES the WAL. Measures the WAL-compacted footprint: the uncompacted WAL (~= the full data
    // size again) otherwise inflates on-disk + mmap'd RAM well beyond the actual vector working set.
    private val compact: Boolean = false,
    // When true, build a RANGE-enabled INTEGER payload index on `timestamp_ms` in [buildIndex] so
    // searchFiltered prunes via the index instead of deserializing every point's payload during the
    // brute-force scan. The index MUST be range-enabled (`IntegerIndexParams(range = true)`): the
    // DEFAULT integer index is LOOKUP-ONLY and does NOT accelerate a range/`gte..lte` condition — the
    // exact form searchFiltered issues — so a lookup-only index would silently reproduce the very
    // artifact this variant exists to remove (mirrors the product store's QdrantEdgeMomentStore).
    // The un-indexed default is the fair-comparison bug: sqlite-vec's vec0 metadata column prunes on
    // timestamp natively, so an un-indexed Qdrant filter measured ~46x slower — an artifact of a
    // missing index, not the engine. This variant restores the apples-to-apples filtered comparison.
    private val payloadIndex: Boolean = false,
) : VectorStore {

    companion object {
        private const val TAG = "QdrantEdgeStore"
        private const val FIELD = "crop"
        const val OBJECT_DIM = 768   // SigLIP2 (Mac endpoint) crop-embedding dimension
        // Binary-quant oversampling: the quantized (1-bit) pass fetches OVERSAMPLE×topK candidates, then
        // the outer query rescores them with the ORIGINAL f32 vectors. Higher = better recall, more rescore
        // cost. 4× is Qdrant's common default starting point; tune from the on-device recall check.
        private const val BINARY_OVERSAMPLE = 32L
    }

    override val name: String get() = when {
        binary -> "qdrant-binary"
        hnsw -> "qdrant-hnsw"
        payloadIndex -> "qdrant-idx"
        storageDatatype == VectorStorageDatatype.FLOAT16 -> "qdrant-f16"
        storageDatatype == VectorStorageDatatype.UINT8 -> "qdrant-uint8"
        else -> "qdrant-edge"
    }

    // Kept as fields so deleteAll() can drop + recreate the shard on the same directory in-process
    // (no app relaunch): close the native handle, wipe the dir, reload from the same config.
    private val dir: String = File(context.filesDir, "objects_shard_$namespace")
        .also { it.mkdirs() }.absolutePath
    private val config = EdgeConfig(
        vectorData = mapOf(
            FIELD to VectorDataConfig(
                size = dim.toULong(), distance = Distance.COSINE,
                // Binary quantization (1 bit/dim). PINNED = keep the tiny codes resident in RAM (64 MB@1M).
                // queryEncoding = BINARY → the query is ALSO 1-bit, so the scan is a cheap Hamming
                // (XOR+popcount) — the whole speed point of binary. (SCALAR8_BITS keeps the query at 8-bit
                // for accuracy, but that turns the scan into an ~8-bit dot ≈ f32 cost, killing the speed;
                // measured slower than f32 at 500k.) Recall is recovered by the oversample+rescore in
                // [buildQuery] instead: fetch BINARY_OVERSAMPLE×topK Hamming candidates, re-rank by exact f32.
                quantizationConfig = if (binary) QuantizationConfig.Binary(BinaryQuantizationParams(
                    memory = Memory.PINNED,
                    encoding = BinaryQuantizationEncoding.ONE_BIT,
                    queryEncoding = BinaryQuantizationQueryEncoding.BINARY,
                )) else null,
                multivectorConfig = null, datatype = storageDatatype,
                // HNSW mode: m/efConstruct are the Edge SDK's own bench defaults; fullScanThreshold
                // 10000 means collections under that size still full-scan (HNSW only kicks in above).
                hnswConfig = if (hnsw) HnswIndexConfig(
                    // maxIndexingThreads = 0 → auto (use all cores). MEASURED (2026-08-29, on-device):
                    // auto really does load all 4 AR1 cores (~373% CPU during optimize), but the build is
                    // STILL ~30min@100k (55 pts/s) — NOT several-fold faster than the old 1-thread number.
                    // So thread count is NOT the bottleneck; on-device HNSW graph build is fundamentally
                    // expensive on this SoC. Do NOT build HNSW on-device — brute-force + payload index is
                    // the edge default; HNSW belongs in the cloud, shipped to the edge as a snapshot.
                    m = 16uL, efConstruct = 100uL, fullScanThreshold = 10000uL,
                    // 0.8: HnswIndexConfig dropped the boolean `onDisk` for a richer `memory: Memory?`
                    // (default null = the old in-memory behavior `onDisk = false` gave us).
                    maxIndexingThreads = 0uL, payloadM = null,
                ) else null,
            )
        ),
        sparseVectorData = emptyMap()
    )

    // The native EdgeShard's thread-safety is unverified and this store is touched concurrently:
    // dedup search + upsert on cropLane, the voice-query search on inferLane, and all()/rail
    // snapshots from the MjpegServer HTTP threads. Serialize every native call through one monitor
    // (reentrant, so close() may call count()). deleteAll() reassigns `shard`, so it is `var` and
    // every read of it happens inside the lock.
    private val lock = Any()
    private var shard: EdgeShard
    // Guards close()/deleteAll() against touching an already-freed native shard — the handle is
    // freed the instant shard.close() returns, so any further call on it is a use-after-free (an
    // uncatchable native abort). Only close() and deleteAll() check this (both idempotent as a
    // result); count()/search()/etc. do NOT — calling one of those after close() is still a
    // use-after-free, it's on the caller not to.
    private var closed = false

    init {
        shard = EdgeShard.load(dir, config)
        Log.i(TAG, "objects shard opened, count=${shard.count(CountRequest(filter = null, exact = false))}")
    }

    override fun upsert(vector: FloatArray, payload: ObjectPayload): String = synchronized(lock) {
        require(vector.size == dim) { "dim ${vector.size} != $dim" }
        val id = UUID.randomUUID().toString()
        val named = Vector.Named(mapOf(FIELD to NamedVector.Dense(vector.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = payload.toJson())
        )))
        shard.flush()
        Log.d(TAG, "upsert: id=$id label=\"${payload.label}\" bbox=${payload.bbox} ts=${payload.timestampMs}")
        id
    }

    override fun upsertBatch(items: List<Pair<FloatArray, ObjectPayload>>): List<String> = synchronized(lock) {
        // The engine's true write throughput: ALL points in one native update, ONE flush at the end
        // (vs upsert()'s flush-per-call). Returns the generated ids in input order.
        val ids = ArrayList<String>(items.size)
        val points = items.map { (vector, payload) ->
            require(vector.size == dim) { "dim ${vector.size} != $dim" }
            val id = UUID.randomUUID().toString()
            ids.add(id)
            Point(
                id = PointId.Uuid(id),
                vector = Vector.Named(mapOf(FIELD to NamedVector.Dense(vector.toList()))),
                payload = payload.toJson(),
            )
        }
        shard.update(UpdateOperation.upsertPoints(points))
        shard.flush()
        ids
    }

    override fun search(vector: FloatArray, topK: Int): List<ObjectHit> = synchronized(lock) {
        val results = shard.query(buildQuery(vector, topK, filter = null))
        val hits = results.map { p -> toHit(p) }
        Log.i(TAG, "search: topK=$topK returned=${hits.size} " +
            hits.take(3).joinToString { "%.3f \"%s\"".format(it.score, it.label.take(20)) })
        hits
    }

    /**
     * Build the kNN query. Brute-force / HNSW modes: a plain Nearest (quantizationConfig=null →
     * full-precision scan/graph). BINARY mode: Qdrant's oversampling + rescore — a [Prefetch] pulls
     * [BINARY_OVERSAMPLE]×topK candidates over the 1-bit codes (`params.exact=false` = the fast
     * quantized pass), then the OUTER query rescores just those candidates with the ORIGINAL f32
     * vectors (`params.exact=true`). The Edge SDK exposes no rescore flag, so this prefetch pattern IS
     * the rescore path — without it binary recall is ~0.4, with it it recovers toward exact.
     */
    private fun buildQuery(vector: FloatArray, topK: Int, filter: Filter?): QueryRequest {
        val nearest = ScoringQuery.Vector(Query.Nearest(vector = NamedVector.Dense(vector.toList()), using = FIELD))
        return if (binary) QueryRequest(
            limit = topK.toULong(), offset = null,
            query = nearest,
            prefetches = listOf(Prefetch(
                limit = (topK.toLong() * BINARY_OVERSAMPLE).toULong(),
                query = nearest,
                prefetches = emptyList(), filter = filter, scoreThreshold = null,
                params = SearchParams(exact = false),
            )),
            withVector = null, withPayload = WithPayload.Bool(true),
            // filter already applied in the prefetch that produced the candidate set to rescore.
            filter = null, scoreThreshold = null, params = SearchParams(exact = true),
        ) else QueryRequest(
            limit = topK.toULong(), offset = null,
            query = nearest,
            prefetches = emptyList(),
            withVector = null, withPayload = WithPayload.Bool(true),
            filter = filter, scoreThreshold = null, params = null,
        )
    }

    override fun searchFiltered(
        vector: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<ObjectHit> = synchronized(lock) {
        // Filter DURING the scan (a post-filtered top-k could return < k): a payload range condition
        // on timestamp_ms. Both bounds optional (null = open end). No bound → no filter at all.
        // Cost depends on the index: with payloadIndex=true (see buildIndex) an INTEGER index on
        // timestamp_ms prunes the candidate set first; without it, the filter is evaluated per point
        // during the brute-force scan (much slower — the un-indexed artifact this variant fixes).
        val filter = if (sinceMs == null && untilMs == null) null else Filter(
            must = listOf(Condition.Field(FieldCondition(
                key = "timestamp_ms",
                match = null,
                range = RangeFloat(
                    gte = sinceMs?.toDouble(), gt = null,
                    lte = untilMs?.toDouble(), lt = null,
                ),
                geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
            ))),
            should = null, mustNot = null,
        )
        val results = shard.query(buildQuery(vector, topK, filter))
        val hits = results.map { p -> toHit(p) }
        Log.i(TAG, "searchFiltered: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        hits
    }

    private fun toHit(p: ScoredPoint): ObjectHit = toHit(p.id, p.payload ?: "{}", p.score)

    private fun toHit(id: PointId?, payload: String, score: Float = 0f): ObjectHit {
        // A single malformed payload must not crash the whole result list (fromJson falls back
        // to an empty JSONObject internally, same as the previous inline try/catch here).
        val p = ObjectPayload.fromJson(payload)
        return ObjectHit(
            id = (id as? PointId.Uuid)?.value ?: "",
            score = score,
            label = p.label,
            bbox = p.bbox,
            timestampMs = p.timestampMs,
            thumbPath = p.thumbPath,
        )
    }

    /**
     * Every stored object, oldest-first — used to rebuild the browser rail when a HUD connects so a
     * fresh app/browser start shows the objects already in memory (not an empty rail). Scroll with
     * no filter and no vectors (payload only), then sort by timestamp so the rail restores in the
     * order things were seen.
     */
    override fun all(limit: Int): List<ObjectHit> = synchronized(lock) {
        val resp = shard.scroll(ScrollRequest(
            offset = null, limit = limit.toULong(), filter = null,
            withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
            orderBy = null,
        ))
        resp.records
            .map { rec -> toHit(rec.id, rec.payload ?: "{}") }
            .sortedBy { it.timestampMs }
            .also { Log.i(TAG, "all(): ${it.size} stored objects") }
    }

    override fun count(): Long = synchronized(lock) { shard.count(CountRequest(filter = null, exact = true)).toLong() }

    override fun deleteAll(): Unit = synchronized(lock) {
        // Drop + recreate in-process (no app relaunch): close the native handle, wipe the shard
        // directory on disk, then reload an empty shard from the same config on the same dir.
        check(!closed) { "deleteAll() called on a closed QdrantEdgeStore" }
        val before = runCatching { shard.count(CountRequest(filter = null, exact = false)).toLong() }.getOrDefault(-1L)
        shard.close()
        // The handle is freed the instant shard.close() returns above: if the wipe or the reload
        // below throws, `shard` is now dangling and MUST NOT be touched again. Mark closed=true
        // right away so a later close()/deleteAll() sees the guard and skips it (an uncatchable
        // native abort otherwise); only flip it back once the reload has actually succeeded.
        closed = true
        val wiped = File(dir).deleteRecursively()
        check(wiped) {
            "deleteAll: failed to fully wipe $dir (a locked/mmap'd file likely survived) — " +
                "reloading a shard on top of leftover files would silently keep old points"
        }
        File(dir).mkdirs()
        shard = EdgeShard.load(dir, config)
        closed = false
        Log.i(TAG, "deleteAll: dropped $before points, shard recreated empty at $dir")
    }

    override fun buildIndex() {
        // HNSW mode only: build the graph over everything inserted (a single native pass, NOT
        // interruptible). MEASURED on-device (2026-08-29, auto-thread, all 4 AR1 cores): ~30min@100k
        // at 55 pts/s — super-linear, so 500k+ runs into hours. Prohibitive on-device (and the auto
        // threads did NOT rescue it; see the maxIndexingThreads note above). Brute-force mode: no
        // vector index → no optimize(). payloadIndex mode still builds the timestamp_ms index below.
        synchronized(lock) {
            if (payloadIndex) {
                // RANGE-enabled INTEGER index on timestamp_ms, built once over everything just loaded.
                // range = true is REQUIRED: searchFiltered issues a gte..lte RangeFloat condition, and
                // the default lookup-only integer index does NOT prune a range predicate. A payload
                // index is an update op (not part of EdgeConfig), so it must be created here — a
                // deleteAll() recreates the shard from config and would drop an init-time index.
                shard.update(UpdateOperation.createFieldIndexWithParams(
                    "timestamp_ms", PayloadIndexParams.Integer(IntegerIndexParams(range = true)),
                ))
                shard.flush()
            }
            if (hnsw || compact) shard.optimize()
        }
    }

    override fun close() = synchronized(lock) {
        // Idempotent: a second close() must NOT touch the already-freed native shard.
        if (closed) return@synchronized
        closed = true
        // Always release the native shard even if the diagnostic count throws.
        runCatching { Log.i(TAG, "close: total objects=${count()}") }
        shard.close()
    }
}
