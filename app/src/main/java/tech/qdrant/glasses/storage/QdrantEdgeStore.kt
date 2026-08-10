package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import tech.qdrant.edge.CountRequest
import tech.qdrant.edge.Distance
import tech.qdrant.edge.EdgeConfig
import tech.qdrant.edge.EdgeShard
import tech.qdrant.edge.Point
import tech.qdrant.edge.UpdateOperation
import tech.qdrant.edge.VectorDataConfig
import tech.qdrant.edge.ffi.Condition
import tech.qdrant.edge.ffi.FieldCondition
import tech.qdrant.edge.ffi.Filter
import tech.qdrant.edge.ffi.HnswIndexConfig
import tech.qdrant.edge.ffi.NamedVector
import tech.qdrant.edge.ffi.PointId
import tech.qdrant.edge.ffi.Query
import tech.qdrant.edge.ffi.QueryRequest
import tech.qdrant.edge.ffi.RangeFloat
import tech.qdrant.edge.ffi.ScoredPoint
import tech.qdrant.edge.ffi.ScoringQuery
import tech.qdrant.edge.ffi.ScrollRequest
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import tech.qdrant.edge.ffi.WithVector
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
) : VectorStore {

    companion object {
        private const val TAG = "QdrantEdgeStore"
        private const val FIELD = "crop"
        const val OBJECT_DIM = 768   // SigLIP2 (Mac endpoint) crop-embedding dimension
    }

    override val name: String get() = if (hnsw) "qdrant-hnsw" else "qdrant-edge"

    // Kept as fields so deleteAll() can drop + recreate the shard on the same directory in-process
    // (no app relaunch): close the native handle, wipe the dir, reload from the same config.
    private val dir: String = File(context.filesDir, "objects_shard_$namespace")
        .also { it.mkdirs() }.absolutePath
    private val config = EdgeConfig(
        vectorData = mapOf(
            FIELD to VectorDataConfig(
                size = dim.toULong(), distance = Distance.COSINE,
                quantizationConfig = null, multivectorConfig = null, datatype = null,
                // HNSW mode: m/efConstruct are the Edge SDK's own bench defaults; fullScanThreshold
                // 10000 means collections under that size still full-scan (HNSW only kicks in above).
                hnswConfig = if (hnsw) HnswIndexConfig(
                    // maxIndexingThreads = 0 → auto (use all cores). Was 1 (copied from the Edge SDK
                    // bench example), which serialized the graph build on ONE of the AR1's 4 cores and
                    // ~3–4×'d the build time — the real cause of the low HNSW ingest, not Qdrant itself.
                    m = 16uL, efConstruct = 100uL, fullScanThreshold = 10000uL,
                    maxIndexingThreads = 0uL, onDisk = false, payloadM = null,
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
    // Guards against a second close(): the native shard is freed on the first, so calling count()
    // or close() on it again is a use-after-free (an uncatchable native abort). Idempotent close.
    private var closed = false

    init {
        shard = EdgeShard.load(dir, config)
        Log.i(TAG, "objects shard opened, count=${shard.count(CountRequest(filter = null, exact = false))}")
    }

    override fun upsert(vector: FloatArray, payload: ObjectPayload): String = synchronized(lock) {
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
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = vector.toList(), using = FIELD)),
            prefetches = emptyList(),
            withVector = null, withPayload = WithPayload.Bool(true),
            filter = null, scoreThreshold = null, params = null
        ))
        val hits = results.map { p -> toHit(p) }
        Log.i(TAG, "search: topK=$topK returned=${hits.size} " +
            hits.take(3).joinToString { "%.3f \"%s\"".format(it.score, it.label.take(20)) })
        hits
    }

    override fun searchFiltered(
        vector: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<ObjectHit> = synchronized(lock) {
        // Filter DURING the scan (a post-filtered top-k could return < k): a payload range condition
        // on timestamp_ms. Both bounds optional (null = open end). No bound → no filter at all.
        // Caveat (see design §2): the shard has NO payload index on timestamp_ms, so this is a
        // filter-during-brute-force-scan, not an indexed filter — the measured cost reflects that.
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
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = vector.toList(), using = FIELD)),
            prefetches = emptyList(),
            withVector = null, withPayload = WithPayload.Bool(true),
            filter = filter, scoreThreshold = null, params = null
        ))
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
        val before = runCatching { shard.count(CountRequest(filter = null, exact = false)).toLong() }.getOrDefault(-1L)
        shard.close()
        File(dir).deleteRecursively()
        File(dir).mkdirs()
        shard = EdgeShard.load(dir, config)
        Log.i(TAG, "deleteAll: dropped $before points, shard recreated empty at $dir")
    }

    override fun buildIndex() {
        // HNSW mode only: build the graph over everything inserted (a single native pass — the pricey
        // step, ~5.5min@100k / ~1h46m@1M measured in the edge-scale bench, and NOT interruptible).
        // Brute-force mode: no index → no-op.
        if (hnsw) synchronized(lock) { shard.optimize() }
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
