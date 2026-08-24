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
import io.qdrant.edge.Condition
import io.qdrant.edge.FieldCondition
import io.qdrant.edge.Filter
import io.qdrant.edge.HnswIndexConfig
import io.qdrant.edge.NamedVector
import io.qdrant.edge.PointId
import io.qdrant.edge.Query
import io.qdrant.edge.QueryRequest
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
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = NamedVector.Dense(vector.toList()), using = FIELD)),
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
        // Caveat: the shard has NO payload index on timestamp_ms, so this is a
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
            query = ScoringQuery.Vector(Query.Nearest(vector = NamedVector.Dense(vector.toList()), using = FIELD)),
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
        // interruptible). The ~5.5min@100k / ~1h46m@1M figures from the edge-scale bench were
        // measured under the OLD maxIndexingThreads=1 config; this store now passes 0 (auto, see
        // above), which should cut the build time several-fold — NOT yet re-measured on this branch,
        // so treat those numbers as stale/an upper bound, not current fact. Brute-force mode: no
        // index → no-op.
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
