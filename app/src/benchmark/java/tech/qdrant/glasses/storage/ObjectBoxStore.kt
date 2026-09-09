package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.kotlin.query
import tech.qdrant.glasses.Config

/**
 * ObjectBox implementation of [VectorStore] (the benchmark's Phase 2 engine) — the HNSW alternative to
 * [QdrantEdgeStore]'s exact brute-force scan. Only instantiated when
 * [VectorStoreFactory.backend] == OBJECTBOX (a benchmark build); the demo default stays Qdrant.
 *
 * Cosine is native ([ObjectBoxMemory]'s `@HnswIndex(distanceType = COSINE)`), so vectors are stored
 * as-is. ObjectBox returns cosine *distance* (nearest-first, smaller = more similar); [toHit] flips
 * it to a higher-is-better similarity so [ObjectHit.score] means the same thing across engines.
 *
 * ObjectBox's `Box` is itself thread-safe, so (unlike the Qdrant shard) no external monitor is
 * needed. The vector dimension is fixed in the entity at codegen (§6), hence the [dim] assert.
 */
class ObjectBoxStore(context: Context, dim: Int, namespace: String) : VectorStore {

    companion object {
        private const val TAG = "ObjectBoxStore"
        const val DIM = 512   // must equal ObjectBoxMemory.embedding @HnswIndex(dimensions=...)
        // Over-fetch factor for searchFiltered: the HNSW maxResultCount bounds the candidate set
        // that the scalar filter then trims, so ask for more candidates than topK to leave enough
        // survivors inside the time window (ObjectBox docs' caveat).
        private const val FILTER_OVERFETCH = 8
        // ObjectBox's nearestNeighbors(vec, maxResultCount) — maxResultCount DOUBLES as the HNSW `ef`
        // search-quality knob. Their docs: use a value WELL ABOVE the desired count (e.g. 100 for a
        // limit of 10). Querying with maxResultCount == topK (ef == topK) is the lowest-quality
        // setting and tanks recall — an earlier version of this store did exactly that. Query with a
        // real ef, then keep the top-K. Overridable per run: `setprop debug.qdrant.dbbench.ef <n>`.
        private val SEARCH_EF = Config.sysprop("qdrant.dbbench.ef").toIntOrNull()?.takeIf { it > 0 } ?: 200
    }

    init {
        require(dim == DIM) {
            "ObjectBox entity dim is compile-time-fixed at $DIM (got $dim) — regenerate the entity for another dim"
        }
    }

    private val store: BoxStore = MyObjectBox.builder()
        .androidContext(context.applicationContext)
        .name("objectbox_$namespace")   // own directory per namespace — never collides with Qdrant's shard
        // Default DB file cap is 1 GB; 512-d f32 vectors + the HNSW graph blow past that around ~500k
        // (DbFullException). Raise it so the on-device ceiling is CPU/build-bound, not an unset limit.
        // Default 2 GiB (covers our range with less small-scale file pre-growth than 4 GiB); override:
        //   setprop debug.qdrant.dbbench.maxdbkb <kb>
        .maxSizeInKByte(Config.sysprop("qdrant.dbbench.maxdbkb").toLongOrNull()?.takeIf { it > 0 } ?: (2L * 1024 * 1024))
        .build()
    private val box: Box<ObjectBoxMemory> = store.boxFor(ObjectBoxMemory::class.java)

    override val name: String = "objectbox"

    override fun upsert(vector: FloatArray, payload: ObjectPayload): String {
        // put() commits its own transaction (durable before returning) — comparable to Qdrant's
        // flush-per-op. box.put(entity) returns the assigned id (and back-fills entity.id).
        val e = ObjectBoxMemory().apply {
            embedding = vector; timestampMs = payload.timestampMs; payloadJson = payload.toJson()
        }
        val id = box.put(e)
        Log.d(TAG, "upsert: id=$id label=\"${payload.label}\" ts=${payload.timestampMs}")
        return id.toString()
    }

    override fun upsertBatch(items: List<Pair<FloatArray, ObjectPayload>>): List<String> {
        // All points in ONE transaction; put(Collection) back-fills each entity's id in order.
        val entities = items.map { (vector, payload) ->
            ObjectBoxMemory().apply {
                embedding = vector; timestampMs = payload.timestampMs; payloadJson = payload.toJson()
            }
        }
        box.put(entities)
        return entities.map { it.id.toString() }
    }

    override fun search(vector: FloatArray, topK: Int): List<ObjectHit> {
        val ef = maxOf(SEARCH_EF, topK)   // maxResultCount == ef; traverse ef nodes, materialize only topK
        // findWithScores(0, topK): keep ef for graph traversal quality, but deserialize only the topK
        // entities (each ~2 KB vector + JSON) instead of all ef — a plain .take(topK) would materialize
        // all ef first, inflating measured search latency.
        val hits = box.query(ObjectBoxMemory_.embedding.nearestNeighbors(vector, ef)).build()
            .use { q -> q.findWithScores(0, topK.toLong()).map { toHit(it.get(), it.score) } }
        Log.i(TAG, "search: topK=$topK ef=$ef returned=${hits.size} " +
            hits.take(3).joinToString { "%.3f \"%s\"".format(it.score, it.label.take(20)) })
        return hits
    }

    override fun searchFiltered(
        vector: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<ObjectHit> {
        if (sinceMs == null && untilMs == null) return search(vector, topK)
        val from = sinceMs ?: Long.MIN_VALUE
        val to = untilMs ?: Long.MAX_VALUE
        // Combine the range filter with the vector search in ONE query (ObjectBox applies both);
        // over-fetch candidates so enough survive the window to still fill topK.
        val candidates = maxOf(SEARCH_EF, topK * FILTER_OVERFETCH, 64)
        val hits = box.query(
            ObjectBoxMemory_.embedding.nearestNeighbors(vector, candidates)
                .and(ObjectBoxMemory_.timestampMs.between(from, to))
        ).build().use { q -> q.findWithScores(0, topK.toLong()).map { toHit(it.get(), it.score) } }
        Log.i(TAG, "searchFiltered: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        return hits
    }

    override fun all(limit: Int): List<ObjectHit> =
        box.query().order(ObjectBoxMemory_.timestampMs).build()
            .use { q -> q.find(0, limit.toLong()).map { toHit(it, distance = null) } }
            .also { Log.i(TAG, "all(): ${it.size} stored objects") }

    override fun count(): Long = box.count()

    override fun deleteAll() {
        val before = box.count()
        box.removeAll()
        Log.i(TAG, "deleteAll: dropped $before points")
    }

    override fun close() = store.close()

    /**
     * [distance] is ObjectBox's cosine DISTANCE (1 − similarity, smaller = closer) from a scored
     * query, or null for a plain scroll ([all]). We publish a higher-is-better cosine SIMILARITY so
     * [ObjectHit.score] compares like-for-like with Qdrant.
     */
    private fun toHit(e: ObjectBoxMemory, distance: Double?): ObjectHit {
        val p = ObjectPayload.fromJson(e.payloadJson)
        return ObjectHit(
            id = e.id.toString(),
            score = if (distance == null) 0f else (1.0 - distance).toFloat(),
            label = p.label,
            bbox = p.bbox,
            timestampMs = p.timestampMs,
            thumbPath = p.thumbPath,
        )
    }
}
