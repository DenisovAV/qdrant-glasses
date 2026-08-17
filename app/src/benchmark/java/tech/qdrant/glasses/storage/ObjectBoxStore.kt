package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.kotlin.query

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
    }

    init {
        require(dim == DIM) {
            "ObjectBox entity dim is compile-time-fixed at $DIM (got $dim) — regenerate the entity for another dim"
        }
    }

    private val store: BoxStore = MyObjectBox.builder()
        .androidContext(context.applicationContext)
        .name("objectbox_$namespace")   // own directory per namespace — never collides with Qdrant's shard
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
        val hits = box.query(ObjectBoxMemory_.embedding.nearestNeighbors(vector, topK)).build()
            .use { q -> q.findWithScores().map { toHit(it.get(), it.score) } }
        Log.i(TAG, "search: topK=$topK returned=${hits.size} " +
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
        val candidates = maxOf(topK * FILTER_OVERFETCH, 64)
        val hits = box.query(
            ObjectBoxMemory_.embedding.nearestNeighbors(vector, candidates)
                .and(ObjectBoxMemory_.timestampMs.between(from, to))
        ).build().use { q -> q.findWithScores().take(topK).map { toHit(it.get(), it.score) } }
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
