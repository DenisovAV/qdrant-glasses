package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import org.json.JSONObject
import tech.qdrant.edge.CountRequest
import tech.qdrant.edge.Distance
import tech.qdrant.edge.EdgeConfig
import tech.qdrant.edge.EdgeShard
import tech.qdrant.edge.Point
import tech.qdrant.edge.UpdateOperation
import tech.qdrant.edge.VectorDataConfig
import tech.qdrant.edge.ffi.NamedVector
import tech.qdrant.edge.ffi.PointId
import tech.qdrant.edge.ffi.Query
import tech.qdrant.edge.ffi.QueryRequest
import tech.qdrant.edge.ffi.ScoredPoint
import tech.qdrant.edge.ffi.ScoringQuery
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import java.io.File
import java.util.UUID

data class ObjectHit(
    val id: String,
    val score: Float,
    val label: String,
    val bbox: String,
    val timestampMs: Long,
    val thumbPath: String,
)

/**
 * Object memory: one Qdrant Edge collection of object crops (dense cosine vectors).
 *
 * [namespace] picks the on-disk shard directory, so different crop-encoder backends keep
 * SEPARATE collections (e.g. "mac" = SigLIP2/768-dim, "ondevice" = TinyCLIP/512-dim). This
 * lets you flip [tech.qdrant.glasses.embedding.CropEncoderFactory.backend] and rebuild without
 * clearing data — each variant indexes into its own collection, and you can switch back and
 * forth comparing search quality without re-scanning. Vectors of one dim never meet a
 * collection built for another.
 */
class ObjectStore(
    context: Context,
    private val dim: Int = OBJECT_DIM,
    namespace: String = "default",
) : AutoCloseable {

    companion object {
        private const val TAG = "ObjectStore"
        private const val FIELD = "crop"
        const val OBJECT_DIM = 768   // SigLIP2 (Mac endpoint) crop-embedding dimension
    }

    private val shard: EdgeShard

    init {
        val dir = File(context.filesDir, "objects_shard_$namespace").also { it.mkdirs() }.absolutePath
        val config = EdgeConfig(
            vectorData = mapOf(
                FIELD to VectorDataConfig(
                    size = dim.toULong(), distance = Distance.COSINE,
                    quantizationConfig = null, multivectorConfig = null, datatype = null
                )
            ),
            sparseVectorData = emptyMap()
        )
        shard = EdgeShard.load(dir, config)
        Log.i(TAG, "objects shard opened, count=${shard.count(CountRequest(filter = null, exact = false))}")
    }

    fun upsert(
        vector: FloatArray,
        label: String,
        bbox: String,
        timestampMs: Long,
        trackId: Int,
        thumbPath: String,
    ): String {
        val id = UUID.randomUUID().toString()
        // caption reserved (empty) for a later hybrid upgrade.
        val payload = JSONObject()
            .put("label", label)
            .put("bbox", bbox)
            .put("timestamp_ms", timestampMs)
            .put("track_id", trackId)
            .put("thumb_path", thumbPath)
            .put("caption", "")
            .toString()
        val named = Vector.Named(mapOf(FIELD to NamedVector.Dense(vector.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = payload)
        )))
        shard.flush()
        Log.d(TAG, "upsert: id=$id label=\"$label\" bbox=$bbox ts=$timestampMs")
        return id
    }

    fun search(vector: FloatArray, topK: Int = 5): List<ObjectHit> {
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
        return hits
    }

    private fun toHit(p: ScoredPoint): ObjectHit {
        // A single malformed payload must not crash the whole result list.
        val o = try { JSONObject(p.payload ?: "{}") } catch (_: Throwable) { JSONObject() }
        return ObjectHit(
            id = (p.id as? PointId.Uuid)?.value ?: "",
            score = p.score,
            label = o.optString("label"),
            bbox = o.optString("bbox"),
            timestampMs = o.optLong("timestamp_ms"),
            thumbPath = o.optString("thumb_path"),
        )
    }

    fun count(): Long = shard.count(CountRequest(filter = null, exact = true)).toLong()

    override fun close() {
        // Always release the native shard even if the diagnostic count throws.
        runCatching { Log.i(TAG, "close: total objects=${count()}") }
        shard.close()
    }
}
