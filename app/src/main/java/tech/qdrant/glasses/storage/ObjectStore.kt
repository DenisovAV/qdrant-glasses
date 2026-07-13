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
import tech.qdrant.edge.ffi.ScrollRequest
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import tech.qdrant.edge.ffi.WithVector
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
 * Typed mirror of the JSON payload stored alongside each vector in the object shard.
 *
 * Keys and defaults are byte-identical to the inline `JSONObject` this replaced, so existing
 * on-disk records (written before this type existed) read back unchanged: [fromJson] mirrors the
 * `optString`/`optLong` defaults (empty string / 0) for any missing key.
 *
 * `internal` (not `private`) only so [ObjectPayloadTest] can exercise it directly.
 */
internal data class ObjectPayload(
    val label: String,
    val bbox: String,
    val timestampMs: Long,
    val trackId: Int,
    val thumbPath: String,
    val caption: String,
) {
    fun toJson(): String = JSONObject()
        .put("label", label)
        .put("bbox", bbox)
        .put("timestamp_ms", timestampMs)
        .put("track_id", trackId)
        .put("thumb_path", thumbPath)
        .put("caption", caption)
        .toString()

    companion object {
        fun fromJson(s: String): ObjectPayload {
            val o = try { JSONObject(s) } catch (_: Throwable) { JSONObject() }
            return ObjectPayload(
                label = o.optString("label"),
                bbox = o.optString("bbox"),
                timestampMs = o.optLong("timestamp_ms"),
                trackId = o.optInt("track_id"),
                thumbPath = o.optString("thumb_path"),
                caption = o.optString("caption"),
            )
        }
    }
}

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
    // The native EdgeShard's thread-safety is unverified and this store is touched concurrently:
    // dedup search + upsert on cropLane, the voice-query search on inferLane, and all()/rail
    // snapshots from the MjpegServer HTTP threads. Serialize every native call through one monitor
    // (reentrant, so close() may call count()).
    private val lock = Any()

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
    ): String = synchronized(lock) {
        val id = UUID.randomUUID().toString()
        // caption reserved (empty) for a later hybrid upgrade.
        val payload = ObjectPayload(
            label = label,
            bbox = bbox,
            timestampMs = timestampMs,
            trackId = trackId,
            thumbPath = thumbPath,
            caption = "",
        ).toJson()
        val named = Vector.Named(mapOf(FIELD to NamedVector.Dense(vector.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = payload)
        )))
        shard.flush()
        Log.d(TAG, "upsert: id=$id label=\"$label\" bbox=$bbox ts=$timestampMs")
        id
    }

    fun search(vector: FloatArray, topK: Int = 5): List<ObjectHit> = synchronized(lock) {
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
    fun all(limit: Int = 500): List<ObjectHit> = synchronized(lock) {
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

    fun count(): Long = synchronized(lock) { shard.count(CountRequest(filter = null, exact = true)).toLong() }

    override fun close() = synchronized(lock) {
        // Always release the native shard even if the diagnostic count throws.
        runCatching { Log.i(TAG, "close: total objects=${count()}") }
        shard.close()
    }
}
