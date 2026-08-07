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
import tech.qdrant.edge.ffi.NamedVector
import tech.qdrant.edge.ffi.PointId
import tech.qdrant.edge.ffi.Query
import tech.qdrant.edge.ffi.ScoredPoint
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import tech.qdrant.edge.ffi.QueryRequest
import tech.qdrant.edge.ffi.ScoringQuery
import tech.qdrant.edge.ffi.Filter
import tech.qdrant.edge.ffi.Condition
import tech.qdrant.edge.ffi.FieldCondition
import tech.qdrant.edge.ffi.Match
import tech.qdrant.edge.ffi.ValueVariants
import tech.qdrant.edge.ffi.ScrollRequest
import tech.qdrant.edge.ffi.WithVector
import java.io.File
import java.util.UUID

data class MemoryFrame(
    val id: String,
    val score: Float,
    val imagePath: String,
    val timestampMs: Long,
    // End of the moment's time span. For a transcript point this is t_end_ms (the
    // utterance end); for an image point it equals timestampMs (a frame is an instant).
    // Lets moment-merge align a frame to an utterance by INTERVAL overlap rather than by
    // comparing two point timestamps that intentionally differ (frame=capture time,
    // transcript=utterance start), which otherwise splits one moment into two cards.
    val tEndMs: Long,
    val type: String = "image",
    val transcript: String? = null,
    // Transcripts spoken near this frame (filled at result time for the shown hit) —
    // lets an image hit display "what was said here", not just the picture.
    val nearbyTranscripts: List<String> = emptyList()
)

class VisionMemoryStore(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "VisionMemory"
        private const val VECTOR_DIM = 512UL
        private const val VECTOR_FIELD = ""
        private const val TEXT_FIELD = "text"
        private const val TEXT_DIM = 384UL
    }

    private val shard: EdgeShard

    init {
        val shardDir = File(context.filesDir, "qdrant_shard").also { it.mkdirs() }.absolutePath
        Log.i(TAG, "init: opening shard at $shardDir")
        val config = EdgeConfig(
            vectorData = mapOf(
                VECTOR_FIELD to VectorDataConfig(
                    size = VECTOR_DIM, distance = Distance.COSINE,
                    quantizationConfig = null, multivectorConfig = null, datatype = null, hnswConfig = null
                ),
                TEXT_FIELD to VectorDataConfig(
                    size = TEXT_DIM, distance = Distance.COSINE,
                    quantizationConfig = null, multivectorConfig = null, datatype = null, hnswConfig = null
                )
            ),
            sparseVectorData = emptyMap()
        )
        shard = EdgeShard.load(shardDir, config)
        Log.i(TAG, "init: shard opened, existing points=${shard.count(CountRequest(filter = null, exact = false))}")
    }

    fun storeImage(imagePath: String, vector: FloatArray, timestampMs: Long): String {
        val id = UUID.randomUUID().toString()
        val payload = """{"type":"image","image_path":"${imagePath.replace("\\", "\\\\")}","timestamp_ms":$timestampMs}"""
        upsert(id, mapOf(VECTOR_FIELD to vector), payload)
        Log.d(TAG, "storeImage: id=$id path=${imagePath.substringAfterLast('/')}")
        return id
    }

    fun storeTranscript(
        text: String, clipVector: FloatArray, bgeVector: FloatArray,
        tStartMs: Long, tEndMs: Long, nearestImagePath: String
    ): String {
        val id = UUID.randomUUID().toString()
        val safeText = text.replace("\\", "\\\\").replace("\"", "\\\"")
        val safePath = nearestImagePath.replace("\\", "\\\\")
        val payload = """{"type":"text","transcript":"$safeText","t_start_ms":$tStartMs,"t_end_ms":$tEndMs,"image_path":"$safePath","timestamp_ms":$tStartMs}"""
        upsert(id, mapOf(VECTOR_FIELD to clipVector, TEXT_FIELD to bgeVector), payload)
        Log.d(TAG, "storeTranscript: id=$id text=\"${text.take(40)}\"")
        return id
    }

    private fun upsert(id: String, vectors: Map<String, FloatArray>, payload: String) {
        val named = Vector.Named(vectors.mapValues { NamedVector.Dense(it.value.toList()) })
        shard.update(
            UpdateOperation.upsertPoints(
                listOf(Point(id = PointId.Uuid(id), vector = named, payload = payload))
            )
        )
        shard.flush()
    }

    data class Hit(val frame: MemoryFrame, val score: Float)

    /** "Saw" channel: CLIP query vector over image points (default vector field). */
    fun searchVision(clipVec: FloatArray, topK: Int = 5): List<Hit> =
        channelQuery(clipVec.toList(), using = null, typeValue = "image", topK = topK)

    /** "Heard" channel: bge query vector over transcript points (field "text"). */
    fun searchHeard(bgeVec: FloatArray, topK: Int = 5): List<Hit> =
        channelQuery(bgeVec.toList(), using = TEXT_FIELD, typeValue = "text", topK = topK)

    private fun channelQuery(q: List<Float>, using: String?, typeValue: String, topK: Int): List<Hit> {
        val results = shard.query(
            QueryRequest(
                limit = topK.toULong(), offset = null,
                query = ScoringQuery.Vector(Query.Nearest(vector = q, using = using)),
                prefetches = emptyList(),
                withVector = null, withPayload = WithPayload.Bool(true),
                filter = Filter(
                    must = listOf(Condition.Field(FieldCondition(
                        key = "type", match = Match.Value(ValueVariants.String(typeValue)),
                        range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null
                    ))),
                    should = null, mustNot = null
                ),
                scoreThreshold = null, params = null
            )
        )
        val hits = results.map { p -> Hit(toFrame(p.payload ?: "{}", p), p.score) }
        Log.i(TAG, "channel[$typeValue]: " + hits.take(3).joinToString {
            "%.3f \"%s\"".format(it.score, (it.frame.transcript ?: it.frame.imagePath.substringAfterLast('/')).take(25))
        })
        return hits
    }

    /** Keyword channel: transcripts containing the query words (binary lexical signal). */
    fun keywordHits(query: String, limit: Int = 5): List<MemoryFrame> {
        val resp = shard.scroll(
            ScrollRequest(
                offset = null, limit = limit.toULong(),
                filter = Filter(
                    must = listOf(
                        Condition.Field(FieldCondition(
                            key = "type", match = Match.Value(ValueVariants.String("text")),
                            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null
                        )),
                        Condition.Field(FieldCondition(
                            key = "transcript", match = Match.Text(query),
                            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null
                        ))
                    ),
                    should = null, mustNot = null
                ),
                withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
                orderBy = null
            )
        )
        return resp.records.map { rec -> toFrame(rec.payload ?: "{}", rec.id, 0f) }
            .also { Log.i(TAG, "keyword(\"$query\"): ${it.size} hits") }
    }

    private fun toFrame(payload: String, scored: ScoredPoint?): MemoryFrame =
        toFrame(payload, scored?.id, scored?.score ?: 0f)

    private fun toFrame(payload: String, pointId: tech.qdrant.edge.ffi.PointId?, score: Float): MemoryFrame {
        val ts = extractLong(payload, "timestamp_ms")
        // t_end_ms exists only on transcript points; image points are instants (tEnd=ts).
        val tEnd = extractLong(payload, "t_end_ms").let { if (it > 0) it else ts }
        return MemoryFrame(
            id = (pointId as? PointId.Uuid)?.value ?: "",
            score = score,
            imagePath = extractString(payload, "image_path"),
            timestampMs = ts,
            tEndMs = tEnd,
            type = extractString(payload, "type").ifEmpty { "image" },
            transcript = extractString(payload, "transcript").ifEmpty { null }
        )
    }

    /**
     * Speech that OVERLAPS a frame in time — the reverse of the transcript→nearest-frame
     * link, done by time RANGE rather than a single nearest point. A long utterance spans
     * several frames; each of those frames should surface it ("overlap" chunking, the RAG
     * best practice), not just the one nearest its midpoint. We scroll all type:"text"
     * points (there are few — one per spoken utterance) and keep those whose
     * [t_start_ms, t_end_ms] window covers the frame's timestamp (± a small slack).
     */
    fun transcriptsOverlappingFrame(frameTimestampMs: Long, slackMs: Long = 1500): List<String> {
        val typeText = Condition.Field(FieldCondition(
            key = "type", match = Match.Value(ValueVariants.String("text")),
            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null
        ))
        val resp = shard.scroll(
            ScrollRequest(
                offset = null, limit = 1000UL,
                filter = Filter(must = listOf(typeText), should = null, mustNot = null),
                withPayload = WithPayload.Bool(true),
                withVector = WithVector.Bool(false),
                orderBy = null
            )
        )
        val hits = resp.records.mapNotNull { rec ->
            val p = rec.payload ?: return@mapNotNull null
            val tStart = extractLong(p, "t_start_ms")
            val tEnd = extractLong(p, "t_end_ms")
            val text = extractString(p, "transcript").ifEmpty { return@mapNotNull null }
            // frame falls within the utterance window (with slack on both sides)
            if (frameTimestampMs in (tStart - slackMs)..(tEnd + slackMs)) text else null
        }
        Log.d(TAG, "transcriptsOverlappingFrame(ts=$frameTimestampMs): ${hits.size} of ${resp.records.size} text points")
        return hits
    }

    /** Diagnostic: scroll the entire base and log every point (type, frame, transcript). */
    fun dumpAll() {
        val resp = shard.scroll(
            ScrollRequest(
                offset = null, limit = 1000UL, filter = null,
                withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
                orderBy = null
            )
        )
        Log.i(TAG, "DUMP: ${resp.records.size} points total")
        resp.records.forEach { rec ->
            val p = rec.payload ?: "{}"
            val type = extractString(p, "type").ifEmpty { "?" }
            val frame = extractString(p, "image_path").substringAfterLast('/')
            val tr = extractString(p, "transcript")
            val ts = extractLong(p, "timestamp_ms")
            Log.i(TAG, "DUMP type=$type frame=$frame ts=$ts transcript=\"$tr\"")
        }
    }

    fun count(): Long =
        shard.count(CountRequest(filter = null, exact = true)).toLong()

    override fun close() {
        Log.i(TAG, "close: total points=${count()}")
        shard.close()
    }

    private fun extractString(json: String, key: String): String =
        Regex("\"$key\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1) ?: ""

    private fun extractLong(json: String, key: String): Long =
        Regex("\"$key\"\\s*:\\s*(\\d+)").find(json)?.groupValues?.get(1)?.toLongOrNull() ?: 0L
}
