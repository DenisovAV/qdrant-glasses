package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import tech.qdrant.edge.CountRequest
import tech.qdrant.edge.Distance
import tech.qdrant.edge.EdgeConfig
import tech.qdrant.edge.EdgeShard
import tech.qdrant.edge.Point
import tech.qdrant.edge.SearchRequest
import tech.qdrant.edge.UpdateOperation
import tech.qdrant.edge.VectorDataConfig
import tech.qdrant.edge.ffi.PointId
import tech.qdrant.edge.ffi.Query
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import java.io.File
import java.util.UUID

data class MemoryFrame(
    val id: String,
    val score: Float,
    val imagePath: String,
    val timestampMs: Long,
    val type: String = "image",
    val transcript: String? = null
)

class VisionMemoryStore(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "VisionMemory"
        private const val VECTOR_DIM = 512UL
        private const val VECTOR_FIELD = ""
    }

    private val shard: EdgeShard

    init {
        val shardDir = File(context.filesDir, "qdrant_shard").also { it.mkdirs() }.absolutePath
        Log.i(TAG, "init: opening shard at $shardDir")
        val config = EdgeConfig(
            vectorData = mapOf(
                VECTOR_FIELD to VectorDataConfig(
                    size = VECTOR_DIM,
                    distance = Distance.COSINE,
                    quantizationConfig = null,
                    multivectorConfig = null,
                    datatype = null
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
        upsert(id, vector, payload)
        Log.d(TAG, "storeImage: id=$id path=${imagePath.substringAfterLast('/')}")
        return id
    }

    fun storeTranscript(
        text: String, vector: FloatArray,
        tStartMs: Long, tEndMs: Long, nearestImagePath: String
    ): String {
        val id = UUID.randomUUID().toString()
        val safeText = text.replace("\\", "\\\\").replace("\"", "\\\"")
        val safePath = nearestImagePath.replace("\\", "\\\\")
        val payload = """{"type":"text","transcript":"$safeText","t_start_ms":$tStartMs,"t_end_ms":$tEndMs,"image_path":"$safePath","timestamp_ms":$tStartMs}"""
        upsert(id, vector, payload)
        Log.d(TAG, "storeTranscript: id=$id text=\"${text.take(40)}\"")
        return id
    }

    private fun upsert(id: String, vector: FloatArray, payload: String) {
        shard.update(
            UpdateOperation.upsertPoints(
                listOf(Point(id = PointId.Uuid(id), vector = Vector.Single(vector.toList()), payload = payload))
            )
        )
        shard.flush()
    }

    fun search(queryVector: FloatArray, topK: Int = 3): List<MemoryFrame> {
        Log.d(TAG, "search: topK=$topK")
        val results = shard.search(
            SearchRequest(
                query = Query.Nearest(vector = queryVector.toList(), using = null),
                limit = topK.toULong(),
                offset = null,
                filter = null,
                params = null,
                withVector = null,
                withPayload = WithPayload.Bool(true),
                scoreThreshold = null
            )
        )
        Log.i(TAG, "search: found ${results.size} results")
        return results.map { point ->
            val payload = point.payload ?: "{}"
            MemoryFrame(
                id = (point.id as? PointId.Uuid)?.value ?: point.id.toString(),
                score = point.score,
                imagePath = extractString(payload, "image_path"),
                timestampMs = extractLong(payload, "timestamp_ms")
            )
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
