package tech.qdrant.glasses.storage

import android.content.Context
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
    val timestampMs: Long
)

class VisionMemoryStore(context: Context) : AutoCloseable {

    companion object {
        private const val VECTOR_DIM = 512UL
        // Empty string = the single unnamed vector field convention in Qdrant Edge
        private const val VECTOR_FIELD = ""
    }

    private val shard: EdgeShard

    init {
        val shardDir = File(context.filesDir, "qdrant_shard").also { it.mkdirs() }.absolutePath
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
    }

    fun store(imagePath: String, vector: FloatArray, timestampMs: Long): String {
        val id = UUID.randomUUID().toString()
        // Escape backslashes in path for JSON safety on Windows paths; fine on Android
        val payload = """{"image_path":"${imagePath.replace("\\", "\\\\")}","timestamp_ms":$timestampMs}"""
        shard.update(
            UpdateOperation.upsertPoints(
                listOf(
                    Point(
                        id = PointId.Uuid(id),
                        vector = Vector.Single(vector.toList()),
                        payload = payload
                    )
                )
            )
        )
        return id
    }

    fun search(queryVector: FloatArray, topK: Int = 3): List<MemoryFrame> {
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
        shard.count(CountRequest(filter = null, exact = false)).toLong()

    override fun close() = shard.close()

    private fun extractString(json: String, key: String): String =
        Regex("\"$key\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1) ?: ""

    private fun extractLong(json: String, key: String): Long =
        Regex("\"$key\"\\s*:\\s*(\\d+)").find(json)?.groupValues?.get(1)?.toLongOrNull() ?: 0L
}
