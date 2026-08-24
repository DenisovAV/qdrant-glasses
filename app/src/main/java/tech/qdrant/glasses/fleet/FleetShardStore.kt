package tech.qdrant.glasses.fleet

import android.util.Log
import io.qdrant.edge.Condition
import io.qdrant.edge.Distance
import io.qdrant.edge.EdgeConfig
import io.qdrant.edge.EdgeShard
import io.qdrant.edge.FieldCondition
import io.qdrant.edge.Filter
import io.qdrant.edge.NamedVector
import io.qdrant.edge.Point
import io.qdrant.edge.PointId
import io.qdrant.edge.Query
import io.qdrant.edge.QueryRequest
import io.qdrant.edge.RangeFloat
import io.qdrant.edge.ScoredPoint
import io.qdrant.edge.ScoringQuery
import io.qdrant.edge.UpdateOperation
import io.qdrant.edge.Vector
import io.qdrant.edge.VectorDataConfig
import io.qdrant.edge.WithPayload
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentType
import java.util.UUID

/**
 * Read-only view over a pulled fleet corpus (plan Task 4, Spec §3/§6): wraps a second `EdgeShard`
 * loaded from a snapshot dir that [io.qdrant.edge.unpackSnapshotAsync] produced, provisioned with the
 * SAME schema [tech.qdrant.glasses.storage.QdrantEdgeMomentStore] uses (named vectors `"clip"`
 * [clipDim]-dim + `"text"` 384-dim, both COSINE) so one on-device query vector searches both shards.
 * One clear job: query the fleet corpus and tag every hit `source="fleet"` (Spec §3) — writes to a
 * pulled shard are never expected from this class, only [seedForTest] (below) creates one, and only
 * for tests.
 *
 * Deliberately narrower than [tech.qdrant.glasses.storage.MomentStore]: no channel (`type`) filter —
 * the fleet corpus is the curated set as a whole, unlike local memory's frame/region/ocr split — only
 * the optional `timestamp_ms` range the caller asks for (mirrors
 * [tech.qdrant.glasses.storage.QdrantEdgeMomentStore.channelSearch]'s query shape minus the type
 * condition). `Task 5` wires this behind a `FleetSource` seam so [MomentSearcher] can fake it in a JVM
 * test without the native `.so`.
 */
class FleetShardStore(private val shard: EdgeShard, private val clipDim: Int) : FleetSource {

    companion object {
        private const val TAG = "FleetShardStore"
        private const val CLIP_FIELD = "clip"
        private const val TEXT_FIELD = "text"
        private const val TEXT_DIM = 384

        // Identical shape to QdrantEdgeMomentStore's `config` — MUST match so a snapshot pulled from
        // the fleet server (provisioned with that same schema, Spec §6) loads cleanly here.
        private fun config(clipDim: Int): EdgeConfig = EdgeConfig(
            vectorData = mapOf(
                CLIP_FIELD to VectorDataConfig(
                    size = clipDim.toULong(), distance = Distance.COSINE,
                    quantizationConfig = null, multivectorConfig = null, datatype = null, hnswConfig = null,
                ),
                TEXT_FIELD to VectorDataConfig(
                    size = TEXT_DIM.toULong(), distance = Distance.COSINE,
                    quantizationConfig = null, multivectorConfig = null, datatype = null, hnswConfig = null,
                ),
            ),
            sparseVectorData = emptyMap(),
        )

        /** Opens an already-unpacked snapshot dir (see [io.qdrant.edge.unpackSnapshotAsync]) read-only. */
        fun load(dir: String, clipDim: Int): FleetShardStore =
            FleetShardStore(EdgeShard.load(dir, config(clipDim)), clipDim)

        /**
         * TEST ONLY (androidTest, [FleetShardStoreTest]): builds a brand-new tiny shard at [dir] and
         * upserts one `type=frame` point, so [load] on the same [dir] has something to find — mirrors
         * [tech.qdrant.glasses.storage.QdrantEdgeMomentStore.storeMoment]'s write shape (same named
         * vector, same payload JSON). [id] is a caller-facing label (stamped into the payload's
         * `moment_id`, same convention `storeMoment` uses); the actual `PointId` must be a valid RFC
         * 4122 UUID, so it's derived deterministically from [id] via [UUID.nameUUIDFromBytes] rather
         * than passed straight through.
         */
        fun seedForTest(dir: String, clipDim: Int, id: String, vec: FloatArray, label: String, ts: Long) {
            val seedShard = EdgeShard.create(dir, config(clipDim))
            val payload = MomentPayload(
                type = MomentType.FRAME, momentId = id, episodeId = 0L, timestampMs = ts, tEndMs = ts,
                thumbPath = "", bbox = "", label = label, yoloConf = 0f, verifyCos = 0f, text = "",
            )
            val pointId = PointId.Uuid(UUID.nameUUIDFromBytes(id.toByteArray(Charsets.UTF_8)).toString())
            seedShard.update(UpdateOperation.upsertPoints(listOf(
                Point(id = pointId, vector = Vector.Named(mapOf(CLIP_FIELD to NamedVector.Dense(vec.toList()))), payload = payload.toJson())
            )))
            seedShard.flush()
            seedShard.close()
        }
    }

    /** Nearest-neighbor search against the fleet corpus's `"clip"` vector; every hit tagged `source="fleet"`. */
    override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> {
        // Mirrors QdrantEdgeMomentStore.channelSearch's guard: reject a malformed query vector before
        // it ever reaches NamedVector.Dense / the native API.
        require(qvec.size == clipDim) { "dim ${qvec.size} != $clipDim" }
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = NamedVector.Dense(qvec.toList()), using = CLIP_FIELD)),
            prefetches = emptyList(),
            withVector = null, withPayload = WithPayload.Bool(true),
            filter = timeFilter(sinceMs, untilMs),
            scoreThreshold = null, params = null,
        ))
        val hits = results.map { toHit(it) }
        Log.i(TAG, "fleet search: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        return hits
    }

    fun close() = shard.close()

    // No bound at all -> no filter (the fleet corpus is small; an unbounded query is the common case).
    private fun timeFilter(sinceMs: Long?, untilMs: Long?): Filter? {
        if (sinceMs == null && untilMs == null) return null
        return Filter(
            must = listOf(Condition.Field(FieldCondition(
                key = "timestamp_ms", match = null,
                range = RangeFloat(gte = sinceMs?.toDouble(), gt = null, lte = untilMs?.toDouble(), lt = null),
                geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
            ))),
            should = null, mustNot = null,
        )
    }

    // Payload-field extraction copied verbatim from QdrantEdgeMomentStore's hit mapping (Spec §6: same
    // payload shape on both sides of the sync) — only `source` differs, always "fleet" here.
    private fun toHit(p: ScoredPoint): MomentHit {
        val payload = MomentPayload.fromJson(p.payload ?: "{}")
        return MomentHit(
            id = (p.id as? PointId.Uuid)?.value ?: "",
            score = p.score,
            type = payload.type,
            momentId = payload.momentId,
            timestampMs = payload.timestampMs,
            thumbPath = payload.thumbPath,
            label = payload.label,
            bbox = payload.bbox,
            yoloConf = payload.yoloConf,
            verifyCos = payload.verifyCos,
            text = payload.text,
            source = "fleet",
        )
    }
}
