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
import tech.qdrant.edge.ffi.Match
import tech.qdrant.edge.ffi.NamedVector
import tech.qdrant.edge.ffi.PointId
import tech.qdrant.edge.ffi.Query
import tech.qdrant.edge.ffi.QueryRequest
import tech.qdrant.edge.ffi.RangeFloat
import tech.qdrant.edge.ffi.ScoredPoint
import tech.qdrant.edge.ffi.ScoringQuery
import tech.qdrant.edge.ffi.ScrollRequest
import tech.qdrant.edge.ffi.ValueVariants
import tech.qdrant.edge.ffi.Vector
import tech.qdrant.edge.ffi.WithPayload
import tech.qdrant.edge.ffi.WithVector
import java.io.File
import java.util.UUID

/**
 * Moment memory: one Qdrant Edge collection holding two point channels distinguished by
 * [MomentPayload.type] — whole-frame keyframes ("frame") and the CLIP-verified YOLO regions within
 * them ("region") (plan Task 1.3, Spec §6). First (and so far only) [MomentStore] implementation,
 * cloned from [QdrantEdgeStore]'s proven mechanics VERBATIM: single monitor [lock] serializing every
 * native call, flush-per-upsert on the live path, idempotent [close], drop+recreate [deleteAll].
 * Type filtering (`Match` on `type`) mirrors [VisionMemoryStore.channelQuery]; time filtering
 * (`RangeFloat` on `timestamp_ms`) mirrors [QdrantEdgeStore.searchFiltered]. Both are combined in one
 * `Filter.must` list (standard Qdrant semantics: `must` = AND) for the filtered searches.
 *
 * **Named vectors (Spec §6/§8.4 unknown, RESOLVED):** the collection provisions BOTH `"clip"`
 * ([dim]-dim cosine, 512 by default — see [dim]'s own doc) and `"text"` (384-dim cosine) at
 * creation. [VisionMemoryStore] already runs a
 * 2-named-vector Edge collection in production on this exact AAR (`""`+`"text"`), and `EdgeConfig`
 * accepted the same `vectorData` map shape here on-device (see `QdrantEdgeMomentStoreTest`) — so the
 * assumed-unsupported fallback (`clip`-only) was NOT needed this stage. Frame/region points write
 * only the `clip` field; `text` sits empty until the speech/OCR channels (Stage 3/4) start using it.
 *
 * [namespace] picks the on-disk shard directory (`moments_shard_<namespace>`), same convention as
 * [QdrantEdgeStore] — different crop-encoder backends get separate collections.
 *
 * [dim] is the `clip` named-vector size (`text` stays fixed at [TEXT_DIM] — Stage 3/4's own
 * encoder, not the crop encoder). Defaults to 512 (TinyCLIP/QNN_B32's space) so the existing
 * instrumented test (which builds 512-dim unit vectors) is unaffected; `GlassesComponents.load()`
 * passes `cropEncoder.dim` explicitly so a MAC_ENDPOINT build (768-dim SigLIP2) doesn't fail
 * [storeMoment]'s dim `require` against a collection provisioned for the wrong backend.
 */
class QdrantEdgeMomentStore(
    context: Context,
    namespace: String = "default",
    dim: Int = 512,
) : MomentStore {

    companion object {
        private const val TAG = "QdrantEdgeMomentStore"
        private const val CLIP_FIELD = "clip"
        private const val TEXT_FIELD = "text"
        private const val TEXT_DIM = 384
        // timeline() pagination page size — see its KDoc for why this scans the whole frame
        // channel instead of asking the shard to order/limit server-side. 256 keeps the number of
        // native scroll() round trips low at demo scale (low hundreds of frames) while capping how
        // much payload JSON is materialized per call.
        private const val SCROLL_PAGE_SIZE = 256UL
    }

    // The `clip` named-vector's dim, promoted from the constructor param to a property so every
    // method below (storeMoment/storeRegion/channelSearch's `require`, plus `config` itself) can
    // see it — a plain constructor param is only visible in property initializers.
    private val clipDim: Int = dim

    // Kept as a field so deleteAll() can drop + recreate the shard on the same directory in-process,
    // exactly as QdrantEdgeStore does (no app relaunch needed for the demo wipe gesture).
    private val dir: String = File(context.filesDir, "moments_shard_$namespace")
        .also { it.mkdirs() }.absolutePath
    private val config = EdgeConfig(
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

    // Same concurrency story as QdrantEdgeStore: the native EdgeShard's thread-safety is unverified
    // and this store will be touched from more than one lane (moment capture writes, HUD timeline
    // reads, voice-query searches) — serialize every native call through one reentrant monitor.
    private val lock = Any()
    private var shard: EdgeShard
    // Guards close()/deleteAll() against a use-after-free on the native handle — see QdrantEdgeStore
    // for the exact reasoning; the discipline is copied verbatim.
    private var closed = false

    init {
        shard = EdgeShard.load(dir, config)
        Log.i(TAG, "moments shard opened, count=${shard.count(CountRequest(filter = null, exact = false))}")
    }

    override fun storeMoment(clipVec: FloatArray, payload: MomentPayload): String = synchronized(lock) {
        require(clipVec.size == clipDim) { "dim ${clipVec.size} != $clipDim" }
        val id = UUID.randomUUID().toString()
        // Spec §6 / MomentPayload KDoc invariant: a frame's moment_id == its OWN id. The caller can't
        // know that id before this call generates it, so storeMoment stamps type+momentId itself
        // rather than asking every call site to pre-generate a UUID and pass it in twice.
        val stamped = payload.copy(type = MomentType.FRAME, momentId = id)
        val named = Vector.Named(mapOf(CLIP_FIELD to NamedVector.Dense(clipVec.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = stamped.toJson())
        )))
        shard.flush()
        Log.d(TAG, "storeMoment: id=$id ts=${payload.timestampMs}")
        id
    }

    override fun storeRegion(clipVec: FloatArray, payload: MomentPayload): String = synchronized(lock) {
        require(clipVec.size == clipDim) { "dim ${clipVec.size} != $clipDim" }
        require(payload.momentId.isNotBlank()) {
            "storeRegion requires payload.momentId = the parent frame's id (Spec §6: region.moment_id = parent's)"
        }
        val id = UUID.randomUUID().toString()
        val stamped = payload.copy(type = MomentType.REGION)
        val named = Vector.Named(mapOf(CLIP_FIELD to NamedVector.Dense(clipVec.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = stamped.toJson())
        )))
        shard.flush()
        Log.d(TAG, "storeRegion: id=$id momentId=${payload.momentId} label=\"${payload.label}\"")
        id
    }

    override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> =
        channelSearch(MomentType.FRAME, qvec, topK, sinceMs, untilMs)

    override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> =
        channelSearch(MomentType.REGION, qvec, topK, sinceMs, untilMs)

    private fun channelSearch(
        typeValue: String,
        qvec: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<MomentHit> = synchronized(lock) {
        require(qvec.size == clipDim) { "dim ${qvec.size} != $clipDim" }
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = qvec.toList(), using = CLIP_FIELD)),
            prefetches = emptyList(),
            withVector = null, withPayload = WithPayload.Bool(true),
            filter = typeAndTimeFilter(typeValue, sinceMs, untilMs),
            scoreThreshold = null, params = null,
        ))
        val hits = results.map { toHit(it) }
        Log.i(TAG, "search[$typeValue]: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        hits
    }

    // AND's a `type` equality condition with an optional `timestamp_ms` range condition in one
    // `Filter.must` list (must = AND, standard Qdrant semantics). No time bound at all → type-only.
    private fun typeAndTimeFilter(typeValue: String, sinceMs: Long?, untilMs: Long?): Filter {
        val conditions = mutableListOf<Condition>(
            Condition.Field(FieldCondition(
                key = "type", match = Match.Value(ValueVariants.String(typeValue)),
                range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
            ))
        )
        if (sinceMs != null || untilMs != null) {
            conditions.add(Condition.Field(FieldCondition(
                key = "timestamp_ms", match = null,
                range = RangeFloat(gte = sinceMs?.toDouble(), gt = null, lte = untilMs?.toDouble(), lt = null),
                geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
            )))
        }
        return Filter(must = conditions, should = null, mustNot = null)
    }

    /**
     * The most-recent [limit] stored `type=frame` moments (payload only, no vectors), oldest-first
     * — rebuilds the HUD timeline rail on connect as a chronological strip ending at "now".
     *
     * **Server-side `orderBy` was tried and rejected on-device**: `ScrollRequest.orderBy` exists in
     * this AAR's FFI, but a live scroll with `OrderBy("timestamp_ms", Direction.DESC)` throws
     * `EdgeException.OperationException("No range index for \`order_by\` key: \`timestamp_ms\`...")`
     * — and `EdgeConfig` (see its fields) exposes no API to create one. So there is no way to get
     * the shard to order by timestamp itself; this pages the ENTIRE `type=frame` channel via
     * [ScrollRequest.offset]/[tech.qdrant.edge.ffi.ScrollResponse.nextOffset] (Qdrant's default
     * point-ID order says nothing about recency, so a single `limit`-sized page could silently
     * drop the newest frames once the channel exceeds `limit`), sorts every point by
     * [MomentHit.timestampMs] client-side, and keeps only the most recent [limit]. Fine at demo
     * scale (low hundreds of frames); a payload index on `timestamp_ms` — if this Edge FFI ever
     * grows one — would let `orderBy` do this server-side instead and drop the full scan.
     */
    override fun timeline(limit: Int): List<MomentHit> = synchronized(lock) {
        val frameFilter = Filter(must = listOf(Condition.Field(FieldCondition(
            key = "type", match = Match.Value(ValueVariants.String(MomentType.FRAME)),
            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
        ))), should = null, mustNot = null)
        val all = mutableListOf<MomentHit>()
        var offset: PointId? = null
        do {
            val resp = shard.scroll(ScrollRequest(
                offset = offset, limit = SCROLL_PAGE_SIZE,
                filter = frameFilter,
                withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
                orderBy = null,
            ))
            resp.records.mapTo(all) { rec -> toHit(rec.id, rec.payload ?: "{}") }
            offset = resp.nextOffset
        } while (offset != null)
        all.sortedBy { it.timestampMs }
            .takeLast(limit)
            .also { Log.i(TAG, "timeline(): scanned ${all.size} stored frames, returning ${it.size} most recent") }
    }

    override fun count(): Long = synchronized(lock) { shard.count(CountRequest(filter = null, exact = true)).toLong() }

    // Whole-branch review fix: `count()` was being reported to callers as "moments stored", but it
    // counts every point across BOTH channels — a keyframe plus its verified regions reads as N+1,
    // not 1. Reuses [typeAndTimeFilter]'s type-only shape (no time bound) rather than hand-rolling a
    // second `Filter.must` list, same as [searchFrames]/[searchRegions] already do via [channelSearch].
    override fun frameCount(): Long = synchronized(lock) {
        shard.count(CountRequest(filter = typeAndTimeFilter(MomentType.FRAME, null, null), exact = true)).toLong()
    }

    override fun deleteAll(): Unit = synchronized(lock) {
        // Drop + recreate in-process, identical discipline to QdrantEdgeStore.deleteAll(): close the
        // native handle, wipe the shard directory on disk, reload an empty shard from the same
        // config. closed=true is set right after shard.close() so a later close()/deleteAll() sees
        // the guard even if the wipe/reload below throws (the handle is dangling from that point on).
        check(!closed) { "deleteAll() called on a closed QdrantEdgeMomentStore" }
        val before = runCatching { shard.count(CountRequest(filter = null, exact = false)).toLong() }.getOrDefault(-1L)
        shard.close()
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

    override fun close() = synchronized(lock) {
        // Idempotent: a second close() must NOT touch the already-freed native shard.
        if (closed) return@synchronized
        closed = true
        runCatching { Log.i(TAG, "close: total points=${count()}") }
        shard.close()
    }

    private fun toHit(p: ScoredPoint): MomentHit = toHit(p.id, p.payload ?: "{}", p.score)

    private fun toHit(id: PointId?, payload: String, score: Float = 0f): MomentHit {
        // A single malformed payload must not crash the whole result list — MomentPayload.fromJson
        // already falls back to an empty JSONObject internally, same discipline as QdrantEdgeStore.
        val p = MomentPayload.fromJson(payload)
        return MomentHit(
            id = (id as? PointId.Uuid)?.value ?: "",
            score = score,
            type = p.type,
            momentId = p.momentId,
            timestampMs = p.timestampMs,
            thumbPath = p.thumbPath,
            label = p.label,
            bbox = p.bbox,
            // Task 2.3 (Spec §3): region-only fields, needed for the soft tag boost. A frame point's
            // payload already stamps both 0f at capture time (MomentCapture.confirmAndStore), so this
            // mapping is unconditional — no `if (p.type == MomentType.REGION)` needed to get "frame hits get
            // 0f", same as [label]/[bbox] above are already mapped straight through regardless of type.
            yoloConf = p.yoloConf,
            verifyCos = p.verifyCos,
        )
    }
}
