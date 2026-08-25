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
import io.qdrant.edge.Match
import io.qdrant.edge.Direction
import io.qdrant.edge.IntegerIndexParams
import io.qdrant.edge.NamedVector
import io.qdrant.edge.OrderBy
import io.qdrant.edge.PayloadIndexParams
import io.qdrant.edge.PayloadSchemaType
import io.qdrant.edge.PointId
import io.qdrant.edge.Query
import io.qdrant.edge.QueryRequest
import io.qdrant.edge.RangeFloat
import io.qdrant.edge.Record
import io.qdrant.edge.ScoredPoint
import io.qdrant.edge.ScoringQuery
import io.qdrant.edge.ScrollRequest
import io.qdrant.edge.ValueVariants
import io.qdrant.edge.Vector
import io.qdrant.edge.WithPayload
import io.qdrant.edge.WithVector
import org.json.JSONObject
import tech.qdrant.glasses.fleet.FleetPoint
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
        ensurePayloadIndexes()
        Log.i(TAG, "moments shard opened, count=${shard.count(CountRequest(filter = null, exact = false))} (payload indexes: timestamp_ms, type, synced)")
    }

    /**
     * (Re)create the 0.8 payload indexes on the CURRENT [shard] (0.7's Edge FFI had NO index-creation
     * API at all — see timeline()'s KDoc). MUST run on every fresh shard handle — both [init] and
     * [deleteAll]'s reload — because `EdgeShard.load` does NOT carry indexes forward: they live in the
     * shard, and deleteAll wipes the shard dir, so an index created only in [init] vanishes the moment
     * deleteAll reloads (which is what made framesInWindow's orderBy throw "No range index" right after
     * the test/demo-wipe deleteAll).
     *
     * `timestamp_ms` needs a RANGE index for SERVER-SIDE orderBy(DESC): the DEFAULT integer index
     * (`createFieldIndex(..., INTEGER)`) is LOOKUP-ONLY and orderBy still throws — it must be
     * `IntegerIndexParams(range = true)`. `type` (KEYWORD lookup) accelerates the channel-equality
     * filter every search AND's in. runCatching: creating an index that already exists throws — swallow.
     */
    private fun ensurePayloadIndexes() {
        runCatching {
            shard.update(UpdateOperation.createFieldIndexWithParams(
                "timestamp_ms", PayloadIndexParams.Integer(IntegerIndexParams(range = true))))
        }.onFailure { Log.d(TAG, "payload index 'timestamp_ms' not (re)created: ${it.message}") }
        runCatching { shard.update(UpdateOperation.createFieldIndex("type", PayloadSchemaType.KEYWORD)) }
            .onFailure { Log.d(TAG, "payload index 'type' not (re)created: ${it.message}") }
        // Fleet-sync upstream flag (Spec §5/§6): accelerates scrollUnsyncedFrames's `synced != true`
        // filter, same reasoning as the `type` KEYWORD index just above.
        runCatching { shard.update(UpdateOperation.createFieldIndex("synced", PayloadSchemaType.BOOL)) }
            .onFailure { Log.d(TAG, "payload index 'synced' not (re)created: ${it.message}") }
        shard.flush()
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

    override fun storeOcr(textVec: FloatArray, payload: MomentPayload): String = synchronized(lock) {
        require(textVec.size == TEXT_DIM) { "dim ${textVec.size} != $TEXT_DIM" }
        require(payload.momentId.isNotBlank()) {
            "storeOcr requires payload.momentId = the parent frame's id (same convention as storeRegion)"
        }
        val id = UUID.randomUUID().toString()
        val stamped = payload.copy(type = MomentType.OCR)
        val named = Vector.Named(mapOf(TEXT_FIELD to NamedVector.Dense(textVec.toList())))
        shard.update(UpdateOperation.upsertPoints(listOf(
            Point(id = PointId.Uuid(id), vector = named, payload = stamped.toJson())
        )))
        shard.flush()
        Log.d(TAG, "storeOcr: id=$id momentId=${payload.momentId} text=\"${payload.text}\"")
        id
    }

    override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> =
        channelSearch(MomentType.FRAME, CLIP_FIELD, clipDim, qvec, topK, sinceMs, untilMs)

    override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> =
        channelSearch(MomentType.REGION, CLIP_FIELD, clipDim, qvec, topK, sinceMs, untilMs)

    override fun searchText(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> =
        channelSearch(MomentType.OCR, TEXT_FIELD, TEXT_DIM, qvec, topK, sinceMs, untilMs)

    // Parameterized over the named-vector field + its dim (Stage 3: the `ocr` channel lives in the
    // BGE `text` 384-dim space, not the crop encoder's `clip` space) so searchFrames/searchRegions/
    // searchText share one query shape instead of three near-identical copies.
    private fun channelSearch(
        typeValue: String,
        field: String,
        dim: Int,
        qvec: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<MomentHit> = synchronized(lock) {
        require(qvec.size == dim) { "dim ${qvec.size} != $dim" }
        val results = shard.query(QueryRequest(
            limit = topK.toULong(), offset = null,
            query = ScoringQuery.Vector(Query.Nearest(vector = NamedVector.Dense(qvec.toList()), using = field)),
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
     * The most-recent [limit] stored `type=frame` moments (payload only, no vectors), OLDEST-first —
     * rebuilds the HUD timeline rail on connect as a chronological strip ending at "now".
     *
     * 0.8: the `timestamp_ms` RANGE index (see [ensurePayloadIndexes]) lets the shard order server-side,
     * so this asks for the `limit` NEWEST frames via orderBy(DESC) and reverses to oldest-first — no more
     * paging the ENTIRE frame channel and sorting every point client-side. (0.7's Edge FFI had NO
     * index-creation API, so orderBy threw "No range index" and the full scan was the only option — that
     * was this method's old body, preserved in git history.)
     */
    override fun timeline(limit: Int): List<MomentHit> = synchronized(lock) {
        val resp = shard.scroll(ScrollRequest(
            offset = null, limit = limit.toULong(),
            filter = typeAndTimeFilter(MomentType.FRAME, null, null),
            withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
            orderBy = OrderBy(key = "timestamp_ms", direction = Direction.DESC, startFrom = null),
        ))
        // orderBy DESC gives newest-first; reverse to the oldest-first order the rail plays back in.
        resp.records.map { rec -> toHit(rec.id, rec.payload ?: "{}") }.reversed()
            .also { Log.i(TAG, "timeline(): orderBy DESC returning ${it.size} most-recent frames (oldest-first)") }
    }

    /**
     * `type=frame` moments whose `timestamp_ms` is in the optional `[sinceMs, untilMs]` range,
     * MOST-RECENT first — the pure-time "what did I see on <day>" retrieval path (query-
     * understanding plan Task 4; no query vector). Reuses [typeAndTimeFilter] (the same combined
     * type+range `Filter.must` [searchFrames] applies) and [timeline]'s scroll-the-whole-matching-
     * set-then-sort-client-side pattern — see [timeline]'s KDoc for why this AAR can't order by
     * `timestamp_ms` server-side. Sorted DESCENDING here (unlike [timeline]'s ascending): this
     * answers "what did I see on <day>", where the newest sighting is what the user wants first,
     * not oldest-first playback of the rail.
     */
    override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> = synchronized(lock) {
        // 0.8: the INTEGER index on `timestamp_ms` (created in init) lets the shard order server-side, so
        // this asks for the `limit` most-recent matches directly — no more scrolling the WHOLE matching
        // set and sorting client-side (0.7's forced pattern; see timeline()'s KDoc for why 0.7 couldn't).
        val resp = shard.scroll(ScrollRequest(
            offset = null, limit = limit.toULong(),
            filter = typeAndTimeFilter(MomentType.FRAME, sinceMs, untilMs),
            withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(false),
            orderBy = OrderBy(key = "timestamp_ms", direction = Direction.DESC, startFrom = null),
        ))
        resp.records.map { rec -> toHit(rec.id, rec.payload ?: "{}") }
            .also { Log.i(TAG, "framesInWindow(): since=$sinceMs until=$untilMs orderBy=DESC returning=${it.size}") }
    }

    override fun count(): Long = synchronized(lock) { shard.count(CountRequest(filter = null, exact = true)).toLong() }

    // Whole-branch review fix: `count()` was being reported to callers as "moments stored", but it
    // counts every point across BOTH channels — a keyframe plus its verified regions reads as N+1,
    // not 1. Reuses [typeAndTimeFilter]'s type-only shape (no time bound) rather than hand-rolling a
    // second `Filter.must` list, same as [searchFrames]/[searchRegions] already do via [channelSearch].
    override fun frameCount(): Long = synchronized(lock) {
        shard.count(CountRequest(filter = typeAndTimeFilter(MomentType.FRAME, null, null), exact = true)).toLong()
    }

    /**
     * The upload backlog for the fleet-sync flag-on-store design (Spec §5): up to [limit] `type=frame`
     * points whose payload's `synced` is not `true` (`Filter.mustNot` on `synced == true`, so a point
     * that never had the key set — pre-fleet-sync capture — matches too, same as the KDoc on
     * [MomentStore.scrollUnsyncedFrames] promises). Unlike every other read here, vectors ARE
     * requested (`withVector = WithVector.Bool(true)`) — the upload needs them.
     *
     * A record whose id isn't a [PointId.Uuid] or whose `clip` vector doesn't parse out of the
     * returned vector JSON is skipped (logged, not thrown) rather than failing the whole batch — this
     * store only ever WRITES `PointId.Uuid` ids with a `clip` vector on a frame point (see
     * [storeMoment]), so neither should happen in practice; the guard is defensive, not expected.
     */
    override fun scrollUnsyncedFrames(limit: Int): List<FleetPoint> = synchronized(lock) {
        val resp = shard.scroll(ScrollRequest(
            offset = null, limit = limit.toULong(),
            filter = unsyncedFrameFilter(),
            withPayload = WithPayload.Bool(true), withVector = WithVector.Bool(true),
            orderBy = null,
        ))
        resp.records.mapNotNull { toFleetPoint(it) }
            .also { Log.i(TAG, "scrollUnsyncedFrames: limit=$limit returned=${it.size}") }
    }

    // `type == frame` AND NOT (`synced == true`) — Filter.must/mustNot both AND into the overall
    // filter (must = AND, mustNot = AND-of-NOT, standard Qdrant semantics), so this reads as
    // "frame points whose synced flag is not true", including points with no `synced` key at all.
    private fun unsyncedFrameFilter(): Filter = Filter(
        must = listOf(Condition.Field(FieldCondition(
            key = "type", match = Match.Value(ValueVariants.String(MomentType.FRAME)),
            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
        ))),
        should = null,
        mustNot = listOf(Condition.Field(FieldCondition(
            key = "synced", match = Match.Value(ValueVariants.Bool(true)),
            range = null, geoBoundingBox = null, geoRadius = null, geoPolygon = null, valuesCount = null,
        ))),
    )

    // A [Record]'s `vector` field is a JSON string keyed by named-vector field (Edge FFI:
    // `vector_struct_internal_to_json` serializes the collection's `VectorStructInternal::Named` map
    // straight to `{"clip": [...], "text": [...]}` — see io.qdrant.edge.EdgeShard.scroll's Rust impl),
    // never a bare FloatArray, so this store must parse it itself for [FleetPoint.vector]. Only the
    // `clip` field is ever wanted here (frame points never carry a `text` vector — see [storeMoment]).
    // `synced` is stripped from the payload before it travels: it's LOCAL-only bookkeeping (Spec §6),
    // never uploaded — [FleetPoint.payload]'s KDoc documents the caller doing exactly this.
    private fun toFleetPoint(rec: Record): FleetPoint? {
        val id = (rec.id as? PointId.Uuid)?.value
        if (id == null) {
            Log.w(TAG, "scrollUnsyncedFrames: skipping a point with no PointId.Uuid (unexpected — this store only writes UUID ids)")
            return null
        }
        val vec = parseClipVector(rec.vector)
        if (vec == null) {
            Log.w(TAG, "scrollUnsyncedFrames: skipping id=$id, missing/unparseable '$CLIP_FIELD' vector")
            return null
        }
        return FleetPoint(id = id, vector = vec, payload = stripSynced(rec.payload))
    }

    private fun parseClipVector(vectorJson: String?): FloatArray? {
        if (vectorJson == null) return null
        return try {
            val arr = JSONObject(vectorJson).optJSONArray(CLIP_FIELD) ?: return null
            FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
        } catch (_: Throwable) {
            null
        }
    }

    private fun stripSynced(payloadJson: String?): String {
        val o = try { JSONObject(payloadJson ?: "{}") } catch (_: Throwable) { JSONObject() }
        o.remove("synced")
        return o.toString()
    }

    /**
     * Flips `synced=true` for exactly [ids] via a payload MERGE patch (`UpdateOperation.setPayload`,
     * NOT `overwritePayload` — every other key on each point's payload is left untouched), called only
     * after [tech.qdrant.glasses.fleet.FleetSync] has a CONFIRMED upsert of those points to the fleet
     * hub (Spec §5's crash-safe invariant). A no-op for an empty [ids] — mirrors
     * [tech.qdrant.glasses.fleet.FleetQdrantClient.upsertPoints]'s same empty-batch guard, so an idle
     * pass that found nothing to sync touches neither the network nor the shard.
     */
    override fun markSynced(ids: List<String>): Unit = synchronized(lock) {
        if (ids.isEmpty()) return@synchronized
        shard.update(UpdateOperation.setPayload(ids.map { PointId.Uuid(it) }, "{\"synced\":true}"))
        shard.flush()
        Log.i(TAG, "markSynced: flipped synced=true for ${ids.size} point(s)")
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
        ensurePayloadIndexes()   // the reload dropped the indexes with the old shard — recreate them
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
            // Stage 3: only a `type=ocr` payload ever has a non-empty text — frame/region points
            // stamp "" at capture time, so this mapping is unconditional like label/bbox above.
            text = p.text,
        )
    }
}
