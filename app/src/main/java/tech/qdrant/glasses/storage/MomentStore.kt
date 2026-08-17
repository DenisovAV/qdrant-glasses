package tech.qdrant.glasses.storage

import org.json.JSONObject

/**
 * The two [MomentPayload.type] wire values, shared so "frame"/"region" stop drifting as independent
 * string literals across [MomentStore]'s implementations and callers (whole-branch review fix —
 * [tech.qdrant.glasses.search.MomentFusion], [tech.qdrant.glasses.pipeline.MomentCapture], and
 * [QdrantEdgeMomentStore] each spelled these out separately before this). The JSON wire values
 * themselves are unchanged — this only centralizes the Kotlin-side string constants.
 */
object MomentType {
    const val FRAME = "frame"
    const val REGION = "region"
}

/**
 * One vector search hit against a moment collection (a frame OR a region point — see [MomentPayload.type]).
 * Mirrors [ObjectHit]'s shape, plus the fields the moment model adds: [type] (channel) and
 * [momentId] (the parent frame a region belongs to; a frame hit's own id), and (Task 2.3, Spec §3)
 * [yoloConf]/[verifyCos] — the region's detector confidence and CLIP label-verification cosine,
 * needed by [tech.qdrant.glasses.search.softBoost]'s `λ·yolo_conf·verify_cos` re-rank. Both default
 * to `0f`: a FRAME hit never carries them (a frame's own [MomentPayload.yoloConf]/[MomentPayload.verifyCos]
 * are stamped 0 at capture — see [tech.qdrant.glasses.pipeline.MomentCapture.confirmAndStore]), and
 * the default keeps every pre-Task-2.3 named-arg [MomentHit] construction site compiling unchanged.
 */
data class MomentHit(
    val id: String,
    val score: Float,
    val type: String,
    val momentId: String,
    val timestampMs: Long,
    val thumbPath: String,
    val label: String,
    val bbox: String,
    val yoloConf: Float = 0f,
    val verifyCos: Float = 0f,
)

/**
 * Typed mirror of the JSON payload stored alongside each vector in a moment collection (Spec §6).
 * One shape covers every point type — `frame`, `region`, and (later stages) `speech`/`ocr` — with
 * fields that don't apply to a given [type] left at their zero value, exactly as [ObjectPayload]
 * leaves unused fields at "" / 0.
 *
 * - `frame`: [momentId] == the point's own id, [episodeId] = the recording session's start ts.
 * - `region`: [momentId] = the parent frame's id; [bbox]/[label]/[yoloConf]/[verifyCos] set.
 * - `speech`/`ocr` (Stage 3/4, not yet stored): [momentId] = nearest/parent frame; [text] set.
 *
 * Engine-independent — the [MomentStore] contract is expressed in terms of this and [MomentHit],
 * so every backend stores and returns the same payload shape (mirrors [VectorStore]/[ObjectPayload]).
 */
data class MomentPayload(
    val type: String,
    val momentId: String,
    val episodeId: Long,
    val timestampMs: Long,
    val tEndMs: Long,
    val thumbPath: String,
    val bbox: String,
    val label: String,
    val yoloConf: Float,
    val verifyCos: Float,
    val text: String,
) {
    fun toJson(): String = JSONObject()
        .put("type", type)
        .put("moment_id", momentId)
        .put("episode_id", episodeId)
        .put("timestamp_ms", timestampMs)
        .put("t_end_ms", tEndMs)
        .put("thumb_path", thumbPath)
        .put("bbox", bbox)
        .put("label", label)
        .put("yolo_conf", yoloConf.toDouble())
        .put("verify_cos", verifyCos.toDouble())
        .put("text", text)
        .toString()

    companion object {
        fun fromJson(s: String): MomentPayload {
            val o = try { JSONObject(s) } catch (_: Throwable) { JSONObject() }
            return MomentPayload(
                type = o.optString("type"),
                momentId = o.optString("moment_id"),
                episodeId = o.optLong("episode_id"),
                timestampMs = o.optLong("timestamp_ms"),
                tEndMs = o.optLong("t_end_ms"),
                thumbPath = o.optString("thumb_path"),
                bbox = o.optString("bbox"),
                label = o.optString("label"),
                yoloConf = o.optDouble("yolo_conf", 0.0).toFloat(),
                verifyCos = o.optDouble("verify_cos", 0.0).toFloat(),
                text = o.optString("text"),
            )
        }
    }
}

/**
 * Moment memory as a small, engine-independent seam — a NEW collection distinct from
 * [VectorStore]'s object crops (Spec §6, Global Constraint: "do NOT widen `VectorStore`" — three
 * engine impls + `VectorStoreBenchmark` depend on its exact shape). One collection holds two point
 * channels distinguished by [MomentPayload.type]: whole-frame keyframes ("frame") and the CLIP-
 * verified YOLO regions within them ("region"), searched separately so region hits can be
 * collapsed to their parent moment (Task 2.3) rather than competing with frames 1:1.
 *
 * `namespace` (a constructor arg of each implementation, as in [VectorStore]) keeps different
 * crop-encoder backends in separate collections.
 *
 * No engine implements this yet — see `QdrantEdgeMomentStore` (plan Task 1.3).
 */
interface MomentStore : AutoCloseable {
    /** Upsert one whole-frame keyframe point, flushing to disk before returning. → id. */
    fun storeMoment(clipVec: FloatArray, payload: MomentPayload): String

    /** Upsert one region point (a CLIP-verified YOLO box within an already-stored moment). → id. */
    fun storeRegion(clipVec: FloatArray, payload: MomentPayload): String

    /** k-nearest among `type=frame` points only, optionally restricted to `timestamp_ms` in
     *  [sinceMs, untilMs] (either bound null = open) — the primary "real memory" recall path. */
    fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit>

    /** k-nearest among `type=region` points only, same time-window semantics as [searchFrames] —
     *  the small-object recall path (Task 2.3 fuses this with [searchFrames] by parent [MomentHit.momentId]). */
    fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit>

    /** Every stored `type=frame` moment, oldest-first (payload only, no vectors) — rebuilds the
     *  HUD timeline rail on connect, mirroring [VectorStore.all]. */
    fun timeline(limit: Int = 500): List<MomentHit>

    /** Total point count across ALL channels (frame + region + future speech/ocr). */
    fun count(): Long

    /** Count of `type=frame` points only — the number of MOMENTS actually stored, distinct from
     *  [count]'s raw point total (whole-branch review fix: a keyframe plus its up to
     *  [tech.qdrant.glasses.pipeline.MomentCapture.REGIONS_MAX_PER_MOMENT] verified regions reads as
     *  N+1 points via [count], but is still ONE moment). A caller reporting "moments stored" to the
     *  user should use this, not [count]; switching the existing HUD/backfill call sites over to it
     *  is a follow-up task, not part of this fix. */
    fun frameCount(): Long

    /** Clear the whole collection in-process (the demo wipe gesture — see `wipe-demo-memory.sh`). */
    fun deleteAll()
}
