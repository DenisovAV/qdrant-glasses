package tech.qdrant.glasses.storage

import org.json.JSONObject

data class ObjectHit(
    val id: String,
    val score: Float,
    val label: String,
    val bbox: String,
    val timestampMs: Long,
    val thumbPath: String,
)

/**
 * Typed mirror of the JSON payload stored alongside each vector in an object collection.
 *
 * Keys and defaults are byte-identical to the inline `JSONObject` this replaced, so existing
 * on-disk records (written before this type existed) read back unchanged: [fromJson] mirrors the
 * `optString`/`optLong` defaults (empty string / 0) for any missing key.
 *
 * Engine-independent — the [VectorStore] contract is expressed in terms of this and [ObjectHit],
 * so every backend (Qdrant Edge, ObjectBox, …) stores and returns the same payload shape.
 * Public (not `internal`) so it can appear in the public [VectorStore] surface and so
 * [ObjectPayloadTest] can exercise it directly.
 */
data class ObjectPayload(
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
 * Object memory as a small, engine-independent seam: one collection of object crops (dense cosine
 * vectors) keyed by a UUID, carrying an [ObjectPayload]. Extracted verbatim from the former
 * `ObjectStore` (now [QdrantEdgeStore]) so alternative on-device vector engines can be measured
 * behind the same contract — mirrors the [tech.qdrant.glasses.embedding.CropEncoder] /
 * [tech.qdrant.glasses.embedding.CropEncoderFactory] pattern (one build-time backend switch).
 *
 * `namespace` (a constructor arg of each implementation) picks the on-disk directory, so different
 * crop-encoder backends keep SEPARATE collections (e.g. "mac" = SigLIP2/768-dim, "qnnb32" =
 * ViT-B/32/512-dim). Vectors of one dim never meet a collection built for another.
 *
 * Thread-safety is each implementation's responsibility (this store is touched concurrently:
 * dedup search + upsert on cropLane, the voice-query search on inferLane, and all()/rail snapshots
 * from the MjpegServer HTTP threads).
 */
interface VectorStore : AutoCloseable {
    /** Upsert one point, flushing to disk before returning (the live [PerceptionPipeline] path). → id. */
    fun upsert(vector: FloatArray, payload: ObjectPayload): String

    /** Upsert many points with a SINGLE flush at the end — the engine's true write throughput. → ids (input order). */
    fun upsertBatch(items: List<Pair<FloatArray, ObjectPayload>>): List<String>

    /** k-nearest by cosine. */
    fun search(vector: FloatArray, topK: Int = 5): List<ObjectHit>

    /** k-nearest by cosine, restricted to points whose `timestamp_ms` is in [[sinceMs], [untilMs]]
     *  (either bound null = open). Backends differ in HOW they filter, so the < k guarantee differs
     *  too: Qdrant Edge and sqlite-vec (both brute-force) apply the range condition DURING the scan,
     *  so a top-k that finds k matches always returns k. ObjectBox (HNSW) over-fetches a candidate
     *  set THEN filters it, so it CAN return fewer than k if too few candidates survive the window. */
    fun searchFiltered(vector: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<ObjectHit>

    /** Every stored object, oldest-first (payload only, no vectors) — used to rebuild the HUD rail. */
    fun all(limit: Int = 500): List<ObjectHit>

    fun count(): Long

    /** Clear the whole collection in-process (bench reset + the B3 wipe gesture). */
    fun deleteAll()

    /**
     * Finalize/build the ANN index after a bulk load. Default no-op: brute-force engines (Qdrant
     * scan, sqlite-vec) have no index to build, and engines that index incrementally on insert
     * (ObjectBox HNSW) already paid the cost during [upsertBatch]. Qdrant Edge in HNSW mode builds
     * the graph here via `optimize()`.
     */
    fun buildIndex() {}

    /** Stable engine name for CSV/MD bench output ("qdrant-edge" / "objectbox" / …). */
    val name: String
}
