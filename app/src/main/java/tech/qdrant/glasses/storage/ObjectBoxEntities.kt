package tech.qdrant.glasses.storage

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.VectorDistanceType

/**
 * ObjectBox entity backing [ObjectBoxStore] — one stored object crop.
 *
 * The [embedding] carries an HNSW index (ObjectBox's ANN). Its `dimensions` is a **compile-time
 * constant** (ObjectBox fixes it at codegen — design §6), so it is HARDCODED to 512 (the current
 * ViT-B/32 space); [ObjectBoxStore] asserts the runtime `dim` matches. `distanceType = COSINE` so
 * ObjectBox computes cosine distance directly (no manual L2-normalize needed) — the query score is
 * cosine *distance* (1 − similarity, nearest-first), which the store converts back to a
 * higher-is-better similarity to match the Qdrant convention.
 *
 * [timestampMs] is its own scalar property so `searchFiltered` can combine a range filter with the
 * vector search in one query; [payloadJson] holds the engine-independent [ObjectPayload] verbatim.
 */
@Entity
class ObjectBoxMemory {
    @Id
    var id: Long = 0
    var timestampMs: Long = 0
    var payloadJson: String = ""

    @HnswIndex(dimensions = 512, distanceType = VectorDistanceType.COSINE)
    var embedding: FloatArray? = null
}
