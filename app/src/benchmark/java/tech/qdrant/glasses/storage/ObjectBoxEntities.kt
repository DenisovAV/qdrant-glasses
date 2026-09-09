package tech.qdrant.glasses.storage

import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.VectorDistanceType

/**
 * ObjectBox entity backing [ObjectBoxStore] — one stored object crop.
 *
 * The [embedding] carries an HNSW index (ObjectBox's ANN). Its `dimensions` is a **compile-time
 * constant** (ObjectBox fixes it at codegen), so it is HARDCODED to 512 (the current
 * ViT-B/32 space); [ObjectBoxStore] asserts the runtime `dim` matches. `distanceType = DOT_PRODUCT`
 * (equivalent to cosine for our unit-normalized vectors, and faster — ObjectBox skips re-deriving
 * magnitudes): the query score is a *distance* in 0..2 (0 = same direction), the SAME range as cosine
 * distance, which the store converts back to a higher-is-better similarity to match Qdrant's convention.
 * `neighborsPerNode = 16` (aka "M") — matched to Qdrant-HNSW's `m = 16` for an apples-to-apples HNSW
 * comparison. An on-device sweep (2026-09-08) showed M=8 also keeps recall@5 = 1.000 at 100k and is
 * ~2.4× faster to ingest (M is the build/recall lever; our clustered data + ef=200 leave recall
 * headroom for a sparser graph), and efConstruction=40 was faster still but dropped recall to 0.94 —
 * so 8 is ObjectBox's optimum here; we keep 16 only for parity with Qdrant. (The old ~100k "ceiling"
 * was our own 150 pts/s DNF floor + the heavy default M=30, not an ObjectBox wall.)
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

    @HnswIndex(dimensions = 512, distanceType = VectorDistanceType.DOT_PRODUCT, neighborsPerNode = 16)
    var embedding: FloatArray? = null
}
