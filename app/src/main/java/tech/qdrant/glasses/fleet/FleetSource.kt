package tech.qdrant.glasses.fleet

import tech.qdrant.glasses.storage.MomentHit

/**
 * The seam `MomentSearcher` (plan Task 5, Spec §3) depends on instead of [FleetShardStore] directly —
 * so a JVM/Robolectric test (`MomentSearcherFleetTest`) can fake the fleet corpus with no native `.so`,
 * the same way [tech.qdrant.glasses.storage.MomentStore] already lets `MomentSearcherTest` fake local
 * memory. [FleetShardStore] is the one real implementation; every hit it returns is tagged
 * `source="fleet"` (see its KDoc) — this interface doesn't enforce that itself, callers/fakes must.
 */
interface FleetSource {
    fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit>
}
