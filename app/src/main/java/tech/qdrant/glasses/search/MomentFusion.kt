package tech.qdrant.glasses.search

import tech.qdrant.glasses.storage.MomentHit

/**
 * Region-channel fusion + soft tag boost (episodic-memory plan Task 2.3, Spec §3). Frame and region
 * vectors are embedded by the SAME [tech.qdrant.glasses.embedding.CropEncoder] into the SAME CLIP
 * space (see [tech.qdrant.glasses.pipeline.MomentCapture.confirmAndStore] — a region crop is
 * embedded with the identical `cropEncoder.encode` call as the whole frame), so collapsing the two
 * channels to one score per moment via a plain client-side max is a legitimate fusion, not an
 * apples-to-oranges mix (Spec §3: "fuse — client-side max is sufficient; Edge-native RRF is an
 * optional optimization", Spec §8 unknown #3).
 *
 * Both functions below are pure — [MomentHit] lists in, [MomentHit] lists out, no store/coroutine/
 * HUD dependency — so the fusion math is fully JVM unit-tested (`MomentFusionTest`) independent of
 * [MomentSearcher]'s wiring, which is the only caller.
 */

/** A tag match can only NUDGE a ranking, never flip a strong vector hit below a weak tag-boosted
 *  one — YOLO labels are unreliable (Spec §3: "vectors absorb its errors"), so the boost is capped
 *  small relative to the cosine range (roughly `[-1,1]`, in practice a much narrower positive band)
 *  it's added to: both `yolo_conf` and `verify_cos` are themselves in `[0,1]`, so the added delta is
 *  bounded by [TAG_BOOST_LAMBDA] itself. UNCALIBRATED starting value (Spec §7/§8 unknown #7) — tuned
 *  at the Stage 2 gate. */
const val TAG_BOOST_LAMBDA = 0.05f

/**
 * Collapse [frameHits] ∪ [regionHits] to ONE [MomentHit] per [MomentHit.momentId]. A moment's fused
 * score is `max(its frame-hit score, its best region-hit score)` (Spec §3). The returned hit:
 *
 * - If a `frame` hit for that moment is in this result set, that hit WINS identity — its own [id]/
 *   [MomentHit.thumbPath]/[label]/[bbox] are kept, only [MomentHit.score] is raised to the max.
 * - Otherwise (the moment was found ONLY via a region — the frame missed it, exactly the
 *   small-object case Task 2.3 exists for) the fused hit is SYNTHESIZED from the best-scoring region
 *   hit: [id]/[type] are overwritten to the PARENT moment's id / `"frame"` so a caller can't tell a
 *   region-only hit apart from a frame hit by shape (the whole point of collapsing is that the
 *   channel split never leaks past this function) — [MomentHit.thumbPath] needs no such rewrite
 *   since a region point's `thumb_path` already IS the parent's thumb (Spec §6 payload table); the
 *   region's own [label]/[bbox]/[MomentHit.yoloConf]/[MomentHit.verifyCos] are kept as-is (there is
 *   no frame-level label/bbox to prefer instead, and the tag is exactly what a region-only find has
 *   to show for itself on the card).
 *
 * Result order: descending by fused score (a `sortedByDescending` on scores that are about to be
 * re-ranked by [softBoost] before the caller applies its own gate/ordering — cheap and never wrong
 * to do here too, and it keeps this function's own output self-consistent for direct callers/tests).
 */
fun fuseAndCollapse(frameHits: List<MomentHit>, regionHits: List<MomentHit>): List<MomentHit> {
    val frameByMoment = frameHits.associateBy { it.momentId }
    val bestRegionByMoment = regionHits
        .groupBy { it.momentId }
        .mapValues { (_, regionsForMoment) -> regionsForMoment.maxBy { it.score } }
    val momentIds = frameByMoment.keys + bestRegionByMoment.keys
    return momentIds.map { momentId ->
        val frame = frameByMoment[momentId]
        val bestRegion = bestRegionByMoment[momentId]
        when {
            frame != null && bestRegion != null -> frame.copy(score = maxOf(frame.score, bestRegion.score))
            frame != null -> frame
            bestRegion != null -> bestRegion.copy(id = momentId, type = "frame", momentId = momentId)
            else -> error("momentId $momentId present in neither frame nor region map")
        }
    }.sortedByDescending { it.score }
}

/**
 * Bounded soft tag re-rank (Spec §3). For each fused moment in [hits], look at that moment's
 * VERIFIED region labels — non-empty [MomentHit.label] entries in [regionHits] (NOT [hits]: after
 * [fuseAndCollapse] a moment carries only its WINNING channel's fields, so the full region label set
 * has to come from the original, un-collapsed [regionHits] list) sharing its [MomentHit.momentId].
 * If any matches a [queryTokens] word ([labelMatchesQuery]), add
 * `[lambda] * yoloConf * verifyCos` of the BEST such match to that moment's score.
 *
 * An UNVERIFIED region (empty [MomentHit.label] — [tech.qdrant.glasses.pipeline.MomentCapture]
 * stores the vector but drops the label when `verify_cos < VERIFY_COS`) never boosts, no matter how
 * well a query token happens to match some other field: it is filtered out of the label pool before
 * [labelMatchesQuery] is even called, and [labelMatchesQuery] on an empty label matches nothing
 * anyway (Spec §3: YOLO tags are "display + a bounded soft re-rank signal only" — an unverified tag
 * isn't even a display signal, let alone a ranking one).
 */
fun softBoost(
    hits: List<MomentHit>,
    regionHits: List<MomentHit>,
    queryTokens: Set<String>,
    lambda: Float,
): List<MomentHit> {
    val verifiedRegionsByMoment = regionHits
        .filter { it.label.isNotEmpty() }
        .groupBy { it.momentId }
    return hits.map { hit ->
        val candidates = verifiedRegionsByMoment[hit.momentId] ?: return@map hit
        val bestMatch = candidates
            .filter { labelMatchesQuery(it.label, queryTokens) }
            .maxByOrNull { it.yoloConf * it.verifyCos }
            ?: return@map hit
        hit.copy(score = hit.score + lambda * bestMatch.yoloConf * bestMatch.verifyCos)
    }
}
