package tech.qdrant.glasses.search

import android.util.Log
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.stream.HudEvents
import tech.qdrant.glasses.stream.HudPublisher

/**
 * Moment-mode voice search (episodic-memory plan Task 1.6 + Task 2.3, Spec §3; query-understanding
 * plan Task 5): normalize the query via [parseQuery] — ONE structured [ParsedQuery] instead of the
 * four separate [searchPhrase]/[stripTimePhrases]/[extractTimeWindow]/[isRecallLocationIntent] calls
 * Task 0.1 put on [ObjectSearcher] (that searcher is retired and still calls them directly — left
 * alone, not migrated) — then either the PURE-TIME path ([ParsedQuery.timeOnly]: a moment scroll
 * over [MomentStore.framesInWindow], no embedding, no gate) or the normal path: text-embed →
 * vector search over BOTH [MomentStore.searchFrames] and [MomentStore.searchRegions] →
 * [fuseAndCollapse] to one hit per moment → [softBoost] a verified-tag match → per-channel score
 * gate → HUD push → map to [MomentCard]s. This is the YOLO-independent "real memory" recall path
 * (Spec §2/§3): the frame channel alone is Stage 1's whole-frame backbone; the region channel
 * (Task 2.3) adds small-object recall on top WITHOUT YOLO ever gating — a region only ever raises a
 * moment's score or attaches a display tag, never filters one out (a moment absent from both
 * channels' hit lists just isn't a candidate at all, same as before).
 *
 * Deliberately reuses [ObjectSearcher.Outcome] rather than defining its own sealed type: the two
 * searchers are mutually exclusive per [tech.qdrant.glasses.Config.MOMENT_MEMORY] (never both
 * live at once — see `GlassesComponents.load`'s "nullable by mode/opt-in" rule for
 * `momentStore`/`objectStore`), so sharing one `Outcome` shape keeps
 * `GlassesViewModel.onVoiceResult`'s two branches structurally identical instead of duplicating a
 * sealed interface that would only ever differ in name.
 *
 * Threading: [search] MUST already be running on `inferLane` (same discipline [ObjectSearcher]
 * documents) — this function does no dispatching of its own.
 */
class MomentSearcher(
    private val cropEncoder: CropEncoder,
    private val store: MomentStore,
    private val hud: HudPublisher,
) {
    companion object {
        private const val TAG = "GlassesVM"

        /** Same widened-pool rationale as [ObjectSearcher]'s RECALL_FETCH_K: a recall-intent
         *  query's true most-recent sighting may not be a top-5 cosine hit. */
        private const val RECALL_FETCH_K = 25

        /**
         * The whole-frame moment-search score gate is PER-BACKEND — [CropEncoderFactory.searchGate]
         * — because the text→image cosine scale differs by encoder (modality gap). CLIP (QNN_B32):
         * 0.25, calibrated from an on-device rehearsal — present-object queries ran 0.26–0.29, the
         * absent junk floor ~0.246, so 0.25 is precision-first. SigLIP2 (SIGLIP_NPU): 0.085 — its
         * scale is ~3× more compressed (measured on-device: present ~0.11, absent ~0.069), so the
         * CLIP 0.25 left EVERY SigLIP moment below the gate and search worked by verified-tag match
         * ONLY. A weak/broad present object that still misses the gate is recovered by an exact
         * VERIFIED-tag match (the tag-accept filter in [search] / [tagAcceptedMomentIds]).
         */
    }

    /** Runs on: inferLane. */
    fun search(query: String): ObjectSearcher.Outcome {
        val pq = parseQuery(query, System.currentTimeMillis())
        if (pq.embedText != query.lowercase())
            Log.i(TAG, "query normalized(moments): \"$query\" → \"${pq.embedText}\" window=${pq.window}")

        // Pure-time query (query-understanding plan Task 5): the query named a time and NOTHING
        // object-like survived parseQuery's strip (see ParsedQuery.timeOnly's KDoc) — "what did I
        // see yesterday" wants that day's moments, not a semantic match, so this skips encodeText
        // and the whole gate/fusion pipeline below entirely and goes straight to a recency-ordered
        // moment scroll. Same store-failure → Unavailable discipline as the normal path (a native
        // EdgeException from framesInWindow must not crash the inferLane coroutine either).
        if (pq.timeOnly) {
            val ordered = try {
                store.framesInWindow(pq.window?.sinceMs, pq.window?.untilMs, limit = 5)
            } catch (e: Throwable) {
                Log.e(TAG, "moment store framesInWindow failed", e)
                return ObjectSearcher.Outcome.Unavailable
            }
            Log.i(TAG, "onVoiceResult(moments, time-only): window=${pq.window} returned=${ordered.size}")
            val resultItems = ordered.map { h ->
                val key = java.io.File(h.thumbPath).nameWithoutExtension
                hud.registerThumb(key, h.thumbPath)
                val tags = if (h.label.isNotEmpty()) listOf(h.label) else emptyList()
                HudEvents.ResultItem(key, h.label, h.score, tags)
            }
            hud.pushEvent(HudEvents.resultsEvent(resultItems))
            return ObjectSearcher.Outcome.Success(ordered.map { toMomentCard(it) })
        }

        val t0 = System.currentTimeMillis()
        val qvec = try {
            cropEncoder.encodeText(pq.embedText)
        } catch (e: Throwable) {
            Log.e(TAG, "moment query embed failed", e)
            return ObjectSearcher.Outcome.Unavailable
        }
        val encMs = System.currentTimeMillis() - t0
        val searchT0 = System.currentTimeMillis()
        // Recall-intent queries need a wider pool than a display top-5 — see RECALL_FETCH_K. Both
        // channels share the same qvec/window/fetchK — a region is only ever a small-object find on
        // the SAME query embedding, never a separately-tuned search.
        val fetchK = if (pq.recallIntent) RECALL_FETCH_K else 5
        // Task 2.3 (Spec §3): collapse the two channels to one hit per moment (client-side max —
        // frame and region vectors share the same CLIP space), then a bounded soft nudge for a
        // query-token match against a VERIFIED region label. Neither step can introduce a moment
        // that wasn't already a hit on one of the two channels — YOLO tags never gate (Spec §3).
        // A native EdgeException (locked/corrupt shard) escaping store.searchFrames/searchRegions
        // would otherwise crash the inferLane coroutine and strand the UI in Processing — wrap the
        // store calls AND the fusion that consumes their results, same honest Unavailable the embed
        // failure above already gets.
        val (allHits, tagAcceptedIds) = try {
            val frameHits = store.searchFrames(qvec, topK = fetchK, sinceMs = pq.window?.sinceMs, untilMs = pq.window?.untilMs)
            val regionHits = store.searchRegions(qvec, topK = fetchK, sinceMs = pq.window?.sinceMs, untilMs = pq.window?.untilMs)
            val fused = fuseAndCollapse(frameHits, regionHits)
            val qTokens = queryTokens(pq.embedText)
            val ranked = softBoost(fused, regionHits, qTokens, TAG_BOOST_LAMBDA)
                .sortedByDescending { it.score }   // softBoost can reorder what fuseAndCollapse sorted
            // Change 2 of the calibration rehearsal: tagAcceptedIds is computed from the PRE-collapse
            // regionHits (fuseAndCollapse can drop an unverified-but-top-scoring region's label — see
            // its KDoc — so the verified label pool has to come from here, same as softBoost above).
            ranked to tagAcceptedMomentIds(regionHits, qTokens)
        } catch (e: Throwable) {
            Log.e(TAG, "moment store search failed", e)
            return ObjectSearcher.Outcome.Unavailable
        }
        // The per-backend search gate is precision-first against the whole-frame junk floor,
        // but it can also drop a moment whose region label we ALREADY verified at capture time,
        // just because that moment's fused vector score happens to miss the gate (broad categories
        // like "person" verify at 0.21–0.23, per VERIFY_COS's KDoc — nowhere near guaranteed to
        // clear 0.25). So accept on EITHER the fused score clearing the gate OR an exact match
        // between a query token and one of this moment's VERIFIED region labels (tagAcceptedIds
        // above) — mirrors the retired ObjectSearcher's gate-OR-labelMatch, but sourced from
        // CLIP-verified region labels, not YOLO's raw output, so it can't be polluted by a raw
        // mislabel the way ObjectSearcher's version could.
        val gate = CropEncoderFactory.searchGate
        val hits = allHits.filter { it.score >= gate || it.momentId in tagAcceptedIds }
        val searchMs = System.currentTimeMillis() - searchT0
        Log.i(TAG, "onVoiceResult(moments): encode=${encMs}ms search=${searchMs}ms " +
            "hits=${hits.size}/${allHits.size} gate=$gate tagAccepted=${tagAcceptedIds.size} " +
            "top=${allHits.firstOrNull()?.score}")
        // "Where did I leave/put X" wants the MOST RECENT moment, not the best cosine match — same
        // pragmatic widen-then-sort-then-trim as ObjectSearcher (see its KDoc for the caveat).
        // Both branches keep the established top-5 contract: recall re-sorts by recency then trims;
        // the non-recall branch is already score-sorted (inherited from allHits above) so trimming
        // here is just the top-5 cap — a two-channel fetchK=5+5 can otherwise surface ~10 distinct
        // moments after fuseAndCollapse.
        val ordered =
            if (pq.recallIntent) hits.sortedByDescending { it.timestampMs }.take(5)
            else hits.take(5)
        val resultItems = ordered.map { h ->
            val key = java.io.File(h.thumbPath).nameWithoutExtension
            hud.registerThumb(key, h.thumbPath)
            // F3 (Spec §5): fuseAndCollapse only ever puts a NON-empty label on a fused hit when a
            // VERIFIED region backs it (see its KDoc) — surface that as a recall-card tag chip. This
            // is the SEARCH-time tag path only; the live timeline's momentEvent tags stay empty
            // regardless (regions are stored after a frame's onMoment fires — see HudEvents.momentEvent's KDoc).
            val tags = if (h.label.isNotEmpty()) listOf(h.label) else emptyList()
            HudEvents.ResultItem(key, h.label, h.score, tags)
        }
        hud.pushEvent(HudEvents.resultsEvent(resultItems))
        return ObjectSearcher.Outcome.Success(ordered.map { toMomentCard(it) })
    }

    private fun toMomentCard(h: MomentHit): MomentCard = MomentCard(
        frame = MemoryFrame(
            id = h.id, score = h.score, imagePath = h.thumbPath,
            timestampMs = h.timestampMs, tEndMs = h.timestampMs,
            type = "moment", transcript = null,
        ),
        fromVision = true, fromHeard = false, strength = h.score,
    )
}
