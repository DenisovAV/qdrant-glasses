package tech.qdrant.glasses.search

import android.util.Log
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.stream.HudEvents
import tech.qdrant.glasses.stream.HudPublisher

/**
 * Moment-mode voice search (episodic-memory plan Task 1.6 + Task 2.3, Spec §3): normalize the query
 * (same [QueryText] helpers [searchPhrase]/[stripTimePhrases]/[extractTimeWindow]/[isRecallLocationIntent]
 * Task 0.1 already put on [ObjectSearcher]) → text-embed → vector search over BOTH
 * [MomentStore.searchFrames] and [MomentStore.searchRegions] → [fuseAndCollapse] to one hit per
 * moment → [softBoost] a verified-tag match → per-channel score gate → HUD push → map to
 * [MomentCard]s. This is the YOLO-independent "real memory" recall path (Spec §2/§3): the frame
 * channel alone is Stage 1's whole-frame backbone; the region channel (Task 2.3) adds small-object
 * recall on top WITHOUT YOLO ever gating — a region only ever raises a moment's score or attaches a
 * display tag, never filters one out (a moment absent from both channels' hit lists just isn't a
 * candidate at all, same as before).
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
         * Per-channel score gate for whole-frame moment search. UNCALIBRATED (Spec §7/§8 unknown
         * #7): [tech.qdrant.glasses.embedding.CropEncoderFactory.searchGate]'s 0.25 was rehearsed
         * against object CROPS, not whole ROOM frames — a whole-frame text→image cosine
         * distribution is a different shape (more context in-frame competing for the same
         * embedding). Seeded at 0.20 as a starting value; final number comes from the Stage 1
         * on-device rehearsal pass (plan Stage Gate 1), not from this constant.
         */
        private const val MOMENT_SEARCH_GATE = 0.20f
    }

    /** Runs on: inferLane. */
    fun search(query: String): ObjectSearcher.Outcome {
        val phrase = searchPhrase(query)
        val window = extractTimeWindow(query, System.currentTimeMillis())
        val embedPhrase = stripTimePhrases(phrase)
        if (embedPhrase != query.lowercase())
            Log.i(TAG, "query normalized(moments): \"$query\" → \"$embedPhrase\" window=$window")
        val t0 = System.currentTimeMillis()
        val qvec = try {
            cropEncoder.encodeText(embedPhrase)
        } catch (e: Throwable) {
            Log.e(TAG, "moment query embed failed", e)
            return ObjectSearcher.Outcome.Unavailable
        }
        val encMs = System.currentTimeMillis() - t0
        val searchT0 = System.currentTimeMillis()
        // Recall-intent queries need a wider pool than a display top-5 — see RECALL_FETCH_K. Both
        // channels share the same qvec/window/fetchK — a region is only ever a small-object find on
        // the SAME query embedding, never a separately-tuned search.
        val fetchK = if (isRecallLocationIntent(query)) RECALL_FETCH_K else 5
        val frameHits = store.searchFrames(qvec, topK = fetchK, sinceMs = window?.sinceMs, untilMs = window?.untilMs)
        val regionHits = store.searchRegions(qvec, topK = fetchK, sinceMs = window?.sinceMs, untilMs = window?.untilMs)
        // Task 2.3 (Spec §3): collapse the two channels to one hit per moment (client-side max —
        // frame and region vectors share the same CLIP space), then a bounded soft nudge for a
        // query-token match against a VERIFIED region label. Neither step can introduce a moment
        // that wasn't already a hit on one of the two channels — YOLO tags never gate (Spec §3).
        val fused = fuseAndCollapse(frameHits, regionHits)
        val allHits = softBoost(fused, regionHits, queryTokens(embedPhrase), TAG_BOOST_LAMBDA)
            .sortedByDescending { it.score }   // softBoost can reorder what fuseAndCollapse sorted
        // Region points carry the only labels this channel has (frame points are unlabeled — Stage
        // 1), so there is no label-match fallback here the way ObjectSearcher has one — cosine gate
        // only (a verified-region label match already contributed via softBoost above, not here).
        val hits = allHits.filter { it.score >= MOMENT_SEARCH_GATE }
        val searchMs = System.currentTimeMillis() - searchT0
        Log.i(TAG, "onVoiceResult(moments): encode=${encMs}ms search=${searchMs}ms " +
            "hits=${hits.size}/${allHits.size} gate=$MOMENT_SEARCH_GATE top=${allHits.firstOrNull()?.score}")
        // "Where did I leave/put X" wants the MOST RECENT moment, not the best cosine match — same
        // pragmatic widen-then-sort-then-trim as ObjectSearcher (see its KDoc for the caveat).
        // Both branches keep the established top-5 contract: recall re-sorts by recency then trims;
        // the non-recall branch is already score-sorted (inherited from allHits above) so trimming
        // here is just the top-5 cap — a two-channel fetchK=5+5 can otherwise surface ~10 distinct
        // moments after fuseAndCollapse.
        val ordered =
            if (isRecallLocationIntent(query)) hits.sortedByDescending { it.timestampMs }.take(5)
            else hits.take(5)
        val resultItems = ordered.map { h ->
            val key = java.io.File(h.thumbPath).nameWithoutExtension
            hud.registerThumb(key, h.thumbPath)
            HudEvents.ResultItem(key, h.label, h.score)
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
