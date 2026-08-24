package tech.qdrant.glasses.search

import android.util.Log
import tech.qdrant.glasses.Config
import tech.qdrant.glasses.embedding.BgeTextEncoder
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.fleet.FleetSource
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
    // Stage 3 "OCR read channel": null disables the text-search fan-out entirely (mirrors
    // MomentCapture's ocrEngine/bgeEncoder nullable-optional-feature contract) — GlassesViewModel
    // only passes a non-null encoder when GlassesComponents.load() built one. Note this is the SAME
    // BgeTextEncoder instance MomentCapture uses to write the `ocr` channel — searching with a
    // different encoder instance would still work (it's a fixed pretrained model, not fine-tuned
    // per-session) but there is exactly one instance in the app either way (GlassesComponents.bgeEncoder).
    private val bgeEncoder: BgeTextEncoder? = null,
    // Fleet-sync Task 5 (Spec §3/§5): the pulled fleet corpus, behind the FleetSource seam so this
    // class stays testable in a plain JVM/Robolectric test (no native EdgeShard) — see FleetSource's
    // KDoc. Null (the default, same nullable-optional-feature contract as bgeEncoder above) means
    // "no fleet configured" and the merge step below is a no-op, byte-for-byte today's local-only
    // search — GlassesComponents only passes non-null when Config.FLEET_URL is set AND a pull
    // succeeded (Task 6).
    private val fleet: FleetSource? = null,
) {
    companion object {
        private const val TAG = "GlassesVM"

        /** Same widened-pool rationale as [ObjectSearcher]'s RECALL_FETCH_K: a recall-intent
         *  query's true most-recent sighting may not be a top-5 cosine hit. */
        private const val RECALL_FETCH_K = 25

        // BGE cosine gate for the OCR "read channel" (Stage 3) — a DIFFERENT scale than
        // CropEncoderFactory.searchGate (BGE sentence embeddings, not CLIP/SigLIP image-text), so
        // this is its own constant, not reused from there. A real text match (query text actually
        // appears in a recognized OCR line) should score high on BGE's own scale; 0.5 is an
        // UNCALIBRATED starting value — TODO: calibrate on-device against real OCR'd text once
        // Stage 3 has been exercised on the glasses (same "starting value pending calibration"
        // status as CONFIRM_COSINE/VERIFY_COS/searchGate/TAG_BOOST_LAMBDA elsewhere in this plan).
        private const val OCR_GATE = 0.5f

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

        /**
         * Gate-then-decay re-rank of gate SURVIVORS (see [Config.RECENCY_TAU_MS]). Re-orders — never
         * re-gates, never drops — [hits] by `score × exp(-Δt / τ)`, so the freshest of several
         * near-equal cosine matches wins. Δt = [nowMs] − hit timestamp, clamped ≥ 0 so a write-time
         * clock skew that stamps a moment slightly in the future can't AMPLIFY it (exp of a positive
         * exponent). [tauMs] is assumed > 0 (the caller gates on that; 0 is the OFF sentinel and would
         * divide into NaN, never passed here). sortedByDescending is STABLE, so hits with equal decayed
         * scores keep their incoming (raw-score-sorted) order — decay only breaks near-ties, it doesn't
         * reshuffle. Pure + static so it unit-tests without the search() deps.
         */
        internal fun recencyRank(hits: List<MomentHit>, nowMs: Long, tauMs: Long): List<MomentHit> =
            hits.sortedByDescending { h ->
                val dt = (nowMs - h.timestampMs).coerceAtLeast(0L).toDouble()
                h.score * kotlin.math.exp(-dt / tauMs)
            }
    }

    /** Runs on: inferLane. */
    fun search(query: String): ObjectSearcher.Outcome {
        val nowMs = System.currentTimeMillis()
        val pq = parseQuery(query, nowMs)
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
        // Stage 3 "OCR read channel": a SEPARATE search — OCR hits live in the BGE 384-dim `text`
        // space, not the crop encoder's `clip` space [allHits] above is scored in, so they can NOT
        // be folded into fuseAndCollapse's client-side max (different scale entirely). Treated like
        // [tagAcceptedIds] instead: collect the momentIds of OCR hits whose score clears [OCR_GATE]
        // and OR-accept those into the final gate below. Additive + optional (mirrors
        // MomentCapture's ocrEngine/bgeEncoder nullable contract on the write side) — a null
        // [bgeEncoder] or an embed/store failure here must never fail the whole voice query, so this
        // gets its OWN try/catch that falls back to an empty set rather than Unavailable.
        val ocrAcceptedIds = try {
            val bge = bgeEncoder
            if (bge != null) {
                val bgeVec = bge.encode(pq.embedText)
                val ocrHits = store.searchText(
                    bgeVec, topK = fetchK, sinceMs = pq.window?.sinceMs, untilMs = pq.window?.untilMs,
                )
                ocrHits.filter { it.score >= OCR_GATE }.mapTo(mutableSetOf()) { it.momentId }
            } else {
                emptySet()
            }
        } catch (e: Throwable) {
            Log.w(TAG, "moment ocr search failed (non-fatal): ${e.message}")
            emptySet()
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
        val hits = allHits.filter {
            it.score >= gate || it.momentId in tagAcceptedIds || it.momentId in ocrAcceptedIds
        }
        val searchMs = System.currentTimeMillis() - searchT0
        Log.i(TAG, "onVoiceResult(moments): encode=${encMs}ms search=${searchMs}ms " +
            "hits=${hits.size}/${allHits.size} gate=$gate tagAccepted=${tagAcceptedIds.size} " +
            "ocrAccepted=${ocrAcceptedIds.size} top=${allHits.firstOrNull()?.score}")
        // Fleet-sync Task 5 (Spec §3/§5): query the SAME qvec/window against the pulled fleet corpus
        // (already tagged source="fleet" by FleetShardStore), gate it with the SAME per-backend gate
        // local hits just cleared, then dedup by id keeping the FIRST occurrence — hits is appended
        // before fleetHits, so a local hit always wins its own id over a fleet duplicate ("local
        // wins", Spec §5's two-shard merge). Gated on `fleet != null` ONLY — Config.FLEET_URL is
        // already the gate one layer up, in GlassesComponents (Task 6): it only ever constructs and
        // passes a non-null `fleet` when the URL is set and a pull succeeded, so re-checking the
        // sysprop here would be redundant (and wrong for a test that injects `fleet` directly, e.g.
        // MomentSearcherFleetTest, without going through Config at all). Wrapped like every other
        // fleet op (Spec §7): a runtime failure (unreachable server, native shard error) falls back
        // to local-only, never fails the voice query.
        val fleetHits = if (fleet != null)
            try { fleet.searchFrames(qvec, fetchK, pq.window?.sinceMs, pq.window?.untilMs).filter { it.score >= gate } }
            catch (e: Throwable) { Log.w(TAG, "fleet search failed (non-fatal): ${e.message}"); emptyList() }
        else emptyList()
        val merged = (hits + fleetHits).distinctBy { it.id }
        // "Where did I leave/put X" wants the MOST RECENT moment, not the best cosine match — same
        // pragmatic widen-then-sort-then-trim as ObjectSearcher (see its KDoc for the caveat).
        // All branches keep the established top-5 contract:
        //   • recall-intent → re-sort by recency (strong preference), trim;
        //   • recency ranker ON ([Config.RECENCY_TAU_MS] > 0) → gate-then-decay re-rank, trim — a
        //     tie-breaker within the noisy above-gate band (see recencyRank / Config KDocs);
        //   • otherwise → the raw score order (inherited from allHits above), trimmed to the top-5 cap
        //     (a two-channel fetchK=5+5 can otherwise surface ~10 distinct moments after fuseAndCollapse).
        val ordered = when {
            pq.recallIntent -> merged.sortedByDescending { it.timestampMs }.take(5)
            Config.RECENCY_TAU_MS > 0L -> recencyRank(merged, nowMs, Config.RECENCY_TAU_MS).take(5)
            else -> merged.take(5)
        }
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
