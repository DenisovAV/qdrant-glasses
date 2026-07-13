package tech.qdrant.glasses.search

import android.util.Log
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.ObjectHit
import tech.qdrant.glasses.storage.ObjectStore
import tech.qdrant.glasses.stream.HudEvents
import tech.qdrant.glasses.stream.HudPublisher

/**
 * OBJECTS-mode voice search: normalize the query → text-embed → vector search → hybrid
 * cosine-gate-OR-label-match filter → HUD push → map to [MomentCard]s. Moved VERBATIM out of
 * [tech.qdrant.glasses.GlassesViewModel.onVoiceResult]'s OBJECTS branch (Task 8 of the
 * God-object decomposition) — same normalization, same gate-OR-labelMatch filter, same
 * push-before-return-value ordering, same empty-result behavior (an empty [Outcome.Success]
 * list is a valid, honest "nothing found" — not [Outcome.Unavailable]).
 *
 * Threading: [search] MUST already be running on `inferLane` (the same lane as the OBJECTS
 * detect/embed pipeline, so a query text-embed serializes with the detect/embed hot path and
 * the store query) — this function does no dispatching of its own.
 */
class ObjectSearcher(
    private val cropEncoder: CropEncoder,
    private val store: ObjectStore,
    private val hud: HudPublisher,
) {
    companion object {
        private const val TAG = "GlassesVM"
    }

    sealed interface Outcome {
        /** [cards] may be EMPTY — the caller maps that to `Results(query, emptyList())`, i.e.
         *  MainActivity's honest "nothing found", not an error state. */
        data class Success(val cards: List<MomentCard>) : Outcome
        /** The text embed threw (encoder not ready / native failure) — caller goes to Idle. */
        object Unavailable : Outcome
    }

    /** Runs on: inferLane. */
    fun search(query: String): Outcome {
        // Strip question boilerplate before embedding: SigLIP2's text→crop scale is
        // compressed, and "where is my laptop" scores ~0.11 vs 0.128 for plain "laptop" —
        // enough to dip under the gate. Search on the object phrase, display the full query.
        val phrase = searchPhrase(query)
        if (phrase != query.lowercase()) Log.i(TAG, "query normalized: \"$query\" → \"$phrase\"")
        val t0 = System.currentTimeMillis()
        val qvec = try {
            cropEncoder.encodeText(phrase)
        } catch (e: Throwable) {
            Log.e(TAG, "query embed failed", e)
            return Outcome.Unavailable
        }
        val encMs = System.currentTimeMillis() - t0
        val searchT0 = System.currentTimeMillis()
        // Per-encoder score gate: without it an absent-object query ("keys" when no keys
        // were ever stored) surfaces junk top-5 around 0.09 — worse than saying "nothing".
        val gate = CropEncoderFactory.searchGate
        val allHits = store.search(qvec, topK = 5)
        // Hybrid acceptance: cosine gate OR detector-label word match. SigLIP2's text→crop
        // scale is compressed AND environment-sensitive (the same "cell phone" query scored
        // 0.117 at home but 0.095-0.106 at the venue against a darker/farther crop), so an
        // absolute gate alone drops real matches. If a query word literally names the stored
        // label ("phone" ⊂ "cell phone"), the object is what was asked for — show it.
        val qTokens = queryTokens(phrase)
        val hits = allHits.filter { it.score >= gate || labelMatchesQuery(it.label, qTokens) }
        val searchMs = System.currentTimeMillis() - searchT0
        Log.i(TAG, "onVoiceResult(objects): encode=${encMs}ms search=${searchMs}ms " +
            "hits=${hits.size}/${allHits.size} gate=$gate top=${allHits.firstOrNull()?.score}")
        val resultItems = hits.map { h ->
            val key = java.io.File(h.thumbPath).nameWithoutExtension
            hud.registerThumb(key, h.thumbPath)
            HudEvents.ResultItem(key, h.label, h.score)
        }
        hud.pushEvent(HudEvents.resultsEvent(resultItems))
        return Outcome.Success(hits.map { toMomentCard(it) })
    }

    private fun toMomentCard(h: ObjectHit): MomentCard = MomentCard(
        frame = MemoryFrame(
            id = h.id, score = h.score, imagePath = h.thumbPath,
            timestampMs = h.timestampMs, tEndMs = h.timestampMs,
            type = "object", transcript = h.label,
        ),
        fromVision = true, fromHeard = false, strength = h.score,
    )
}
