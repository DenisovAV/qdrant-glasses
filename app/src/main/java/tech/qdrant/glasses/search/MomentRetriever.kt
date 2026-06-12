package tech.qdrant.glasses.search

import android.util.Log
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.VisionMemoryStore

/** One result card: a moment = frame + the speech of that moment. */
data class MomentCard(
    val frame: MemoryFrame,           // carries imagePath + timestampMs (+ transcript for heard hits)
    val fromVision: Boolean,
    val fromHeard: Boolean,
    val strength: Float,
)

/**
 * Dual-channel retrieval with strength gates (canon design, 2026-06-12 spec):
 *  - vision: ABSOLUTE score gate (room frames are mutually similar; margins are noise)
 *  - heard: MARGIN gate (top1−top2) OR keyword override (exact word present)
 * Surviving hits collapse into moments: same frame, or |frame.ts − hit.ts| within
 * MERGE_MS, count as one moment with summed strength (double-confirmed ranks first).
 */
class MomentRetriever(private val store: VisionMemoryStore) {
    companion object {
        private const val TAG = "MomentRetriever"
        // Calibrated from 13 on-device DIAG queries (TinyCLIP-40M):
        const val VISION_MIN_SCORE = 0.28f   // relevant ~0.30+, background 0.19–0.25
        const val HEARD_MIN_MARGIN = 0.10f   // real hits ≥0.16, noise ≤0.054
        const val MERGE_MS = 5_000L
        const val MAX_CARDS = 3
    }

    fun retrieve(queryText: String, clipVec: FloatArray, bgeVec: FloatArray): List<MomentCard> {
        val vision = store.searchVision(clipVec)
        val heard = store.searchHeard(bgeVec)
        val keyword = store.keywordHits(queryText)

        val visionOk = vision.firstOrNull()?.let { it.score >= VISION_MIN_SCORE } == true
        val heardMargin = if (heard.size >= 2) heard[0].score - heard[1].score
                          else if (heard.size == 1) 1f else 0f
        val heardOk = keyword.isNotEmpty() || heardMargin >= HEARD_MIN_MARGIN
        Log.i(TAG, "gates: vision=$visionOk (top=%.3f) heard=$heardOk (margin=%.3f, kw=${keyword.size})"
            .format(vision.firstOrNull()?.score ?: 0f, heardMargin))

        val cards = ArrayList<MomentCard>()
        if (visionOk) {
            vision.takeWhile { it.score >= VISION_MIN_SCORE }.take(MAX_CARDS).forEach {
                cards.add(MomentCard(it.frame, fromVision = true, fromHeard = false, strength = it.score))
            }
        }
        if (heardOk) {
            // keyword hits are certain (strength 1.0); semantic top only if margin passed
            keyword.forEach { cards.add(MomentCard(it, fromVision = false, fromHeard = true, strength = 1.0f)) }
            if (heardMargin >= HEARD_MIN_MARGIN) heard.firstOrNull()?.let {
                cards.add(MomentCard(it.frame, fromVision = false, fromHeard = true, strength = it.score))
            }
        }

        // merge into moments: same imagePath or timestamps within MERGE_MS
        val merged = ArrayList<MomentCard>()
        for (c in cards.sortedByDescending { it.strength }) {
            val near = merged.indexOfFirst {
                it.frame.imagePath == c.frame.imagePath ||
                kotlin.math.abs(it.frame.timestampMs - c.frame.timestampMs) <= MERGE_MS
            }
            if (near == -1) merged.add(c)
            else merged[near] = merged[near].let { m ->
                m.copy(
                    fromVision = m.fromVision || c.fromVision,
                    fromHeard = m.fromHeard || c.fromHeard,
                    strength = m.strength + c.strength,
                    // prefer the frame that carries a transcript (heard hit) for display
                    frame = if (m.frame.transcript == null && c.frame.transcript != null) c.frame else m.frame
                )
            }
        }
        val top = ArrayList(merged.sortedByDescending { it.strength }.take(MAX_CARDS))

        // BACKFILL: never return fewer than MAX_CARDS when candidates exist — an AR
        // display renders "nothing" as a transparent hole, and a single card leaves
        // nothing to page through. Confident (gated) moments rank first; remaining
        // slots are topped up with low-confidence vision hits marked "?" (no badges).
        if (top.size < MAX_CARDS) {
            for (h in vision + heard) {
                if (top.size >= MAX_CARDS) break
                val dup = top.any {
                    it.frame.imagePath == h.frame.imagePath ||
                    kotlin.math.abs(it.frame.timestampMs - h.frame.timestampMs) <= MERGE_MS
                }
                if (!dup) top.add(MomentCard(h.frame, fromVision = false, fromHeard = false, strength = h.score))
            }
        }
        Log.i(TAG, "moments: " + top.joinToString { m ->
            "[%s %.2f %s]".format(
                when { m.fromVision && m.fromHeard -> "S+H"; m.fromVision -> "S"; m.fromHeard -> "H"; else -> "?" },
                m.strength, m.frame.imagePath.substringAfterLast('/'))
        }.ifEmpty { "none (empty base)" })
        return top
    }
}
