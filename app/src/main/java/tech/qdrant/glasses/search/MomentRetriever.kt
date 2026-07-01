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
 * Surviving hits collapse into moments: same frame, or their time SPANS overlap (an
 * image is an instant, a transcript covers its utterance), count as one moment with
 * summed strength (double-confirmed ranks first). See [spansOverlap].
 */
class MomentRetriever(
    private val store: VisionMemoryStore,
    // Per-encoder vision gate (see CropEncoder.visionMinScore). Defaults to the TinyCLIP value for
    // LEGACY mode / callers that don't supply one.
    private val visionMinScore: Float = VISION_MIN_SCORE,
) {
    companion object {
        private const val TAG = "MomentRetriever"
        // Calibrated from on-device DIAG (TinyCLIP-40M). Absolute CLIP cosine sits in
        // ~0.15–0.40 (modality gap; random pairs avg ~0.22). A small object in a cluttered
        // scene (a person ~10% of the frame) scores only ~0.20 even as the #1-ranked correct
        // frame, while background sits ~0.19–0.20 — separation is tiny (~0.006), below noise,
        // so no single threshold cleanly splits them. Set to 0.20 to let correct-but-small
        // objects through (e.g. "woman"→sofa frame = 0.203); the cost is occasional background
        // false positives. The heard channel remains the strong, reliable signal.
        const val VISION_MIN_SCORE = 0.20f   // real-but-small object ~0.20, background ~0.19
        const val HEARD_MIN_MARGIN = 0.10f   // real hits ≥0.16, noise ≤0.054
        const val MERGE_MS = 5_000L
        const val MAX_CARDS = 3
    }

    fun retrieve(queryText: String, clipVec: FloatArray, bgeVec: FloatArray): List<MomentCard> {
        val vision = store.searchVision(clipVec)
        val heard = store.searchHeard(bgeVec)
        val keyword = store.keywordHits(queryText)

        val visionOk = vision.firstOrNull()?.let { it.score >= visionMinScore } == true
        val heardMargin = if (heard.size >= 2) heard[0].score - heard[1].score
                          else if (heard.size == 1) 1f else 0f
        val heardOk = keyword.isNotEmpty() || heardMargin >= HEARD_MIN_MARGIN
        Log.i(TAG, "gates: vision=$visionOk (top=%.3f) heard=$heardOk (margin=%.3f, kw=${keyword.size})"
            .format(vision.firstOrNull()?.score ?: 0f, heardMargin))

        val cards = ArrayList<MomentCard>()
        if (visionOk) {
            vision.takeWhile { it.score >= visionMinScore }.take(MAX_CARDS).forEach {
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

        // merge into moments: same imagePath, or their time SPANS overlap (within slack).
        // Interval overlap — not |Δtimestamp| — because a frame's timestamp is its capture
        // instant while a transcript's is its utterance START; comparing the two points
        // directly splits one moment into two cards once an utterance runs longer than
        // MERGE_MS. Each MemoryFrame carries [timestampMs, tEndMs] (an image is an instant,
        // a transcript spans its utterance), so we test span overlap, the canonical
        // frame↔utterance alignment for multimodal retrieval.
        val merged = ArrayList<MomentCard>()
        for (c in cards.sortedByDescending { it.strength }) {
            val near = merged.indexOfFirst {
                samePath(it.frame, c.frame) || spansOverlap(it.frame, c.frame)
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

        // Enrich the SAW/HEARD badge: a heard signal that didn't clear its own margin
        // gate still confirms the source channel of an ALREADY-CONFIDENT moment. Mark
        // fromHeard on any merged moment that overlaps a heard/keyword hit in time, so a
        // frame matched both visually and by speech shows "SAW+HEARD", not just "SAW".
        // This only annotates existing moments — it never adds a card on its own.
        // Use ONLY the strongest heard match (top-1) for the overlap, not the whole
        // result tail: with continuous speech a frame is almost always within MERGE_MS
        // of SOME low-score transcript, which would falsely badge everything SAW+HEARD.
        val heardFrames = (keyword + listOfNotNull(heard.firstOrNull()?.frame))
        for (i in merged.indices) {
            val m = merged[i]
            if (m.fromHeard) continue
            val overlapsHeard = heardFrames.any { hf ->
                samePath(hf, m.frame) || spansOverlap(hf, m.frame)
            }
            if (overlapsHeard) merged[i] = m.copy(fromHeard = true)
        }

        // Only confident, gated moments are returned. When every gate closed the list is
        // empty and the caller shows an honest "NOTHING FOUND" (no closest-frame backfill).
        val top = merged.sortedByDescending { it.strength }.take(MAX_CARDS)
        Log.i(TAG, "moments: " + top.joinToString { m ->
            "[%s %.2f %s]".format(
                when { m.fromVision && m.fromHeard -> "S+H"; m.fromVision -> "S"; m.fromHeard -> "H"; else -> "ERR" },
                m.strength, m.frame.imagePath.substringAfterLast('/'))
        }.ifEmpty { "none (empty base)" })
        return top
    }

    /**
     * Same cover frame? Only when both paths are non-empty and equal. A transcript stored
     * on a static (deduped) scene has an EMPTY image_path; two such transcripts must NOT
     * collapse just because "" == "" — they merge only if their time spans overlap.
     */
    private fun samePath(a: MemoryFrame, b: MemoryFrame): Boolean =
        a.imagePath.isNotEmpty() && a.imagePath == b.imagePath

    /**
     * Do two moments cover the same point in time? Tests overlap of their [timestampMs,
     * tEndMs] spans widened by MERGE_MS on each side. An image span is an instant
     * (start == end); a transcript span covers its whole utterance — so a 12s utterance
     * still aligns to a frame captured anywhere inside it, which |Δtimestamp| would miss.
     * The effective merge window for a transcript is therefore [start − MERGE_MS,
     * end + MERGE_MS] — it scales with utterance length (~20s for a 10s utterance), not a
     * fixed 2·MERGE_MS, which is what makes a long utterance group all the frames it spans.
     */
    private fun spansOverlap(a: MemoryFrame, b: MemoryFrame): Boolean =
        a.timestampMs - MERGE_MS <= b.tEndMs && b.timestampMs - MERGE_MS <= a.tEndMs
}
