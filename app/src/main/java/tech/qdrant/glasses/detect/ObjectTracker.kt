package tech.qdrant.glasses.detect

import android.graphics.RectF

data class Track(
    val trackId: Int,
    val label: String,
    val bbox: RectF,
    val sightings: Int,
    val embedded: Boolean,
)

/**
 * IoU tracker. Each frame, detection↔track pairs (same label, IoU >= threshold) are claimed
 * GLOBALLY best-first — the highest-IoU pair wins, so detection order can't make one detection
 * steal a track that fits another detection better (important when two same-label objects are
 * in frame). Unmatched detections start new tracks. A track becomes eligible for embedding
 * once seen [confirmSightings] times, and is embedded at most once (dedup via [markEmbedded]).
 *
 * NOT thread-safe: [update], [confirmedUnembedded], and [markEmbedded] must all be called from
 * the same thread/coroutine lane (the ViewModel's inference lane).
 */
class ObjectTracker(
    private val confirmSightings: Int = 3,
    private val iouThreshold: Float = 0.4f,
) {
    private companion object {
        const val MAX_MISSED_TICKS = 15   // drop a track unseen for this many frames
    }

    private data class State(
        val id: Int, val label: String, var bbox: RectF,
        var sightings: Int, var embedded: Boolean, var lastSeenTick: Int,
    )
    private data class Candidate(val score: Float, val detIdx: Int, val trackIdx: Int)

    private val tracks = mutableListOf<State>()
    private var nextId = 1
    private var tick = 0

    fun update(detections: List<Detection>): List<Track> {
        tick++
        // All eligible (detection, track) pairs, claimed in descending-IoU order so the best
        // global match wins regardless of detection iteration order.
        val candidates = buildList {
            for (di in detections.indices) for (ti in tracks.indices) {
                if (tracks[ti].label != detections[di].label) continue
                val s = iou(tracks[ti].bbox, detections[di].bbox)
                if (s >= iouThreshold) add(Candidate(s, di, ti))
            }
        }.sortedByDescending { it.score }

        val claimedDet = BooleanArray(detections.size)
        val claimedTrack = BooleanArray(tracks.size)
        for (c in candidates) {
            if (claimedDet[c.detIdx] || claimedTrack[c.trackIdx]) continue
            claimedDet[c.detIdx] = true; claimedTrack[c.trackIdx] = true
            tracks[c.trackIdx].apply { bbox = detections[c.detIdx].bbox; sightings++; lastSeenTick = tick }
        }
        for (di in detections.indices) {
            if (!claimedDet[di]) {
                val d = detections[di]
                tracks.add(State(nextId++, d.label, d.bbox, 1, false, tick))
            }
        }
        tracks.removeAll { tick - it.lastSeenTick > MAX_MISSED_TICKS }
        return tracks.map { Track(it.id, it.label, it.bbox, it.sightings, it.embedded) }
    }

    fun confirmedUnembedded(): List<Track> =
        tracks.filter { it.sightings >= confirmSightings && !it.embedded }
            .map { Track(it.id, it.label, it.bbox, it.sightings, it.embedded) }

    fun markEmbedded(trackId: Int) {
        tracks.firstOrNull { it.id == trackId }?.embedded = true
    }

    /** Undo markEmbedded — e.g. when the async embed failed, so the object is retried. */
    fun unmarkEmbedded(trackId: Int) {
        tracks.firstOrNull { it.id == trackId }?.embedded = false
    }
}
