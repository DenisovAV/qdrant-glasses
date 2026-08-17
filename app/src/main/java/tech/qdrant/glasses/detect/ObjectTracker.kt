package tech.qdrant.glasses.detect

import android.graphics.RectF

data class Track(
    val trackId: Int,
    val label: String,
    val bbox: RectF,
    val sightings: Int,
    val embedded: Boolean,
    // The MOST RECENT matched detection's confidence (Task 2.2 region layer — a region's
    // `yolo_conf` payload field). Defaulted so every existing positional `Track(...)` call site
    // (tests, HudEventsTest) keeps compiling unchanged; 0f for a track this field was never set on.
    val conf: Float = 0f,
)

/**
 * IoU tracker. Each frame, detection↔track pairs (same label, IoU >= threshold) are claimed
 * GLOBALLY best-first — the highest-IoU pair wins, so detection order can't make one detection
 * steal a track that fits another detection better (important when two same-label objects are
 * in frame). Unmatched detections start new tracks. A track becomes eligible for embedding
 * once seen [confirmSightings] times, and is embedded at most once (dedup via [markEmbedded]).
 *
 * Doc-rot note: the embed-dedup pair above ([confirmedUnembedded]/[markEmbedded]/
 * [unmarkEmbedded]) describes the RETIRED crop-embed-and-store path (Task 2.4) — nothing in the
 * shipped OBJECTS pipeline calls them anymore; their only callers today are [ObjectTrackerTest].
 * [confirmed] (below) is the live region-candidate source instead — see its own KDoc.
 *
 * NOT thread-safe: [update], [confirmedUnembedded], [markEmbedded], and [confirmed] must all be
 * called from the same thread/coroutine lane (the ViewModel's inference lane).
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
        var sightings: Int, var embedded: Boolean, var lastSeenTick: Int, var conf: Float,
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
            tracks[c.trackIdx].apply {
                bbox = detections[c.detIdx].bbox; sightings++; lastSeenTick = tick
                conf = detections[c.detIdx].score
            }
        }
        for (di in detections.indices) {
            if (!claimedDet[di]) {
                val d = detections[di]
                tracks.add(State(nextId++, d.label, d.bbox, 1, false, tick, d.score))
            }
        }
        tracks.removeAll { tick - it.lastSeenTick > MAX_MISSED_TICKS }
        return tracks.map { Track(it.id, it.label, it.bbox, it.sightings, it.embedded, it.conf) }
    }

    fun confirmedUnembedded(): List<Track> =
        tracks.filter { it.sightings >= confirmSightings && !it.embedded }
            .map { Track(it.id, it.label, it.bbox, it.sightings, it.embedded, it.conf) }

    /** Confirmed tracks (`sightings >= confirmSightings`), regardless of [Track.embedded] — the
     *  region-candidate source for [tech.qdrant.glasses.pipeline.MomentCapture] (Task 2.2). Unlike
     *  [confirmedUnembedded], this NEVER touches the `embedded` flag: it is a read-only snapshot for
     *  a caller that doesn't participate in the crop-store path's dedup bookkeeping (see
     *  [tech.qdrant.glasses.pipeline.RegionCandidate]'s KDoc — confirmation here gates tag quality,
     *  not memory admission, so there is nothing to mark/unmark). */
    fun confirmed(): List<Track> =
        tracks.filter { it.sightings >= confirmSightings }
            .map { Track(it.id, it.label, it.bbox, it.sightings, it.embedded, it.conf) }

    fun markEmbedded(trackId: Int) {
        tracks.firstOrNull { it.id == trackId }?.embedded = true
    }

    /** Undo markEmbedded — e.g. when the async embed failed, so the object is retried. */
    fun unmarkEmbedded(trackId: Int) {
        tracks.firstOrNull { it.id == trackId }?.embedded = false
    }
}
