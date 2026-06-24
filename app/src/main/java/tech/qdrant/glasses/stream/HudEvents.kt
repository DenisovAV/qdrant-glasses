package tech.qdrant.glasses.stream

import org.json.JSONArray
import org.json.JSONObject
import tech.qdrant.glasses.detect.Track

/**
 * Pure builders for the 5 SSE event lines the HUD consumes. Each returns one compact
 * JSON line (no trailing newline — the server adds the SSE "data: ...\n\n" framing).
 * Boxes carry the tracker's trackId as `id`; stored/results carry the thumb KEY as `id`.
 */
object HudEvents {
    data class ResultItem(val thumbKey: String, val label: String, val score: Float)

    /** scores maps trackId -> detection score for this frame; absent → score omitted. */
    fun boxesEvent(tracks: List<Track>, scores: Map<Int, Float>, frameW: Int, frameH: Int): String {
        val items = JSONArray()
        for (t in tracks) {
            val o = JSONObject()
                .put("id", t.trackId).put("label", t.label)
                .put("x", t.bbox.left.toInt()).put("y", t.bbox.top.toInt())
                .put("w", (t.bbox.right - t.bbox.left).toInt())
                .put("h", (t.bbox.bottom - t.bbox.top).toInt())
            scores[t.trackId]?.let { o.put("score", it.toDouble()) }
            items.put(o)
        }
        return JSONObject().put("t", "boxes")
            .put("frame", JSONArray().put(frameW).put(frameH))
            .put("items", items).toString()
    }

    fun storedEvent(thumbKey: String, label: String, count: Long): String =
        JSONObject().put("t", "stored").put("id", thumbKey).put("label", label).put("count", count).toString()

    fun tickEvent(detectMs: Long, embedMs: Long, storeMs: Long, count: Long): String =
        JSONObject().put("t", "tick").put("detect", detectMs).put("embed", embedMs)
            .put("store", storeMs).put("count", count).toString()

    fun modeEvent(value: String, query: String? = null): String {
        val o = JSONObject().put("t", "mode").put("value", value)
        if (query != null) o.put("query", query)
        return o.toString()
    }

    fun resultsEvent(items: List<ResultItem>): String {
        val arr = JSONArray()
        for (r in items) arr.put(JSONObject().put("id", r.thumbKey).put("label", r.label).put("score", r.score.toDouble()))
        return JSONObject().put("t", "results").put("items", arr).toString()
    }
}
