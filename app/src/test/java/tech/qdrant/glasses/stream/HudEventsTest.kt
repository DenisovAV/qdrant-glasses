package tech.qdrant.glasses.stream

import android.graphics.RectF
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.detect.Track

@RunWith(RobolectricTestRunner::class)
class HudEventsTest {
    @Test fun boxesEvent_shape() {
        val t = Track(3, "laptop", RectF(120f, 80f, 320f, 230f), 4, false)
        val json = JSONObject(HudEvents.boxesEvent(listOf(t), mapOf(3 to 0.91f), 640, 480))
        assertEquals("boxes", json.getString("t"))
        assertEquals(640, json.getJSONArray("frame").getInt(0))
        val it0 = json.getJSONArray("items").getJSONObject(0)
        assertEquals(3, it0.getInt("id"))
        assertEquals("laptop", it0.getString("label"))
        assertEquals(120, it0.getInt("x"))
        assertEquals(200, it0.getInt("w"))   // right-left = 320-120
        assertEquals(0.91, it0.getDouble("score"), 0.001)
    }

    @Test fun storedEvent_shape() {
        val json = JSONObject(HudEvents.storedEvent("obj_3_171", "laptop", 7))
        assertEquals("stored", json.getString("t"))
        assertEquals("obj_3_171", json.getString("id"))
        assertEquals("laptop", json.getString("label"))
        assertEquals(7, json.getInt("count"))
    }

    @Test fun modeEvent_searchCarriesQuery() {
        val json = JSONObject(HudEvents.modeEvent("search", "where is the cup"))
        assertEquals("mode", json.getString("t"))
        assertEquals("search", json.getString("value"))
        assertEquals("where is the cup", json.getString("query"))
    }

    @Test fun modeEvent_idleHasNoQuery() {
        val json = JSONObject(HudEvents.modeEvent("idle"))
        assertEquals("idle", json.getString("value"))
        assertEquals(false, json.has("query"))
    }

    @Test fun resultsEvent_shape() {
        val json = JSONObject(HudEvents.resultsEvent(listOf(HudEvents.ResultItem("obj_3_171", "cup", 0.82f))))
        assertEquals("results", json.getString("t"))
        val r0 = json.getJSONArray("items").getJSONObject(0)
        assertEquals("obj_3_171", r0.getString("id"))
        assertEquals(0.82, r0.getDouble("score"), 0.001)
    }
}
