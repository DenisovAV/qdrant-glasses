package tech.qdrant.glasses.storage

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ObjectPayloadTest {

    @Test fun roundTrip_preservesAllFields() {
        val payload = ObjectPayload(
            label = "laptop",
            bbox = "120,80,320,230",
            timestampMs = 171234L,
            trackId = 3,
            thumbPath = "/data/thumbs/obj_3.jpg",
            caption = "",
        )
        val restored = ObjectPayload.fromJson(payload.toJson())
        assertEquals(payload.label, restored.label)
        assertEquals(payload.bbox, restored.bbox)
        assertEquals(payload.timestampMs, restored.timestampMs)
        assertEquals(payload.trackId, restored.trackId)
        assertEquals(payload.thumbPath, restored.thumbPath)
        assertEquals(payload.caption, restored.caption)
    }

    @Test fun toJson_usesExactOnDiskKeys() {
        val payload = ObjectPayload(
            label = "cup",
            bbox = "1,2,3,4",
            timestampMs = 42L,
            trackId = 7,
            thumbPath = "/data/thumbs/obj_7.jpg",
            caption = "",
        )
        val json = JSONObject(payload.toJson())
        assertTrue(json.has("label"))
        assertTrue(json.has("bbox"))
        assertTrue(json.has("timestamp_ms"))
        assertTrue(json.has("track_id"))
        assertTrue(json.has("thumb_path"))
        assertTrue(json.has("caption"))
        // pin exact values too, not just key presence
        assertEquals("cup", json.getString("label"))
        assertEquals("1,2,3,4", json.getString("bbox"))
        assertEquals(42L, json.getLong("timestamp_ms"))
        assertEquals(7, json.getInt("track_id"))
        assertEquals("/data/thumbs/obj_7.jpg", json.getString("thumb_path"))
        assertEquals("", json.getString("caption"))
    }

    @Test fun fromJson_missingKeys_defaultToEmptyStringAndZero() {
        val restored = ObjectPayload.fromJson("{}")
        assertEquals("", restored.label)
        assertEquals("", restored.bbox)
        assertEquals(0L, restored.timestampMs)
        assertEquals(0, restored.trackId)
        assertEquals("", restored.thumbPath)
        assertEquals("", restored.caption)
    }
}
