package tech.qdrant.glasses.storage

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class MomentPayloadTest {

    @Test fun roundTrip_preservesAllFields() {
        val payload = MomentPayload(
            type = "region",
            momentId = "moment-abc",
            episodeId = 1723900000000L,
            timestampMs = 1723900123456L,
            tEndMs = 1723900123456L,
            thumbPath = "/data/thumbs/moment_abc.jpg",
            bbox = "120,80,320,230",
            label = "cup",
            yoloConf = 0.73f,
            verifyCos = 0.31f,
            text = "",
        )
        val restored = MomentPayload.fromJson(payload.toJson())
        assertEquals(payload.type, restored.type)
        assertEquals(payload.momentId, restored.momentId)
        assertEquals(payload.episodeId, restored.episodeId)
        assertEquals(payload.timestampMs, restored.timestampMs)
        assertEquals(payload.tEndMs, restored.tEndMs)
        assertEquals(payload.thumbPath, restored.thumbPath)
        assertEquals(payload.bbox, restored.bbox)
        assertEquals(payload.label, restored.label)
        assertEquals(payload.yoloConf, restored.yoloConf, 1e-6f)
        assertEquals(payload.verifyCos, restored.verifyCos, 1e-6f)
        assertEquals(payload.text, restored.text)
        assertEquals(payload, restored)
    }

    @Test fun toJson_usesExactOnDiskKeys() {
        val payload = MomentPayload(
            type = "frame",
            momentId = "moment-xyz",
            episodeId = 42L,
            timestampMs = 100L,
            tEndMs = 100L,
            thumbPath = "/data/thumbs/moment_xyz.jpg",
            bbox = "",
            label = "",
            yoloConf = 0f,
            verifyCos = 0f,
            text = "",
        )
        val json = JSONObject(payload.toJson())
        assertTrue(json.has("type"))
        assertTrue(json.has("moment_id"))
        assertTrue(json.has("episode_id"))
        assertTrue(json.has("timestamp_ms"))
        assertTrue(json.has("t_end_ms"))
        assertTrue(json.has("thumb_path"))
        assertTrue(json.has("bbox"))
        assertTrue(json.has("label"))
        assertTrue(json.has("yolo_conf"))
        assertTrue(json.has("verify_cos"))
        assertTrue(json.has("text"))
        // pin exact values too, not just key presence
        assertEquals("frame", json.getString("type"))
        assertEquals("moment-xyz", json.getString("moment_id"))
        assertEquals(42L, json.getLong("episode_id"))
        assertEquals(100L, json.getLong("timestamp_ms"))
        assertEquals(100L, json.getLong("t_end_ms"))
        assertEquals("/data/thumbs/moment_xyz.jpg", json.getString("thumb_path"))
    }

    @Test fun fromJson_malformedJson_fallsBackToDefaults() {
        val restored = MomentPayload.fromJson("{ not valid json ]")
        assertEquals("", restored.type)
        assertEquals("", restored.momentId)
        assertEquals(0L, restored.episodeId)
        assertEquals(0L, restored.timestampMs)
        assertEquals(0L, restored.tEndMs)
        assertEquals("", restored.thumbPath)
        assertEquals("", restored.bbox)
        assertEquals("", restored.label)
        assertEquals(0f, restored.yoloConf, 1e-6f)
        assertEquals(0f, restored.verifyCos, 1e-6f)
        assertEquals("", restored.text)
    }

    @Test fun fromJson_missingKeys_defaultToEmptyStringAndZero() {
        val restored = MomentPayload.fromJson("{}")
        assertEquals("", restored.type)
        assertEquals("", restored.momentId)
        assertEquals(0L, restored.episodeId)
        assertEquals(0L, restored.timestampMs)
        assertEquals(0L, restored.tEndMs)
        assertEquals("", restored.thumbPath)
        assertEquals("", restored.bbox)
        assertEquals("", restored.label)
        assertEquals(0f, restored.yoloConf, 1e-6f)
        assertEquals(0f, restored.verifyCos, 1e-6f)
        assertEquals("", restored.text)
    }
}
