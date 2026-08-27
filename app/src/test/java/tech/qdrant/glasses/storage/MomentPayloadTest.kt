package tech.qdrant.glasses.storage

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
            thumbB64 = "/9j/4AAQSkZJRgABAQ==",
            synced = true,
        )
        val restored = MomentPayload.fromJson(payload.toJson())
        assertEquals(payload.thumbB64, restored.thumbB64)
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
        assertEquals(payload.synced, restored.synced)
        assertEquals(payload, restored)
    }

    @Test fun synced_defaultsFalse_whenOmittedFromConstructor() {
        val payload = MomentPayload(
            type = "frame", momentId = "moment-def", episodeId = 1L, timestampMs = 1L, tEndMs = 1L,
            thumbPath = "", bbox = "", label = "", yoloConf = 0f, verifyCos = 0f, text = "",
        )
        assertFalse(payload.synced)
        assertFalse(MomentPayload.fromJson(payload.toJson()).synced)
    }

    @Test fun synced_falseRoundTrips_notJustTrue() {
        val payload = MomentPayload(
            type = "frame", momentId = "moment-ghi", episodeId = 1L, timestampMs = 1L, tEndMs = 1L,
            thumbPath = "", bbox = "", label = "", yoloConf = 0f, verifyCos = 0f, text = "",
            synced = false,
        )
        val json = JSONObject(payload.toJson())
        assertTrue(json.has("synced"))
        assertFalse(json.getBoolean("synced"))
        assertFalse(MomentPayload.fromJson(payload.toJson()).synced)
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
        assertTrue(json.has("thumb_b64"))
        assertTrue(json.has("synced"))
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
        assertFalse(restored.synced)
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
        assertEquals("", restored.thumbB64)
        assertFalse(restored.synced)
    }

    @Test fun thumbB64_defaultsEmpty_andReadsBackUnchangedFromOldPayloads() {
        // A frame that carries no fleet thumb (regions, OCR, and every pre-Phase-4 point) leaves
        // thumb_b64 == "" — and a stored payload written BEFORE this field existed (no thumb_b64 key)
        // must still read back as "" (optString default), never a crash.
        val noThumb = MomentPayload(
            type = "frame", momentId = "m", episodeId = 1L, timestampMs = 1L, tEndMs = 1L,
            thumbPath = "", bbox = "", label = "", yoloConf = 0f, verifyCos = 0f, text = "",
        )
        assertEquals("", noThumb.thumbB64)
        assertEquals("", MomentPayload.fromJson(noThumb.toJson()).thumbB64)
        // legacy on-disk JSON with every OTHER key but no thumb_b64:
        val legacy = """{"type":"frame","moment_id":"m","timestamp_ms":5,"thumb_path":"/x.jpg"}"""
        assertEquals("", MomentPayload.fromJson(legacy).thumbB64)
    }
}
