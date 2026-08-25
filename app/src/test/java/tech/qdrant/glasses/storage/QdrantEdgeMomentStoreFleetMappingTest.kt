package tech.qdrant.glasses.storage

import io.qdrant.edge.PointId
import io.qdrant.edge.Record
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * [recordToFleetPoint]/[stripSyncedPayload] are the pure `Record -> FleetPoint?` mapping behind
 * [QdrantEdgeMomentStore.scrollUnsyncedFrames] (Spec §5/§6) — factored to file scope precisely so
 * this JSON/id/vector parsing is testable without a live native `EdgeShard`. [Record] and [PointId]
 * are plain UniFFI data classes (no native init in their constructors), so they're safe to build
 * directly in a JVM test.
 *
 * Review regression (Task 2 fix): a null or malformed payload used to be silently coerced into a
 * syntactically valid but EMPTY `"{}"`, producing a degraded [FleetPoint] the upload client would
 * happily accept — after which the point could get flagged `synced=true` despite its real
 * [MomentPayload] never reaching the fleet hub. The fix is to skip (return null) instead, so the
 * point stays `synced=false` and is retried next idle pass.
 */
@RunWith(RobolectricTestRunner::class)
class QdrantEdgeMomentStoreFleetMappingTest {

    private val id = "11111111-1111-1111-1111-111111111111"
    private val vectorJson = """{"clip":[0.1,0.2,0.3]}"""
    private val validPayload = """{"type":"frame","moment_id":"$id","synced":false}"""

    private fun record(payload: String?, vector: String? = vectorJson, pointId: PointId? = PointId.Uuid(id)) =
        Record(id = pointId ?: PointId.Uuid(id), payload = payload, vector = vector)

    // --- the review bug: null/malformed payload must SKIP, never fall back to "{}" ---

    @Test fun nullPayload_isSkipped_notCoercedToEmptyObject() {
        val result = recordToFleetPoint(record(payload = null), clipField = "clip", tag = "test")
        assertNull(result)
    }

    @Test fun malformedPayload_isSkipped_notCoercedToEmptyObject() {
        val result = recordToFleetPoint(record(payload = "{not valid json"), clipField = "clip", tag = "test")
        assertNull(result)
    }

    @Test fun stripSyncedPayload_returnsNull_forNullInput() {
        assertNull(stripSyncedPayload(null))
    }

    @Test fun stripSyncedPayload_returnsNull_forMalformedInput() {
        assertNull(stripSyncedPayload("{not valid json"))
    }

    // --- the happy path: still works, `synced` still stripped ---

    @Test fun validPayload_isMappedAndSyncedIsStripped() {
        val result = recordToFleetPoint(record(payload = validPayload), clipField = "clip", tag = "test")
        assertTrue(result != null)
        val payloadJson = JSONObject(result!!.payload)
        assertFalse("synced must be stripped before upload (Spec §6, LOCAL-only)", payloadJson.has("synced"))
        assertEquals("frame", payloadJson.getString("type"))
        assertEquals(id, result.id)
        assertEquals(3, result.vector.size)
    }

    @Test fun stripSyncedPayload_removesOnlySyncedKey() {
        val stripped = stripSyncedPayload("""{"type":"frame","synced":true,"label":"cup"}""")
        assertTrue(stripped != null)
        val o = JSONObject(stripped!!)
        assertFalse(o.has("synced"))
        assertEquals("frame", o.getString("type"))
        assertEquals("cup", o.getString("label"))
    }

    // --- pre-existing skip cases, unaffected by this fix ---

    @Test fun missingUuidId_isSkipped() {
        val result = recordToFleetPoint(
            record(payload = validPayload, pointId = PointId.NumId(7uL)),
            clipField = "clip", tag = "test",
        )
        assertNull(result)
    }

    @Test fun missingVector_isSkipped() {
        val result = recordToFleetPoint(record(payload = validPayload, vector = null), clipField = "clip", tag = "test")
        assertNull(result)
    }

    @Test fun unparseableVector_isSkipped() {
        val result = recordToFleetPoint(
            record(payload = validPayload, vector = "not json"),
            clipField = "clip", tag = "test",
        )
        assertNull(result)
    }
}
