package tech.qdrant.glasses.fleet
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
// org.json.JSONObject is an unmocked Android stub under plain JVM unit tests (throws on any real
// call) — Robolectric provides the real implementation, same pattern as MomentPayloadTest/HudEventsTest.
@RunWith(RobolectricTestRunner::class)
class FleetQdrantClientTest {
    @Test fun createReturnsSnapshotName() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":{"name":"snap-1.snapshot"},"status":"ok"}"""))
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        assertEquals("snap-1.snapshot", c.createShardSnapshot("fleet_curated"))
        val req = srv.takeRequest()
        assertEquals("POST", req.method)
        assertEquals("/collections/fleet_curated/shards/0/snapshots", req.path)
        srv.shutdown()
    }
    @Test fun downloadWritesFile() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("SNAPSHOT-BYTES"))
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val f = File.createTempFile("snap", ".snapshot")
        c.downloadSnapshot("fleet_curated", 0, "snap-1.snapshot", f)
        assertEquals("SNAPSHOT-BYTES", f.readText())
        assertEquals("/collections/fleet_curated/shards/0/snapshots/snap-1.snapshot", srv.takeRequest().path)
        srv.shutdown()
    }
    @Test fun deleteSnapshotSendsDelete() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":true,"status":"ok"}"""))
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        c.deleteSnapshot("fleet_curated", 0, "snap-1.snapshot")
        val req = srv.takeRequest()
        assertEquals("DELETE", req.method)
        assertEquals("/collections/fleet_curated/shards/0/snapshots/snap-1.snapshot", req.path)
        srv.shutdown()
    }
    @Test fun upsertPointsSendsPutBatch() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":{"operation_id":1,"status":"completed"},"status":"ok"}"""))
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val points = listOf(
            FleetPoint(
                id = "11111111-1111-1111-1111-111111111111",
                vector = floatArrayOf(0.1f, 0.2f, 0.3f),
                payload = """{"label":"cup","timestamp_ms":42}""",
            ),
            FleetPoint(
                id = "22222222-2222-2222-2222-222222222222",
                vector = floatArrayOf(0.4f, 0.5f, 0.6f),
                payload = """{"label":"mug","timestamp_ms":43}""",
            ),
        )
        c.upsertPoints("fleet_inbox", points)
        val req = srv.takeRequest()
        assertEquals("PUT", req.method)
        assertEquals("/collections/fleet_inbox/points?wait=true", req.path)
        val body = JSONObject(req.body.readUtf8())
        val pointsJson = body.getJSONArray("points")
        assertEquals(2, pointsJson.length())
        val p0 = pointsJson.getJSONObject(0)
        assertEquals("11111111-1111-1111-1111-111111111111", p0.getString("id"))
        assertEquals("cup", p0.getJSONObject("payload").getString("label"))
        assertEquals(42, p0.getJSONObject("payload").getLong("timestamp_ms"))
        val vec = p0.getJSONObject("vector").getJSONArray("clip")
        assertEquals(3, vec.length())
        assertEquals(0.1, vec.getDouble(0), 1e-6)
        assertEquals(0.2, vec.getDouble(1), 1e-6)
        assertEquals(0.3, vec.getDouble(2), 1e-6)
        val p1 = pointsJson.getJSONObject(1)
        assertEquals("22222222-2222-2222-2222-222222222222", p1.getString("id"))
        assertEquals("mug", p1.getJSONObject("payload").getString("label"))
        srv.shutdown()
    }
    @Test fun upsertPointsEmptyListSendsNoRequest() {
        val srv = MockWebServer()
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        c.upsertPoints("fleet_inbox", emptyList())
        assertEquals(0, srv.requestCount)
        srv.shutdown()
    }
    // Regression: an unparseable payload must never be silently substituted with `{}` and upserted —
    // a "successful" upsert of empty JSON would get the local point wrongly flagged synced=true
    // despite its real data never reaching the hub. The call must throw and send NO request at all,
    // leaving every point in the batch (including the well-formed one) synced=false for retry.
    @Test fun upsertPointsWithUnparseablePayloadThrowsAndSendsNoRequest() {
        val srv = MockWebServer()
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val points = listOf(
            FleetPoint(
                id = "11111111-1111-1111-1111-111111111111",
                vector = floatArrayOf(0.1f, 0.2f, 0.3f),
                payload = "{not valid json",
            ),
            FleetPoint(
                id = "22222222-2222-2222-2222-222222222222",
                vector = floatArrayOf(0.4f, 0.5f, 0.6f),
                payload = """{"label":"mug","timestamp_ms":43}""",
            ),
        )
        try {
            c.upsertPoints("fleet_inbox", points)
            org.junit.Assert.fail("expected IllegalArgumentException for unparseable payload")
        } catch (_: IllegalArgumentException) {
            // expected
        }
        assertEquals(0, srv.requestCount)
        srv.shutdown()
    }
}
