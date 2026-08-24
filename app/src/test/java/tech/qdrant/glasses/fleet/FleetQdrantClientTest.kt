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
    @Test fun upsertPointsPutsBatch() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":{"status":"acknowledged"},"status":"ok"}"""))
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val point = QueuedPoint(id = "m1", clip = floatArrayOf(0.1f, 0.2f), payloadJson = """{"label":"cup"}""")
        c.upsertPoints("fleet_inbox", listOf(point))
        val req = srv.takeRequest()
        assertEquals("PUT", req.method)
        assertEquals("/collections/fleet_inbox/points?wait=true", req.path)
        val body = JSONObject(req.body.readUtf8())
        val points = body.getJSONArray("points")
        assertEquals(1, points.length())
        val p0 = points.getJSONObject(0)
        assertEquals("m1", p0.getString("id"))
        assertEquals("cup", p0.getJSONObject("payload").getString("label"))
        val vec = p0.getJSONObject("vector").getJSONArray("clip")
        assertEquals(0.1, vec.getDouble(0), 1e-6)
        assertEquals(0.2, vec.getDouble(1), 1e-6)
        srv.shutdown()
    }
    @Test fun upsertPointsSkipsRequestWhenEmpty() {
        val srv = MockWebServer()
        srv.start()
        val c = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        c.upsertPoints("fleet_inbox", emptyList())
        assertEquals(0, srv.requestCount)
        srv.shutdown()
    }
}
