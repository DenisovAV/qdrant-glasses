package tech.qdrant.glasses.fleet
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
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
}
