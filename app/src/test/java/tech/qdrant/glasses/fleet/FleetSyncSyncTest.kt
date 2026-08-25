package tech.qdrant.glasses.fleet

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import java.io.File

/**
 * Unit-tests [FleetSync.syncOnce] — the UP half's single pass (Spec §5, "flag-on-store") — against
 * a fake [MomentStore] (records `scrollUnsyncedFrames`/`markSynced` calls, no native shard needed)
 * and a REAL [FleetQdrantClient] backed by [MockWebServer], so the wire shape and the
 * confirmed-upsert-BEFORE-flag-flip ordering are actually exercised, not just assumed.
 *
 * Robolectric: [FleetQdrantClient] parses/builds real `org.json.JSONObject`, an unmocked Android
 * stub under plain JVM unit tests (same reason [FleetQdrantClientTest] uses this runner).
 */
@RunWith(RobolectricTestRunner::class)
class FleetSyncSyncTest {

    /** Records every [scrollUnsyncedFrames]/[markSynced] call so a test can assert BOTH "a
     *  confirmed upsert happened" (via the MockWebServer request) AND "the flag flip happened
     *  after it, with exactly the batch's ids" — and, in the failure test, that it did NOT happen
     *  at all. Every other [MomentStore] method is unused by [FleetSync.syncOnce]. */
    private class FakeMomentStore(private val backlog: List<FleetPoint>) : MomentStore {
        val markSyncedCalls = mutableListOf<List<String>>()
        var lastScrollLimit: Int? = null; private set
        override fun storeMoment(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeRegion(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeOcr(textVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = error("not used")
        override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = error("not used")
        override fun searchText(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = error("not used")
        override fun timeline(limit: Int): List<MomentHit> = error("not used")
        override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> = error("not used")
        override fun count(): Long = error("not used")
        override fun frameCount(): Long = error("not used")
        override fun deleteAll() = error("not used")
        override fun scrollUnsyncedFrames(limit: Int): List<FleetPoint> {
            lastScrollLimit = limit
            return backlog.take(limit)
        }
        override fun markSynced(ids: List<String>) { markSyncedCalls += ids }
        override fun close() {}
    }

    private fun client(srv: MockWebServer) = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))

    private fun point(id: String) = FleetPoint(
        id = id,
        vector = floatArrayOf(0.1f, 0.2f, 0.3f),
        payload = """{"label":"cup","timestamp_ms":42}""",
    )

    private fun sync(srv: MockWebServer, store: FakeMomentStore) = FleetSync(
        client(srv), filesDir = File("."), clipDim = 8, momentStore = store, isRecording = { false },
    )

    @Test fun emptyBacklogSyncsNothingAndSendsNoRequest() {
        val srv = MockWebServer(); srv.start()
        val store = FakeMomentStore(backlog = emptyList())

        val n = runBlocking { sync(srv, store).syncOnce() }

        assertEquals(0, n)
        assertEquals(0, srv.requestCount)
        assertTrue(store.markSyncedCalls.isEmpty())
        srv.shutdown()
    }

    @Test fun nonEmptyBacklogUpsertsThenMarksSyncedWithSameIds() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":{"operation_id":1,"status":"completed"},"status":"ok"}"""))
        srv.start()
        val backlog = listOf(
            point("11111111-1111-1111-1111-111111111111"),
            point("22222222-2222-2222-2222-222222222222"),
        )
        val store = FakeMomentStore(backlog = backlog)

        val n = runBlocking { sync(srv, store).syncOnce() }

        assertEquals(2, n)
        assertEquals(1, srv.requestCount)
        val req = srv.takeRequest()
        assertEquals("PUT", req.method)
        assertEquals("/collections/fleet_inbox/points?wait=true", req.path)
        assertEquals(100, store.lastScrollLimit)   // FleetSync's UP_BATCH_SIZE default
        assertEquals(1, store.markSyncedCalls.size)
        assertEquals(backlog.map { it.id }, store.markSyncedCalls[0])
        srv.shutdown()
    }

    // Fail-soft + crash-safety (Spec §5/§7): a failed upsert must NEVER reach markSynced — the
    // batch stays synced=false for retry — and syncOnce degrades to 0 rather than throwing.
    @Test fun failedUpsertNeverMarksSyncedAndReturnsZero() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setResponseCode(500).setBody("boom"))
        srv.start()
        val store = FakeMomentStore(backlog = listOf(point("11111111-1111-1111-1111-111111111111")))

        val n = runBlocking { sync(srv, store).syncOnce() }

        assertEquals(0, n)
        assertTrue(store.markSyncedCalls.isEmpty())
        srv.shutdown()
    }

    @Test fun unreachableServerNeverMarksSyncedAndReturnsZero() {
        val srv = MockWebServer(); srv.start()
        val store = FakeMomentStore(backlog = listOf(point("11111111-1111-1111-1111-111111111111")))
        val badClient = FleetQdrantClient("http://127.0.0.1:1")   // nothing listening — connect refused
        val sync = FleetSync(badClient, filesDir = File("."), clipDim = 8, momentStore = store, isRecording = { false })

        val n = runBlocking { sync.syncOnce() }

        assertEquals(0, n)
        assertTrue(store.markSyncedCalls.isEmpty())
        srv.shutdown()
    }

    @Test fun collectionOverrideIsRespected() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setBody("""{"result":{"operation_id":1,"status":"completed"},"status":"ok"}"""))
        srv.start()
        val store = FakeMomentStore(backlog = listOf(point("11111111-1111-1111-1111-111111111111")))

        runBlocking { sync(srv, store).syncOnce(collection = "fleet_inbox_test") }

        assertEquals("/collections/fleet_inbox_test/points?wait=true", srv.takeRequest().path)
        srv.shutdown()
    }
}
