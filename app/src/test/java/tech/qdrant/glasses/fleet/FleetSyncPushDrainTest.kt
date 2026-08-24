package tech.qdrant.glasses.fleet

import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Round-2 review regression test for Finding 2 (FleetSync.pushDrain must be single-flight): before
 * that fix, two concurrent [FleetSync.pushDrain] calls could each independently drain()/upsert()/
 * ack() the SAME queued batch, racing their [UploadQueue.ack] writes (see [FleetSync]'s `pushMutex`
 * doc and [UploadQueue]'s class doc for the exact mechanism). This test drives two concurrent
 * [FleetSync.pushDrain] calls against a real (temp-file-backed) [UploadQueue] and a mock server that
 * holds each request open just long enough to make an unguarded race observable, then asserts (a)
 * the server never saw two upsert requests in flight at once (the Mutex actually serialized them)
 * and (b) every enqueued point was eventually drained — none lost, none stuck.
 */
@RunWith(RobolectricTestRunner::class)
class FleetSyncPushDrainTest {

    @Test
    fun concurrentPushDrainCallsAreSingleFlightAndDrainEverything() {
        val srv = MockWebServer()
        val inFlight = AtomicInteger(0)
        val sawOverlap = AtomicBoolean(false)
        srv.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                if (inFlight.incrementAndGet() > 1) sawOverlap.set(true)
                try {
                    // Widen the window so a second, unguarded pushDrain call WOULD overlap this
                    // request if the Mutex weren't actually serializing the two callers.
                    Thread.sleep(80)
                } finally {
                    inFlight.decrementAndGet()
                }
                return MockResponse().setResponseCode(200).setBody("""{"status":"ok"}""")
            }
        }
        srv.start()

        val client = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val dir = kotlin.io.path.createTempDirectory("fleetsync-pushdrain-test").toFile()
        val queue = UploadQueue(File(dir, "fleet_queue.jsonl"))
        repeat(6) { i -> queue.enqueue("id-$i", floatArrayOf(0.1f * i), """{"label":"x$i"}""") }
        val sync = FleetSync(client, dir, clipDim = 1, uploadQueue = queue)

        runBlocking {
            val a = async(Dispatchers.Default) { sync.pushDrain() }
            val b = async(Dispatchers.Default) { sync.pushDrain() }
            awaitAll(a, b)
        }

        assertFalse(
            "pushDrain must be single-flight: two concurrent calls must never have an upsert " +
                "in flight at the same time (Finding 2's Mutex fix)",
            sawOverlap.get(),
        )
        assertEquals(
            "every enqueued point must end up drained — none lost, none left stuck behind a " +
                "racing ack()",
            emptyList<QueuedPoint>(),
            queue.drain(100),
        )
        srv.shutdown()
    }

    /**
     * Round-4 review regression test (blocker + major findings): `thumb_b64` must be read off disk
     * and base64-encoded by [FleetSync.pushDrain] itself — via `thumb_path`, from the payload
     * [MomentCapture] enqueued WITHOUT a thumbnail — not by the capture path. This drives that
     * whole path: enqueue a point whose payload carries only `thumb_path` (the enqueue-time shape
     * after the round-4 fix, see `MomentCaptureUploadTest`), run a real [pushDrain], and assert the
     * bytes actually PUT to the server were read from that file and base64-encoded.
     */
    @Test
    fun pushDrainReadsAndBase64EncodesTheThumbnailFromDiskBeforeUpserting() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setResponseCode(200).setBody("""{"status":"ok"}"""))
        srv.start()

        val dir = kotlin.io.path.createTempDirectory("fleetsync-thumb-test").toFile()
        val thumbBytes = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9)
        val thumbFile = File(dir, "thumb.jpg").apply { writeBytes(thumbBytes) }
        val queue = UploadQueue(File(dir, "fleet_queue.jsonl"))
        val payloadJson = JSONObject().put("thumb_path", thumbFile.absolutePath).put("label", "cup").toString()
        queue.enqueue("id-1", floatArrayOf(0.1f), payloadJson)

        val client = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val sync = FleetSync(client, dir, clipDim = 1, uploadQueue = queue)

        runBlocking { sync.pushDrain() }

        val sentPayload = JSONObject(srv.takeRequest().body.readUtf8())
            .getJSONArray("points").getJSONObject(0).getJSONObject("payload")
        assertEquals(
            "the HTTP-PUT payload's thumb_b64 must be the thumb_path file's bytes, base64-encoded " +
                "(read lazily by FleetSync, off embedLane, right before the PUT)",
            Base64.encodeToString(thumbBytes, Base64.NO_WRAP),
            sentPayload.getString("thumb_b64"),
        )
        assertEquals("cup", sentPayload.getString("label"))
        assertEquals(emptyList<QueuedPoint>(), queue.drain(10))
        srv.shutdown()
    }

    /** Fail-soft companion to the test above: a `thumb_path` that no longer resolves (deleted,
     *  moved, whatever) must degrade that ONE point to `thumb_b64=""`, not fail the whole batch's
     *  upsert (Spec §7). */
    @Test
    fun pushDrainDegradesToEmptyThumbB64WhenTheFileIsMissing() {
        val srv = MockWebServer()
        srv.enqueue(MockResponse().setResponseCode(200).setBody("""{"status":"ok"}"""))
        srv.start()

        val dir = kotlin.io.path.createTempDirectory("fleetsync-thumb-missing-test").toFile()
        val queue = UploadQueue(File(dir, "fleet_queue.jsonl"))
        val payloadJson = JSONObject().put("thumb_path", File(dir, "does-not-exist.jpg").absolutePath).toString()
        queue.enqueue("id-1", floatArrayOf(0.1f), payloadJson)

        val client = FleetQdrantClient(srv.url("/").toString().trimEnd('/'))
        val sync = FleetSync(client, dir, clipDim = 1, uploadQueue = queue)

        runBlocking { sync.pushDrain() }

        val sentPayload = JSONObject(srv.takeRequest().body.readUtf8())
            .getJSONArray("points").getJSONObject(0).getJSONObject("payload")
        assertEquals(
            "a missing thumbnail must degrade this ONE point to an empty thumb_b64, never fail " +
                "the whole batch's upsert",
            "", sentPayload.getString("thumb_b64"),
        )
        assertEquals(emptyList<QueuedPoint>(), queue.drain(10))
        srv.shutdown()
    }
}
