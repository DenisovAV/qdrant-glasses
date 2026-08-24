package tech.qdrant.glasses.fleet

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

// org.json.JSONObject is an unmocked Android stub under plain JVM unit tests (throws on any real
// call) — Robolectric provides the real implementation, same pattern as MomentPayloadTest/FleetQdrantClientTest.
@RunWith(RobolectricTestRunner::class)
class UploadQueueTest {

    private fun tempQueueFile(): File {
        val dir = kotlin.io.path.createTempDirectory("upload-queue-test").toFile()
        return File(dir, "fleet_queue.jsonl")
    }

    @Test fun drainReturnsEnqueuedPointsFifo() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f, 0.2f), """{"label":"cup"}""")
        q.enqueue("b", floatArrayOf(0.3f, 0.4f), """{"label":"mug"}""")
        q.enqueue("c", floatArrayOf(0.5f, 0.6f), """{"label":"bottle"}""")

        val drained = q.drain(max = 10)

        assertEquals(listOf("a", "b", "c"), drained.map { it.id })
        assertEquals(listOf(0.1f, 0.2f), drained[0].clip.toList())
        assertEquals("cup", JSONObject(drained[0].payloadJson).getString("label"))
    }

    @Test fun drainRespectsMax() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")
        q.enqueue("c", floatArrayOf(0.3f), "{}")

        val drained = q.drain(max = 2)

        assertEquals(listOf("a", "b"), drained.map { it.id })
    }

    @Test fun drainDoesNotRemoveUntilAck() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")

        q.drain(max = 10)
        val stillThere = q.drain(max = 10)

        assertEquals(listOf("a"), stillThere.map { it.id })
    }

    @Test fun ackRemovesOnlyAcknowledgedIds() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")
        q.enqueue("c", floatArrayOf(0.3f), "{}")

        q.ack(listOf("a", "c"))

        val remaining = q.drain(max = 10)
        assertEquals(listOf("b"), remaining.map { it.id })
    }

    @Test fun survivesReopen_enqueueThenReopenThenDrain() {
        val file = tempQueueFile()
        UploadQueue(file).enqueue("a", floatArrayOf(0.1f, 0.2f), """{"label":"cup"}""")
        UploadQueue(file).enqueue("b", floatArrayOf(0.3f, 0.4f), """{"label":"mug"}""")

        // Fresh instance over the same file (simulates a process restart) — the queue must be a
        // durable JSONL file on disk, not in-memory state on the UploadQueue object.
        val reopened = UploadQueue(file)
        val drained = reopened.drain(max = 10)

        assertEquals(listOf("a", "b"), drained.map { it.id })
    }

    @Test fun ackThenReopen_ackedPointsStayGone() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")
        q.ack(listOf("a"))

        val reopened = UploadQueue(file)
        val drained = reopened.drain(max = 10)

        assertEquals(listOf("b"), drained.map { it.id })
    }

    @Test fun drainOnMissingFile_returnsEmpty() {
        val file = tempQueueFile()
        assertFalse(file.exists())
        val q = UploadQueue(file)

        assertTrue(q.drain(max = 10).isEmpty())
    }

    @Test fun ackOnMissingFile_doesNotThrow() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.ack(listOf("nonexistent"))
        assertTrue(q.drain(max = 10).isEmpty())
    }

    @Test fun ackDrainingEverything_deletesTheFile() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.ack(listOf("a"))

        assertFalse(file.exists())
    }
}
