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

    // --- Review fixes: fail-soft enqueue, non-blocking/concurrency, atomic ack, bounded growth ---

    @Test fun enqueueWithMalformedPayloadIsDroppedNotThrown() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("bad", floatArrayOf(0.2f), "not-json{{{")   // must not throw, must not corrupt the file
        q.enqueue("b", floatArrayOf(0.3f), "{}")

        val drained = q.drain(max = 10)

        assertEquals(listOf("a", "b"), drained.map { it.id })
    }

    @Test fun enqueueDuringAConcurrentAckIsNotLostOrDeadlocked() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")

        val ackThread = Thread { q.ack(listOf("a")) }
        ackThread.start()
        // A point enqueued while ack() is mid-rewrite must land in the file, not be silently
        // dropped by ack()'s swap (review fix — ack() folds in anything appended after its snapshot).
        q.enqueue("b", floatArrayOf(0.2f), "{}")
        ackThread.join(5_000)
        assertFalse("ack() thread should have finished", ackThread.isAlive)

        assertEquals(listOf("b"), q.drain(max = 10).map { it.id })
    }

    @Test fun ackLeavesNoOrphanedTempFileBehind() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")

        q.ack(listOf("a"))

        val tmp = File(file.parentFile, file.name + ".tmp")
        assertFalse(tmp.exists())
        assertEquals(listOf("b"), q.drain(max = 10).map { it.id })
    }

    @Test fun enqueueDropsOnceQueueIsFull() {
        val file = tempQueueFile()
        val q = UploadQueue(file, maxEntries = 2)
        q.enqueue("a", floatArrayOf(0f), "{}")
        q.enqueue("b", floatArrayOf(0f), "{}")
        q.enqueue("c", floatArrayOf(0f), "{}")   // over the cap -> dropped, not queued

        assertEquals(listOf("a", "b"), q.drain(max = 10).map { it.id })
    }

    @Test fun enqueueHasRoomAgainAfterAckFreesSpace() {
        val file = tempQueueFile()
        val q = UploadQueue(file, maxEntries = 2)
        q.enqueue("a", floatArrayOf(0f), "{}")
        q.enqueue("b", floatArrayOf(0f), "{}")
        q.ack(listOf("a"))
        q.enqueue("c", floatArrayOf(0f), "{}")   // room again after ack freed a slot

        assertEquals(listOf("b", "c"), q.drain(max = 10).map { it.id })
    }

    @Test fun ackReturnsTrueOnSuccessfulRewrite() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")

        assertTrue("ack() must report success so callers (FleetSync.pushDrain) can trust it",
            q.ack(listOf("a")))
        assertEquals(listOf("b"), q.drain(max = 10).map { it.id })
    }

    @Test fun ackReturnsFalseInsteadOfSilentlySucceedingWhenTheRewriteFails() {
        val file = tempQueueFile()
        val q = UploadQueue(file)
        q.enqueue("a", floatArrayOf(0.1f), "{}")
        q.enqueue("b", floatArrayOf(0.2f), "{}")

        // Force the durable rewrite in ack() to fail: replace the live queue file with a DIRECTORY
        // at the same path, so writeAtomic's tmp-file write/rename can't complete. Round-1 fix:
        // ack() must now REPORT this failure (false) rather than the old Unit-returning version,
        // which let a failed rewrite look identical to a successful one to FleetSync.pushDrain (see
        // its "no-progress guard" fix) — and must not throw back up to the caller either.
        file.delete()
        file.mkdir()

        assertFalse("a failed rewrite must be reported, not silently treated as success",
            q.ack(listOf("a")))
    }

    @Test fun capIsRespectedAcrossReopen() {
        val file = tempQueueFile()
        val q1 = UploadQueue(file, maxEntries = 2)
        q1.enqueue("a", floatArrayOf(0f), "{}")
        q1.enqueue("b", floatArrayOf(0f), "{}")

        // A fresh instance (simulating a process restart) must seed its in-memory count from the
        // file it inherits, not start back at zero — otherwise a restart would silently blow the cap.
        val q2 = UploadQueue(file, maxEntries = 2)
        q2.enqueue("c", floatArrayOf(0f), "{}")   // still over cap -> dropped

        assertEquals(listOf("a", "b"), q2.drain(max = 10).map { it.id })
    }
}
