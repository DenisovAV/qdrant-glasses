package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.fleet.UploadQueue
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import java.io.File

/**
 * Drives the REAL [MomentCapture] (not just [buildUploadPayloadJson] in isolation) through one
 * full capture — [MomentCapture.onFrame] twice, arming and closing the sharpness-selection window
 * — with FAKE [CropEncoder]/[MomentStore] deps (both interfaces, same no-mocking-framework style
 * as `MomentSearcherTest`) and a REAL, temp-file-backed [UploadQueue] (Robolectric-backed, same as
 * `UploadQueueTest`). Covers plan Task 10's two required assertions directly: storing a moment
 * enqueues exactly one upload point (id + clip vector + payload incl. `sync_ts`/`thumb_b64`), and
 * the LOCAL store still keeps its own point afterward (decision C — no delete).
 *
 * [scope]/[embedLane] both use [Dispatchers.Unconfined]: neither [MomentCapture.process] nor
 * [MomentCapture.confirmAndStore] ever suspends, so an Unconfined-dispatched `launch` runs to
 * completion synchronously within the call that posted it — no need to await/join a background
 * coroutine to observe the store+enqueue side effects below. The injected [nowMs] clock (a mutable
 * local, not the wall clock) is what actually drives the gate/window timing deterministically.
 */
@RunWith(RobolectricTestRunner::class)
class MomentCaptureUploadTest {

    private class FakeCropEncoder(override val dim: Int = 4) : CropEncoder {
        override fun encode(crop: Bitmap): FloatArray = FloatArray(dim) { 0.5f }
        override fun encodeText(query: String): FloatArray = error("not used by this test")
        override val visionMinScore: Float = 0f
    }

    /** Records every [storeMoment] call; [deleteAll] is never invoked by [MomentCapture] itself —
     *  there IS no "delete one point" method on [MomentStore] for it to call (decision C is, in
     *  part, structurally enforced by this interface's shape). */
    private class FakeMomentStore : MomentStore {
        val stored = mutableListOf<Pair<FloatArray, MomentPayload>>()
        private var nextId = 0
        override fun storeMoment(clipVec: FloatArray, payload: MomentPayload): String {
            val id = "moment-${nextId++}"
            stored += clipVec to payload
            return id
        }
        override fun storeRegion(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeOcr(textVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = emptyList()
        override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = emptyList()
        override fun searchText(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?): List<MomentHit> = emptyList()
        override fun timeline(limit: Int): List<MomentHit> = emptyList()
        override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> = emptyList()
        override fun count(): Long = stored.size.toLong()
        override fun frameCount(): Long = stored.size.toLong()
        override fun deleteAll() { stored.clear() }
        override fun close() {}
    }

    private fun freshDir(prefix: String): File = kotlin.io.path.createTempDirectory(prefix).toFile()

    /** Arms the sharpness window (first-ever frame → unconditional CAPTURE, [decide]'s
     *  `prevGrid == null` case) then closes it past [SELECT_WINDOW_MS] so [MomentCapture]
     *  actually runs [MomentCapture.confirmAndStore] — the store+enqueue side effects under
     *  test only happen once the window closes, not on the arming frame alone. */
    private fun captureOneMoment(capture: MomentCapture, clockRef: LongArray) {
        capture.startSession(clockRef[0])
        capture.onFrame(Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888))
        clockRef[0] += 900L   // past SELECT_WINDOW_MS(800ms)
        capture.onFrame(Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888))
    }

    @Test fun storingAMomentEnqueuesOneUploadPointAndKeepsTheLocalPoint() {
        val store = FakeMomentStore()
        val thumbsDir = freshDir("moment-thumbs").also { it.mkdirs() }
        val queue = UploadQueue(File(freshDir("fleet-queue"), "fleet_queue.jsonl"))
        val clockRef = longArrayOf(10_000L)

        val capture = MomentCapture(
            scope = CoroutineScope(Dispatchers.Unconfined),
            embedLane = Dispatchers.Unconfined,
            cropEncoder = FakeCropEncoder(),
            store = store,
            momentThumbsDir = thumbsDir,
            isRecording = { true },
            uploadQueue = queue,
            fleetLane = Dispatchers.Unconfined,   // enqueue now runs on scope.launch(fleetLane); Unconfined → synchronous, observable
            nowMs = { clockRef[0] },
        )

        captureOneMoment(capture, clockRef)

        assertEquals("local point kept (decision C)", 1, store.stored.size)
        val (storedVec, storedPayload) = store.stored[0]
        assertEquals("frame", storedPayload.type)

        val drained = queue.drain(max = 10)
        assertEquals("exactly one upload point enqueued", 1, drained.size)
        assertEquals("same clip vector as the local store", storedVec.toList(), drained[0].clip.toList())

        val uploadPayload = JSONObject(drained[0].payloadJson)
        assertEquals("frame", uploadPayload.getString("type"))
        assertTrue("sync_ts present", uploadPayload.has("sync_ts"))
        assertTrue("thumb_b64 present", uploadPayload.has("thumb_b64"))
        // Round-1 regression: the upload payload's moment_id must match the enqueued point's OWN
        // id (Spec §6 frame invariant) — framePayload is built with a momentId="" placeholder
        // BEFORE storeMoment stamps its own internal copy, so reusing it unstamped silently
        // shipped moment_id:"" to fleet_inbox on every upload until this was fixed.
        assertEquals("upload payload moment_id must equal the point's own id",
            drained[0].id, uploadPayload.getString("moment_id"))
    }

    @Test fun nullUploadQueueSkipsTheUploadSideButStillStoresLocally() {
        val store = FakeMomentStore()
        val thumbsDir = freshDir("moment-thumbs-2").also { it.mkdirs() }
        val clockRef = longArrayOf(10_000L)

        val capture = MomentCapture(
            scope = CoroutineScope(Dispatchers.Unconfined),
            embedLane = Dispatchers.Unconfined,
            cropEncoder = FakeCropEncoder(),
            store = store,
            momentThumbsDir = thumbsDir,
            isRecording = { true },
            uploadQueue = null,   // Config.FLEET_URL unset (Global Constraint) — GlassesComponents' default shape
            nowMs = { clockRef[0] },
        )

        captureOneMoment(capture, clockRef)

        assertEquals("uploadQueue=null only skips the upload side, not the local store", 1, store.stored.size)
    }
}
