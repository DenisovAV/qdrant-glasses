package tech.qdrant.glasses.stream

import android.util.Log
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * WIRELESS FrameSink: pushes frames/events OUTBOUND to the Mac relay via OkHttp POST. The RayNeo
 * firmware blocks inbound TCP to the glasses over WiFi, but outbound is fine (same path the embed
 * encoder already uses), so the glasses push and the browser reads from the Mac. No USB cable.
 *
 * Transport: per-frame multipart POST over OkHttp's persistent connection pool. At 20–27 FPS the
 * pool amortises the TCP handshake; per-frame cost is one HTTP round-trip (~2–5 ms on LAN WiFi) —
 * simpler than a WebSocket, same measured latency at this frame rate. offerFrame uses async
 * enqueue() so it NEVER blocks the caller's streamLane; backpressure stays in the ViewModel's
 * streamBusy gate (this just fires-and-forgets, dropping on failure).
 */
class MjpegPusher(private val relayBaseUrl: String) : FrameSink {

    companion object {
        private const val TAG = "MjpegPusher"
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(2, TimeUnit.SECONDS)
        .writeTimeout(4, TimeUnit.SECONDS)
        .build()

    private val jpegType = "image/jpeg".toMediaType()
    private val jsonType = "application/json".toMediaType()

    // Only MjpegServer's /events replay uses this; the relay handles reconnect replay itself.
    override var railSnapshotProvider: (() -> List<MjpegServer.RailItem>)? = null
    // Same story as railSnapshotProvider just above, for moments (F2, whole-branch review fix):
    // only MjpegServer's /events replay reads this.
    override var momentSnapshotProvider: (() -> List<MjpegServer.MomentItem>)? = null

    /** Fire-and-forget async POST — returns immediately so streamLane isn't held on the network. */
    override fun offerFrame(jpeg: ByteArray) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("frame", "frame.jpg", jpeg.toRequestBody(jpegType))
            .build()
        val req = Request.Builder().url("$relayBaseUrl/push_frame").post(body).build()
        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.w(TAG, "push_frame: ${e.message}") }
            override fun onResponse(call: Call, response: Response) {
                response.use { if (!it.isSuccessful) Log.w(TAG, "push_frame ${it.code}") }
            }
        })
    }

    /** HUD events are rare (<1/s); same async POST. Fanned out to browsers by the relay. */
    override fun pushEvent(line: String) {
        val req = Request.Builder().url("$relayBaseUrl/push_event")
            .post(line.toRequestBody(jsonType)).build()
        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.w(TAG, "push_event: ${e.message}") }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }

    /**
     * Ship the crop JPEG bytes to the Mac so it can serve /thumb/<key> (the Mac has no access to
     * the glasses filesystem — [absPath] is only valid on-device). Fire-and-forget, same as
     * offerFrame/pushEvent: a dropped thumb just leaves one rail card broken, not fatal.
     */
    override fun registerThumb(key: String, absPath: String) {
        val bytes = try { java.io.File(absPath).readBytes() } catch (e: Exception) {
            Log.w(TAG, "registerThumb: read failed for $absPath: ${e.message}"); return
        }
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("thumb", "$key.jpg", bytes.toRequestBody(jpegType))
            .build()
        val req = Request.Builder().url("$relayBaseUrl/push_thumb/$key").post(body).build()
        client.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.w(TAG, "push_thumb: ${e.message}") }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }

    /**
     * The glasses are a CLIENT — they can't know when a browser opens /events on the Mac. So
     * instead of replaying per-connection (which is the Mac's job, since only it has connections),
     * push the whole known rail ONCE via ordinary `stored` events. The Mac remembers the latest
     * `stored` event per id (a dict, not just a fan-out queue) and replays that dict to every new
     * /events client — see embed_server.py's _rail_state. Called once from GlassesViewModel.init
     * after objectStore finishes loading (~10s), which is the only place the full set is known.
     */
    override fun broadcastRailSnapshot() {
        val items = railSnapshotProvider?.invoke() ?: return
        if (items.isEmpty()) return
        val count = items.size.toLong()
        for (it in items) {
            registerThumb(it.key, it.thumbPath)
            pushEvent(HudEvents.storedEvent(it.key, it.label, count))
        }
        Log.i(TAG, "broadcastRailSnapshot: pushed $count objects to relay")
    }

    override fun close() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
}
