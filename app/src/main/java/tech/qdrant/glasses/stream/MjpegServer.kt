package tech.qdrant.glasses.stream

import android.content.res.AssetManager
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import java.io.InputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue

/**
 * An InputStream fed by discrete byte[] messages from any number of writer threads.
 * read(b,off,len) blocks until a message is queued, then returns it in one shot.
 *
 * This replaces PipedInputStream/PipedOutputStream, whose sparse-write wakeup is unreliable:
 * NanoHTTPD's sendBody does data.read(buf,0,16384) in a loop and emits one HTTP chunk per read.
 * High-rate MJPEG kept the piped reader cycling so it drained; low-rate SSE events (stored/tick)
 * left the parked reader un-retriggered and sat undelivered (PipedOutputStream.flush() is
 * notify-only — it doesn't push bytes). With a queue, each message → one read → one chunk → one
 * immediate socket flush, for one byte or one megabyte, any number of writers.
 */
private class MessageStream : InputStream() {
    private val queue = LinkedBlockingQueue<ByteArray>()
    private val poison = ByteArray(0)
    @Volatile private var closed = false
    private var current: ByteArray? = null
    private var pos = 0

    fun offer(bytes: ByteArray) { if (!closed) queue.offer(bytes) }

    override fun read(): Int {
        val one = ByteArray(1); val n = read(one, 0, 1)
        return if (n <= 0) -1 else (one[0].toInt() and 0xFF)
    }

    override fun read(b: ByteArray, off: Int, len: Int): Int {
        if (len == 0) return 0
        var cur = current
        if (cur == null || pos >= cur.size) {
            val next = try { queue.take() } catch (e: InterruptedException) {
                Thread.currentThread().interrupt(); return -1
            }
            if (next === poison) return -1
            cur = next; current = cur; pos = 0
        }
        val n = minOf(len, cur.size - pos)
        System.arraycopy(cur, pos, b, off, n)
        pos += n
        if (pos >= cur.size) current = null
        return n
    }

    override fun available(): Int = current?.let { it.size - pos } ?: 0
    override fun close() { closed = true; queue.offer(poison) }
}

/**
 * Minimal MJPEG (multipart/x-mixed-replace) HTTP server.
 *
 * GET /stream  → an endless multipart stream of JPEG frames. This is exactly the
 *                format OpenCV's `cv2.VideoCapture("http://host:PORT/stream")` reads,
 *                so the desktop edge-mission-control pipeline can consume it unchanged.
 * GET /        → a tiny status/landing page.
 *
 * The camera thread pushes the latest JPEG via [offerFrame]; each connected client
 * gets whatever the most recent frame is (no per-client queue — newest wins, lowest
 * latency, which is what a live demo wants).
 */
class MjpegServer(port: Int, private val assets: AssetManager) : NanoHTTPD("0.0.0.0", port) {

    companion object {
        private const val TAG = "MjpegServer"
        private const val BOUNDARY = "frameboundary"
    }

    /** One open client connection — backed by a message queue, not a pipe. */
    private class Client(val stream: MessageStream) {
        @Volatile var alive = true
    }

    private val clients = CopyOnWriteArrayList<Client>()
    private val thumbs = ConcurrentHashMap<String, String>()

    @Volatile private var latestJpeg: ByteArray? = null
    // A standby frame shown when no live camera frame is flowing (app idle / not recording).
    // Without it a freshly-connected browser <img> stays blank-white until the first real
    // frame. Set once by the Activity; offerFrame() takes over the moment recording starts.
    @Volatile private var placeholderJpeg: ByteArray? = null

    /** One already-stored object, replayed into the rail when a HUD connects. */
    data class RailItem(val key: String, val label: String, val thumbPath: String)

    // Supplies the objects already in memory so a freshly-connected /events client rebuilds its
    // rail instead of starting empty (fixes: rail blank after an app/browser restart). Set by the
    // Activity to point at ObjectStore.all(); each item's thumb is registered for /thumb/<key>.
    @Volatile var railSnapshotProvider: (() -> List<RailItem>)? = null

    /** Register a crop thumbnail file for serving via /thumb/<key>. */
    fun registerThumb(key: String, absPath: String) {
        thumbs[key] = absPath
    }

    /** Provide a standby JPEG shown to viewers until live frames start (and after they stop). */
    fun setPlaceholder(jpeg: ByteArray) {
        placeholderJpeg = jpeg
        // Push it to anyone already watching so they leave the blank state immediately.
        if (latestJpeg == null) broadcast(jpeg)
    }

    /** Called from the camera thread with each freshly-encoded JPEG frame. */
    fun offerFrame(jpeg: ByteArray) {
        latestJpeg = jpeg
        broadcast(jpeg)
    }

    private fun broadcast(jpeg: ByteArray) {
        if (clients.isEmpty()) return
        val header = ("--$BOUNDARY\r\n" +
                "Content-Type: image/jpeg\r\n" +
                "Content-Length: ${jpeg.size}\r\n\r\n").toByteArray()
        // One whole multipart part = one queued message = one contiguous chunk.
        val part = header + jpeg + "\r\n".toByteArray()
        for (c in clients) {
            if (!c.alive) { clients.remove(c); continue }
            c.stream.offer(part)
        }
    }

    fun clientCount(): Int = clients.size

    // NEVER gzip any response. NanoHTTPD's send() calls useGzipWhenAccepted() and re-enables gzip
    // whenever the client sends Accept-Encoding: gzip (which browsers and fetch always do),
    // overriding any per-Response setGzipEncoding(false). gzip BUFFERS small inputs and emits
    // nothing until a block fills, so the browser's fetch/ReadableStream on /events and /stream
    // received zero chunks — the rail/counter stayed empty in the browser while curl (no
    // Accept-Encoding) worked. Forcing this false for the whole server fixes both live streams.
    override fun useGzipWhenAccepted(r: Response?): Boolean = false

    /** One open Server-Sent-Events connection (the HUD's /events channel). */
    private val eventClients = CopyOnWriteArrayList<Client>()

    /** Fan out one SSE event line to every connected HUD. Safe to call from any thread. */
    fun pushEvent(line: String) {
        if (eventClients.isEmpty()) return
        // One complete SSE event = one queued message = one HTTP chunk = one socket flush.
        val payload = "data: $line\n\n".toByteArray()
        for (c in eventClients) {
            if (!c.alive) { eventClients.remove(c); continue }
            c.stream.offer(payload)
        }
    }

    fun eventClientCount(): Int = eventClients.size

    /**
     * Push the current object memory into EVERY connected HUD's rail. Called once the ObjectStore
     * finishes its (async, ~10s) init, because a browser typically connects before the store is
     * ready — at connect time the snapshot is empty, so we resend here to fill those early clients.
     * Late clients (connecting after the store is ready) get theirs via [serveEvents]'s replay.
     */
    fun broadcastRailSnapshot() {
        val items = railSnapshotProvider?.invoke() ?: return
        if (items.isEmpty() || eventClients.isEmpty()) return
        val count = items.size.toLong()
        for (it in items) thumbs[it.key] = it.thumbPath
        for (c in eventClients) {
            if (!c.alive) { eventClients.remove(c); continue }
            for (it in items) {
                c.stream.offer("data: ${HudEvents.storedEvent(it.key, it.label, count)}\n\n".toByteArray())
            }
        }
        Log.i(TAG, "broadcast rail snapshot: $count objects → ${eventClients.size} HUD(s)")
    }

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        return when {
            uri == "/stream" -> serveStream()
            uri == "/events" -> serveEvents()
            uri == "/" -> serveAsset("web/index.html", "text/html")
            uri.startsWith("/static/") -> {
                val name = uri.removePrefix("/static/").substringBefore('?')
                serveAsset("web/$name", mimeFor(name))
            }
            uri.startsWith("/thumb/") -> serveThumb(uri.removePrefix("/thumb/").substringBefore('?'))
            else -> newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "not found")
        }
    }

    private fun mimeFor(name: String): String = when {
        name.endsWith(".js")  -> "application/javascript"
        name.endsWith(".css") -> "text/css"
        name.endsWith(".html")-> "text/html"
        name.endsWith(".svg") -> "image/svg+xml"
        else -> "application/octet-stream"
    }

    private fun serveAsset(path: String, mime: String): Response = try {
        val bytes = assets.open(path).use { it.readBytes() }
        newFixedLengthResponse(Response.Status.OK, mime, java.io.ByteArrayInputStream(bytes), bytes.size.toLong())
            // The HUD assets change between builds; never let the browser serve a stale app.js/css.
            .apply { addHeader("Cache-Control", "no-store, must-revalidate") }
    } catch (e: Exception) {
        Log.w(TAG, "asset missing: $path (${e.message})")
        newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "not found")
    }

    private fun serveThumb(key: String): Response {
        val path = thumbs[key] ?: return newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "no thumb")
        val f = java.io.File(path)
        if (!f.exists()) return newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "gone")
        val bytes = f.readBytes()
        return newFixedLengthResponse(Response.Status.OK, "image/jpeg", java.io.ByteArrayInputStream(bytes), bytes.size.toLong())
    }

    /**
     * NanoHTTPD streams a chunked response by handing us a PipedOutputStream via a
     * generator InputStream. Simpler and more robust here: take over the socket with a
     * custom Response whose data is produced by writing directly to the client stream.
     * We use the documented approach: a chunked Response backed by a piped stream that
     * a writer thread feeds. Each registered Client receives [offerFrame] writes.
     */
    private fun serveStream(): Response {
        val stream = MessageStream()
        val client = Client(stream)
        clients.add(client)
        Log.i(TAG, "client connected (${clients.size} total)")

        // Prime the new client immediately so the viewer isn't blank: a live frame if we have
        // one, otherwise the standby placeholder.
        (latestJpeg ?: placeholderJpeg)?.let { jpeg ->
            val header = ("--$BOUNDARY\r\n" +
                    "Content-Type: image/jpeg\r\n" +
                    "Content-Length: ${jpeg.size}\r\n\r\n").toByteArray()
            stream.offer(header + jpeg + "\r\n".toByteArray())
        }

        val resp = newChunkedResponse(
            Response.Status.OK,
            "multipart/x-mixed-replace; boundary=$BOUNDARY",
            stream
        )
        // CRITICAL: never gzip a live stream. NanoHTTPD gzips text-ish responses when the client
        // sends Accept-Encoding: gzip — gzip BUFFERS small inputs and emits nothing until a block
        // fills, so the browser's fetch/ReadableStream sees zero chunks and the stream looks dead.
        // curl works only because it omits Accept-Encoding by default. Disable gzip here.
        resp.addHeader("Cache-Control", "no-cache, private")
        resp.addHeader("Connection", "close")
        return resp
    }

    private fun serveEvents(): Response {
        val stream = MessageStream()
        val client = Client(stream)
        eventClients.add(client)
        Log.i(TAG, "event client connected (${eventClients.size} total)")
        // SSE preamble (a comment line) so the client's reader opens cleanly.
        stream.offer(": connected\n\n".toByteArray())
        // Replay objects already in memory into THIS client's stream so the rail isn't empty after
        // an app/browser restart. Registering each thumb makes /thumb/<key> resolve. Sent only to
        // the connecting client (not broadcast) so reconnecting one HUD doesn't disturb others.
        railSnapshotProvider?.invoke()?.let { items ->
            val count = items.size.toLong()
            for (it in items) {
                thumbs[it.key] = it.thumbPath
                stream.offer("data: ${HudEvents.storedEvent(it.key, it.label, count)}\n\n".toByteArray())
            }
            Log.i(TAG, "replayed $count stored objects to new HUD")
        }
        val resp = newChunkedResponse(Response.Status.OK, "text/event-stream", stream)
        // CRITICAL: never gzip SSE. text/event-stream is "text/*" so NanoHTTPD gzips it when the
        // browser sends Accept-Encoding: gzip; gzip buffers the tiny events and the browser's
        // fetch reader receives nothing. This is THE bug that left the rail/counter empty in the
        // browser while curl (no Accept-Encoding) worked. Disable gzip so events flush immediately.
        resp.addHeader("Cache-Control", "no-cache, private")
        resp.addHeader("Connection", "keep-alive")
        return resp
    }
}
