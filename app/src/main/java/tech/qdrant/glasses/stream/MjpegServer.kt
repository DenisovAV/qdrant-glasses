package tech.qdrant.glasses.stream

import android.content.res.AssetManager
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import java.io.OutputStream
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ConcurrentHashMap

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

    /** One open client connection that we keep writing JPEG parts to. */
    private class Client(val out: OutputStream) {
        @Volatile var alive = true
    }

    private val clients = CopyOnWriteArrayList<Client>()
    private val thumbs = ConcurrentHashMap<String, String>()

    @Volatile private var latestJpeg: ByteArray? = null
    // A standby frame shown when no live camera frame is flowing (app idle / not recording).
    // Without it a freshly-connected browser <img> stays blank-white until the first real
    // frame. Set once by the Activity; offerFrame() takes over the moment recording starts.
    @Volatile private var placeholderJpeg: ByteArray? = null

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
        for (c in clients) {
            if (!c.alive) { clients.remove(c); continue }
            try {
                synchronized(c.out) {
                    c.out.write(header)
                    c.out.write(jpeg)
                    c.out.write("\r\n".toByteArray())
                    c.out.flush()
                }
            } catch (e: Exception) {
                c.alive = false
                clients.remove(c)
                Log.d(TAG, "client dropped: ${e.message}")
            }
        }
    }

    fun clientCount(): Int = clients.size

    /** One open Server-Sent-Events connection (the HUD's /events channel). */
    private val eventClients = CopyOnWriteArrayList<Client>()

    /** Fan out one SSE event line to every connected HUD. Safe to call from any thread. */
    fun pushEvent(line: String) {
        if (eventClients.isEmpty()) return
        val payload = "data: $line\n\n".toByteArray()
        for (c in eventClients) {
            if (!c.alive) { eventClients.remove(c); continue }
            try {
                synchronized(c.out) { c.out.write(payload); c.out.flush() }
            } catch (e: Exception) {
                c.alive = false; eventClients.remove(c)
                Log.d(TAG, "event client dropped: ${e.message}")
            }
        }
    }

    fun eventClientCount(): Int = eventClients.size

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
        val pipeIn = java.io.PipedInputStream(64 * 1024)
        val pipeOut = java.io.PipedOutputStream(pipeIn)
        val client = Client(pipeOut)
        clients.add(client)
        Log.i(TAG, "client connected (${clients.size} total)")

        // Prime the new client immediately so the viewer isn't blank: a live frame if we have
        // one, otherwise the standby placeholder.
        (latestJpeg ?: placeholderJpeg)?.let { jpeg ->
            try {
                val header = ("--$BOUNDARY\r\n" +
                        "Content-Type: image/jpeg\r\n" +
                        "Content-Length: ${jpeg.size}\r\n\r\n").toByteArray()
                synchronized(client.out) {
                    client.out.write(header); client.out.write(jpeg); client.out.write("\r\n".toByteArray()); client.out.flush()
                }
            } catch (_: Exception) {}
        }

        val resp = newChunkedResponse(
            Response.Status.OK,
            "multipart/x-mixed-replace; boundary=$BOUNDARY",
            pipeIn
        )
        resp.addHeader("Cache-Control", "no-cache, private")
        resp.addHeader("Connection", "close")
        return resp
    }

    private fun serveEvents(): Response {
        val pipeIn = java.io.PipedInputStream(64 * 1024)
        val pipeOut = java.io.PipedOutputStream(pipeIn)
        val client = Client(pipeOut)
        eventClients.add(client)
        Log.i(TAG, "event client connected (${eventClients.size} total)")
        // SSE preamble so the browser's EventSource opens cleanly.
        try { synchronized(client.out) { client.out.write(": connected\n\n".toByteArray()); client.out.flush() } } catch (_: Exception) {}
        val resp = newChunkedResponse(Response.Status.OK, "text/event-stream", pipeIn)
        resp.addHeader("Cache-Control", "no-cache, private")
        resp.addHeader("Connection", "keep-alive")
        return resp
    }
}
