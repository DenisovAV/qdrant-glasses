package tech.qdrant.glasses.stream

import android.util.Log
import fi.iki.elonen.NanoHTTPD
import java.io.OutputStream
import java.util.concurrent.CopyOnWriteArrayList

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
class MjpegServer(port: Int) : NanoHTTPD("0.0.0.0", port) {

    companion object {
        private const val TAG = "MjpegServer"
        private const val BOUNDARY = "frameboundary"
    }

    /** One open client connection that we keep writing JPEG parts to. */
    private class Client(val out: OutputStream) {
        @Volatile var alive = true
    }

    private val clients = CopyOnWriteArrayList<Client>()

    @Volatile private var latestJpeg: ByteArray? = null

    /** Called from the camera thread with each freshly-encoded JPEG frame. */
    fun offerFrame(jpeg: ByteArray) {
        latestJpeg = jpeg
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

    override fun serve(session: IHTTPSession): Response {
        return when (session.uri) {
            "/stream" -> serveStream()
            else -> newFixedLengthResponse(
                Response.Status.OK, "text/html",
                "<html><body style='font-family:sans-serif'>" +
                        "<h2>Glasses camera stream</h2>" +
                        "<p>MJPEG: <a href='/stream'>/stream</a></p>" +
                        "<img src='/stream' style='max-width:100%'/>" +
                        "</body></html>"
            )
        }
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

        // Prime the new client with the latest frame immediately so the viewer isn't blank.
        latestJpeg?.let { jpeg ->
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
}
