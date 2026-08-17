package tech.qdrant.glasses.stream

/**
 * Where the ViewModel sends live frames + HUD events. Two implementations:
 *  - [MjpegServer]  — the WIRED path: an on-glasses NanoHTTPD server the browser reaches over USB
 *                     `adb forward` (RayNeo firmware blocks inbound TCP over WiFi, so it can't be
 *                     reached wirelessly).
 *  - [MjpegPusher]  — the WIRELESS path: pushes frames/events OUTBOUND to a Mac relay (outbound is
 *                     NOT blocked), which serves the browser. No cable.
 *
 * MainActivity picks one at startup; the ViewModel targets this interface and is oblivious to which.
 */
interface FrameSink : AutoCloseable {
    fun offerFrame(jpeg: ByteArray)
    fun pushEvent(line: String)
    fun registerThumb(key: String, absPath: String)
    fun broadcastRailSnapshot()
    var railSnapshotProvider: (() -> List<MjpegServer.RailItem>)?
    // Moment-timeline analogue of [railSnapshotProvider] (F2, whole-branch review fix): replayed
    // per-connection by MjpegServer.serveEvents() the same way railSnapshotProvider already is, but
    // through HudEvents.momentEvent instead of storedEvent — see MjpegServer.MomentItem's KDoc for
    // why the two can't share one shape. MjpegPusher never reads this: the Mac relay owns reconnect
    // replay for the wireless path itself, same as railSnapshotProvider there.
    var momentSnapshotProvider: (() -> List<MjpegServer.MomentItem>)?
}
