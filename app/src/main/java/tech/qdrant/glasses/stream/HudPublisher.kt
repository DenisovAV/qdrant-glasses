package tech.qdrant.glasses.stream

/**
 * Wraps the late-bound [FrameSink]: the HUD (browser dashboard) attaches after the ViewModel is
 * already running, and may never attach at all (no HUD client connected). Every method below is a
 * no-op when detached (`sink == null`) — same `?.` semantics as the field it replaces.
 */
class HudPublisher(
    private val railItems: () -> List<MjpegServer.RailItem>,
    // Moment-timeline analogue of [railItems] (F2, whole-branch review fix): supplies the moments
    // already in memory so [attach] can wire a per-connection replay for the timeline the same way
    // it already does for the object rail — see FrameSink.momentSnapshotProvider's KDoc for why
    // this is a SEPARATE provider rather than [railItems] reused.
    private val momentItems: () -> List<MjpegServer.MomentItem>,
) {
    @Volatile private var sink: FrameSink? = null      // fully private — nothing reads the raw sink
    val hasClient: Boolean get() = sink != null        // Runs on: any thread (perception's drop-gate)

    /** Runs on: main. Installs the rail-snapshot AND moment-snapshot providers so a HUD
     *  connecting/reconnecting after a restart or mid-session gets the objects/moments already in
     *  memory (empty lists if the stores aren't loaded yet). */
    fun attach(s: FrameSink) {
        sink = s
        s.railSnapshotProvider = { railItems() }
        s.momentSnapshotProvider = { momentItems() }
    }
    fun offerFrame(jpeg: ByteArray) { sink?.offerFrame(jpeg) }              // Runs on: streamLane
    fun pushEvent(line: String) { sink?.pushEvent(line) }                  // Runs on: any thread
    fun registerThumb(key: String, path: String) { sink?.registerThumb(key, path) }  // cropLane/inferLane/embedLane/IO
    fun broadcastRailSnapshot() { sink?.broadcastRailSnapshot() }          // Runs on: init (IO)
}
