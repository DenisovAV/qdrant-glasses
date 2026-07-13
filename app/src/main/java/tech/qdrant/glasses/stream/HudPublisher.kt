package tech.qdrant.glasses.stream

/**
 * Wraps the late-bound [FrameSink]: the HUD (browser dashboard) attaches after the ViewModel is
 * already running, and may never attach at all (no HUD client connected). Every method below is a
 * no-op when detached (`sink == null`) — same `?.` semantics as the field it replaces.
 */
class HudPublisher(private val railItems: () -> List<MjpegServer.RailItem>) {
    @Volatile private var sink: FrameSink? = null      // fully private — nothing reads the raw sink
    val hasClient: Boolean get() = sink != null        // Runs on: any thread (perception's drop-gate)

    /** Runs on: main. Installs the rail-snapshot provider so a HUD connecting after a restart
     *  gets the objects already in memory (empty list if the store isn't loaded yet). */
    fun attach(s: FrameSink) {
        sink = s
        s.railSnapshotProvider = { railItems() }
    }
    fun offerFrame(jpeg: ByteArray) { sink?.offerFrame(jpeg) }              // Runs on: streamLane
    fun pushEvent(line: String) { sink?.pushEvent(line) }                  // Runs on: any thread
    fun registerThumb(key: String, path: String) { sink?.registerThumb(key, path) }  // cropLane/inferLane
    fun broadcastRailSnapshot() { sink?.broadcastRailSnapshot() }          // Runs on: init (IO)

    // TEMPORARY: consumed only by GlassesViewModel.onObjectFrame until Task 7 moves that block; delete then.
    internal val sinkOrNull: FrameSink? get() = sink
}
