package tech.qdrant.glasses

/**
 * Demo-time wiring constants.
 *
 * The Mac runs both the SigLIP2 embed endpoint AND (in wireless mode) the MJPEG/HUD relay, on the
 * same port. In WIRED mode the glasses reach it via `adb reverse tcp:9000` (localhost); in WIRELESS
 * mode they reach it directly on the Mac's LAN IP. Flip [WIRELESS] to switch the demo path.
 */
object Config {
    /** true = push frames to the Mac relay over WiFi (no cable). false = on-glasses MjpegServer + USB. */
    const val WIRELESS = true

    /** Compiled-in fallback: home-LAN Mac IP (wireless) or localhost (wired via adb reverse). */
    private val DEFAULT_MAC_BASE_URL: String =
        if (WIRELESS) "http://192.168.1.100:9000" else "http://localhost:9000"

    /**
     * Mac relay/embed base URL. Overridable WITHOUT a rebuild for venue networks (stage hotspot):
     *   adb shell setprop persist.qdrant.relay http://172.20.10.2:9000
     * persist.* survives reboots (debug.* does NOT — a glasses reboot silently reverted the app
     * to the compiled-in home IP mid-rehearsal, twice). debug.qdrant.relay still works as a
     * volatile override and wins over persist. Read once at startup (app restart applies it).
     */
    val MAC_BASE_URL: String = sysprop("qdrant.relay").ifBlank { DEFAULT_MAC_BASE_URL }

    /**
     * Whether to run the HUD stream (browser dashboard). **OPT-IN — off by default.**
     *
     *   adb shell setprop debug.qdrant.hud 1     → enable the dashboard stream (our stage setup)
     *
     * Default OFF because the pusher otherwise keeps trying to reach the Mac relay ([MAC_BASE_URL],
     * a compiled-in LAN IP) on a device that has no relay — spamming `MjpegPusher: Failed to
     * connect` and phoning home to a private IP that means nothing on someone else's unit (this
     * confused a tester into thinking it was a Mac/host build). It's also not free: it downscales +
     * JPEG-encodes every frame on the same cores the CLIP encoder needs. Enable it only when the
     * relay is actually up.
     */
    val HUD_STREAM: Boolean = sysprop("qdrant.hud") == "1"

    /**
     * Whether to run the whole-frame + CLIP-verified-region keyframe memory path
     * (`PerceptionPipeline`'s [tech.qdrant.glasses.pipeline.MomentCapture] branch).
     * **DEFAULT ON** as of the episodic-memory plan's Task 2.4 (Stage 2): [MomentCapture] is now
     * the ONLY OBJECTS-mode memory path — the old YOLO-crop-embed-and-store block it used to run
     * alongside is retired. Turning this off is therefore a kill switch for A/B/regression
     * testing, not a fallback to the old behavior (which no longer exists in this build): with it
     * off, OBJECTS mode still detects/tracks/streams boxes but stores nothing at all.
     *
     *   adb shell setprop debug.qdrant.memory 0     → disable MomentCapture (A/B / regression only)
     */
    val MOMENT_MEMORY: Boolean = sysprop("qdrant.memory") != "0"

    /**
     * Reads `debug.<name>` (volatile, wins) then `persist.<name>` (survives reboots).
     * persist.* matters because a glasses reboot silently reverted the app to compiled-in defaults
     * mid-rehearsal — twice. Read once at startup; an app restart applies a change.
     */
    internal fun sysprop(name: String): String = runCatching {
        val get = Class.forName("android.os.SystemProperties")
            .getMethod("get", String::class.java, String::class.java)
        fun prop(key: String) = (get.invoke(null, key, "") as String)
        prop("debug.$name").ifBlank { prop("persist.$name") }
    }.getOrDefault("")
}
