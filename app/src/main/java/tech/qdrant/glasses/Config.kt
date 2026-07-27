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
     * Whether to run the HUD stream (browser dashboard) at all.
     *
     *   adb shell setprop debug.qdrant.hud 0     → no HUD; on stage, or when measuring
     *
     * The stream is not free: it downscales and JPEG-encodes every frame (~30-40ms each) on the
     * same cores the CLIP encoder needs, so a crop embed measured with the dashboard open is NOT
     * the latency the demo has without it. Turning it off is also the honest way to benchmark the
     * encoder — and a legitimate stage lever when nobody is watching the dashboard.
     */
    val HUD_STREAM: Boolean = sysprop("qdrant.hud") != "0"

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
