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
     *   adb shell setprop debug.qdrant.relay http://172.20.10.2:9000
     * Empty/unset property → compiled-in default. Read once at startup (app restart applies it).
     */
    val MAC_BASE_URL: String = runCatching {
        val get = Class.forName("android.os.SystemProperties")
            .getMethod("get", String::class.java, String::class.java)
        (get.invoke(null, "debug.qdrant.relay", "") as String).ifBlank { DEFAULT_MAC_BASE_URL }
    }.getOrDefault(DEFAULT_MAC_BASE_URL)
}
