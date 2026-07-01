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

    /** Mac LAN IP (wireless) or localhost (wired via adb reverse). Serves /embed_* and the relay. */
    val MAC_BASE_URL: String = if (WIRELESS) "http://192.168.1.100:9000" else "http://localhost:9000"
}
