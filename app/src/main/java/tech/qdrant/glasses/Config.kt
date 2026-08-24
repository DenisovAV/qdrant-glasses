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
     * OCR recognizer backend. **OPT-IN NPU — OFF by default (the long-proven CPU CRNN path).**
     *
     *   adb shell setprop debug.qdrant.ocr_npu 1     → ViTSTR recognizer on the Hexagon NPU (~24ms/line)
     *   (unset / anything else)                      → PP-OCR CRNN recognizer on the CPU (~440ms/line)
     *
     * This flag swaps ONLY the recognizer — detection is the CPU DBNet@1536 either way, so small/far
     * text coverage is identical. NPU rec is ViTSTR (non-recurrent ViT; the CRNN's LSTM can't run on
     * the AR1 HTP). Kept switchable for A/B on real captures; default is the CPU path that already
     * worked well, so a fresh install "just works" and NPU is a deliberate opt-in.
     */
    val OCR_NPU: Boolean = sysprop("qdrant.ocr_npu") == "1"

    /**
     * Recency re-ranking of moment-search survivors ("gate-then-decay"). The per-backend score gate
     * ([tech.qdrant.glasses.embedding.CropEncoderFactory.searchGate]) is a PRESENCE detector; ABOVE
     * it the surviving cosine band is only ~0.025 wide — mostly noise — so raw-score order is a weak
     * ranker of the survivors. When this is a POSITIVE number of seconds, the normal (non-recall,
     * non-time-only) search re-ranks its gate survivors by `score × exp(-Δt / τ)`, τ = this many
     * seconds, so the freshest of several near-equal matches surfaces first. **UNSET / 0 → OFF** (raw
     * cosine order, the calibrated status quo) — opt-in like [OCR_NPU], so a fresh install is unchanged.
     *
     * Recall-intent ("where did I leave X") and pure-time ("what did I see yesterday") queries are
     * DELIBERATELY unaffected: both already order by recency directly (a stronger preference than a
     * tie-breaker). τ must match the capture timescale to reorder at all — a demo session spans
     * minutes-to-hours, so 1800 (30 min) is the suggested starting value; raise it for multi-day memory.
     *
     *   adb shell setprop debug.qdrant.recency_tau_s 1800   → recency ranker on, τ = 30 min
     *   (unset / 0)                                          → off (raw score order)
     */
    val RECENCY_TAU_MS: Long =
        sysprop("qdrant.recency_tau_s").toLongOrNull()?.takeIf { it > 0 }?.times(1000L) ?: 0L

    /**
     * Private-Qdrant base URL for the OPTIONAL fleet tier (Sovereign Fleet Memory PoC). BLANK = the
     * app is pure on-device/offline, exactly as before. When set, the glasses pull a curated fleet
     * corpus and (P2) push their moments up. Reachable from the glasses via `adb reverse tcp:6333`.
     *
     *   adb shell setprop debug.qdrant.fleet_url http://localhost:6333   → fleet tier on
     *   (unset)                                                          → off (local-only)
     */
    val FLEET_URL: String = sysprop("qdrant.fleet_url")

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
