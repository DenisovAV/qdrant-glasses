package tech.qdrant.glasses.pipeline

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Drives the PURE [decide] gate (plan Task 1.4, Spec §4 steps 2 + 5) with a fake clock — plain
 * `Long` millis passed straight into [decide], since it takes `lastStoreMs`/`nowMs` as parameters
 * and never touches a real clock. Mirrors [SceneDiffTest]'s style: plain `FloatArray` grids, no
 * Bitmap/coroutine/store involved (that half is [MomentCapture], verified on-device at the Stage
 * Gate, not here).
 *
 * The thresholds under test are `MomentCapture.kt`'s file-private constants — not importable, so
 * (same convention [CropGeometryTest] uses for `PerceptionPipeline`'s `CROP_PADDING`) their values
 * are hardcoded here with a comment: `PREGATE_SIMILARITY=0.98`, `CAPTURE_COOLDOWN_MS=3000`,
 * `HEARTBEAT_MS=45000` (the similarity/cooldown pair was loosened from 0.85/8000 by the SigLIP
 * scene-dedup calibration — keep these mirrors in sync with MomentCapture.kt's constants).
 */
class MomentCaptureGateTest {
    // Two SEPARATE arrays with equal values (not the same reference) so similarity() does a real
    // elementwise comparison rather than short-circuiting on identity.
    private val prevGrid = FloatArray(1024) { 0.5f }
    private val sameGrid = FloatArray(1024) { 0.5f }
    // Alternating 0/1 vs a flat 0.5 baseline → meanAbsDiff=0.5 → similarity=0.5, comfortably below
    // PREGATE_SIMILARITY(0.98) — "the scene changed" for gate purposes.
    private val changedGrid = FloatArray(1024) { i -> if (i % 2 == 0) 0f else 1f }

    @Test fun identicalGridsWithinCooldownSkip() {
        // 2s since the last store, CAPTURE_COOLDOWN_MS=3s → cooldown alone forces Skip, regardless
        // of similarity (which is also 1.0 here — an unchanged scene).
        val d = decide(prevGrid, sameGrid, lastStoreMs = 0L, nowMs = 2_000L)
        assertEquals(Decision.SKIP, d)
    }

    @Test fun dissimilarGridsPastCooldownCapture() {
        // 9s since the last store (past CAPTURE_COOLDOWN_MS=3s) and similarity=0.5 < 0.98 → the
        // scene genuinely changed → Capture.
        val d = decide(prevGrid, changedGrid, lastStoreMs = 0L, nowMs = 9_000L)
        assertEquals(Decision.CAPTURE, d)
    }

    @Test fun identicalGridsPastHeartbeatFireHeartbeat() {
        // 46s since the last store (past HEARTBEAT_MS=45s) but the scene never changed
        // (similarity=1.0) → Heartbeat, not Skip — a static scene must still get a keyframe
        // eventually so time-window queries stay answerable.
        val d = decide(prevGrid, sameGrid, lastStoreMs = 0L, nowMs = 46_000L)
        assertEquals(Decision.HEARTBEAT, d)
    }

    @Test fun dissimilarGridsWithinCooldownStillSkip() {
        // Scene changed (similarity=0.5 < 0.98) but only 1.5s since the last store (< 3s cooldown) —
        // the cooldown is checked BEFORE the pixel diff and caps the flood regardless of how
        // different the scene looks (Spec §4 step 5). nowMs must be STRICTLY under CAPTURE_COOLDOWN_MS:
        // the gate is `< cooldown`, so 3_000L (== 3s) is already PAST it and would Capture.
        val d = decide(prevGrid, changedGrid, lastStoreMs = 0L, nowMs = 1_500L)
        assertEquals(Decision.SKIP, d)
    }

    @Test fun nullPrevGridAlwaysCaptures() {
        // Nothing stored yet this session (the first-ever frame) → Capture unconditionally, no
        // baseline to diff against.
        val d = decide(prevGrid = null, candGrid = changedGrid, lastStoreMs = 0L, nowMs = 0L)
        assertEquals(Decision.CAPTURE, d)
    }
}
