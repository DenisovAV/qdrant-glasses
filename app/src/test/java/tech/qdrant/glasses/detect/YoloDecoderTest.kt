package tech.qdrant.glasses.detect

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class YoloDecoderTest {
    private val N = 8400
    private fun blank(): Triple<Array<Array<FloatArray>>, Array<FloatArray>, IntArray> =
        Triple(Array(1) { Array(N) { FloatArray(4) } }, Array(1) { FloatArray(N) }, IntArray(N))

    private fun put(b: Array<Array<FloatArray>>, s: Array<FloatArray>, c: IntArray, i: Int,
                    x1: Float, y1: Float, x2: Float, y2: Float, cls: Int, score: Float) {
        b[0][i][0] = x1; b[0][i][1] = y1; b[0][i][2] = x2; b[0][i][3] = y2; s[0][i] = score; c[i] = cls
    }

    @Test fun decodesOneBox_labelScoreScaledBbox() {
        val (b, s, c) = blank()
        // xyxy in 640 grid: 288,216 .. 352,264 ; class 41 ("cup"); score 0.9
        put(b, s, c, 0, 288f, 216f, 352f, 264f, 41, 0.9f)
        // frame 640x480 → x scale 1.0, y scale 0.75
        val dets = YoloDecoder().decode(b, s, c, modelSize = 640, frameW = 640, frameH = 480)
        assertEquals(1, dets.size)
        assertEquals("cup", dets[0].label)
        assertEquals(0.9f, dets[0].score, 1e-4f)
        assertEquals(288f, dets[0].bbox.left, 0.5f)
        assertEquals(352f, dets[0].bbox.right, 0.5f)
        assertEquals(162f, dets[0].bbox.top, 0.5f)      // 216 * 0.75
        assertEquals(198f, dets[0].bbox.bottom, 0.5f)   // 264 * 0.75
    }

    @Test fun thresholdDropsLowConf() {
        val (b, s, c) = blank(); put(b, s, c, 0, 100f, 100f, 200f, 200f, 41, 0.30f)
        assertTrue(YoloDecoder(scoreThreshold = 0.4f).decode(b, s, c, 640, 640, 480).isEmpty())
    }

    @Test fun nmsSuppressesDuplicate() {
        val (b, s, c) = blank()
        put(b, s, c, 0, 288f, 216f, 352f, 264f, 41, 0.9f)
        put(b, s, c, 1, 290f, 216f, 354f, 264f, 41, 0.8f)   // overlapping, same class
        val dets = YoloDecoder().decode(b, s, c, 640, 640, 480)
        assertEquals(1, dets.size); assertEquals(0.9f, dets[0].score, 1e-4f)
    }

    @Test fun clampsOutOfBounds() {
        val (b, s, c) = blank()
        put(b, s, c, 0, -10f, -5f, 700f, 660f, 0, 0.9f)     // exceeds 640 grid
        val d = YoloDecoder().decode(b, s, c, 640, 640, 480)[0]
        assertEquals(0f, d.bbox.left, 0.5f); assertEquals(640f, d.bbox.right, 0.5f)
        assertEquals(0f, d.bbox.top, 0.5f); assertEquals(480f, d.bbox.bottom, 0.5f)
    }
}
