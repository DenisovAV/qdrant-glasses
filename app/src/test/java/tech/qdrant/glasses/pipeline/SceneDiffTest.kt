package tech.qdrant.glasses.pipeline

import org.junit.Assert.*
import org.junit.Test

class SceneDiffTest {
    @Test fun identicalGridsAreSimilar1() {
        val g = FloatArray(1024) { 0.5f }
        assertEquals(1.0f, similarity(g, g), 1e-4f)
    }
    @Test fun oppositeGridsAreDissimilar() {
        val a = FloatArray(1024) { 0f }; val b = FloatArray(1024) { 1f }
        assertTrue(similarity(a, b) < 0.1f)
    }
    @Test fun downscaleAveragesToGrid() {
        // 2x2 all-white image → single-cell grid (out=1) ≈ 1.0 luma
        val argb = IntArray(4) { 0xFFFFFFFF.toInt() }
        val g = downscaleLuma(argb, 2, 2, out = 1)
        assertEquals(1, g.size); assertEquals(1.0f, g[0], 1e-2f)
    }
    @Test fun sharpEdgeScoresHigherThanFlat() {
        val side = 8
        val flat = FloatArray(side * side) { 0.5f }
        val edge = FloatArray(side * side) { i -> if ((i % side) < side / 2) 0f else 1f }
        assertTrue(sharpness(edge, side) > sharpness(flat, side))
    }
    @Test fun downscaleHandlesSourceSmallerThanOut() {
        // 2x2 source, default out=32 (source dims < out) must not throw and must fill every cell.
        val argb = IntArray(4) { 0xFFFFFFFF.toInt() }
        val g = downscaleLuma(argb, 2, 2, out = 32)
        assertEquals(32 * 32, g.size)
        assertTrue(g.all { it in 0f..1f })
        assertEquals(1.0f, g[0], 1e-2f)
    }
}
