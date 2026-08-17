package tech.qdrant.glasses.pipeline

import kotlin.math.abs

/**
 * Pure scene-change + sharpness math for keyframe gating (Spec §4). Operates on plain
 * `IntArray`/`FloatArray` — no `Bitmap` — so it is a JVM unit test, not an instrumented one.
 *
 * [downscaleLuma]/[similarity] are the SAME math as [FrameCaptureManager]'s
 * `downscalePixels`/`similarityBetween` (identical luminance weights 0.299/0.587/0.114), lifted out
 * so [MomentCapture]'s pre-gate can reuse it against a moment baseline instead of a Bitmap-scaled
 * one. [sharpness] is new: Laplacian-variance scoring for the sharpness-selection window (Spec §4
 * step 3), which [FrameCaptureManager] does not have.
 */

/** Mean luma (0..1, luminance/255) per cell of an [out]×[out] grid, block-averaged from an
 *  ARGB [argb] pixel buffer of size [w]×[h]. Cell bounds are computed with integer division of
 *  the source dimensions, so [w]/[h] need not divide evenly by [out] (and may be smaller than
 *  [out], e.g. a tiny crop upsampled into the grid). Every cell's start/end index is clamped into
 *  `[0, w)` / `[0, h)` so no combination of [w]/[h]/[out] reads past the end of [argb]; a
 *  zero-sized source returns an all-zero grid instead of indexing into an empty array. */
fun downscaleLuma(argb: IntArray, w: Int, h: Int, out: Int = 32): FloatArray {
    val grid = FloatArray(out * out)
    if (w <= 0 || h <= 0) return grid
    for (gy in 0 until out) {
        val y0 = (gy * h / out).coerceIn(0, h - 1)
        val y1 = maxOf(y0 + 1, (gy + 1) * h / out).coerceAtMost(h)
        for (gx in 0 until out) {
            val x0 = (gx * w / out).coerceIn(0, w - 1)
            val x1 = maxOf(x0 + 1, (gx + 1) * w / out).coerceAtMost(w)
            var sum = 0L
            var n = 0
            for (y in y0 until y1) {
                val row = y * w
                for (x in x0 until x1) {
                    sum += luminance(argb[row + x])
                    n++
                }
            }
            grid[gy * out + gx] = (sum.toFloat() / n) / 255f
        }
    }
    return grid
}

/** 1.0 == identical, lower == more different. [a]/[b] must be the same length and already
 *  normalized to 0..1 (e.g. two [downscaleLuma] grids) — this is `1 - meanAbsDiff`. */
fun similarity(a: FloatArray, b: FloatArray): Float {
    var diffSum = 0f
    for (i in a.indices) diffSum += abs(a[i] - b[i])
    return 1f - diffSum / a.size
}

/** Laplacian-variance sharpness of a [side]×[side] luma grid: higher = crisper edges, lower =
 *  motion-blur smear. Border cells use clamped (replicated-edge) neighbors so every cell of the
 *  grid contributes, not just the interior. */
fun sharpness(luma: FloatArray, side: Int): Float {
    require(luma.size == side * side) { "luma must have side*side=${side * side} cells, got ${luma.size}" }
    val laplacian = FloatArray(luma.size)
    for (y in 0 until side) {
        for (x in 0 until side) {
            val center = lumaAt(luma, side, x, y)
            val neighborSum = lumaAt(luma, side, x - 1, y) + lumaAt(luma, side, x + 1, y) +
                lumaAt(luma, side, x, y - 1) + lumaAt(luma, side, x, y + 1)
            laplacian[y * side + x] = neighborSum - 4f * center
        }
    }
    val mean = laplacian.average().toFloat()
    var sumSq = 0f
    for (v in laplacian) { val d = v - mean; sumSq += d * d }
    return sumSq / laplacian.size
}

private fun lumaAt(luma: FloatArray, side: Int, x: Int, y: Int): Float {
    val cx = x.coerceIn(0, side - 1)
    val cy = y.coerceIn(0, side - 1)
    return luma[cy * side + cx]
}

private fun luminance(argb: Int): Int {
    val r = argb shr 16 and 0xFF
    val g = argb shr 8 and 0xFF
    val b = argb and 0xFF
    return (0.299 * r + 0.587 * g + 0.114 * b).toInt()
}
