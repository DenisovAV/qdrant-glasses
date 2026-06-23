package tech.qdrant.glasses.detect

import android.graphics.RectF

/** Intersection-over-union of two boxes (same coordinate space). 0f if disjoint. */
fun iou(a: RectF, b: RectF): Float {
    val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
    val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
    val inter = ix * iy
    if (inter <= 0f) return 0f
    val areaA = (a.right - a.left) * (a.bottom - a.top)
    val areaB = (b.right - b.left) * (b.bottom - b.top)
    val union = areaA + areaB - inter
    return if (union <= 0f) 0f else inter / union
}
