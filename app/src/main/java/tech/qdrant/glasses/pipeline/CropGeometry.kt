package tech.qdrant.glasses.pipeline

import android.graphics.Rect
import android.graphics.RectF

/** Grow [box] by [padding] of its own size on each side, clamped to [w]x[h]; null if degenerate. */
fun paddedCropRect(box: RectF, padding: Float, w: Int, h: Int): Rect? {
    val padX = box.width() * padding
    val padY = box.height() * padding
    val l = (box.left - padX).toInt().coerceIn(0, w - 1)
    val t = (box.top - padY).toInt().coerceIn(0, h - 1)
    val r = (box.right + padX).toInt().coerceIn(l + 1, w)
    val b = (box.bottom + padY).toInt().coerceIn(t + 1, h)
    return if (r > l && b > t) Rect(l, t, r, b) else null
}
