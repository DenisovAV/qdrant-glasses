package tech.qdrant.glasses.stream

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import tech.qdrant.glasses.detect.Detection

private val boxPaint = Paint().apply {
    color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true
}
private val textPaint = Paint().apply {
    color = Color.GREEN; textSize = 28f; isAntiAlias = true
}
// Distinct from the GREEN object boxes: CYAN for OCR text-regions (Stage 3 live highlighting).
private val textBoxPaint = Paint().apply {
    color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true
}

/** Returns a mutable copy of [frame] with each detection's box + label drawn on. */
fun drawBoxes(frame: Bitmap, detections: List<Detection>): Bitmap {
    val out = frame.copy(Bitmap.Config.ARGB_8888, true)
    drawBoxesInPlace(out, detections)
    return out
}

/**
 * Draws boxes directly onto [frame] — NO copy. [frame] MUST be mutable (Canvas throws otherwise).
 * Use this on the stream lane where we already own a fresh mutable bitmap, to avoid the ~2.8MB
 * per-frame copy that [drawBoxes] pays. [detections] coordinates must already be in [frame]'s
 * pixel space (scale them to the stream size before calling if the frame was downscaled).
 */
fun drawBoxesInPlace(frame: Bitmap, detections: List<Detection>) {
    val c = Canvas(frame)
    for (d in detections) {
        c.drawRect(d.bbox, boxPaint)
        c.drawText("${d.label} ${(d.score * 100).toInt()}%", d.bbox.left, maxOf(28f, d.bbox.top - 6f), textPaint)
    }
}

/**
 * Draws DBNet text-region [boxes] onto [frame] (mutable) in CYAN, distinct from the GREEN object
 * boxes. Coordinate convention DIFFERS from [drawBoxesInPlace]: DBNet returns NORMALIZED [0,1]
 * fractions, so each box is scaled to [frame]'s pixel size HERE (not by the caller's pixel factors).
 */
fun drawTextBoxesInPlace(frame: Bitmap, boxes: List<RectF>) {
    if (boxes.isEmpty()) return
    val c = Canvas(frame)
    val w = frame.width.toFloat(); val h = frame.height.toFloat()
    for (b in boxes) c.drawRect(b.left * w, b.top * h, b.right * w, b.bottom * h, textBoxPaint)
}
