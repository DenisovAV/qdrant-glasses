package tech.qdrant.glasses.stream

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import tech.qdrant.glasses.detect.Detection

private val boxPaint = Paint().apply {
    color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true
}
private val textPaint = Paint().apply {
    color = Color.GREEN; textSize = 28f; isAntiAlias = true
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
