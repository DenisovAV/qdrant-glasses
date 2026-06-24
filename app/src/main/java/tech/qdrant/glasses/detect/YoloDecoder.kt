package tech.qdrant.glasses.detect

import android.graphics.RectF

/**
 * Decodes the qai-hub YOLOv8 export's 3 outputs into pixel-space [Detection]s with NMS.
 * boxes[0][i] = (x1,y1,x2,y2) in the modelSize grid; scores[0][i] = max class score (graph
 * already did softmax+argmax); classIdx[i] = class index 0–79. We just filter, scale+clamp
 * to the frame, label, and NMS.
 */
class YoloDecoder(
    private val scoreThreshold: Float = 0.4f,
    private val iouThreshold: Float = 0.45f,
) {
    fun decode(
        boxes: Array<Array<FloatArray>>, scores: Array<FloatArray>, classIdx: IntArray,
        modelSize: Int, frameW: Int, frameH: Int,
    ): List<Detection> {
        val s = scores[0]; val b = boxes[0]
        val sx = frameW.toFloat() / modelSize; val sy = frameH.toFloat() / modelSize
        val fw = frameW.toFloat(); val fh = frameH.toFloat()
        val candidates = ArrayList<Detection>()
        for (i in s.indices) {
            if (s[i] < scoreThreshold) continue
            val r = b[i]
            val left = (r[0] * sx).coerceIn(0f, fw)
            val top = (r[1] * sy).coerceIn(0f, fh)
            val right = (r[2] * sx).coerceIn(0f, fw)
            val bottom = (r[3] * sy).coerceIn(0f, fh)
            if (right <= left || bottom <= top) continue
            candidates.add(Detection(RectF(left, top, right, bottom), CocoLabels[classIdx[i]], s[i]))
        }
        return nms(candidates)
    }

    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        val suppressed = BooleanArray(sorted.size)
        for (a in sorted.indices) {
            if (suppressed[a]) continue
            kept.add(sorted[a])
            for (j in a + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (sorted[j].label == sorted[a].label && iou(sorted[a].bbox, sorted[j].bbox) > iouThreshold) suppressed[j] = true
            }
        }
        return kept
    }
}
