package tech.qdrant.glasses.detect

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class GeometryTest {
    @Test fun identicalBoxesIouIsOne() {
        val r = RectF(0f, 0f, 10f, 10f)
        assertEquals(1f, iou(r, r), 1e-4f)
    }

    @Test fun disjointBoxesIouIsZero() {
        assertEquals(0f, iou(RectF(0f, 0f, 1f, 1f), RectF(5f, 5f, 6f, 6f)), 1e-4f)
    }

    @Test fun halfOverlap() {
        // a=[0,0,2,2] area4, b=[1,0,3,2] area4, intersection=[1,0,2,2] area2, union=6
        assertEquals(2f / 6f, iou(RectF(0f, 0f, 2f, 2f), RectF(1f, 0f, 3f, 2f)), 1e-4f)
    }
}
