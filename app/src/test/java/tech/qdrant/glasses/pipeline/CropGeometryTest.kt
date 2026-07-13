package tech.qdrant.glasses.pipeline

import android.graphics.Rect
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class CropGeometryTest {
    @Test fun middleBoxGrowsByPaddingOnEachSide() {
        // 100x100 box, 0.20 padding = 20px each side, well inside a 1000x1000 frame → no clamping.
        val r = paddedCropRect(RectF(100f, 100f, 200f, 200f), 0.20f, 1000, 1000)
        assertEquals(Rect(80, 80, 220, 220), r)
    }

    @Test fun boxAtTopLeftEdgeClampsToZero() {
        // Box hangs off the top-left; padded left/top go negative and clamp to 0.
        // width=60,height=60 → padX=padY=12; l=-22→0, t=-22→0, r=62, b=62.
        val r = paddedCropRect(RectF(-10f, -10f, 50f, 50f), 0.20f, 100, 100)
        assertEquals(Rect(0, 0, 62, 62), r)
    }

    @Test fun boxAtBottomRightEdgeClampsToFrame() {
        // Box hugs the bottom-right; padded right/bottom exceed the frame and clamp to w/h.
        // width=50,height=50 → padX=padY=10; l=940, t=940, r=1010→1000, b=1010→1000.
        val r = paddedCropRect(RectF(950f, 950f, 1000f, 1000f), 0.20f, 1000, 1000)
        assertEquals(Rect(940, 940, 1000, 1000), r)
    }

    @Test fun degenerateZeroAreaBoxStillProducesValidRect() {
        // A zero-size box with no padding: the coerceIn(l+1,w)/coerceIn(t+1,h) lower bounds
        // force a 1x1 rect, so the result is non-null with r>l and b>t.
        val r = paddedCropRect(RectF(50f, 50f, 50f, 50f), 0f, 100, 100)
        assertNotNull(r)
        r!!
        assertTrue("r>l", r.right > r.left)
        assertTrue("b>t", r.bottom > r.top)
        assertEquals(Rect(50, 50, 51, 51), r)
    }
}
