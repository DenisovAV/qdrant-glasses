package tech.qdrant.glasses.detect

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ObjectTrackerTest {
    private fun det(l: String, x: Float, y: Float) =
        Detection(RectF(x, y, x + 10f, y + 10f), l, 0.9f)

    @Test fun sameObjectKeepsOneTrackAndCountsSightings() {
        val t = ObjectTracker(confirmSightings = 3, iouThreshold = 0.4f)
        t.update(listOf(det("cup", 0f, 0f)))
        t.update(listOf(det("cup", 1f, 0f)))          // overlaps → same track
        val tracks = t.update(listOf(det("cup", 0f, 1f)))
        assertEquals(1, tracks.size)
        assertEquals(3, tracks[0].sightings)
    }

    @Test fun differentLabelStartsNewTrack() {
        val t = ObjectTracker()
        t.update(listOf(det("cup", 0f, 0f)))
        val tracks = t.update(listOf(det("cup", 0f, 0f), det("book", 0f, 0f)))
        assertEquals(2, tracks.size)
    }

    @Test fun disjointSameLabelStartsNewTrack() {
        val t = ObjectTracker()
        t.update(listOf(det("cup", 0f, 0f)))
        val tracks = t.update(listOf(det("cup", 100f, 100f)))  // no overlap
        assertEquals(2, tracks.size)
    }

    @Test fun confirmedAfterNThenDedup() {
        val t = ObjectTracker(confirmSightings = 2)
        t.update(listOf(det("cup", 0f, 0f)))
        t.update(listOf(det("cup", 0f, 0f)))
        assertEquals(1, t.confirmedUnembedded().size)
        t.markEmbedded(t.confirmedUnembedded().first().trackId)
        assertTrue(t.confirmedUnembedded().isEmpty())  // embedded once, not again
    }

    @Test fun confirmedIncludesEmbeddedTracksUnlikeConfirmedUnembedded() {
        val t = ObjectTracker(confirmSightings = 2)
        t.update(listOf(det("cup", 0f, 0f)))
        t.update(listOf(det("cup", 0f, 0f)))
        t.markEmbedded(t.confirmedUnembedded().first().trackId)
        assertTrue(t.confirmedUnembedded().isEmpty())  // embedded once, dedup'd out
        assertEquals(1, t.confirmed().size)             // confirmed() doesn't filter on `embedded`
    }

    @Test fun twoSameLabelObjectsKeepSeparateTracks() {
        // Two cups, far apart, seeded as two tracks. Next frame both reappear (slightly moved).
        // Global best-first matching must keep each cup on its own track — no cross-claim,
        // no spurious third track. (Detection order is deliberately A-then-B both frames.)
        val t = ObjectTracker(iouThreshold = 0.4f)
        t.update(listOf(det("cup", 0f, 0f), det("cup", 100f, 100f)))   // two tracks
        val tracks = t.update(listOf(det("cup", 1f, 0f), det("cup", 101f, 100f)))
        assertEquals(2, tracks.size)
        assertTrue(tracks.all { it.sightings == 2 })  // each matched its own track
    }
}
