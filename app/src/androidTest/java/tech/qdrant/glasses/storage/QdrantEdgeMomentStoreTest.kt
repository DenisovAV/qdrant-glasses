package tech.qdrant.glasses.storage

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Random
import kotlin.math.sqrt

/**
 * On-device verification of [QdrantEdgeMomentStore] (plan Task 1.3): stores frame + region points
 * in one collection and checks the type/time filters actually separate the channels, mirroring
 * [QdrantFfiOverheadTest]'s bare-store-against-real-FFI style (no mocks — this IS the unknown being
 * resolved, Spec §8.4: does a 2-named-vector `EdgeConfig` load and query correctly on-device).
 * Logged under tag "MOMENTSTORE" — capture with: adb logcat -s MOMENTSTORE:I
 */
@RunWith(AndroidJUnit4::class)
class QdrantEdgeMomentStoreTest {

    private val TAG = "MOMENTSTORE"

    @Test fun frameAndRegionChannelsAndTimeline() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store = QdrantEdgeMomentStore(ctx, "momentstoretest")
        try {
            store.deleteAll()
            val rnd = Random(7)

            // Three frames at distinct, well-separated timestamps.
            val t0 = 1_000_000L
            val t1 = 2_000_000L
            val t2 = 3_000_000L
            val f0 = store.storeMoment(unit(rnd), framePayload(t0))
            val f1 = store.storeMoment(unit(rnd), framePayload(t1))
            val f2 = store.storeMoment(unit(rnd), framePayload(t2))
            Log.i(TAG, "stored frames: $f0@$t0 $f1@$t1 $f2@$t2")

            // A frame's own moment_id must equal its own id (Spec §6 / MomentPayload invariant).
            val f0hit = store.searchFrames(unit(rnd), topK = 10, sinceMs = null, untilMs = null)
                .first { it.id == f0 }
            assertEquals(f0, f0hit.momentId)

            // Two regions sharing ONE parent moment_id (f0).
            val r0 = store.storeRegion(unit(rnd), regionPayload(momentId = f0, ts = t0, label = "mug"))
            val r1 = store.storeRegion(unit(rnd), regionPayload(momentId = f0, ts = t0, label = "cup"))
            Log.i(TAG, "stored regions: $r0 $r1 (momentId=$f0)")

            assertEquals(5L, store.count())

            // searchFrames returns ONLY frame-type points.
            val allFrames = store.searchFrames(unit(rnd), topK = 10, sinceMs = null, untilMs = null)
            assertEquals(3, allFrames.size)
            assertTrue(allFrames.all { it.type == "frame" })
            assertEquals(setOf(f0, f1, f2), allFrames.map { it.id }.toSet())

            // searchFrames respects a time window (only t0, t1 in range).
            val windowed = store.searchFrames(unit(rnd), topK = 10, sinceMs = t0, untilMs = t1 + 1)
            assertEquals(setOf(f0, f1), windowed.map { it.id }.toSet())

            // searchRegions returns ONLY region-type points, both sharing momentId=f0.
            val regions = store.searchRegions(unit(rnd), topK = 10, sinceMs = null, untilMs = null)
            assertEquals(2, regions.size)
            assertTrue(regions.all { it.type == "region" })
            assertTrue(regions.all { it.momentId == f0 })
            assertEquals(setOf(r0, r1), regions.map { it.id }.toSet())

            // timeline() returns frames only, oldest-first.
            val tl = store.timeline(500)
            assertEquals(listOf(f0, f1, f2), tl.map { it.id })
            assertEquals(listOf(t0, t1, t2), tl.map { it.timestampMs })

            // deleteAll() empties the whole collection (both channels).
            store.deleteAll()
            assertEquals(0L, store.count())
            assertTrue(store.timeline(500).isEmpty())

            Log.i(TAG, "frameAndRegionChannelsAndTimeline: PASS")
        } finally {
            store.deleteAll(); store.close()
        }
    }

    /**
     * Regression for the Codex P2: timeline(limit) used to scroll `limit` points in Qdrant's
     * default point-ID order and THEN sort that arbitrary page by timestamp, so once the frame
     * count exceeded `limit`, the rail could silently drop recent frames. With 5 stored frames and
     * limit=3, the only correct answer is the 3 NEWEST, oldest-first — never an arbitrary subset.
     */
    @Test fun timelineReturnsNewestFramesWithinLimit() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store = QdrantEdgeMomentStore(ctx, "momentstoretest_timeline")
        try {
            store.deleteAll()
            val rnd = Random(11)

            val timestamps = listOf(1_000_000L, 2_000_000L, 3_000_000L, 4_000_000L, 5_000_000L)
            val ids = timestamps.map { ts -> store.storeMoment(unit(rnd), framePayload(ts)) }
            Log.i(TAG, "stored ${ids.size} frames at $timestamps")

            val tl = store.timeline(limit = 3)
            // The 3 NEWEST frames (ts=3e6,4e6,5e6 / ids[2..4]), oldest-first.
            assertEquals(listOf(ids[2], ids[3], ids[4]), tl.map { it.id })
            assertEquals(listOf(timestamps[2], timestamps[3], timestamps[4]), tl.map { it.timestampMs })

            Log.i(TAG, "timelineReturnsNewestFramesWithinLimit: PASS")
        } finally {
            store.deleteAll(); store.close()
        }
    }

    /**
     * [QdrantEdgeMomentStore.framesInWindow] (query-understanding plan Task 4): the pure-time "what
     * did I see on <day>" path — a window filter with NO query vector, newest-first (the opposite
     * order from [timeline], which is oldest-first for rail playback).
     */
    @Test fun framesInWindowFiltersAndOrdersNewestFirst() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val store = QdrantEdgeMomentStore(ctx, "momentstoretest_window")
        try {
            store.deleteAll()
            val rnd = Random(13)

            // Four frames at distinct, well-separated timestamps.
            val t0 = 1_000_000L
            val t1 = 2_000_000L
            val t2 = 3_000_000L
            val t3 = 4_000_000L
            val f0 = store.storeMoment(unit(rnd), framePayload(t0))
            val f1 = store.storeMoment(unit(rnd), framePayload(t1))
            val f2 = store.storeMoment(unit(rnd), framePayload(t2))
            val f3 = store.storeMoment(unit(rnd), framePayload(t3))
            // A REGION point INSIDE the [t1,t2] window and the open-ended set — framesInWindow is a
            // frame-only path (type=frame filter, review gap T6), so it must never surface here even
            // though it falls squarely within every window below.
            val r2 = store.storeRegion(unit(rnd), regionPayload(f2, t2, "laptop"))
            Log.i(TAG, "stored frames: $f0@$t0 $f1@$t1 $f2@$t2 $f3@$t3 + region $r2@$t2")

            // A closed [t1, t2] window returns only f1, f2 — newest (f2) first; the region at t2 is
            // excluded despite being in range.
            val windowed = store.framesInWindow(sinceMs = t1, untilMs = t2)
            assertEquals(listOf(f2, f1), windowed.map { it.id })
            assertEquals(listOf(t2, t1), windowed.map { it.timestampMs })

            // Open-ended bounds (both null) return every frame, newest-first — the same set as
            // timeline() but reversed (timeline() is oldest-first for rail playback).
            val all = store.framesInWindow(sinceMs = null, untilMs = null)
            assertEquals(listOf(f3, f2, f1, f0), all.map { it.id })

            // A one-sided lower bound (sinceMs only) excludes everything before it.
            val sinceOnly = store.framesInWindow(sinceMs = t2, untilMs = null)
            assertEquals(listOf(f3, f2), sinceOnly.map { it.id })

            // A one-sided upper bound (untilMs only) excludes everything after it.
            val untilOnly = store.framesInWindow(sinceMs = null, untilMs = t1)
            assertEquals(listOf(f1, f0), untilOnly.map { it.id })

            // limit is applied AFTER the newest-first sort, not before.
            val limited = store.framesInWindow(sinceMs = null, untilMs = null, limit = 2)
            assertEquals(listOf(f3, f2), limited.map { it.id })

            Log.i(TAG, "framesInWindowFiltersAndOrdersNewestFirst: PASS")
        } finally {
            store.deleteAll(); store.close()
        }
    }

    private fun framePayload(ts: Long) = MomentPayload(
        type = "frame", momentId = "", episodeId = ts, timestampMs = ts, tEndMs = ts,
        thumbPath = "/sdcard/moments/$ts.jpg", bbox = "", label = "", yoloConf = 0f, verifyCos = 0f, text = "",
    )

    private fun regionPayload(momentId: String, ts: Long, label: String) = MomentPayload(
        type = "region", momentId = momentId, episodeId = ts, timestampMs = ts, tEndMs = ts,
        thumbPath = "/sdcard/moments/$ts.jpg", bbox = "0.1,0.1,0.4,0.4", label = label,
        yoloConf = 0.8f, verifyCos = 0.3f, text = "",
    )

    private fun unit(rnd: Random): FloatArray {
        val v = FloatArray(512) { rnd.nextGaussian().toFloat() }
        var n = 0.0; for (x in v) n += (x * x).toDouble()
        val inv = if (n > 0) (1.0 / sqrt(n)).toFloat() else 0f
        for (i in v.indices) v[i] *= inv
        return v
    }
}
