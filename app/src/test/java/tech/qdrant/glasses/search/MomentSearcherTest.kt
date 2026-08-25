package tech.qdrant.glasses.search

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.stream.HudPublisher

/**
 * [MomentSearcher.search]'s gate-OR-tag filter (episodic-memory calibration rehearsal, Change 2):
 * a moment is accepted if its fused vector score clears [MomentSearcher]'s 0.25 gate OR its id is
 * in the verified-tag-accepted set — see `MOMENT_SEARCH_GATE`'s KDoc. This is the highest-value
 * coverage gap flagged by review: [MomentFusionTest] covers the two pure functions in isolation,
 * this covers them WIRED TOGETHER through `search()`, with FAKE [CropEncoder]/[MomentStore] deps
 * (both are interfaces — no mocking framework needed) and a real, un-attached [HudPublisher] (its
 * `sink` stays null with no `attach()` call, so `registerThumb`/`pushEvent` are genuine no-ops —
 * see its KDoc).
 */
@RunWith(RobolectricTestRunner::class)
class MomentSearcherTest {

    private class FakeCropEncoder(override val dim: Int = 4) : CropEncoder {
        override fun encode(crop: Bitmap): FloatArray = error("not used by MomentSearcher.search")
        override fun encodeText(query: String): FloatArray = FloatArray(dim) { 1f }
        override val visionMinScore: Float = 0f
    }

    private class FakeMomentStore(
        private val frameHits: List<MomentHit>,
        private val regionHits: List<MomentHit>,
        private val windowHits: List<MomentHit> = emptyList(),
    ) : MomentStore {
        // Records the exact bounds the pure-time branch passed, so a test can catch a since/until
        // swap or drop (which returning canned windowHits unconditionally would otherwise hide).
        var lastWindowSince: Long? = null; private set
        var lastWindowUntil: Long? = null; private set
        override fun storeMoment(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeRegion(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeOcr(textVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = frameHits
        override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = regionHits
        // Stage 3 OCR channel — MomentSearcherTest never wires a bgeEncoder into MomentSearcher, so
        // this branch is never reached; kept as `error()` like the other unused write-path methods
        // above so a future test that DOES wire OCR search finds out immediately if it forgets to
        // stub this.
        override fun searchText(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = error("not used")
        override fun timeline(limit: Int): List<MomentHit> = emptyList()
        override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> {
            lastWindowSince = sinceMs; lastWindowUntil = untilMs
            return windowHits
        }
        override fun count(): Long = (frameHits.size + regionHits.size).toLong()
        override fun frameCount(): Long = frameHits.size.toLong()
        override fun deleteAll() {}
        override fun scrollUnsyncedFrames(limit: Int) = error("not used")
        override fun markSynced(ids: List<String>) = error("not used")
        override fun close() {}
    }

    private fun frameHit(momentId: String, score: Float) = MomentHit(
        id = momentId, score = score, type = "frame", momentId = momentId, timestampMs = 0L,
        thumbPath = "$momentId.jpg", label = "", bbox = "",
    )

    private fun regionHit(momentId: String, score: Float, label: String, yoloConf: Float, verifyCos: Float) = MomentHit(
        id = "$momentId-region", score = score, type = "region", momentId = momentId, timestampMs = 0L,
        thumbPath = "$momentId.jpg", label = label, bbox = "0.1,0.1,0.2,0.2",
        yoloConf = yoloConf, verifyCos = verifyCos,
    )

    private fun noopHud() = HudPublisher(railItems = { emptyList() }, momentItems = { emptyList() })

    @Test fun searchAcceptsGatePassAndTagAccept_rejectsBelowGateNoTag() {
        // The gate is PER-BACKEND ([CropEncoderFactory.searchGate]), so fixtures are expressed
        // RELATIVE to it — a pinned "0.15 = below gate" silently became ABOVE-gate when the default
        // backend moved CLIP→SigLIP2 (gate 0.25→0.085). softBoost's max lift is
        // TAG_BOOST_LAMBDA·yoloConf·verifyCos = 0.05·0.9·0.8 ≈ 0.036, so (c)'s region (0.3·gate) sits
        // far enough below that even boosted it stays under the gate, proving (c) is admitted by the
        // tag-accept OR-clause, not the vector score.
        val gate = CropEncoderFactory.searchGate
        // (a) ABOVE gate, no region — accepted on score alone.
        val aboveGate = frameHit("a", score = gate + 0.05f)
        // (b) BELOW gate, no matching region at all — must be rejected.
        val belowGateNoTag = frameHit("b", score = gate * 0.5f)
        // (c) BELOW gate on BOTH the frame hit and its region (fused max stays below gate, and the
        // bounded softBoost can't close the gap), but the region's label ("cup") verifies
        // (verifyCos > 0) and matches the query — accepted via the tag-accept OR-clause.
        val belowGateFrame = frameHit("c", score = gate * 0.2f)
        val belowGateVerifiedRegion = regionHit("c", score = gate * 0.3f, label = "cup", yoloConf = 0.9f, verifyCos = 0.8f)

        val searcher = MomentSearcher(
            cropEncoder = FakeCropEncoder(),
            store = FakeMomentStore(
                frameHits = listOf(aboveGate, belowGateNoTag, belowGateFrame),
                regionHits = listOf(belowGateVerifiedRegion),
            ),
            hud = noopHud(),
        )

        val outcome = searcher.search("cup") as ObjectSearcher.Outcome.Success
        val acceptedIds = outcome.cards.map { it.frame.id }.toSet()

        assertEquals(setOf("a", "c"), acceptedIds)
    }

    @Test fun searchPureTimeQuerySkipsGateAndReturnsWindowFrames() {
        // Both BELOW MomentSearcher's 0.25 gate — if the pure-time branch fell through to the
        // normal gate/fusion path (e.g. because parseQuery.timeOnly was ignored), these would be
        // filtered out. framesInWindow returning them unfiltered proves the gate was skipped.
        val windowHits = listOf(frameHit("d", score = 0.05f), frameHit("e", score = 0.01f))

        // frameHits/regionHits deliberately non-empty: if the pure-time branch mistakenly fell
        // through to searchFrames/searchRegions instead of framesInWindow, this hit ("a", 0.30,
        // clears the gate on its own) would leak into the result and fail the id assertion below.
        val store = FakeMomentStore(
            frameHits = listOf(frameHit("a", score = 0.30f)),
            regionHits = emptyList(),
            windowHits = windowHits,
        )
        val searcher = MomentSearcher(cropEncoder = FakeCropEncoder(), store = store, hud = noopHud())

        // "yesterday" strips to a blank embed phrase (parseQuery.timeOnly) — no object named.
        val outcome = searcher.search("what did i see yesterday") as ObjectSearcher.Outcome.Success
        val returnedIds = outcome.cards.map { it.frame.id }

        assertEquals(listOf("d", "e"), returnedIds)
        // Review gap T4: prove the window BOUNDS actually reached framesInWindow, not just that the
        // branch was taken — a since/until swap or drop would pass the id assertion above unchanged.
        // "yesterday" is a bounded past day, so both ends are non-null and sinceMs strictly precedes
        // untilMs; a swapped call (untilMs, sinceMs) would invert this.
        val since = store.lastWindowSince!!
        val until = store.lastWindowUntil!!
        assertTrue("since must precede until", since < until)
    }

    // --- gate-then-decay recency ranker (Config.RECENCY_TAU_MS) — pure companion fn, no search() deps ---

    @Test fun recencyRankFavorsFresherAmongNearEqualScores() {
        val now = 1_000_000_000L
        val min = 60_000L
        // "old" is marginally STRONGER (0.30 vs 0.28) but 30 min stale; "fresh" is 1 min old. With
        // τ = 10 min: old → 0.30·exp(-3)=0.015, fresh → 0.28·exp(-0.1)=0.253. Recency flips the order.
        val old   = frameHit("old", score = 0.30f).copy(timestampMs = now - 30 * min)
        val fresh = frameHit("fresh", score = 0.28f).copy(timestampMs = now - 1 * min)
        val ranked = MomentSearcher.recencyRank(listOf(old, fresh), now, tauMs = 10 * min)
        assertEquals(listOf("fresh", "old"), ranked.map { it.id })
    }

    @Test fun recencyRankKeepsScoreOrderAtEqualAgeAndClampsFutureStamps() {
        val now = 1_000_000_000L
        // Equal timestamps → identical decay factor → the STABLE sort keeps raw-score order.
        val strong = frameHit("strong", score = 0.30f).copy(timestampMs = now - 60_000L)
        val weak   = frameHit("weak", score = 0.20f).copy(timestampMs = now - 60_000L)
        assertEquals(listOf("strong", "weak"),
            MomentSearcher.recencyRank(listOf(weak, strong), now, tauMs = 600_000L).map { it.id })
        // A hit stamped in the FUTURE (write-time clock skew) must NOT be amplified: Δt clamps to 0
        // (decay = 1), so it ranks on its (lower) score alone. Without the clamp exp(+1)=2.72 would
        // rocket future (0.22) above strong (0.30·exp(-0.1)=0.271) — this asserts strong still wins.
        val future = frameHit("future", score = 0.22f).copy(timestampMs = now + 600_000L)
        val ranked = MomentSearcher.recencyRank(listOf(future, strong), now, tauMs = 600_000L)
        assertEquals("strong", ranked.first().id)
    }
}
