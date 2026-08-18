package tech.qdrant.glasses.search

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.embedding.CropEncoder
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
        override fun storeMoment(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeRegion(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = frameHits
        override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = regionHits
        override fun timeline(limit: Int): List<MomentHit> = emptyList()
        override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> = windowHits
        override fun count(): Long = (frameHits.size + regionHits.size).toLong()
        override fun frameCount(): Long = frameHits.size.toLong()
        override fun deleteAll() {}
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
        // (a) ABOVE MomentSearcher's 0.25 gate, no region — accepted on score alone.
        val aboveGate = frameHit("a", score = 0.30f)
        // (b) BELOW gate, no matching region at all — must be rejected.
        val belowGateNoTag = frameHit("b", score = 0.15f)
        // (c) BELOW gate on BOTH the frame hit and its region (fused score also stays below gate,
        // and the bounded softBoost — max 0.05 — can't close the gap either: 0.20 + 0.05*0.9*0.8 =
        // 0.236, still < 0.25), but the region's label ("cup") verifies (verifyCos > 0) and matches
        // the query — accepted via the tag-accept OR-clause, not the vector gate.
        val belowGateFrame = frameHit("c", score = 0.10f)
        val belowGateVerifiedRegion = regionHit("c", score = 0.20f, label = "cup", yoloConf = 0.9f, verifyCos = 0.8f)

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

        val searcher = MomentSearcher(
            cropEncoder = FakeCropEncoder(),
            // frameHits/regionHits deliberately non-empty: if the pure-time branch mistakenly fell
            // through to searchFrames/searchRegions instead of framesInWindow, this hit ("a", 0.30,
            // clears the gate on its own) would leak into the result and fail the id assertion below.
            store = FakeMomentStore(
                frameHits = listOf(frameHit("a", score = 0.30f)),
                regionHits = emptyList(),
                windowHits = windowHits,
            ),
            hud = noopHud(),
        )

        // "yesterday" strips to a blank embed phrase (parseQuery.timeOnly) — no object named.
        val outcome = searcher.search("what did i see yesterday") as ObjectSearcher.Outcome.Success
        val returnedIds = outcome.cards.map { it.frame.id }

        assertEquals(listOf("d", "e"), returnedIds)
    }
}
