package tech.qdrant.glasses.search

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import tech.qdrant.glasses.storage.MomentHit

/** Pure-function tests for [fuseAndCollapse]/[softBoost] (episodic-memory plan Task 2.3, Spec §3). */
class MomentFusionTest {

    private fun frameHit(
        momentId: String,
        score: Float,
        thumbPath: String = "$momentId.jpg",
        id: String = momentId,
        timestampMs: Long = 0L,
    ) = MomentHit(
        id = id, score = score, type = "frame", momentId = momentId, timestampMs = timestampMs,
        thumbPath = thumbPath, label = "", bbox = "",
    )

    private fun regionHit(
        momentId: String,
        score: Float,
        label: String = "",
        yoloConf: Float = 0f,
        verifyCos: Float = 0f,
        thumbPath: String = "$momentId.jpg",
        id: String = "$momentId-region",
    ) = MomentHit(
        id = id, score = score, type = "region", momentId = momentId, timestampMs = 0L,
        thumbPath = thumbPath, label = label, bbox = "0.1,0.1,0.2,0.2",
        yoloConf = yoloConf, verifyCos = verifyCos,
    )

    // ---- fuseAndCollapse -------------------------------------------------------------------

    @Test fun collapsesFrameAndRegionsOfSameMomentToMaxScore() {
        val frame = frameHit("m1", score = 0.30f)
        val r1 = regionHit("m1", score = 0.50f, label = "cup")
        val r2 = regionHit("m1", score = 0.20f, label = "mug", id = "m1-region-2")
        val fused = fuseAndCollapse(frameHits = listOf(frame), regionHits = listOf(r1, r2))

        assertEquals(1, fused.size)
        assertEquals("m1", fused[0].momentId)
        assertEquals(0.50f, fused[0].score, 1e-6f)
        // Identity/thumb come from the FRAME hit, not the winning region.
        assertEquals(frame.id, fused[0].id)
        assertEquals(frame.thumbPath, fused[0].thumbPath)
    }

    @Test fun regionOnlyMomentStillAppearsExactlyOnce() {
        val r1 = regionHit("m2", score = 0.40f, label = "lamp", thumbPath = "parent-m2.jpg")
        val r2 = regionHit("m2", score = 0.35f, label = "lamp", id = "m2-region-2", thumbPath = "parent-m2.jpg")
        val fused = fuseAndCollapse(frameHits = emptyList(), regionHits = listOf(r1, r2))

        assertEquals(1, fused.size)
        val hit = fused[0]
        assertEquals("m2", hit.momentId)
        assertEquals(0.40f, hit.score, 1e-6f)
        // Synthesized from the parent moment: id == momentId, type == "frame", thumb == parent thumb
        // (a region's own thumb_path already IS the parent's — Spec §6).
        assertEquals("m2", hit.id)
        assertEquals("frame", hit.type)
        assertEquals("parent-m2.jpg", hit.thumbPath)
    }

    @Test fun distinctMomentsStayDistinct() {
        val frame1 = frameHit("m1", score = 0.30f)
        val frame2 = frameHit("m2", score = 0.60f)
        val region3 = regionHit("m3", score = 0.10f, label = "key")
        val fused = fuseAndCollapse(listOf(frame1, frame2), listOf(region3))

        assertEquals(3, fused.size)
        assertEquals(setOf("m1", "m2", "m3"), fused.map { it.momentId }.toSet())
        // Sorted descending by fused score.
        assertEquals(listOf("m2", "m1", "m3"), fused.map { it.momentId })
    }

    // ---- softBoost --------------------------------------------------------------------------

    @Test fun matchingVerifiedTagBoostsScore() {
        val hit = frameHit("m1", score = 0.30f)
        val region = regionHit("m1", score = 0.10f, label = "cup", yoloConf = 0.9f, verifyCos = 0.8f)
        val boosted = softBoost(listOf(hit), regionHits = listOf(region), queryTokens = setOf("cup"), lambda = 0.05f)

        val expected = 0.30f + 0.05f * 0.9f * 0.8f
        assertEquals(expected, boosted[0].score, 1e-6f)
        assertTrue(boosted[0].score > hit.score)
    }

    @Test fun boundedBoostCannotOutrankAMuchStrongerFrameHit() {
        // A strong, correct frame-vector hit vs a weak vector match that also happens to carry a
        // matching verified tag — the tag must nudge, never flip the ranking (Spec §3).
        val strongFrame = frameHit("strong", score = 0.90f)
        val weakFrameWithTag = frameHit("weak", score = 0.21f)
        val weakTagRegion = regionHit("weak", score = 0.21f, label = "cup", yoloConf = 1.0f, verifyCos = 1.0f)

        val boosted = softBoost(
            hits = listOf(strongFrame, weakFrameWithTag),
            regionHits = listOf(weakTagRegion),
            queryTokens = setOf("cup"),
            lambda = TAG_BOOST_LAMBDA,
        )
        val strongResult = boosted.first { it.momentId == "strong" }
        val weakResult = boosted.first { it.momentId == "weak" }

        // Max possible boost is TAG_BOOST_LAMBDA * 1.0 * 1.0 == TAG_BOOST_LAMBDA (0.05) — nowhere
        // near enough to close a 0.69 gap.
        assertEquals(0.21f + TAG_BOOST_LAMBDA, weakResult.score, 1e-6f)
        assertTrue(strongResult.score > weakResult.score)
    }

    @Test fun nonMatchingTagDoesNotBoost() {
        val hit = frameHit("m1", score = 0.30f)
        val region = regionHit("m1", score = 0.10f, label = "cup", yoloConf = 0.9f, verifyCos = 0.8f)
        val boosted = softBoost(listOf(hit), regionHits = listOf(region), queryTokens = setOf("laptop"), lambda = 0.05f)

        assertEquals(0.30f, boosted[0].score, 1e-6f)
    }

    @Test fun emptyLabelRegionNeverBoostsEvenOnATokenMatch() {
        // An UNVERIFIED region (verify_cos < VERIFY_COS at capture time) is stored with an empty
        // label — MomentCapture drops the tag but keeps the vector. A query token that happens to
        // match some OTHER field must not boost via this region: labelMatchesQuery("", …) is checked
        // against the (empty) label, not the bbox/anything else.
        val hit = frameHit("m1", score = 0.30f)
        val unverifiedRegion = regionHit("m1", score = 0.10f, label = "", yoloConf = 0.95f, verifyCos = 0.95f)
        val boosted = softBoost(listOf(hit), regionHits = listOf(unverifiedRegion), queryTokens = setOf("cup"), lambda = 0.05f)

        assertEquals(0.30f, boosted[0].score, 1e-6f)
    }

    @Test fun regionOnlyMomentCanAlsoBeBoosted() {
        // fuseAndCollapse then softBoost, chained — the realistic MomentSearcher pipeline for a
        // small object the frame channel never surfaced.
        val region = regionHit("m2", score = 0.40f, label = "lamp", yoloConf = 0.8f, verifyCos = 0.5f)
        val fused = fuseAndCollapse(emptyList(), listOf(region))
        val boosted = softBoost(fused, listOf(region), queryTokens = setOf("lamp"), lambda = 0.05f)

        val expected = 0.40f + 0.05f * 0.8f * 0.5f
        assertEquals(expected, boosted[0].score, 1e-6f)
    }
}
