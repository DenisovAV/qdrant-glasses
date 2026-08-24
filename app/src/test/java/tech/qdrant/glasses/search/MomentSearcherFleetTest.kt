package tech.qdrant.glasses.search

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowSystemProperties
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.fleet.FleetSource
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.stream.HudPublisher

/**
 * Task 5 (plan, Spec §3/§5): [MomentSearcher] merges local + fleet hits behind the [FleetSource] seam
 * — dedup by `id` (local wins on a collision: local hits are appended before fleet hits, and
 * `distinctBy` keeps the first), each fleet hit tagged `source="fleet"`, gated by the SAME
 * `searchGate` as local hits, AND by `Config.FLEET_URL.isNotBlank()` (review fix: the plan's
 * documented offline/config gate). Deliberately duplicates the small fakes [MomentSearcherTest]
 * defines (they're private nested classes there, not shared) so this stays a plain JVM/Robolectric
 * test with no native `.so` — [FleetSource] is exactly the seam that makes that possible for the
 * fleet side too.
 *
 * [Config.FLEET_URL] is a `val` on a Kotlin `object`, computed once from `android.os.SystemProperties`
 * the first time [Config] is touched in a given Robolectric sandbox — so every test here overrides the
 * sysprop via [ShadowSystemProperties] BEFORE constructing/searching, in [setUp], rather than trying to
 * vary it per-test (a later override cannot un-cache an already-evaluated `val`).
 */
@RunWith(RobolectricTestRunner::class)
class MomentSearcherFleetTest {

    @Before
    fun setUp() {
        // Belt-and-suspenders for the `Config.FLEET_URL.isNotBlank() && fleet != null` gate
        // MomentSearcher.search checks: every test in this file wants the fleet tier ON.
        ShadowSystemProperties.override("debug.qdrant.fleet_url", "http://fleet.test")
    }

    private class FakeCropEncoder(override val dim: Int = 4) : CropEncoder {
        override fun encode(crop: Bitmap): FloatArray = error("not used by MomentSearcher.search")
        override fun encodeText(query: String): FloatArray = FloatArray(dim) { 1f }
        override val visionMinScore: Float = 0f
    }

    private class FakeMomentStore(
        private val frameHits: List<MomentHit>,
        private val regionHits: List<MomentHit>,
    ) : MomentStore {
        override fun storeMoment(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeRegion(clipVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun storeOcr(textVec: FloatArray, payload: MomentPayload) = error("not used")
        override fun searchFrames(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = frameHits
        override fun searchRegions(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = regionHits
        override fun searchText(qvec: FloatArray, topK: Int, sinceMs: Long?, untilMs: Long?) = error("not used")
        override fun timeline(limit: Int): List<MomentHit> = emptyList()
        override fun framesInWindow(sinceMs: Long?, untilMs: Long?, limit: Int): List<MomentHit> = emptyList()
        override fun count(): Long = (frameHits.size + regionHits.size).toLong()
        override fun frameCount(): Long = frameHits.size.toLong()
        override fun deleteAll() {}
        override fun close() {}
    }

    private fun frameHit(momentId: String, score: Float) = MomentHit(
        id = momentId, score = score, type = "frame", momentId = momentId, timestampMs = 0L,
        thumbPath = "$momentId.jpg", label = "", bbox = "",
    )

    private fun noopHud() = HudPublisher(railItems = { emptyList() }, momentItems = { emptyList() })

    @Test fun mergesFleetHitsTaggedAndDeduped() {
        val gate = CropEncoderFactory.searchGate
        val local = FakeMomentStore(frameHits = listOf(frameHit("a", gate + 0.05f)), regionHits = emptyList())
        val fleet = object : FleetSource {
            override fun searchFrames(q: FloatArray, k: Int, s: Long?, u: Long?) = listOf(
                frameHit("a", gate + 0.9f).copy(source = "fleet"),   // dup id -> local kept, fleet dropped
                frameHit("z", gate + 0.06f).copy(source = "fleet"),  // new -> appears, tagged
            )
        }
        val searcher = MomentSearcher(FakeCropEncoder(), local, noopHud(), fleet = fleet)
        val cards = (searcher.search("cup") as ObjectSearcher.Outcome.Success).cards
        val ids = cards.map { it.frame.id }
        assertTrue(ids.contains("a") && ids.contains("z"))
        assertEquals(1, ids.count { it == "a" })   // deduped
    }

    /**
     * Review fix (issue 3): a duplicate id must resolve to the LOCAL copy's own content — not
     * "whichever score is higher". The fleet duplicate for "a" scores far above the local one
     * (gate+0.9 vs gate+0.05); an implementation that dedups by picking the higher score (or that
     * reversed the local/fleet precedence entirely) would surface fleet's score here instead, and
     * the old assertions (id-set + count only) could not tell the difference.
     */
    @Test fun localDuplicateWinsContentOverHigherScoringFleetDuplicate() {
        val gate = CropEncoderFactory.searchGate
        val localScore = gate + 0.05f
        val local = FakeMomentStore(frameHits = listOf(frameHit("a", localScore)), regionHits = emptyList())
        val fleet = object : FleetSource {
            override fun searchFrames(q: FloatArray, k: Int, s: Long?, u: Long?) = listOf(
                frameHit("a", gate + 0.9f).copy(source = "fleet", label = "fleet-label"),
                frameHit("z", gate + 0.06f).copy(source = "fleet"),
            )
        }
        val searcher = MomentSearcher(FakeCropEncoder(), local, noopHud(), fleet = fleet)
        val cards = (searcher.search("cup") as ObjectSearcher.Outcome.Success).cards

        val winner = cards.first { it.frame.id == "a" }
        assertEquals("local content, not fleet's higher score, must win the duplicate", localScore, winner.frame.score)

        // Non-duplicate fleet result "z" must still be preserved in the merged output.
        assertTrue("non-duplicate fleet hit must be preserved", cards.any { it.frame.id == "z" })
    }

    /**
     * Review fix (issue 2): a higher-scoring fleet hit must not be crowded out of the top-5
     * truncation merely because several lower-scoring LOCAL hits were listed first in the
     * concatenation. Five local hits all clear the gate but score well below one distinct
     * (non-duplicate) fleet hit; a naive `(local + fleet).distinctBy{id}.take(5)` merge — local
     * first, fleet appended, no re-sort — drops the fleet hit entirely here, since the five local
     * hits alone already fill the top-5 cap.
     */
    @Test fun higherScoringFleetHitOutranksWeakerLocalHitsInsteadOfBeingTruncated() {
        val gate = CropEncoderFactory.searchGate
        val localHits = (1..5).map { i -> frameHit("l$i", gate + 0.01f * i) }  // l5 is the strongest local
        val local = FakeMomentStore(frameHits = localHits, regionHits = emptyList())
        val fleet = object : FleetSource {
            override fun searchFrames(q: FloatArray, k: Int, s: Long?, u: Long?) = listOf(
                frameHit("f1", gate + 0.5f).copy(source = "fleet"),  // clearly beats every local hit
            )
        }
        val searcher = MomentSearcher(FakeCropEncoder(), local, noopHud(), fleet = fleet)
        val cards = (searcher.search("cup") as ObjectSearcher.Outcome.Success).cards
        val ids = cards.map { it.frame.id }

        assertEquals(5, cards.size)
        assertEquals("the highest-scoring hit overall must rank first", "f1", ids.first())
        assertFalse("the weakest local hit must be the one truncated, not the fleet hit", ids.contains("l1"))
    }
}
