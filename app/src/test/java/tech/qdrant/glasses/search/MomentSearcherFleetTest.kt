package tech.qdrant.glasses.search

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import tech.qdrant.glasses.embedding.CropEncoder
import tech.qdrant.glasses.embedding.CropEncoderFactory
import tech.qdrant.glasses.fleet.FleetSource
import tech.qdrant.glasses.storage.MomentHit
import tech.qdrant.glasses.storage.MomentPayload
import tech.qdrant.glasses.storage.MomentStore
import tech.qdrant.glasses.stream.HudPublisher

/**
 * Task 5 (plan, Spec §3/§5): [MomentSearcher] merges local + fleet hits behind the [FleetSource] seam
 * — dedup by `id` (local wins on a tie: local hits are appended before fleet hits, and
 * `distinctBy` keeps the first), each fleet hit tagged `source="fleet"`, gated by the SAME
 * `searchGate` as local hits. Deliberately duplicates the small fakes [MomentSearcherTest] defines
 * (they're private nested classes there, not shared) so this stays a plain JVM/Robolectric test with
 * no native `.so` — [FleetSource] is exactly the seam that makes that possible for the fleet side too.
 */
@RunWith(RobolectricTestRunner::class)
class MomentSearcherFleetTest {

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
}
