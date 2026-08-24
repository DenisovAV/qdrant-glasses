package tech.qdrant.glasses.fleet

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * On-device verification of [FleetShardStore] (plan Task 4): builds a tiny Edge shard on disk via
 * [FleetShardStore.seedForTest] (the same `EdgeConfig`/schema [tech.qdrant.glasses.storage
 * .QdrantEdgeMomentStore] provisions — named vectors `"clip"` 768-dim + `"text"` 384-dim, COSINE),
 * loads it read-only with [FleetShardStore.load], and checks a nearest-neighbor query against that
 * seeded point comes back tagged `source="fleet"` (Spec §3/§6). No mocks — this IS the unknown being
 * resolved: does a pulled/local Edge shard round-trip through the SAME schema the local moment store
 * uses. NOT run by this task (no physical device attached) — verified via
 * `./gradlew :app:compileDemoDebugKotlin` only.
 */
class FleetShardStoreTest {
    @Test fun searchReturnsFleetTaggedHit() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = ctx.filesDir.resolve("fleet_test").absolutePath
        // Build a shard with one "clip" point via the same EdgeShard API QdrantEdgeMomentStore uses.
        FleetShardStore.seedForTest(dir, clipDim = 768, id = "f1",
            vec = FloatArray(768) { 0.01f }, label = "fleet-cup", ts = 111L)
        val store = FleetShardStore.load(dir, clipDim = 768)
        val hits = store.searchFrames(FloatArray(768) { 0.01f }, topK = 5, sinceMs = null, untilMs = null)
        assertTrue(hits.isNotEmpty())
        assertEquals("fleet", hits.first().source)
        assertEquals("fleet-cup", hits.first().label)
        store.close()
    }
}
