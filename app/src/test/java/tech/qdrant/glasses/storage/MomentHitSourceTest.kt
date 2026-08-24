package tech.qdrant.glasses.storage
import org.junit.Assert.assertEquals
import org.junit.Test
class MomentHitSourceTest {
    @Test fun defaultsToLocal() {
        val h = MomentHit(id="a", score=0.5f, type="frame", momentId="a",
            timestampMs=0L, thumbPath="a.jpg", label="", bbox="")
        assertEquals("local", h.source)
    }
    @Test fun canBeTaggedFleet() {
        val h = MomentHit(id="a", score=0.5f, type="frame", momentId="a",
            timestampMs=0L, thumbPath="a.jpg", label="", bbox="", source="fleet")
        assertEquals("fleet", h.source)
    }
}
