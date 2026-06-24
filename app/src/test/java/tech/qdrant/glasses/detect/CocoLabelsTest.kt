package tech.qdrant.glasses.detect

import org.junit.Assert.assertEquals
import org.junit.Test

class CocoLabelsTest {
    @Test fun has80Labels() = assertEquals(80, CocoLabels.NAMES.size)
    @Test fun firstIsPerson() = assertEquals("person", CocoLabels[0])
    @Test fun knownIndices() {
        assertEquals("bicycle", CocoLabels[1]); assertEquals("car", CocoLabels[2]); assertEquals("toothbrush", CocoLabels[79])
    }
    @Test fun outOfRangeIsSafe() = assertEquals("unknown", CocoLabels[999])
}
