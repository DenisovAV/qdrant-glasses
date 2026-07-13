package tech.qdrant.glasses.search

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QueryTextTest {
    @Test fun stripsQuestionBoilerplate() {
        assertEquals("laptop", searchPhrase("where is my laptop"))
        assertEquals("keys", searchPhrase("find the keys"))
        assertEquals("phone", searchPhrase("show me a phone"))
    }
    @Test fun stripsLeadingArticleOnly() {
        assertEquals("cup", searchPhrase("the cup"))
    }
    @Test fun blankResultFallsBackToOriginal() {
        assertEquals("where is", searchPhrase("where is"))   // nothing left after strip → original
    }
    @Test fun labelMatchHandlesContainmentBothWays() {
        val t = queryTokens(searchPhrase("phone"))
        assertTrue(labelMatchesQuery("cell phone", t))                    // "phone" == token in "cell phone"
        assertTrue(labelMatchesQuery("laptops", queryTokens("laptop")))   // >=4-char containment both ways
        // Guard: the matcher requires length >= 4 for CONTAINMENT (tuned to avoid junk substrings),
        // so a 3-char query word matches a longer label ONLY via exact equality, not containment.
        // (The production comment cites "cups ⊃ cup" but the code's length guard actually rejects it.)
        assertFalse(labelMatchesQuery("cups", queryTokens("cup")))
    }
    @Test fun labelMatchRejectsShortJunkSubstrings() {
        assertFalse(labelMatchesQuery("cat", queryTokens("laptop")))
    }
}
