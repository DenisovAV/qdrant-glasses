package tech.qdrant.glasses.search

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

class QueryTextTest {
    @Test fun stripsQuestionBoilerplate() {
        assertEquals("laptop", searchPhrase("where is my laptop"))
        assertEquals("keys", searchPhrase("find the keys"))
        assertEquals("phone", searchPhrase("show me a phone"))
    }
    @Test fun stripsRecallBoilerplate() {
        assertEquals("wallet", searchPhrase("where did i leave my wallet"))
        assertEquals("keys", searchPhrase("where did i put the keys"))
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

    // Fixed reference instant: 2026-08-17 15:00:00 UTC.
    private val nowMs = 1_786_633_200_000L
    private val utc = TimeZone.getTimeZone("UTC")
    private val DAY = 86_400_000L
    private val HOUR = 3_600_000L

    @Test fun noTimePhraseReturnsNull() {
        assertNull(extractTimeWindow("where is my laptop", nowMs, utc))
    }
    @Test fun yesterdayIsTheFullPreviousCalendarDay() {
        val w = extractTimeWindow("what did I see yesterday", nowMs, utc)!!
        // start of today (UTC) = 2026-08-17 00:00:00 UTC = nowMs - 15h
        val startToday = nowMs - 15 * HOUR
        assertEquals(startToday - DAY, w.sinceMs)
        // Exclusive of today's midnight (whole-branch review fix) — a keyframe stored at exactly
        // startToday belongs to "today"'s window, not "yesterday"'s.
        assertEquals(startToday - 1, w.untilMs)
    }
    @Test fun todayIsStartOfDayUntilNow() {
        val w = extractTimeWindow("what did I see today", nowMs, utc)!!
        assertEquals(nowMs - 15 * HOUR, w.sinceMs)
        assertEquals(nowMs, w.untilMs)
    }
    @Test fun yesterdaySpansTheCorrectLocalDayAcrossDst() {
        // 2026-03-08 is the US spring-forward day in America/New_York (clocks jump 02:00 →
        // 03:00 EST → EDT), so that local day is only 23h long. A naive `sod - 24h` (the old
        // implementation) would land an hour into March 7 instead of at March 8's midnight.
        val nyZone = TimeZone.getTimeZone("America/New_York")
        val cal = Calendar.getInstance(nyZone)
        cal.set(2026, Calendar.MARCH, 9, 15, 0, 0)
        cal.set(Calendar.MILLISECOND, 0)
        val nowMsNy = cal.timeInMillis
        val w = extractTimeWindow("what did I see yesterday", nowMsNy, nyZone)!!
        val spanMs = w.untilMs!! - w.sinceMs!!
        assertNotEquals(DAY, spanMs)
        // -1ms for the exclusive-of-midnight upper bound (whole-branch review fix).
        assertEquals(23 * HOUR - 1, spanMs)
    }
    @Test fun lastHourIsRelative() {
        val w = extractTimeWindow("where did I put my keys an hour ago", nowMs, utc)!!
        assertEquals(nowMs - HOUR, w.sinceMs)
        assertEquals(nowMs, w.untilMs)
    }
    @Test fun stripTimePhrasesLeavesTheObject() {
        assertEquals("wallet", stripTimePhrases("wallet yesterday"))
        assertEquals("keys", stripTimePhrases("keys an hour ago"))
    }
    @Test fun recallIntentDetectsLeavePut() {
        assertTrue(isRecallLocationIntent("where did I leave my wallet"))
        assertTrue(isRecallLocationIntent("where did i put the keys"))
        assertFalse(isRecallLocationIntent("show me a laptop"))
    }
}
