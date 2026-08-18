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

    // Fixed reference instant: 2026-08-13 15:00:00 UTC (verified from the epoch ms).
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

    // Reference: 2026-08-17 12:00:00 UTC (a Monday, verified from the epoch ms). DAY = 86_400_000.
    private val now2 = 1_786_968_000_000L

    private fun dayWindow(y: Int, mo: Int, d: Int, zone: TimeZone): TimeWindow {
        val c = java.util.Calendar.getInstance(zone)
        c.clear(); c.set(y, mo, d, 0, 0, 0)          // mo is 0-based
        val start = c.timeInMillis
        return TimeWindow(start, start + 86_400_000L - 1)
    }

    @Test fun englishNamedMonthDate() {
        val m = extractAbsoluteDate("what did I see on August 1", now2, utc)!!
        assertEquals(dayWindow(2026, 7, 1, utc), m.window)     // Aug 1 already passed this year
    }
    @Test fun futureBareDateResolvesToLastYear() {
        val m = extractAbsoluteDate("what did I see on September 5", now2, utc)!!
        assertEquals(dayWindow(2025, 8, 5, utc), m.window)     // Sept 5 hasn't happened in 2026 yet
    }
    @Test fun russianNamedMonthDate() {
        val m = extractAbsoluteDate("что я видел 5 сентября", now2, utc)!!
        assertEquals(dayWindow(2025, 8, 5, utc), m.window)
    }
    @Test fun abbreviatedMonth() {
        val m = extractAbsoluteDate("anything from sept 5", now2, utc)!!
        assertEquals(dayWindow(2025, 8, 5, utc), m.window)
    }
    @Test fun absoluteDateWindowSpansTheCorrectLocalDayAcrossDst() {
        // Review fix: extractAbsoluteDate's upper bound used to be a fixed `+ 86_400_000L - 1`,
        // the same DST bug yesterdaySpansTheCorrectLocalDayAcrossDst caught in extractTimeWindow
        // (2026-03-08 is the US spring-forward day in America/New_York — a 23h local day, not
        // 24h). Query "now" from the day after so the matched date resolves to March 8 itself.
        val nyZone = TimeZone.getTimeZone("America/New_York")
        val cal = Calendar.getInstance(nyZone)
        cal.set(2026, Calendar.MARCH, 9, 15, 0, 0)
        cal.set(Calendar.MILLISECOND, 0)
        val nowMsNy = cal.timeInMillis
        val m = extractAbsoluteDate("what did I see on March 8", nowMsNy, nyZone)!!
        val spanMs = m.window.untilMs!! - m.window.sinceMs!!
        assertNotEquals(DAY, spanMs)
        assertEquals(23 * HOUR - 1, spanMs)
    }
    @Test fun noDateReturnsNull() {
        assertNull(extractAbsoluteDate("where is my laptop", now2, utc))
        assertNull(extractAbsoluteDate("what did I see yesterday", now2, utc))   // relative, not absolute
    }
    @Test fun matchedSpanCoversTheDateTokens() {
        val m = extractAbsoluteDate("wallet on september 5 please", now2, utc)!!
        assertEquals("september 5", "wallet on september 5 please".substring(m.matchedSpan))
    }

    @Test fun stripDateLeavesTheObject() {
        val m = extractAbsoluteDate("wallet on september 5", now2, utc)!!
        assertEquals("wallet on", stripDateSpan("wallet on september 5", m.matchedSpan))
    }
    @Test fun stripDatePureTimeGoesBlankish() {
        val m = extractAbsoluteDate("what did i see on september 5", now2, utc)!!
        // after removing the date span, only boilerplate remains — searchPhrase will finish the job
        assertEquals("what did i see on", stripDateSpan("what did i see on september 5", m.matchedSpan))
    }

    @Test fun parseObjectPlusAbsoluteDate() {
        val p = parseQuery("wallet on september 5", now2, utc)
        assertEquals("wallet", p.embedText); assertNotNull(p.window); assertFalse(p.timeOnly)
    }
    @Test fun parsePureTimeQuery() {
        val p = parseQuery("what did i see on september 5", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }
    @Test fun parseRelativeStillWorks() {
        val p = parseQuery("where did i leave my keys yesterday", now2, utc)
        assertEquals("keys", p.embedText); assertNotNull(p.window); assertTrue(p.recallIntent); assertFalse(p.timeOnly)
    }
    @Test fun parsePlainObjectNoTime() {
        val p = parseQuery("laptop", now2, utc)
        assertEquals("laptop", p.embedText); assertNull(p.window); assertFalse(p.timeOnly)
    }

    // Review fix: a RELATIVE pure-time query ("yesterday") must collapse to timeOnly too, the
    // same way an ABSOLUTE pure-time query already does (parsePureTimeQuery above) — before this
    // fix, embedText stayed "what did i see" (the un-stripped question stem) and timeOnly stayed
    // false, so the searcher would CLIP-embed garbage instead of running a clean time-only scan.
    @Test fun parseRelativePureTimeQueryIsTimeOnly() {
        val p = parseQuery("what did i see yesterday", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }
    @Test fun parseRelativePureTimeQueryTodayIsTimeOnly() {
        val p = parseQuery("what did i see today", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }

    // --- Review fix (Codex P2 #1): an IMPOSSIBLE calendar day must be REJECTED, not silently rolled
    // by a lenient Calendar into a different real day (Feb 31 → Mar 3), which would both pollute the
    // strip and filter the search to the wrong day. extractAbsoluteDate returns null → no window.
    @Test fun impossibleDayReturnsNull() {
        assertNull(extractAbsoluteDate("what did i see on february 31", now2, utc))   // Feb never has 31
        assertNull(extractAbsoluteDate("what did i see on february 30", now2, utc))   // Feb never has 30
        assertNull(extractAbsoluteDate("what did i see on april 31", now2, utc))      // April has 30
    }
    // Feb 29 is a REAL recurring date, not an impossible one: with no year, it resolves to the most
    // recent PAST leap year. now2 is 2026 (common), so the answer is Feb 29 2024, NOT null.
    @Test fun leapDayResolvesToMostRecentPastLeapYear() {
        val m = extractAbsoluteDate("что я видел 29 февраля", now2, utc)!!
        assertEquals(dayWindow(2024, 1, 29, utc), m.window)
        assertEquals(dayWindow(2024, 1, 29, utc), extractAbsoluteDate("february 29", now2, utc)!!.window)
    }
    @Test fun impossibleDateFallsThroughParseQueryWithNoWindow() {
        // The whole pipeline must degrade gracefully: an impossible date isn't a window, and the
        // leftover text is embedded normally rather than the query silently searching March 3.
        val p = parseQuery("laptop on february 31", now2, utc)
        assertNull(p.window); assertFalse(p.timeOnly)
        // An impossible date is NOT a date: its span is left in the query as ordinary text (not
        // stripped as if valid), and with no window the searcher just embeds it — no wrong-day filter.
        assertTrue("february" in p.embedText)
    }

    // --- Review fix (Codex P2 #2): trailing punctuation and natural capitalization must not defeat
    // the pure-time path. "What did I see on September 5?" is the user's literal use case.
    @Test fun punctuatedPureTimeQueryIsTimeOnly() {
        val p = parseQuery("what did i see on september 5?", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }
    @Test fun naturalCasePureTimeQueryIsTimeOnly() {
        val p = parseQuery("What did I see on September 5?", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }
    @Test fun punctuatedObjectPlusDateStripsCleanly() {
        val p = parseQuery("laptop on July 31?", now2, utc)
        assertEquals("laptop", p.embedText); assertNotNull(p.window); assertFalse(p.timeOnly)
    }

    // --- Coverage gap T8: symmetric leading-preposition strip for a date-FIRST phrasing.
    @Test fun dateFirstPhrasingStripsLeadingPreposition() {
        val p = parseQuery("on september 5 wallet", now2, utc)
        assertEquals("wallet", p.embedText); assertNotNull(p.window); assertFalse(p.timeOnly)
    }

    // --- Coverage gap T1: a BARE object + relative time (no recall wrapper) must keep the object.
    // This is the concrete over-strip guard for stripDateAdjacentBoilerplate's unconditional
    // TIME_PHRASE strip — "laptop" must survive "yesterday" being removed.
    @Test fun bareObjectPlusRelativeTimeKeepsObject() {
        val p1 = parseQuery("laptop yesterday", now2, utc)
        assertEquals("laptop", p1.embedText); assertNotNull(p1.window); assertFalse(p1.timeOnly)
        val p2 = parseQuery("keys today", now2, utc)
        assertEquals("keys", p2.embedText); assertNotNull(p2.window); assertFalse(p2.timeOnly)
    }

    // --- Coverage gap T2: the year-rollback comparison is `>` (strictly future), so asking about
    // TODAY'S own date by name must resolve to today, not roll back a year. now2 is 2026-08-17 UTC
    // (verified from the epoch ms — the class's "2026-08-18" comment is stale), so "august 17" == today.
    @Test fun namedDateEqualToTodayStaysThisYear() {
        val m = extractAbsoluteDate("what did i see on august 17", now2, utc)!!
        assertEquals(dayWindow(2026, 7, 17, utc), m.window)
    }

    // --- Coverage gap T3: ordinal day suffixes ("5th"), a KDoc-claimed supported form.
    @Test fun ordinalDaySuffix() {
        val m = extractAbsoluteDate("what did i see on sept 5th", now2, utc)!!
        assertEquals(dayWindow(2025, 8, 5, utc), m.window)
        assertEquals("sept 5th", "what did i see on sept 5th".substring(m.matchedSpan))
    }

    // --- Coverage gap T5: English DAY-FIRST order ("5 September" / "5th September"), previously only
    // the Russian "5 сентября" exercised the day1/month1 regex branch.
    @Test fun englishDayFirstOrder() {
        assertEquals(dayWindow(2025, 8, 5, utc), extractAbsoluteDate("5 september", now2, utc)!!.window)
        assertEquals(dayWindow(2025, 8, 5, utc), extractAbsoluteDate("the 5th september", now2, utc)!!.window)
    }

    // --- Coverage gap T7: when a query carries BOTH an absolute date and a relative phrase, the
    // absolute date wins (parseQuery's `dateMatch?.window ?: extractTimeWindow`).
    @Test fun absoluteDateWinsOverRelativePhrase() {
        val p = parseQuery("keys yesterday on september 5", now2, utc)
        assertEquals(dayWindow(2025, 8, 5, utc), p.window)   // Sept 5, NOT yesterday
        assertEquals("keys", p.embedText); assertFalse(p.timeOnly)
    }
}
