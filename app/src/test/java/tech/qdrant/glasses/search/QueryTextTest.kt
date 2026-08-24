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
    // Re-review guard: a day whose LOCAL MIDNIGHT is skipped by a DST spring-forward is still a valid
    // day and must resolve to THIS year, not walk back. America/Sao_Paulo sprang forward at 00:00 on
    // 2018-11-04, so "november 4" asked days later must resolve to Nov 4 2018 (day-start = 01:00 local,
    // the first valid instant), NOT Nov 4 2017. Validating at noon (never skipped) is what fixes this.
    @Test fun dayWithSkippedLocalMidnightResolvesToThisYear() {
        val sp = TimeZone.getTimeZone("America/Sao_Paulo")
        val nowCal = Calendar.getInstance(sp).apply { clear(); set(2018, Calendar.NOVEMBER, 10, 12, 0, 0) }
        val m = extractAbsoluteDate("what did i see on november 4", nowCal.timeInMillis, sp)!!
        val refNoon = Calendar.getInstance(sp).apply { clear(); set(2018, Calendar.NOVEMBER, 4, 12, 0, 0) }
        assertEquals(startOfDay(refNoon.timeInMillis, sp), m.window.sinceMs)   // Nov 4 2018, not 2017
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

    // --- Coverage gap T8 + re-review: a date-FIRST phrasing keeps a harmless leading stopword ("on")
    // rather than risk stripping a meaningful leading preposition from an object/location phrase. The
    // window and object are what matter; CLIP largely ignores the connector.
    @Test fun dateFirstPhrasingKeepsWindowAndObject() {
        val p = parseQuery("on september 5 wallet", now2, utc)
        assertEquals(dayWindow(2025, 8, 5, utc), p.window)
        assertTrue("wallet" in p.embedText); assertFalse(p.timeOnly)
    }
    // Re-review regression guard: a leading preposition that belongs to the OBJECT/LOCATION phrase
    // must survive even when the query ALSO carries an absolute date later — only the date's own
    // stranded trailing connector ("on") is removed ("at home on september 5" → "at home").
    @Test fun objectWithLeadingPrepositionPlusDate() {
        val p = parseQuery("at home on september 5", now2, utc)
        assertEquals("at home", p.embedText)
        assertEquals(dayWindow(2025, 8, 5, utc), p.window); assertFalse(p.timeOnly)
    }
    // And an ordinary NO-date query keeps its meaningful leading location word untouched.
    @Test fun noDateQueryPreservesLeadingPreposition() {
        assertEquals("in my backpack", parseQuery("in my backpack", now2, utc).embedText)
        assertEquals("at home", parseQuery("at home", now2, utc).embedText)
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

    // --- Relative temporal grammar: weekday / week / month / N-ago / day-before-yesterday.
    // now2 = 2026-08-17 12:00 UTC, a MONDAY. Windows in UTC to match dayWindow.
    private fun monthStart(mo0: Int) =
        Calendar.getInstance(utc).apply { clear(); set(2026, mo0, 1, 0, 0, 0) }.timeInMillis

    @Test fun lastWeekdayResolvesToPastOccurrence() {
        // Aug 11 is the Tuesday before Monday Aug 17.
        assertEquals(dayWindow(2026, 7, 11, utc), extractTimeWindow("what did I see last Tuesday", now2, utc))
        assertEquals(dayWindow(2026, 7, 11, utc), extractTimeWindow("the report on Tuesday", now2, utc))
    }
    @Test fun lastWeekIsPreviousMondayToSunday() {
        val w = extractTimeWindow("desk last week", now2, utc)!!
        val monAug10 = Calendar.getInstance(utc).apply { clear(); set(2026, Calendar.AUGUST, 10, 0, 0, 0) }.timeInMillis
        assertEquals(monAug10, w.sinceMs)                 // Mon Aug 10 00:00
        assertEquals(monAug10 + 7 * DAY - 1, w.untilMs)   // .. Sun Aug 16 23:59:59.999
    }
    @Test fun thisWeekIsMondayUntilNow() {
        val w = extractTimeWindow("what did I see this week", now2, utc)!!
        assertEquals(dayWindow(2026, 7, 17, utc).sinceMs, w.sinceMs)   // Mon Aug 17 00:00
        assertEquals(now2, w.untilMs)
    }
    @Test fun lastMonthIsWholePreviousMonth() {
        val w = extractTimeWindow("what did I see last month", now2, utc)!!
        assertEquals(monthStart(Calendar.JULY), w.sinceMs)
        assertEquals(monthStart(Calendar.AUGUST) - 1, w.untilMs)
    }
    @Test fun inMonthIsWholeNamedMonth() {
        val w = extractTimeWindow("the MCP talk in April", now2, utc)!!
        assertEquals(monthStart(Calendar.APRIL), w.sinceMs)
        assertEquals(monthStart(Calendar.MAY) - 1, w.untilMs)
    }
    @Test fun nDaysAgoIsThatDay() {
        assertEquals(dayWindow(2026, 7, 14, utc), extractTimeWindow("3 days ago", now2, utc))   // 17 - 3
    }
    @Test fun dayBeforeYesterdayIsTwoDaysBack() {
        assertEquals(dayWindow(2026, 7, 15, utc), extractTimeWindow("the day before yesterday", now2, utc))
    }

    // parseQuery integration for the user's three example shapes.
    @Test fun parseObjectPlusRelativeWeek() {
        val p = parseQuery("desk last week", now2, utc)
        assertEquals("desk", p.embedText); assertNotNull(p.window); assertFalse(p.timeOnly)
    }
    @Test fun parsePureTimeWeekdayIsTimeOnly() {
        val p = parseQuery("what did I see last Tuesday", now2, utc)
        assertEquals("", p.embedText); assertNotNull(p.window); assertTrue(p.timeOnly)
    }
    @Test fun parseTopicPlusInMonth() {
        val p = parseQuery("the MCP talk in April", now2, utc)
        assertEquals("mcp talk", p.embedText); assertNotNull(p.window); assertFalse(p.timeOnly)
    }
    // Guard: relative-date additions must not eat a plain object/location query.
    @Test fun plainQueriesUnaffectedByRelativeGrammar() {
        assertNull(extractTimeWindow("where is my laptop", now2, utc))
        assertEquals("in my backpack", parseQuery("in my backpack", now2, utc).embedText)
    }

    // Coverage stress test: 100 realistic queries → dump each parse to a file for human review.
    // Not an assertion suite (always "passes"); the point is to eyeball which phrasings the grammar
    // MISSES (no window where one is meant, boilerplate left in embedText, wrong intent).
    @Test fun dumpHundredQueryParsesForReview() {
        val fmt = java.text.SimpleDateFormat("MM-dd HH:mm").apply { timeZone = utc }
        fun d(ms: Long?) = ms?.let { fmt.format(java.util.Date(it)) } ?: "?"
        val queries = listOf(
            // object recall (no time)
            "where's my laptop", "find my keys", "where did I put my wallet", "my phone",
            "show me my glasses", "where are my headphones", "find the charger", "where's the remote",
            "my backpack", "find my umbrella", "where's my water bottle", "my notebook",
            "show me the scissors", "where did I last see my passport", "the red mug",
            // object + relative time
            "laptop yesterday", "cup this morning", "keys last week", "phone last Tuesday",
            "wallet an hour ago", "bag today", "mug last month", "charger 3 days ago",
            "glasses this week", "bottle last night", "notebook this afternoon", "badge earlier today",
            "book two weeks ago", "remote a few minutes ago", "umbrella the day before yesterday",
            "headphones on Monday", "scissors last Friday", "watch last weekend", "folder this evening",
            "the desk last week",
            // object + absolute date
            "wallet on September 5", "laptop on August 1", "keys on July 4", "phone on March 8",
            "cup on 5 september",
            // pure time
            "what did I see yesterday", "what did I see last week", "what did I see last Tuesday",
            "what did I see this morning", "what did I see in April", "what did I see on September 5",
            "what did I see last month", "what did I see today", "what did I see 3 days ago",
            "what did I see this week", "what did I see last night", "what did I see over the weekend",
            "what did I see a week ago", "what did I see this month", "what did I see on Monday",
            // topic / OCR text
            "the MCP talk in April", "slides about kubernetes", "the whiteboard with the diagram",
            "notes on the roadmap", "the presentation about AI", "the poster in the hallway",
            "the menu at the restaurant", "the sign that said exit", "the receipt from lunch",
            "the code on the screen", "the meeting notes last week", "the error message",
            "the phone number on the card", "the address on the envelope", "the title of the book",
            // recall intent
            "where did I leave my laptop", "where did I put my keys", "where did I last see my wallet",
            "where did I drop my phone", "where did I leave my badge yesterday", "where did I put the charger",
            "where did I last see the remote", "where did I leave my glasses this morning",
            "where did I put my umbrella", "where did I drop my keys last Tuesday",
            // count / aggregation
            "how many cups did I see", "how many times did I see my laptop", "how many people did I see today",
            "how many bottles last week", "how many times did I see the whiteboard",
            // tricky / edge phrasings
            "a couple days ago", "two months ago", "earlier this week", "a while ago", "recently",
            "just now", "over the weekend", "last night", "this evening", "tonight", "a few hours ago",
            "yesterday morning", "sometime last week", "around noon", "at lunch",
        )
        val sb = StringBuilder("n=${queries.size}  now2 = Mon 2026-08-17 12:00 UTC\n\n")
        queries.forEachIndexed { i, q ->
            val p = parseQuery(q, now2, utc)
            val win = p.window?.let { "${d(it.sinceMs)}..${d(it.untilMs)}" } ?: "—"
            sb.appendLine("%3d | %-38s | embed='%s' | win=%s | timeOnly=%s recall=%s"
                .format(i + 1, q, p.embedText, win, p.timeOnly, p.recallIntent))
        }
        java.io.File("/private/tmp/claude-501/-Users-sashadenisov-Work-qdrant-glasses/f9efee29-f096-4cd5-8759-1877d8e07302/scratchpad/query_parses.txt")
            .writeText(sb.toString())
    }

    // Second, adversarial batch (fable-generated): future time, "a fortnight/back", seasons, event-
    // relative, holidays, ISO dates, and false-trigger words ("may"/"march"/"May issue"). Same dump-for-
    // review format; the point is to see false-positives + uncovered phrasings, not to assert.
    @Test fun dumpAgentQueryParsesForReview() {
        val fmt = java.text.SimpleDateFormat("MM-dd HH:mm").apply { timeZone = utc }
        fun d(ms: Long?) = ms?.let { fmt.format(java.util.Date(it)) } ?: "?"
        val queries = listOf(
            "Where did I leave my keys?", "Where's my wallet?", "Where'd I put my phone charger?", "Wheres my badge",
            "What did I see yesterday?", "What did I see last Thursday?",
            "What am I supposed to see this coming Friday on my calendar?", "What did I see the day before yesterday?",
            "Show me what I saw a fortnight ago", "What did I look at 10 days ago?", "What was I looking at half an hour ago?",
            "What did I see a couple weeks back?", "What did I see earlier this afternoon?", "What was on my desk last week?",
            "What did I see in March?", "Show me the whiteboard from early April", "What was I reading at the start of the month?",
            "What did I photograph mid-July?", "Where did I put my gloves last winter?", "What did I see at the beach this summer?",
            "I saw a poster a while back, find it", "That restaurant menu from the other day", "The bike I saw ages ago",
            "What did I see recently?", "What was I just looking at?", "What did I see just now?",
            "Where was my laptop before lunch?", "What was on the screen after the meeting?", "The slides from during the conference",
            "What did I pass on my way home?", "What did I see on my birthday?", "The decorations I saw around Christmas",
            "What did I see on 2026-03-15?", "What was I doing at 3pm yesterday?", "The talk about MCP",
            "The diagram on the whiteboard", "What was the error code on the screen?", "What's the wifi password I saw at the cafe?",
            "What was my gate number?", "The price tag on that jacket", "The slide about vector databases",
            "What did the street sign say near the station?", "The phone number on the flyer",
            "The license plate of the car that hit the pole", "What was written on the meeting room door?",
            "The QR code at the conference booth", "The red notebook on my desk last Tuesday", "The blue mug in the kitchen this morning",
            "The black backpack I had at the airport last month", "The green sticky note on my monitor on Monday",
            "The white cable on the conference table yesterday afternoon", "How many times did I see my keys today?",
            "How often was the laptop open this week?", "How many cups of coffee did I have yesterday?",
            "How many people were in the meeting room?", "How many slides were in that deck?", "What did I not see today?",
            "The last time I saw my wallet", "When did I last see my passport?", "The first slide of the deck",
            "The first time I saw that dog this month", "Which room have I not been in today?", "I may have left my keys at the gym",
            "Did I march to the office or take the bus?", "The May issue of the magazine on the shelf",
            "A date on the calendar I circled", "What's on my second monitor?", "The march schedule for the parade",
            "Was there a sale sign in the window?", "Keys?", "Wallet", "Yesterday", "Glasses case", "Umbrella?",
            "Where did I park the car this morning?", "Where's the remote?", "Where'd I leave my headphones last night?",
            "Did I lock the front door before I left?", "What book was I reading on the train last Wednesday?",
            "Show me the receipt from the pharmacy two days ago", "What did I eat for dinner on Sunday?",
            "The plant I saw at the garden center a month ago", "What was on the kitchen counter an hour ago?",
            "The address on the package that arrived this week", "What did the doctor's prescription say?",
            "The chart from the standup this morning", "Where were my sunglasses the weekend before last?",
            "What did I see between 9 and 10 this morning?", "What was I looking at around noon?",
            "The graffiti I walked past sometime in June", "Did I see my neighbor's cat this evening?",
            "What was the room number of my hotel?", "Show me everything from last Friday evening",
            "The blue folder — did I have it at the office or at home?", "What jacket was I wearing on New Year's Eve?",
            "The screwdriver I used the weekend I fixed the shelf", "What did the parking meter display say?",
            "Anything with a dog in it from this month", "The total on the grocery bill from Tuesday",
            "What was the last thing I saw before my glasses died last night?",
        )
        val sb = StringBuilder("n=${queries.size}  now2 = Mon 2026-08-17 12:00 UTC\n\n")
        queries.forEachIndexed { i, q ->
            val p = parseQuery(q, now2, utc)
            val win = p.window?.let { "${d(it.sinceMs)}..${d(it.untilMs)}" } ?: "—"
            sb.appendLine("%3d | %-52s | embed='%s' | win=%s | tOnly=%s rc=%s"
                .format(i + 1, q.take(52), p.embedText, win, p.timeOnly, p.recallIntent))
        }
        java.io.File("/private/tmp/claude-501/-Users-sashadenisov-Work-qdrant-glasses/f9efee29-f096-4cd5-8759-1877d8e07302/scratchpad/query_parses_agent.txt")
            .writeText(sb.toString())
    }
}
