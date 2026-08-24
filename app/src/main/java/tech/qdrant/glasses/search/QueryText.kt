package tech.qdrant.glasses.search

/** "where is my laptop" → "laptop", "where did i leave my wallet" → "wallet". SigLIP2's
 *  text→crop scale is compressed, so question boilerplate ("where is …", "where did i leave …")
 *  dips a real match under the gate — search the object phrase, display the full query.
 *  Falls back to the original query if stripping leaves nothing. */
fun searchPhrase(rawQuery: String): String =
    rawQuery.lowercase()
        .replace(Regex("^(where(?:\\s+did|'?d)\\s+i\\s+(leave|put|last\\s+see|drop)|where'?s|what'?s|when'?s|where\\s+(is|are)|what\\s+(is|are)|when\\s+(is|are)|that\\s+is|this\\s+is|find|show\\s+me|look\\s+for|search\\s+for)\\s+"), "")
        .replace(Regex("^(my|the|a|an)\\s+"), "")
        .trim().ifBlank { rawQuery }

fun queryTokens(searchPhrase: String): Set<String> =
    searchPhrase.split(Regex("\\W+")).filter { it.length > 2 }.toSet()

/** Hybrid-acceptance label match: token equality or ≥4-char containment either way
 *  ("smartphone" ⊃ "phone" ⊂ "cell phone", "cups" ⊃ "cup"). */
fun labelMatchesQuery(label: String, queryTokens: Set<String>): Boolean {
    val lTokens = label.lowercase().split(Regex("\\W+")).filter { it.length > 2 }
    return lTokens.any { lt ->
        queryTokens.any { qt ->
            qt == lt || (lt.length >= 4 && qt.contains(lt)) || (qt.length >= 4 && lt.contains(qt))
        }
    }
}

data class TimeWindow(val sinceMs: Long?, val untilMs: Long?)

fun startOfDay(nowMs: Long, zone: java.util.TimeZone): Long {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = nowMs
    c.set(java.util.Calendar.HOUR_OF_DAY, 0); c.set(java.util.Calendar.MINUTE, 0)
    c.set(java.util.Calendar.SECOND, 0); c.set(java.util.Calendar.MILLISECOND, 0)
    return c.timeInMillis
}

/** Start of the previous local day. A fixed 86_400_000ms subtraction from [startOfDay] is
 *  wrong across a DST transition (the prior local day is 23h or 25h, not 24h) — step the
 *  Calendar a whole date back instead, so it lands on the correct local midnight either way. */
private fun startOfPreviousDay(nowMs: Long, zone: java.util.TimeZone): Long {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = startOfDay(nowMs, zone)
    c.add(java.util.Calendar.DATE, -1)
    return c.timeInMillis
}

/** Start of the day AFTER [nowMs]'s local day — same DST-safety rationale as
 *  [startOfPreviousDay] (a fixed 86_400_000ms addition is wrong across a DST transition; the
 *  local day can be 23h or 25h, not 24h), mirrored forward instead of back. */
private fun startOfNextDay(nowMs: Long, zone: java.util.TimeZone): Long {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = startOfDay(nowMs, zone)
    c.add(java.util.Calendar.DATE, 1)
    return c.timeInMillis
}

/** Local Monday-00:00 of the week containing [nowMs]. Walks back a whole DATE at a time (DST-safe,
 *  same rationale as [startOfPreviousDay]) rather than subtracting fixed 24h chunks. */
private fun startOfWeek(nowMs: Long, zone: java.util.TimeZone): Long {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = startOfDay(nowMs, zone)
    while (c.get(java.util.Calendar.DAY_OF_WEEK) != java.util.Calendar.MONDAY) c.add(java.util.Calendar.DATE, -1)
    return c.timeInMillis
}

/** Local 1st-of-month 00:00 of the month containing [nowMs]. */
private fun startOfMonth(nowMs: Long, zone: java.util.TimeZone): Long {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = startOfDay(nowMs, zone)
    c.set(java.util.Calendar.DAY_OF_MONTH, 1)
    return c.timeInMillis
}

/** Step [start] (a day/week/month-start instant) forward/back by [field] units, DST-safe via Calendar. */
private fun shift(start: Long, zone: java.util.TimeZone, field: Int, amount: Int): Long {
    val c = java.util.Calendar.getInstance(zone); c.timeInMillis = start; c.add(field, amount); return c.timeInMillis
}

private val WEEKDAYS: Map<String, Int> = mapOf(
    "monday" to java.util.Calendar.MONDAY, "tuesday" to java.util.Calendar.TUESDAY,
    "wednesday" to java.util.Calendar.WEDNESDAY, "thursday" to java.util.Calendar.THURSDAY,
    "friday" to java.util.Calendar.FRIDAY, "saturday" to java.util.Calendar.SATURDAY,
    "sunday" to java.util.Calendar.SUNDAY,
)

/** The single-day window for the most-recent occurrence of weekday [dow] that is not in the future.
 *  [strictlyPast] (the "last <weekday>" form) skips today even when today IS that weekday, landing a
 *  whole week back — bare "<weekday>"/"on <weekday>" resolves to today in that case. */
private fun weekdayWindow(nowMs: Long, zone: java.util.TimeZone, dow: Int, strictlyPast: Boolean): TimeWindow {
    val c = java.util.Calendar.getInstance(zone)
    c.timeInMillis = startOfDay(nowMs, zone)
    if (strictlyPast) c.add(java.util.Calendar.DATE, -1)
    var guard = 0
    while (c.get(java.util.Calendar.DAY_OF_WEEK) != dow && guard < 7) { c.add(java.util.Calendar.DATE, -1); guard++ }
    val start = c.timeInMillis
    return TimeWindow(start, startOfNextDay(start, zone) - 1)
}

/** Whole-month window for the most-recent PAST occurrence of month [monthIndex] (0-based), in [zone] —
 *  the "in April" / bare-month form (a named month with NO day; [extractAbsoluteDate] handles day+month
 *  and runs first in [parseQuery], so this only ever fires when no day accompanies the month). Resolves
 *  the year like [extractAbsoluteDate]: current year, stepped back one if that month hasn't started yet. */
private fun monthOnlyWindow(nowMs: Long, zone: java.util.TimeZone, monthIndex: Int): TimeWindow {
    val cal = java.util.Calendar.getInstance(zone); cal.timeInMillis = nowMs
    var year = cal.get(java.util.Calendar.YEAR)
    val startThisYear = java.util.Calendar.getInstance(zone).apply { clear(); set(year, monthIndex, 1, 0, 0, 0) }.timeInMillis
    if (startThisYear > startOfMonth(nowMs, zone)) year -= 1   // month is still ahead this year → last year
    val start = java.util.Calendar.getInstance(zone).apply { clear(); set(year, monthIndex, 1, 0, 0, 0) }.timeInMillis
    return TimeWindow(start, shift(start, zone, java.util.Calendar.MONTH, 1) - 1)
}

// Relative expressions handled by [extractTimeWindow] beyond the fixed phrases: an OPTIONAL "last "/"on "
// prefix + a weekday; "last/this week|month"; "N days/weeks/hours/minutes ago"; "the day before yesterday";
// and a bare month name ("in April"). Kept as one alternation so [stripRelativeDate] can remove exactly
// what matched from the embed text. `(?<!\p{L})`/`(?!\p{L})` are Unicode-letter boundaries (plain `\b` is
// ASCII-only — see MONTH_TOKEN's KDoc).
private val WEEKDAY_ALT = WEEKDAYS.keys.joinToString("|")
private val REL_WEEKDAY = Regex("\\b(?:(last|this|on)\\s+)?($WEEKDAY_ALT)(?!\\p{L})")
private val REL_WEEK    = Regex("\\b(last|this)\\s+week\\b")
private val REL_MONTH   = Regex("\\b(last|this)\\s+month\\b")
// "the weekend before last" = the weekend TWO weeks back — matched (and stripped) before REL_WEEKEND,
// which would otherwise claim its leading "weekend" and resolve the wrong (most-recent) weekend.
private val REL_WEEKEND_BEFORE_LAST = Regex("\\b(?:the\\s+)?weekend\\s+before\\s+last\\b")
// A bare "the weekend" is deliberately NOT here: it collides with event-relative phrasing ("the weekend
// I fixed the shelf") where "weekend" is a noun the following clause qualifies, not "last weekend". Only
// an explicit qualifier (last/this/over the) reads as a time window.
private val REL_WEEKEND = Regex("\\b(?:last\\s+|this\\s+|over\\s+the\\s+)weekend\\b")
// "a fortnight ago" = two weeks back (British for 14 days). Same week-window shape as "two weeks ago".
private val REL_FORTNIGHT = Regex("\\ba\\s+fortnight\\s+ago\\b")
// "last night"/"this evening"/"tonight" — evening-of-day windows (handled in extractTimeWindow).
private val REL_EVENING = Regex("\\b(last\\s+night|this\\s+evening|tonight)\\b")
// FUTURE weekday ("next Friday", "this coming Monday") — there are no memories ahead of now, so this
// must NOT resolve to the most-recent PAST weekday (a silent wrong window). Detected to SUPPRESS the
// REL_WEEKDAY branch, not to build a window.
private val FUTURE_WEEKDAY = Regex("\\b(next|coming|upcoming)\\s+($WEEKDAY_ALT)\\b")

// Spelled-out counts so "two weeks ago" / "a couple days ago" / "a few minutes ago" / "a week ago"
// parse like "3 days ago" (the regex only understood \d+). Longest alternatives first so "a couple of"
// wins over "a couple" wins over bare "a". `month` added to the units (was day/week/hour/minute only).
private val NUM_WORDS = mapOf(
    "a couple of" to 2, "a couple" to 2, "a few" to 3, "several" to 3,
    "a" to 1, "an" to 1, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6,
    "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10, "eleven" to 11, "twelve" to 12,
)
private val NUM_ALT = "a couple of|a couple|a few|several|\\d+|" +
    "an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
// "…ago" or the equally natural "…back" ("two weeks back", "a couple weeks back"). "back" is only
// reached AFTER a <count> <unit> prefix, so a stray "back" ("the back room", "bring it back") can't fire.
private val REL_N_AGO = Regex("\\b($NUM_ALT)\\s+(day|week|month|hour|minute)s?\\s+(?:ago|back)\\b")
private fun countWord(s: String): Int = s.toIntOrNull() ?: NUM_WORDS[s] ?: 1

private val REL_DAY_BEFORE = Regex("\\bthe\\s+day\\s+before\\s+yesterday\\b")

/** Set [dayStart] (a local midnight) to [hour]:00 the same day. Used for evening-of-day windows. */
private fun atHour(dayStart: Long, zone: java.util.TimeZone, hour: Int): Long {
    val c = java.util.Calendar.getInstance(zone); c.timeInMillis = dayStart
    c.set(java.util.Calendar.HOUR_OF_DAY, hour); return c.timeInMillis
}

/** Map a few spoken time phrases to a [TimeWindow]; null if the query names no time.
 *  Calendar-accurate for "today"/"yesterday"/weekdays/weeks/months in [zone]; relative for hour-scale. */
fun extractTimeWindow(
    raw: String,
    nowMs: Long,
    zone: java.util.TimeZone = java.util.TimeZone.getDefault(),
): TimeWindow? {
    val q = raw.lowercase()
    val sod = startOfDay(nowMs, zone)
    // --- Relative dates beyond the fixed phrases below (early-return on the first match). "the day
    // before yesterday" is tested BEFORE the fixed "yesterday" (which it literally contains).
    REL_DAY_BEFORE.find(q)?.let {
        val s = startOfPreviousDay(startOfPreviousDay(nowMs, zone), zone)
        return TimeWindow(s, startOfNextDay(s, zone) - 1)
    }
    REL_WEEK.find(q)?.let {
        val thisWeek = startOfWeek(nowMs, zone)
        return if (it.groupValues[1] == "this") TimeWindow(thisWeek, nowMs)
               else TimeWindow(shift(thisWeek, zone, java.util.Calendar.DATE, -7), thisWeek - 1)
    }
    REL_MONTH.find(q)?.let {
        val thisMonth = startOfMonth(nowMs, zone)
        return if (it.groupValues[1] == "this") TimeWindow(thisMonth, nowMs)
               else TimeWindow(shift(thisMonth, zone, java.util.Calendar.MONTH, -1), thisMonth - 1)
    }
    REL_EVENING.find(q)?.let { m ->
        if (m.value == "last night") return TimeWindow(atHour(startOfPreviousDay(nowMs, zone), zone, 18), sod)
        // "this evening" / "tonight" → today 18:00 → now. Asked BEFORE 18:00 the evening hasn't happened,
        // so 18:00 > now would build an inverted (since > until) window the range filter reads as empty —
        // strictly worse than no filter. Drop the time constraint in that case and let it search all time.
        val eveningStart = atHour(sod, zone, 18)
        return if (eveningStart < nowMs) TimeWindow(eveningStart, nowMs) else null
    }
    REL_WEEKEND_BEFORE_LAST.find(q)?.let {
        // Two weekends back: this week's Monday minus 9 days is that Saturday; the following Monday
        // (minus 7 days from this one) closes it one ms early.
        val mon = startOfWeek(nowMs, zone)
        return TimeWindow(shift(mon, zone, java.util.Calendar.DATE, -9), shift(mon, zone, java.util.Calendar.DATE, -7) - 1)
    }
    REL_WEEKEND.find(q)?.let {
        // Most-recent PAST weekend = Sat 00:00 .. Sun 23:59:59.999 of the week just gone: this week's
        // Monday minus 2 days is that Saturday, minus 1ms is that Sunday's last instant.
        val mon = startOfWeek(nowMs, zone)
        return TimeWindow(shift(mon, zone, java.util.Calendar.DATE, -2), mon - 1)
    }
    REL_FORTNIGHT.find(q)?.let {
        val s = shift(startOfWeek(nowMs, zone), zone, java.util.Calendar.DATE, -14)
        return TimeWindow(s, shift(s, zone, java.util.Calendar.DATE, 7) - 1)
    }
    REL_N_AGO.find(q)?.let { m ->
        val n = countWord(m.groupValues[1])
        return when (m.groupValues[2]) {
            "minute" -> TimeWindow(nowMs - n * 60_000L, nowMs)
            "hour"   -> TimeWindow(nowMs - n * 3_600_000L, nowMs)
            "day"    -> shift(sod, zone, java.util.Calendar.DATE, -n).let { s -> TimeWindow(s, startOfNextDay(s, zone) - 1) }
            "week"   -> shift(startOfWeek(nowMs, zone), zone, java.util.Calendar.DATE, -7 * n).let { s -> TimeWindow(s, shift(s, zone, java.util.Calendar.DATE, 7) - 1) }
            "month"  -> shift(startOfMonth(nowMs, zone), zone, java.util.Calendar.MONTH, -n).let { s -> TimeWindow(s, shift(s, zone, java.util.Calendar.MONTH, 1) - 1) }
            else     -> null
        }
    }
    // A future weekday ("next Friday", "this coming Monday") has no memories — skip the weekday branch
    // entirely rather than resolve it to the most-recent PAST weekday.
    if (!FUTURE_WEEKDAY.containsMatchIn(q)) REL_WEEKDAY.find(q)?.let { m ->
        WEEKDAYS[m.groupValues[2]]?.let { dow -> return weekdayWindow(nowMs, zone, dow, m.groupValues[1] == "last") }
    }
    // Bare month name, "in April" form (whole month). Requires the "in " prefix so the common-word
    // months ("may", "march") can't false-match a non-date sentence. MONTH_ALT/MONTH_NAMES live below
    // in the file but are init'd before this fn is ever called, so referencing them at runtime is fine.
    Regex("\\bin\\s+((?:$MONTH_ALT))(?!\\p{L})").find(q)?.let { m ->
        MONTH_NAMES[m.groupValues[1]]?.let { return monthOnlyWindow(nowMs, zone, it) }
    }
    return when {
        // Whole-branch review fix: `untilMs` is applied as an INCLUSIVE upper bound by the storage
        // layer (RangeFloat's `lte`), so `sod` itself (today's midnight) used to double as
        // yesterday's closing instant too — a keyframe stored at exactly local midnight matched
        // BOTH "yesterday" and "today". `sod - 1` closes yesterday's window one ms before the
        // boundary "today"'s window (which correctly starts AT `sod`, inclusive) opens.
        Regex("\\byesterday\\b").containsMatchIn(q) -> TimeWindow(startOfPreviousDay(nowMs, zone), sod - 1)
        Regex("\\b(today|this morning|this afternoon|earlier today)\\b").containsMatchIn(q) ->
            TimeWindow(sod, nowMs)
        Regex("\\b(an hour ago|last hour|in the last hour)\\b").containsMatchIn(q) ->
            TimeWindow(nowMs - 3_600_000L, nowMs)
        Regex("\\b(just now|a moment ago|recently|a minute ago)\\b").containsMatchIn(q) ->
            TimeWindow(nowMs - 600_000L, nowMs)
        else -> null
    }
}

/** Month name → 0-based month index. EN full + 3-letter (and "sept") abbreviations + RU genitive
 *  case ("5 сентября" is genitive — the form Russian actually uses after a day number). Lowercase
 *  keys only; callers must lowercase the query before lookup. */
private val MONTH_NAMES: Map<String, Int> = mapOf(
    "january" to 0, "february" to 1, "march" to 2, "april" to 3, "may" to 4, "june" to 5,
    "july" to 6, "august" to 7, "september" to 8, "october" to 9, "november" to 10, "december" to 11,
) + mapOf(
    "jan" to 0, "feb" to 1, "mar" to 2, "apr" to 3, "jun" to 5, "jul" to 6, "aug" to 7,
    "sep" to 8, "sept" to 8, "oct" to 9, "nov" to 10, "dec" to 11,
) + mapOf(
    "января" to 0, "февраля" to 1, "марта" to 2, "апреля" to 3, "мая" to 4, "июня" to 5,
    "июля" to 6, "августа" to 7, "сентября" to 8, "октября" to 9, "ноября" to 10, "декабря" to 11,
)

// Longest-name-first so a full "september" isn't cut short by the "sep"/"sept" alternative —
// though the `(?!\p{L})` boundary guard below would reject a partial match anyway (backtracking
// into the next alternative), this keeps the common case matching on the first try.
private val MONTH_ALT = MONTH_NAMES.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }

// `\p{L}` (Unicode "any letter") rather than `\b` — `\b` is ASCII-`\w`-based in Java/Kotlin regex
// by default and does NOT treat Cyrillic letters as word characters, so it would fail to find a
// boundary around a Russian month name at all (neither the letter nor the surrounding space is a
// `\w` char, so there's no transition for `\b` to anchor on).
private val MONTH_TOKEN = "(?<!\\p{L})(?:$MONTH_ALT)(?!\\p{L})"
private val DAY_TOKEN = "(?<!\\d)\\d{1,2}(?:st|nd|rd|th)?(?!\\d)"

// Either order: "<day> <month>" (RU "5 сентября", EN "5 september") or "<month> <day>"
// ("september 5", "sept 5th"). Named groups pick out which side is which after a match.
private val ABSOLUTE_DATE = Regex(
    "(?<day1>$DAY_TOKEN)\\s+(?<month1>$MONTH_TOKEN)|(?<month2>$MONTH_TOKEN)\\s+(?<day2>$DAY_TOKEN)"
)

// ISO 8601 calendar date "2026-03-15" — unambiguous (explicit year, no D/M-vs-M/D guessing), so unlike
// the bare-numeric case it IS parsed. The negative look-arounds keep it from biting into a longer digit
// run (a timestamp, a serial number). Checked before ABSOLUTE_DATE in [extractAbsoluteDate].
private val ISO_DATE = Regex("(?<![\\d-])(\\d{4})-(\\d{1,2})-(\\d{1,2})(?![\\d-])")

/** The day-window a matched absolute date resolves to, plus the character range it occupied in
 *  the raw query (so [stripDateSpan] can remove exactly that text before embedding). */
data class DateMatch(val window: TimeWindow, val matchedSpan: IntRange)

/** Absolute NAMED-MONTH date parsing ("5 сентября" / "September 5" / "sept 5th") — bare numeric
 *  dates ("5/9") are deliberately NOT matched (D/M vs M/D is ambiguous with no locale signal;
 *  deferred, not a bug). No year in the query, so the year resolves to the most-recent PAST
 *  occurrence: the date is built in the CURRENT year in [zone], and if that day's [startOfDay] is
 *  still after today's, it steps back a year (and "February 29" keeps stepping back to the most
 *  recent PAST leap year — a real recurring date, not an impossible one).
 *  Returns null if the query names no absolute date, OR if the named day exists in NO year for that
 *  month (Feb 31, Feb 30, Apr 31) — see the non-lenient walk-back below.
 *  KNOWN LIMIT (deferred to a future mini-LLM parser, not a bug at this scale): the day+month
 *  regex has no part-of-speech signal, so a query that happens to place a 1–2 digit number before
 *  an English month-word that is also a common word ("5 may be enough", "3 march …") false-matches
 *  as a date. No realistic object-memory voice query mixes a bare number with such a word, so this
 *  is left unhandled rather than papered over with a brittle stop-word list. */
fun extractAbsoluteDate(
    raw: String,
    nowMs: Long,
    zone: java.util.TimeZone = java.util.TimeZone.getDefault(),
): DateMatch? {
    val q = raw.lowercase()   // length-preserving for EN/RU, so match.range still indexes `raw`
    // ISO date first: it carries its own year, so it resolves directly (no most-recent-past walk-back)
    // and returns before the yearless named-month path below.
    ISO_DATE.find(q)?.let { m ->
        val y = m.groupValues[1].toInt(); val mo = m.groupValues[2].toInt(); val d = m.groupValues[3].toInt()
        if (mo in 1..12 && d in 1..31) {
            val cal = java.util.Calendar.getInstance(zone).apply { clear(); isLenient = false }
            val start = try { cal.set(y, mo - 1, d, 12, 0, 0); startOfDay(cal.timeInMillis, zone) }
                        catch (e: IllegalArgumentException) { null }   // impossible day (e.g. 2026-02-31)
            if (start != null) return DateMatch(TimeWindow(start, startOfNextDay(start, zone) - 1), m.range)
        }
    }
    val match = ABSOLUTE_DATE.find(q) ?: return null
    val day = (match.groups["day1"] ?: match.groups["day2"])?.value?.let { it.filter(Char::isDigit).toInt() }
        ?: return null
    val monthName = (match.groups["month1"] ?: match.groups["month2"])?.value ?: return null
    val monthIndex = MONTH_NAMES[monthName] ?: return null

    val cal = java.util.Calendar.getInstance(zone)
    cal.timeInMillis = nowMs
    val currentYear = cal.get(java.util.Calendar.YEAR)
    val todayStart = startOfDay(nowMs, zone)

    // Resolve the (yearless) date to the MOST-RECENT occurrence of this month/day that is not in
    // the future. Walk back year by year from the current year, skipping a year where the date is
    // either still ahead of today OR does not exist that year (Feb 29 in a common year — a LENIENT
    // Calendar would silently roll it to Mar 1, so leniency is off and the read throws instead). The
    // first year that yields a real, non-future date wins:
    //   • normal past date ("august 1") → current year;
    //   • future-so-far this year ("september 5" asked in August) → last year;
    //   • Feb 29 → the most recent PAST leap year (2024 when asked in 2026), NOT null — it is a real
    //     recurring date, not an impossible one.
    // A day that exists in NO year (Feb 31, Apr 31, Feb 30) never resolves and falls through to null.
    // Bounded to 10 steps: only Feb 29 needs more than one, and its leap gap is at most 8 years
    // (across a skipped century leap like 2100).
    //
    // The strict Calendar validates the day at NOON, not midnight: some zones spring forward AT local
    // midnight (e.g. America/Sao_Paulo on 2018-11-04), so 00:00 is a skipped wall-clock instant on an
    // otherwise-valid day and a strict read there would throw — misread as an impossible date and
    // walked back a year. Noon is never skipped by any real DST transition, so it validates the
    // day-of-month cleanly; startOfDay (lenient) then resolves the actual local day-start, rolling a
    // skipped midnight forward to the day's first valid instant.
    var year = currentYear
    var resolvedStart: Long? = null
    var guard = 0
    while (guard < 10) {
        cal.clear()
        cal.isLenient = false
        cal.set(year, monthIndex, day, 12, 0, 0)
        val sod = try { startOfDay(cal.timeInMillis, zone) } catch (e: IllegalArgumentException) { null }
        if (sod != null && sod <= todayStart) { resolvedStart = sod; break }
        year--
        guard++
    }
    val dayStart = resolvedStart ?: return null
    // DST-safe upper bound — same reasoning as startOfPreviousDay: a fixed +86_400_000ms lands
    // short/long of midnight across a DST transition, so step a Calendar a whole date forward
    // instead and close one ms before that boundary.
    val dayEndExclusive = startOfNextDay(dayStart, zone)
    return DateMatch(TimeWindow(dayStart, dayEndExclusive - 1), match.range)
}

/** Remove the [span] a [DateMatch] occupied (e.g. "september 5") from [raw] and collapse the
 *  resulting whitespace, so the date doesn't pollute the CLIP text embedding. Relative-phrase
 *  stripping ("yesterday", "an hour ago") stays in [stripTimePhrases] — this is absolute-date-only. */
fun stripDateSpan(raw: String, span: IntRange): String =
    (raw.substring(0, span.first) + raw.substring(span.last + 1))
        .replace(Regex("\\s+"), " ").trim()

private val TIME_PHRASE = Regex(
    "\\b(yesterday|today|this morning|this afternoon|earlier today|an hour ago|last hour|" +
    "in the last hour|just now|a moment ago|a minute ago|recently)\\b")

/** Remove time words so they don't pollute the CLIP text embedding. */
fun stripTimePhrases(phrase: String): String =
    phrase.replace(TIME_PHRASE, "").replace(Regex("\\s+"), " ").trim().ifBlank { phrase }

private val RECALL_INTENT = Regex("where(?:\\s+did|'?d)\\s+i\\s+(leave|put|last\\s+see|drop)")

/** "where did I leave/put my X" — answer with the most RECENT sighting, not the best score. */
fun isRecallLocationIntent(raw: String): Boolean = RECALL_INTENT.containsMatchIn(raw.lowercase())

// [searchPhrase] and [stripTimePhrases] each fall back to their OWN input whenever their
// internal stripping would otherwise leave nothing (see searchPhrase's kdoc) — by design,
// neither ever turns a non-blank string blank. A pure-time query needs the opposite: once
// [stripDateSpan] removes an absolute date's words, whatever question boilerplate is left
// ("what did i see on") must collapse to genuinely blank BEFORE reaching those two, so
// [parseQuery] can tell the searcher to skip embedding entirely instead of encoding "". A
// RELATIVE pure-time query ("what did i see yesterday") has no [stripDateSpan] step to remove
// "yesterday" first, so this function also strips [TIME_PHRASE] itself here, unguarded (unlike
// [stripTimePhrases]'s own call on the same regex) — otherwise "yesterday" would survive as the
// sole leftover token and [stripTimePhrases]'s ifBlank-fallback would hand it right back instead
// of letting it go blank. Order matters: the dangling preposition ("... on") is stripped first
// so the question-prefix regex's trailing boundary isn't left stranded on a leftover connector
// word.
//
// KNOWN LIMIT (deferred to a future mini-LLM parser, not a bug): exactly ONE recall stem
// ("what did i see") collapses to blank here; another natural pure-time phrasing ("show me
// september 5", "anything from sept 5") leaves a non-blank leftover, so [parseQuery] falls to the
// normal vector path (window still applied, just semantically ranked) rather than the chronological
// scroll. Graceful — no crash, correct window — so not chased with an ever-growing stem list.
private val DATE_QUESTION_PREFIX = Regex("^what\\s+did\\s+i\\s+see\\b\\s*")
private val DATE_DANGLING_PREPOSITION = Regex("\\s+(on|in|at|from)$")
// A trailing "?"/"!"/"." the date strip stranded ("what did i see on september 5?" → after the
// date span is cut → "what did i see on ?") would otherwise block the $-anchored dangling-preposition
// strip below, leaving "on" behind so the query never collapses to blank and [timeOnly] stays false.
private val TRAILING_PUNCTUATION = Regex("[\\s?!.,;]+$")

private fun stripDateAdjacentBoilerplate(text: String, dateWasStripped: Boolean): String {
    // Lowercase first: the final embedText is lowercased anyway, and the case-sensitive prefix/
    // preposition regexes below must fire on a natural-case query ("What did I see …") too, not just
    // the lowercased smoke-test inputs. Trailing punctuation is always safe to drop (it is never a
    // query term).
    var t = text.lowercase().replace(TRAILING_PUNCTUATION, "")
    // Only the DANGLING (trailing) preposition is stripped, and only when a DATE span was removed —
    // it cleans up the connector the date left behind ("... on <date>" → "... on" → "..."). We do
    // NOT strip a LEADING preposition: a query-initial "on/in/at/from" usually belongs to the object
    // or location phrase, not the date ("at home on september 5" → keep "at home"; "in my backpack"
    // → keep "in"), and only a contrived date-FIRST phrasing ("on september 5 wallet") would benefit
    // — at the cost of corrupting the common case. A leading connector left by a date-first query is
    // a harmless stopword the CLIP text encoder largely ignores.
    if (dateWasStripped)
        t = t.replace(DATE_DANGLING_PREPOSITION, "")
    return t.replace(DATE_QUESTION_PREFIX, "")
        // Relative-date words (extractTimeWindow's new branches) must ALSO leave the embed text, or a
        // "desk last week" would embed "desk last week". REL_DAY_BEFORE runs before TIME_PHRASE because
        // it contains "yesterday" (TIME_PHRASE would strip that first and orphan "the day before").
        .replace(REL_DAY_BEFORE, " ").replace(REL_EVENING, " ")
        .replace(REL_WEEKEND_BEFORE_LAST, " ").replace(REL_WEEKEND, " ").replace(REL_FORTNIGHT, " ")
        .replace(REL_WEEK, " ").replace(REL_MONTH, " ").replace(REL_N_AGO, " ").replace(REL_WEEKDAY, " ")
        .replace(Regex("\\bin\\s+((?:$MONTH_ALT))(?!\\p{L})"), " ")
        .replace(TIME_PHRASE, "")
        // Leftover day-part / vague-time fillers a partial match strands ("yesterday morning" → "morning";
        // "sometime last week" → "sometime"). Runs LAST so the phrase matchers above claim their words first.
        .replace(Regex("\\b(morning|afternoon|evening|night|sometime|earlier|roughly|approximately)\\b"), " ")
        .replace(Regex("\\s+"), " ").trim()
}

/** One structured query intent — the seam a future on-device mini-LLM parser can implement
 *  without touching the searcher (`MomentSearcher` consumes this struct, not four separate
 *  helper calls). [embedText] is what CLIP embeds; it is BLANK for a pure-time query — callers
 *  must check [timeOnly] and skip `encodeText` rather than embedding an empty string. [window]
 *  is an absolute date's window if the query named one, else a relative one, else null. */
data class ParsedQuery(
    val embedText: String,
    val window: TimeWindow?,
    val recallIntent: Boolean,
    val timeOnly: Boolean,
)

/** Parse a raw voice/typed query into one structured intent. Composition order: (1) [window] —
 *  an absolute date wins over a relative phrase if both somehow appear; (2) [embedText] — strip
 *  the absolute-date span (if matched), then date/time-adjacent boilerplate
 *  ([stripDateAdjacentBoilerplate] — the question stem AND any relative time word, run
 *  unconditionally so a RELATIVE pure-time query collapses to blank the same way an absolute one
 *  does), then question/article boilerplate ([searchPhrase]), then any residual relative-time
 *  words ([stripTimePhrases]), lowercase, trim; (3) [recallIntent] and [timeOnly] are read off
 *  independently ([timeOnly] = a window was found and nothing object-like survived the strip). */
fun parseQuery(
    raw: String,
    nowMs: Long,
    zone: java.util.TimeZone = java.util.TimeZone.getDefault(),
): ParsedQuery {
    val dateMatch = extractAbsoluteDate(raw, nowMs, zone)
    val window = dateMatch?.window ?: extractTimeWindow(raw, nowMs, zone)
    val spanStripped = if (dateMatch != null) stripDateSpan(raw, dateMatch.matchedSpan) else raw
    val dateStripped = stripDateAdjacentBoilerplate(spanStripped, dateWasStripped = dateMatch != null)
    val embedText = stripTimePhrases(searchPhrase(dateStripped)).lowercase().trim()
    val recallIntent = isRecallLocationIntent(raw)
    val timeOnly = window != null && embedText.isBlank()
    return ParsedQuery(embedText, window, recallIntent, timeOnly)
}
