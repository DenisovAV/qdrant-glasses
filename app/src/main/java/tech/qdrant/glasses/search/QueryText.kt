package tech.qdrant.glasses.search

/** "where is my laptop" → "laptop", "where did i leave my wallet" → "wallet". SigLIP2's
 *  text→crop scale is compressed, so question boilerplate ("where is …", "where did i leave …")
 *  dips a real match under the gate — search the object phrase, display the full query.
 *  Falls back to the original query if stripping leaves nothing. */
fun searchPhrase(rawQuery: String): String =
    rawQuery.lowercase()
        .replace(Regex("^(where\\s+did\\s+i\\s+(leave|put|last\\s+see|drop)|where\\s+(is|are)|what\\s+(is|are)|when\\s+(is|are)|that\\s+is|this\\s+is|find|show\\s+me|look\\s+for|search\\s+for)\\s+"), "")
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

/** Map a few spoken time phrases to a [TimeWindow]; null if the query names no time.
 *  Calendar-accurate for "today"/"yesterday" in [zone]; relative for hour-scale phrases. */
fun extractTimeWindow(
    raw: String,
    nowMs: Long,
    zone: java.util.TimeZone = java.util.TimeZone.getDefault(),
): TimeWindow? {
    val q = raw.lowercase()
    val sod = startOfDay(nowMs, zone)
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
    var year = currentYear
    var resolvedStart: Long? = null
    var guard = 0
    while (guard < 10) {
        cal.clear()
        cal.isLenient = false
        cal.set(year, monthIndex, day, 0, 0, 0)
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

private val RECALL_INTENT = Regex("where\\s+did\\s+i\\s+(leave|put|last\\s+see|drop)")

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
// Symmetric to [DATE_DANGLING_PREPOSITION] for a date-FIRST phrasing ("on september 5 wallet" →
// after the span is cut → "on wallet"): drop the leading connector so the object embeds clean.
private val DATE_LEADING_PREPOSITION = Regex("^(on|in|at|from)\\s+")
// A trailing "?"/"!"/"." the date strip stranded ("what did i see on september 5?" → after the
// date span is cut → "what did i see on ?") would otherwise block the $-anchored dangling-preposition
// strip below, leaving "on" behind so the query never collapses to blank and [timeOnly] stays false.
private val TRAILING_PUNCTUATION = Regex("[\\s?!.,;]+$")

private fun stripDateAdjacentBoilerplate(text: String): String =
    // Lowercase first: the final embedText is lowercased anyway, and the case-sensitive prefix/
    // preposition regexes below must fire on a natural-case query ("What did I see …") too, not just
    // the lowercased smoke-test inputs.
    text.lowercase()
        .replace(TRAILING_PUNCTUATION, "")
        .replace(DATE_DANGLING_PREPOSITION, "")
        .replace(DATE_LEADING_PREPOSITION, "")
        .replace(DATE_QUESTION_PREFIX, "")
        .replace(TIME_PHRASE, "").replace(Regex("\\s+"), " ").trim()

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
    val dateStripped = stripDateAdjacentBoilerplate(spanStripped)
    val embedText = stripTimePhrases(searchPhrase(dateStripped)).lowercase().trim()
    val recallIntent = isRecallLocationIntent(raw)
    val timeOnly = window != null && embedText.isBlank()
    return ParsedQuery(embedText, window, recallIntent, timeOnly)
}
