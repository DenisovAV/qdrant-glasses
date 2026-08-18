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
 *  still after today's, it steps back one year — memory is retrospective, a future date is wrong.
 *  Returns null if the query names no absolute date. */
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
    cal.clear()
    cal.set(currentYear, monthIndex, day, 0, 0, 0)
    if (startOfDay(cal.timeInMillis, zone) > startOfDay(nowMs, zone)) {
        cal.set(java.util.Calendar.YEAR, currentYear - 1)
    }
    val dayStart = startOfDay(cal.timeInMillis, zone)
    return DateMatch(TimeWindow(dayStart, dayStart + 86_400_000L - 1), match.range)
}

private val TIME_PHRASE = Regex(
    "\\b(yesterday|today|this morning|this afternoon|earlier today|an hour ago|last hour|" +
    "in the last hour|just now|a moment ago|a minute ago|recently)\\b")

/** Remove time words so they don't pollute the CLIP text embedding. */
fun stripTimePhrases(phrase: String): String =
    phrase.replace(TIME_PHRASE, "").replace(Regex("\\s+"), " ").trim().ifBlank { phrase }

private val RECALL_INTENT = Regex("where\\s+did\\s+i\\s+(leave|put|last\\s+see|drop)")

/** "where did I leave/put my X" — answer with the most RECENT sighting, not the best score. */
fun isRecallLocationIntent(raw: String): Boolean = RECALL_INTENT.containsMatchIn(raw.lowercase())
