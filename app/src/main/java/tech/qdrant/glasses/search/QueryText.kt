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

private fun startOfDay(nowMs: Long, zone: java.util.TimeZone): Long {
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

private val TIME_PHRASE = Regex(
    "\\b(yesterday|today|this morning|this afternoon|earlier today|an hour ago|last hour|" +
    "in the last hour|just now|a moment ago|a minute ago|recently)\\b")

/** Remove time words so they don't pollute the CLIP text embedding. */
fun stripTimePhrases(phrase: String): String =
    phrase.replace(TIME_PHRASE, "").replace(Regex("\\s+"), " ").trim().ifBlank { phrase }

private val RECALL_INTENT = Regex("where\\s+did\\s+i\\s+(leave|put|last\\s+see|drop)")

/** "where did I leave/put my X" — answer with the most RECENT sighting, not the best score. */
fun isRecallLocationIntent(raw: String): Boolean = RECALL_INTENT.containsMatchIn(raw.lowercase())
