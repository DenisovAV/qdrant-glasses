package tech.qdrant.glasses.search

/** "where is my laptop" → "laptop". SigLIP2's text→crop scale is compressed, so question
 *  boilerplate ("where is …") dips a real match under the gate — search the object phrase,
 *  display the full query. Falls back to the original query if stripping leaves nothing. */
fun searchPhrase(rawQuery: String): String =
    rawQuery.lowercase()
        .replace(Regex("^(where\\s+(is|are)|what\\s+(is|are)|when\\s+(is|are)|that\\s+is|this\\s+is|find|show\\s+me|look\\s+for|search\\s+for)\\s+"), "")
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
