package tech.qdrant.glasses.embedding

import android.util.Log
import java.io.File
import kotlin.math.sqrt

/**
 * Lazy, memoized cache of CLIP text-embedding vectors for a small, mostly-fixed label vocabulary
 * (plan Task 2.1/2.2: the ~80 COCO labels used to CLIP-verify YOLO regions — Spec §2/§7). Every
 * label is run through [encodeText] AT MOST ONCE per cache instance, however many times it is
 * asked for: [B32ClipTextEncoder]'s text tower is the cold ~180ms CPU-int8 path (Global
 * Constraints), and a region-heavy moment (Task 2.2's `REGIONS_MAX_PER_MOMENT`) would otherwise
 * pay that cost repeatedly per capture for labels that recur constantly ("cup", "person", …).
 *
 * [persistFile], if given, survives the first-time cost across process restarts too — pay the
 * ~180ms×80 labels ONCE per install, not once per app launch (Spec §7). On construction the file
 * (one `label\tcsv-floats` line per cached vector — chosen over JSON for a trivial, streaming-
 * append-friendly format) is loaded into memory; each newly-computed label is appended to it. A
 * malformed/partial line (e.g. a torn write from a process killed mid-append) is skipped, not
 * fatal — that one label just falls back to re-embedding, the rest of the cache stays usable.
 *
 * [dim] is the expected vector length ([CropEncoder.dim] of whichever encoder backs
 * [encodeText]). A torn append (process killed mid-write) can leave a persisted line with a
 * short-but-parseable float list ("cup\t0.1,0.2" instead of the full 512) — well-formed enough to
 * NOT hit the malformed-line checks below, but wrong enough to index past the end when later
 * compared against a full-length vector (Codex P1 fix). Any persisted line whose float count
 * doesn't match [dim] is skipped exactly like a malformed one; [cosine] is ALSO defensive against
 * length mismatches independently, so a bad vector that somehow gets into the cache by another
 * path still can't crash a lookup.
 *
 * Label keys are normalized (trim + lowercase) so "Cup", " cup ", and "cup" share one cache entry
 * and one [encodeText] call — callers don't need to pre-normalize COCO labels themselves.
 */
class LabelVectorCache(
    private val encodeText: (String) -> FloatArray,
    private val dim: Int,
    private val persistFile: File? = null,
) {
    // One coarse lock: labels are ~80 distinct strings total and gets are infrequent (per stored
    // moment's regions, not per frame) — no need for finer-grained per-key locking. Guards against
    // two concurrent callers for the SAME uncached label both paying the ~180ms encode.
    private val lock = Any()
    private val cache = HashMap<String, FloatArray>()

    init {
        persistFile?.let { loadPersisted(it) }
    }

    /** The cached (or freshly-embedded-then-cached) text vector for [label]. */
    fun get(label: String): FloatArray {
        val key = normalize(label)
        synchronized(lock) {
            cache[key]?.let { return it }
            val t0 = System.currentTimeMillis()
            val vec = encodeText(key)
            Log.i(TAG, "encodeText label=\"$key\" ${System.currentTimeMillis() - t0}ms " +
                "(cache miss, ${cache.size + 1} labels cached)")
            cache[key] = vec
            persistFile?.let { appendPersisted(it, key, vec) }
            return vec
        }
    }

    /** Cosine similarity between a region embedding and [label]'s (cached) text vector — the
     *  CLIP-verify signal region tagging is built on (Spec §2, plan Task 2.2). 0f if either vector
     *  is degenerate (all-zero) rather than dividing by zero, or if the two vectors' lengths don't
     *  match (Codex P1 fix — see [cosine]) — "not verified" is a safer failure than a crash or a
     *  NaN silently poisoning the comparison. */
    fun verify(regionVec: FloatArray, label: String): Float = cosine(regionVec, get(label))

    private fun normalize(label: String): String = label.trim().lowercase()

    private fun loadPersisted(file: File) {
        if (!file.exists()) return
        try {
            var loaded = 0
            var skipped = 0
            var skippedWrongDim = 0
            file.forEachLine { line ->
                if (line.isBlank()) return@forEachLine
                val tab = line.indexOf('\t')
                val floats = if (tab < 0) null else line.substring(tab + 1).split(',').map { it.toFloatOrNull() }
                if (tab < 0 || floats.isNullOrEmpty() || floats.any { it == null }) {
                    skipped++
                    return@forEachLine
                }
                if (floats.size != dim) {
                    // A torn append (process killed mid-write) can leave a short-but-parseable
                    // line ("cup\t0.1,0.2") — well-formed enough to pass the checks above, wrong
                    // enough to index past the end if cached as-is and later compared against a
                    // full-length vector (Codex P1 fix). Drop it exactly like a malformed line:
                    // the label just falls back to a fresh [encodeText] call on the next [get].
                    skippedWrongDim++
                    return@forEachLine
                }
                cache[line.substring(0, tab)] = FloatArray(floats.size) { floats[it]!! }
                loaded++
            }
            Log.i(TAG, "loaded $loaded persisted label vectors from ${file.path} " +
                "($skipped malformed, $skippedWrongDim wrong-dim lines skipped)")
        } catch (e: Throwable) {
            // Best-effort load (Codex P2 fix), matching appendPersisted's write-side convention:
            // if persistFile exists but isn't actually readable (a directory, a permission error),
            // forEachLine throws right out of the constructor — that must not make the WHOLE cache
            // unusable for the rest of the process. Proceed with an empty in-memory cache; every
            // label just re-embeds on its first [get], same as a fresh install with no persisted
            // file at all.
            Log.w(TAG, "failed to load persisted label vectors from ${file.path}", e)
        }
    }

    private fun appendPersisted(file: File, key: String, vec: FloatArray) {
        try {
            file.parentFile?.mkdirs()
            file.appendText("$key\t${vec.joinToString(",")}\n")
        } catch (e: Throwable) {
            // Persistence is a nice-to-have (survives restarts); an unwritable file must not lose
            // the in-memory entry [get] already cached for the rest of this process's lifetime.
            Log.w(TAG, "failed to persist label=\"$key\" to ${file.path}", e)
        }
    }

    companion object {
        private const val TAG = "LabelVectorCache"
    }
}

/** Plain cosine similarity. 0f if either vector is degenerate (all-zero) rather than dividing by
 *  zero — same convention as the private `cosine` in [tech.qdrant.glasses.pipeline.MomentCapture].
 *  Also 0f on a length mismatch (Codex P1 fix, defense-in-depth alongside [dim]-checked loading
 *  above): a mismatch bails out BEFORE the indexing loop below ever runs, so a truncated vector
 *  that somehow still made it into the cache can't index [a] past [b]'s length (or vice versa) —
 *  it just reads as "not verified" instead of crashing. */
private fun cosine(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    var dot = 0f
    var na = 0f
    var nb = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    if (na <= 0f || nb <= 0f) return 0f
    return (dot / (sqrt(na.toDouble()) * sqrt(nb.toDouble()))).toFloat()
}
