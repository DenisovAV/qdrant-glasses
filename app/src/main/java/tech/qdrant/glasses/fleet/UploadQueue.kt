package tech.qdrant.glasses.fleet

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * One point queued for upload to the private Qdrant fleet hub (Spec §4 UP flow / §5 dual-write,
 * plan Task 8): the clip vector plus its already-serialized payload JSON. [payloadJson] is opaque
 * to [UploadQueue] — the caller ([tech.qdrant.glasses.pipeline.MomentCapture], Task 10) stamps in
 * `sync_ts`/`thumb_b64` before enqueuing; this class only persists and replays what it's given.
 */
data class QueuedPoint(val id: String, val clip: FloatArray, val payloadJson: String)

/**
 * Persistent, append-only upload queue for the UP half of fleet sync (plan Task 8, Spec §4:
 * "new local moment --> upload queue --> (online) batch upsert --> private Qdrant [fleet_inbox]").
 * Backed by one JSONL file (`filesDir/fleet_queue.jsonl`, per Spec §4/plan) — one line per queued
 * point, `{"id":..,"clip":[..],"payload":{..}}` — so the queue survives a process restart: a moment
 * captured offline stays queued until the next successful [FleetSync.pushDrain], across reboots.
 * [enqueue] is synchronous and durable: a crash the instant after it returns can never lose the
 * point (unlike a fire-and-forget async append would — see "Concurrency" below for why that was
 * rejected).
 *
 * [drain] is a peek, not a pop: it returns up to [max] points without removing them, so a crash
 * between drain and a successful server upsert never loses a point — the caller only removes a
 * batch via [ack] AFTER the upsert actually lands (see plan Task 11). This mirrors Decision C for
 * the OTHER direction: exactly as a local moment is never deleted after a successful upload, a
 * queued upload is never speculatively removed before one.
 *
 * **Concurrency (review fix, post-Task-8):** [tech.qdrant.glasses.pipeline.MomentCapture] calls
 * [enqueue] from the capture lane, while [FleetSync.pushDrain] calls [drain]/[ack] from the fleet
 * lane (Spec §7: fleet ops run off the inference lane) — but Spec §7 ALSO requires that fleet sync
 * "never crash or block capture", and the original design shared ONE lock across all three methods,
 * so a slow [ack] (read + JSON-parse + rewrite of the WHOLE file) could stall a concurrent capture-
 * side [enqueue] for that entire operation. [ack] now does its slow part — reading, parsing, and
 * filtering the file — OUTSIDE [lock] entirely; [lock] only wraps the small fixed-cost tail: folding
 * in anything [enqueue] appended in the meantime (see below) and swapping the rewritten file in.
 * [enqueue]'s own critical section is symmetrically small (a cap check, one append, a counter bump).
 * So the longest either side can ever block the other for is that short tail, never "read+parse+
 * rewrite the whole queue" — that's what actually satisfies the "must not block capture" requirement
 * (an async fire-and-forget [enqueue] would satisfy it too, but at the cost of durability — see
 * above — which is the worse trade for a queue whose entire point is surviving a crash/reboot).
 *
 * A synchronous [enqueue] that ISN'T serialized with [ack] at all would risk a subtler bug: if an
 * [enqueue] appends a line while [ack] is mid-read, [ack]'s rewrite (built from an earlier snapshot)
 * could silently overwrite the file and drop that brand-new point. [ack] guards against exactly this:
 * right before it swaps the rewritten file in, it re-reads whatever now sits past the point it
 * originally snapshotted and folds those lines back in — so a point [enqueue] appends during an
 * in-flight [ack] is never lost, only possibly (harmlessly) re-ordered relative to it in the file.
 *
 * **Fail-soft (review fix):** every file/JSON operation runs inside a try/catch that logs and
 * swallows — a malformed [payloadJson], a full disk, or any other I/O failure is dropped with a
 * warning, never thrown back at the caller (Spec §7). [drain]/[ack] additionally tolerate a single
 * unreadable line (e.g. one torn by a concurrent append-in-progress, or any other corruption) by
 * skipping just that line rather than failing the whole batch.
 *
 * **Durability of [ack] (review fix):** the old `file.writeText(remaining)` truncated the live queue
 * file in place before writing the new content — a crash between truncate and the write completing
 * would silently discard every unacknowledged (not-yet-uploaded) entry, with no retry path. [ack] now
 * writes the new content to a sibling `.tmp` file first and only swaps it in via [File.renameTo] (an
 * atomic same-filesystem rename), so a crash at any point before the rename leaves the ORIGINAL file
 * — every entry, acked or not — fully intact.
 *
 * **Bounded growth (review fix):** an unreachable fleet hub means [enqueue] runs forever with no
 * [ack] ever removing anything. [maxEntries] caps how many points this queue will hold; once full,
 * [enqueue] drops the newest point (logging a warning) rather than growing the file without limit.
 * This never loses the moment itself — decision C keeps the local mutable shard as the point's
 * permanent home regardless of whether it ever reaches the fleet — it only means that one moment
 * won't be offered up for sync. [maxEntries] defaults to a PoC-grade bound, not a tuned figure.
 */
class UploadQueue(private val file: File, private val maxEntries: Int = DEFAULT_MAX_ENTRIES) {

    companion object {
        private const val TAG = "UploadQueue"

        /** PoC-grade cap on queued-but-unsent points (see class doc "Bounded growth"). */
        const val DEFAULT_MAX_ENTRIES = 2000
    }

    // Guards ONLY the short, fixed-cost sections described in the class doc's "Concurrency" section:
    // enqueue()'s cap-check+append+counter-bump, and ack()'s final tail-fold+atomic-swap. Deliberately
    // does NOT cover drain()'s read (read-only, benign under a torn concurrent append — parseLine
    // just skips it) or the bulk of ack()'s work (reading + JSON-parsing + filtering the whole file).
    private val lock = Any()

    // In-memory queue depth, mutated only under [lock] (by enqueue's append and ack's swap), so no
    // extra synchronization beyond holding [lock] wherever it's touched. -1 = not yet seeded from
    // disk; seeded lazily on first touch (not in the constructor), so simply constructing a queue
    // never touches the filesystem.
    private var count = -1

    /**
     * Appends one point to the queue. Synchronous and durable (see class doc): the point is on disk
     * — visible to any future [drain], including from a freshly (re)constructed [UploadQueue] over
     * the same file — by the time this call returns. Never throws (fail-soft): a malformed
     * [payloadJson] or any I/O failure is logged and the point is dropped, not propagated to the
     * caller. If the queue is already at [maxEntries], the point is dropped too (logged) — the local
     * copy of the moment is unaffected either way (decision C).
     */
    fun enqueue(id: String, clipVec: FloatArray, payloadJson: String) {
        try {
            // Building the line (JSON parse of payloadJson included) is the one part of enqueue that
            // can meaningfully fail on bad input — do it OUTSIDE [lock] so a malformed payload never
            // holds the lock a concurrent ack() might be waiting on.
            val line = JSONObject()
                .put("id", id)
                .put("clip", JSONArray(clipVec.map { it.toDouble() }))
                .put("payload", JSONObject(payloadJson))
                .toString()
            synchronized(lock) {
                if (count < 0) count = if (file.exists()) file.readLines().count { it.isNotBlank() } else 0
                if (count >= maxEntries) {
                    Log.w(TAG, "upload queue full ($count/$maxEntries) — dropping id=$id (local copy is kept regardless, decision C)")
                    return@synchronized
                }
                file.parentFile?.mkdirs()
                file.appendText(line + "\n")
                count++
            }
        } catch (e: Throwable) {
            Log.w(TAG, "upload queue enqueue failed (non-fatal, id=$id dropped, local copy is kept regardless): ${e.message}")
        }
    }

    /**
     * Returns up to [max] queued points, oldest-first (FIFO — append order on disk). Does NOT
     * remove them from the queue; call [ack] once they've actually been upserted server-side.
     * A missing file (nothing ever queued) is an empty queue, not an error; any other read/parse
     * failure (fail-soft) returns an empty list rather than throwing.
     */
    fun drain(max: Int): List<QueuedPoint> = try {
        if (!file.exists()) emptyList()
        else file.readLines().asSequence().filter { it.isNotBlank() }.mapNotNull(::parseLine).take(max).toList()
    } catch (e: Throwable) {
        Log.w(TAG, "upload queue drain failed (non-fatal, returning no points this round): ${e.message}")
        emptyList()
    }

    /**
     * Durably removes the given [ids] from the queue (rewrites the file without them, atomically —
     * see class doc) — call after a batch [drain] has been confirmed upserted server-side. A missing
     * file or an empty [ids] is a no-op, not an error (mirrors [drain]'s soft-fail-safe shape — an
     * ack after a queue already emptied by a prior run must never throw). Fail-soft like [enqueue]/
     * [drain]: any failure is logged and swallowed, leaving the entries queued so they retry later.
     */
    fun ack(ids: Collection<String>) {
        if (ids.isEmpty()) return
        try {
            if (!file.exists()) return
            val acked = ids.toSet()
            // The slow part — full read + per-line JSON parse + filter — runs OFF [lock] (see class
            // doc "Concurrency"), so a concurrent enqueue() is never stuck behind it.
            val snapshot = file.readLines()
            val remaining = snapshot.filter { it.isNotBlank() }.mapNotNull(::parseLine).filter { it.id !in acked }
            synchronized(lock) {
                // Fold in anything enqueue() appended AFTER `snapshot` was taken (appends only ever
                // grow the file, so any lines beyond `snapshot.size` are new arrivals) — without this,
                // the swap below would silently overwrite a point that landed mid-ack.
                val fresh = file.readLines()
                val extra = fresh.drop(snapshot.size).filter { it.isNotBlank() }.mapNotNull(::parseLine)
                val finalRemaining = remaining + extra
                writeAtomic(finalRemaining)
                count = finalRemaining.size
            }
        } catch (e: Throwable) {
            Log.w(TAG, "upload queue ack failed (non-fatal, entries stay queued and will retry): ${e.message}")
        }
    }

    /** Write-temp-then-rename so a crash never leaves the live file partially truncated. */
    private fun writeAtomic(remaining: List<QueuedPoint>) {
        if (remaining.isEmpty()) {
            file.delete()
            return
        }
        val body = remaining.joinToString("\n") { it.toLine() } + "\n"
        val tmp = File(file.absoluteFile.parentFile, file.name + ".tmp")
        tmp.writeText(body)
        if (!tmp.renameTo(file)) {
            // Same-filesystem rename should always succeed for an app-private filesDir path; this
            // is a last-resort fallback so the queue never gets stuck behind an orphaned .tmp file.
            Log.w(TAG, "upload queue: atomic rename failed, falling back to a direct write")
            tmp.copyTo(file, overwrite = true)
            tmp.delete()
        }
    }

    private fun QueuedPoint.toLine(): String = JSONObject()
        .put("id", id)
        .put("clip", JSONArray(clip.map { it.toDouble() }))
        .put("payload", JSONObject(payloadJson))
        .toString()

    /** Returns null (logged, skipped) for a single unreadable line rather than failing the whole read. */
    private fun parseLine(line: String): QueuedPoint? = try {
        val o = JSONObject(line)
        val arr = o.getJSONArray("clip")
        val clip = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
        QueuedPoint(id = o.getString("id"), clip = clip, payloadJson = o.getJSONObject("payload").toString())
    } catch (e: Throwable) {
        Log.w(TAG, "upload queue: skipping unreadable line (non-fatal): ${e.message}")
        null
    }
}
