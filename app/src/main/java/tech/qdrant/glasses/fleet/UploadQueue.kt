package tech.qdrant.glasses.fleet

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption

/**
 * One point queued for upload to the private Qdrant fleet hub (Spec §4 UP flow / §5 dual-write,
 * plan Task 8): the clip vector plus its already-serialized payload JSON. [payloadJson] is opaque
 * to [UploadQueue] — the caller ([tech.qdrant.glasses.pipeline.MomentCapture], Task 10) stamps in
 * `sync_ts` before enqueuing; this class only persists and replays what it's given. `thumb_b64` is
 * NOT part of the enqueued payload (round-4 review fix, see [UploadQueue]'s "Concurrency" doc below)
 * — [tech.qdrant.glasses.fleet.FleetSync] stamps that in lazily, right before the HTTP upsert.
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
 * side [enqueue] for that entire operation. [ack] now does its slow part — PARSING and FILTERING the
 * file — OUTSIDE [lock] entirely; [lock] only wraps the two small fixed-cost pieces: the plain file
 * reads themselves (both the initial snapshot and the later "fresh" re-read — round-1 fix, see below,
 * this is what keeps them from ever observing a torn write) and the final fold-in-extras+atomic-swap.
 * [enqueue]'s own critical section is symmetrically small (a cap check, one append, a counter bump).
 * So the longest either side can ever block the other for is a plain file read/append/rename, never
 * "parse+filter the whole queue" — that's what actually satisfies the "must not block capture" requirement
 * (an async fire-and-forget [enqueue] would satisfy it too, but at the cost of durability — see
 * above — which is the worse trade for a queue whose entire point is surviving a crash/reboot).
 *
 * **Round-4 review fix — this invariant briefly broke, then was restored.** Between Task 10 (which
 * added a thumbnail to the fleet-upload payload) and this round, [enqueue]'s "small, fixed-cost"
 * framing above stopped being true: the caller was stamping a base64-encoded JPEG (tens-to-hundreds
 * of KB, not the "multi-KB clip vector" this doc means) into [QueuedPoint.payloadJson] before ever
 * calling [enqueue], so every JSONL line — and therefore every [ack]'s `writeAtomic` rewrite of the
 * remaining backlog, still done under [lock] — carried that same blob. `thumb_b64` is no longer part
 * of the enqueued payload at all (see [QueuedPoint]'s doc) — [FleetSync] reads and stamps it in
 * lazily, off this class and off `embedLane` entirely, right before the HTTP upsert — so this
 * invariant is accurate again for every point [MomentCapture] enqueues from here on.
 *
 * A synchronous [enqueue] that ISN'T serialized with [ack] at all would risk a subtler bug: if an
 * [enqueue] appends a line while [ack] is mid-read, [ack]'s rewrite (built from an earlier snapshot)
 * could silently overwrite the file and drop that brand-new point. [ack] guards against exactly this:
 * right before it swaps the rewritten file in, it re-reads whatever now sits past the point it
 * originally snapshotted and folds those lines back in — so a point [enqueue] appends during an
 * in-flight [ack] is never lost, only possibly (harmlessly) re-ordered relative to it in the file.
 *
 * **This fold is NOT, by itself, safe against two CONCURRENT [ack] calls** (round-2 review finding):
 * each [ack] computes its snapshot/remaining/fresh triple independently, so two overlapping [ack]s
 * can race their `writeAtomic` swaps — the second can resurrect ids the first already removed
 * (harmless, Qdrant upsert-by-id is idempotent) or, worse, undercount past `snapshot.size` and
 * silently drop an entry the OTHER [ack] hadn't folded in yet. [UploadQueue] itself does not
 * single-flight [ack] — that's a caller responsibility, kept out of this class because only the
 * caller knows whether it ever calls [ack] from more than one place at once. The one caller in this
 * codebase, [tech.qdrant.glasses.fleet.FleetSync.pushDrain], enforces single-flight itself (a
 * `Mutex` around its whole drain/upsert/ack body) — see that class's `pushMutex` doc. A future
 * second caller of [ack] MUST provide the same guarantee, or serialize through the same [FleetSync].
 *
 * **Round-1 fix — [ack]'s OWN snapshot read must not be torn.** The fold above assumes [ack]'s
 * first read either fully sees a concurrently-appended line or doesn't see it at all. That assumption
 * broke because the read used to run via a plain unlocked `file.readLines()` while [enqueue]'s
 * `file.appendText` (a multi-KB write for a real clip vector, spanning more than one filesystem
 * page) was in flight: a read landing mid-write could observe a HALF-written last line, which
 * [parseLine] then drops as unparseable — but the fold's "extra = lines past `snapshot.size`" logic
 * still counted that half-written line towards `snapshot.size`, so once the write completed the now-
 * valid line sat at an index BELOW the fold's cutoff and was never re-included — permanently and
 * silently dropped. Since [enqueue]'s write itself runs entirely inside [lock], the fix is for
 * [ack]'s snapshot read to take [lock] too (a plain `file.readLines()`, not the JSON parse) — the two
 * can then never interleave, so the read is always either fully-before or fully-after any given
 * `appendText`, never mid-write. Only the read moved under [lock]; the actually slow part (per-line
 * JSON parse + filter) still runs OUTSIDE it, exactly as before.
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
 * — every entry, acked or not — fully intact. **Round-1 fix:** a failed [File.renameTo] used to fall
 * back to `tmp.copyTo(file, overwrite = true)` — a non-atomic truncate-then-copy of the LIVE file
 * that defeated the entire point of the tmp-swap (a crash mid-copy could still destroy unacked
 * entries). That fallback is gone: a rename failure now throws, [ack] returns `false` (see below)
 * with the original file untouched, and the whole rewrite is retried from scratch on the next [ack]
 * call — "leave it queued and retry" instead of "maybe corrupt it now," matching every other
 * fail-soft path in this class.
 *
 * **Round-4 fix — this was atomic-VISIBILITY, not crash-DURABLE.** Write-then-rename alone
 * guarantees no reader/JVM-crash ever sees a torn file, but says nothing about a real hard-power-loss
 * event (UVLO battery brownout — a documented, recurring failure mode on this exact glasses hardware,
 * CLAUDE.md): a rename that already returned to the caller can still lose the tmp file's bytes, or in
 * rarer cases the rename itself, if power dies before the filesystem journals it. [writeAtomic] now
 * `fsync`s the tmp file before the rename (durable content) and best-effort `fsync`s the parent
 * directory after it (durable rename), closing that gap — see [writeAtomic]'s own doc for exactly
 * what's guaranteed vs. best-effort.
 *
 * **[ack]'s return value (round-1 fix):** [ack] used to swallow every failure silently (`Unit`
 * return), so [FleetSync.pushDrain] could not tell a successful rewrite from a failed one — it
 * treated a failed [ack] as done and immediately re-[drain]ed + re-upserted the SAME still-queued
 * batch, an unbounded tight retry loop against the server on a persistent local I/O failure (e.g. a
 * full disk). [ack] now returns `true` only when the queue file was durably updated to reflect the
 * acked ids (or there was nothing to do), `false` on any failure (still logged, still non-throwing to
 * the caller) — [pushDrain] uses this to stop its pass instead of spinning.
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
    // enqueue()'s cap-check+append+counter-bump, ack()'s two plain file.readLines() calls (round-1
    // fix — never the JSON parse/filter that follows each one), and ack()'s final tail-fold+atomic-
    // swap. Deliberately does NOT cover drain()'s read (read-only, benign under a torn concurrent
    // append — parseLine just skips it) or ack()'s JSON-parsing/filtering of the whole file.
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
     * file or an empty [ids] is a no-op (returns `true`), not an error (mirrors [drain]'s soft-fail-
     * safe shape — an ack after a queue already emptied by a prior run must never throw). Fail-soft
     * like [enqueue]/[drain]: any failure is logged and swallowed (never thrown to the caller), the
     * entries stay queued to retry later, and this returns `false` so the caller (see [FleetSync.
     * pushDrain]) can tell the difference instead of assuming every ack landed (round-1 fix — see
     * class doc "[ack]'s return value").
     */
    fun ack(ids: Collection<String>): Boolean {
        if (ids.isEmpty()) return true
        return try {
            if (!file.exists()) return true
            val acked = ids.toSet()
            // The READ itself runs under [lock] (round-1 fix — see class doc "ack's OWN snapshot
            // read must not be torn"): enqueue()'s file.appendText also runs entirely under [lock],
            // so this can never observe a half-written line. Only the read is locked here — the
            // actually slow part (per-line JSON parse + filter) still runs OFF [lock], same as before.
            val snapshot = synchronized(lock) { if (file.exists()) file.readLines() else emptyList() }
            val remaining = snapshot.filter { it.isNotBlank() }.mapNotNull(::parseLine).filter { it.id !in acked }
            synchronized(lock) {
                // Fold in anything enqueue() appended AFTER `snapshot` was taken (appends only ever
                // grow the file, so any lines beyond `snapshot.size` are new arrivals) — without this,
                // the swap below would silently overwrite a point that landed mid-ack. This second
                // read is ALSO under [lock] (mutual exclusion with enqueue()'s append), so it's whole
                // and complete too.
                val fresh = file.readLines()
                val extra = fresh.drop(snapshot.size).filter { it.isNotBlank() }.mapNotNull(::parseLine)
                val finalRemaining = remaining + extra
                writeAtomic(finalRemaining)   // throws (round-1 fix) rather than falling back unsafe
                count = finalRemaining.size
            }
            true
        } catch (e: Throwable) {
            Log.w(TAG, "upload queue ack failed (non-fatal, entries stay queued and will retry): ${e.message}")
            false
        }
    }

    /**
     * Write-temp-then-rename so a crash never leaves the live file partially truncated. Throws if
     * the atomic rename fails (round-1 fix) — callers must NOT fall back to a non-atomic copy-then-
     * overwrite of the live file (see class doc "Durability of ack"); [ack]'s own try/catch turns
     * this into a logged, fail-soft `false` return with the ORIGINAL file left untouched.
     *
     * **Round-4 review fix — fsync before the rename.** Write-then-rename alone is atomic w.r.t.
     * VISIBILITY (a reader, or a crash in the JVM itself, never observes a torn file — the original
     * claim this KDoc made) but is NOT crash-DURABLE on its own: a rename that has already returned
     * to the caller is not guaranteed to have the tmp file's bytes on flash yet, and on a real
     * hard-power-loss event the write can simply vanish. That gap matters on THIS hardware
     * specifically — UVLO battery-brownout is a documented, recurring failure mode on the AR1
     * glasses (CLAUDE.md), not a hypothetical. [FileOutputStream.getFD] + [java.io.FileDescriptor.sync]
     * is `fsync(2)`: it blocks until the tmp file's data AND metadata are durable, so by the time the
     * rename is requested there is nothing left for a crash to lose from the write itself. A sync
     * failure here throws (same contract as a rename failure — [ack] turns it into a logged, fail-
     * soft `false`, original file untouched, retried next time).
     *
     * A best-effort fsync of the PARENT DIRECTORY follows the rename, so the directory-entry update
     * (the rename itself) is also durable, not just the tmp file's bytes — same UVLO concern, the
     * other half of it. This is deliberately best-effort (caught, logged, never rethrown): opening a
     * directory for `force()` is standard POSIX but its exact guarantees are filesystem-dependent and
     * this path is not exercised on-device as part of this fix (no adb/on-device I/O experiments were
     * run to confirm it), so a platform/filesystem that rejects it must degrade to "no worse than
     * before this fix," not turn an otherwise-successful [ack] into a failure.
     */
    private fun writeAtomic(remaining: List<QueuedPoint>) {
        if (remaining.isEmpty()) {
            file.delete()
            return
        }
        val body = remaining.joinToString("\n") { it.toLine() } + "\n"
        val tmp = File(file.absoluteFile.parentFile, file.name + ".tmp")
        FileOutputStream(tmp).use { fos ->
            fos.write(body.toByteArray(Charsets.UTF_8))
            fos.fd.sync()   // fsync(2) — durable on disk before we ever ask for the rename
        }
        if (!tmp.renameTo(file)) {
            // Same-filesystem rename should always succeed for an app-private filesDir path; if it
            // somehow doesn't, throw rather than silently truncating the live file — see class doc.
            throw IOException("upload queue: atomic rename of ${tmp.name} -> ${file.name} failed")
        }
        try {
            // parentFile is nullable in general (a bare root path has none) but never null in
            // practice here — [file] is always constructed with an explicit parent directory
            // (`filesDir`/`fleet_queue.jsonl` in production, a temp dir in tests) — the `?.let` is
            // just the null-safe way to express "skip this best-effort step" for that edge case.
            file.absoluteFile.parentFile?.toPath()?.let { dir ->
                FileChannel.open(dir, StandardOpenOption.READ).use { it.force(true) }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "upload queue: parent-directory fsync after rename failed (non-fatal, best-effort durability hardening): ${e.message}")
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
