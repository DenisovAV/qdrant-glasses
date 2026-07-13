package tech.qdrant.glasses.legacy

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import tech.qdrant.glasses.GlassesComponents
import tech.qdrant.glasses.search.MomentCard
import java.io.File
import java.io.FileOutputStream

/**
 * Quarantine for the DORMANT `appMode == AppMode.LEGACY` path (Task 6 of the `GlassesViewModel`
 * God-object decomposition): whole-frame JPEG capture + CLIP encode queue, the ambient
 * (heard-channel) transcriber, and whole-frame + transcript voice search.
 *
 * [tech.qdrant.glasses.GlassesViewModel] only *constructs* this class when `appMode == LEGACY` —
 * in the shipped OBJECTS config the VM's `legacy` field stays null and every method below is
 * unreachable. Moved verbatim from `GlassesViewModel` — do NOT "fix" the LEGACY quirks here (e.g.
 * the retriever coming back null in LEGACY mode is a pre-existing, deliberately-preserved
 * oddity of [GlassesComponents.load]'s per-mode gating).
 *
 * Threading:
 *  - [onFrame] runs on the camera analyzer thread. It takes ownership of `bitmap` and does NOT
 *    recycle it (unlike the OBJECTS path's `onObjectFrame`) — the queued bitmap is only ever
 *    read by the encode worker below, which never recycles it either (a pre-existing LEGACY
 *    leak, left unchanged by this move).
 *  - [onRecordingStarted] / [onRecordingStopped] run on main (called from the VM's guarded
 *    `startRecording`/`stopRecording` entry points).
 *  - [search] MUST already be running on `inferLane` (same lane as the encode/ambient workers,
 *    so it serializes with them) — it does no dispatching of its own.
 *  - [destroyAmbient] runs on main (called from the VM's `onCleared`).
 */
class LegacyMomentPipeline(
    private val scope: CoroutineScope,
    private val inferLane: CoroutineDispatcher,
    private val isRecording: () -> Boolean,
    private val onMemoryIndexed: () -> Unit,
    private val imagesDir: File,
    private val components: () -> GlassesComponents?,
    private val app: Application,
) {
    companion object {
        private const val TAG = "GlassesVM"
    }

    private var savedCount = 0L      // frames captured (internal log only)
    private var encodeQueue = Channel<Pair<File, Bitmap>>(Channel.UNLIMITED)
    private var encodeWorker: Job? = null
    private val recentFrames = ArrayDeque<Pair<String, Long>>()  // (imagePath, t_ms), newest last
    private val recentFramesMax = 64
    // Reject transcript↔frame associations farther apart than this — a "nearest" frame
    // from minutes ago (camera stalled / session boundary) is worse than no frame.
    private val maxFrameAssocMs = 30_000L
    private var ambient: tech.qdrant.glasses.search.AmbientTranscriber? = null

    /** Runs on: main. Resets the per-session counter/queue/ring and starts the encode worker
     *  + ambient transcriber. Called by the VM's `startRecording` after `session.beginRecording`. */
    fun onRecordingStarted() {
        savedCount = 0L
        encodeQueue = Channel(Channel.UNLIMITED)
        // Fresh session — frames of the previous session must not become "nearest"
        // for this session's first transcripts.
        synchronized(recentFrames) { recentFrames.clear() }

        encodeWorker = scope.launch(inferLane) {
            for ((file, bitmap) in encodeQueue) {
                val enc = components()?.visionEncoder ?: continue
                val db  = components()?.store ?: continue
                try {
                    val timestampMs = file.nameWithoutExtension.removePrefix("frame_").toLongOrNull()
                        ?: System.currentTimeMillis()
                    val vector = enc.encode(bitmap)
                    db.storeImage(file.absolutePath, vector, timestampMs)
                    onMemoryIndexed()
                    Log.d(TAG, "indexed frame (total=${db.count()})")
                } catch (e: Exception) {
                    Log.e(TAG, "encode/store frame failed, dropping ${file.name}", e)
                }
            }
        }

        // Heard channel (ambient transcription → text embed → store): the VM only constructs
        // this pipeline at all when appMode == LEGACY (see class KDoc), so unconditionally
        // spinning up the Sherpa VAD+ASR mic pipeline here is safe — this line is only ever
        // reached on a real LEGACY build, never in the shipped OBJECTS config.
        ambient = tech.qdrant.glasses.search.AmbientTranscriber(app) { text, tStart, tEnd ->
            scope.launch(inferLane) {
                val enc = components()?.textEncoder ?: run { Log.d(TAG, "ambient drop: textEncoder not ready"); return@launch }
                val db = components()?.store ?: run { Log.d(TAG, "ambient drop: store not ready"); return@launch }
                val mid = (tStart + tEnd) / 2
                // The speech is valuable on its own (the heard channel searches transcripts);
                // a frame is only an "episode cover". On a static scene the frame-dedup drops
                // near-identical frames, so recentFrames can be empty for this window — store
                // the transcript anyway with an empty image_path rather than losing the speech.
                val nearest = nearestFramePath(mid)
                if (nearest.isEmpty()) Log.d(TAG, "ambient: no nearby frame (deduped?), storing transcript without a cover")
                try {
                    val vec = enc.encode(text.take(300))  // CLIP truncates ~77 tokens; cap chars
                    val bge = components()?.bgeEncoder?.encode(text) ?: run {
                        Log.d(TAG, "ambient drop: bge not ready"); return@launch
                    }
                    db.storeTranscript(text, vec, bge, tStart, tEnd, nearest)
                    onMemoryIndexed()
                    Log.d(TAG, "ambient segment stored: \"${text.take(40)}\"")
                } catch (e: Exception) {
                    // An encoder/FFI throw must not kill the recording session.
                    Log.e(TAG, "ambient segment failed, dropping \"${text.take(40)}\"", e)
                }
            }
        }.also { it.start() }
    }

    /** Runs on: main. Called by the VM's `stopRecording` after `session.endRecording`. */
    fun onRecordingStopped() {
        encodeQueue.close()
        ambient?.stop()
        ambient = null
        Log.d(TAG, "legacy stopRecording: captured frames=$savedCount")
    }

    /**
     * LEGACY per-frame processing: JPEG-save the whole frame to [imagesDir], track it in the
     * recent-frames ring (for transcript↔frame association), and enqueue it for CLIP encoding.
     *
     * Called from the camera analyzer thread (see class KDoc re: bitmap ownership).
     */
    fun onFrame(bitmap: Bitmap) {
        if (!isRecording()) return
        val timestampMs = System.currentTimeMillis()
        val file = File(imagesDir, "frame_$timestampMs.jpg")
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 70, it) }
        savedCount++
        synchronized(recentFrames) {
            recentFrames.addLast(file.absolutePath to timestampMs)
            while (recentFrames.size > recentFramesMax) recentFrames.removeFirst()
        }
        Log.d(TAG, "frame captured: $savedCount (queued for indexing)")
        encodeQueue.trySend(file to bitmap)
    }

    private fun nearestFramePath(midMs: Long): String {
        synchronized(recentFrames) {
            val best = recentFrames.minByOrNull { kotlin.math.abs(it.second - midMs) } ?: return ""
            // A frame minutes away is a wrong memory, not a near one — reject it.
            return if (kotlin.math.abs(best.second - midMs) <= maxFrameAssocMs) best.first else ""
        }
    }

    /**
     * LEGACY voice search: whole-frame CLIP text encode + BGE/CLIP transcript retrieval,
     * enriched with overlapping transcripts. MUST already be running on `inferLane` (same lane
     * as the encode/ambient workers above) — this function does no dispatching of its own.
     *
     * Returns null when a required component isn't ready, the retriever isn't built (LEGACY
     * mode's retriever is a pre-existing null today — see [GlassesComponents.load]'s per-mode
     * gating), or the search throws. The caller (VM) maps null → `session.setIdle()` and a
     * non-null (possibly empty) list → `session.setResults(query, cards)`.
     */
    fun search(query: String): List<MomentCard>? {
        val c = components() ?: return null
        val enc = c.textEncoder ?: return null
        val db = c.store
        return try {
            val t0 = System.currentTimeMillis()
            val clipVec = enc.encode(query)
            val bgeVec = c.bgeEncoder.encode(query)
            val ret = c.retriever
            if (ret == null) { Log.w(TAG, "retriever not ready"); return null }
            val encMs = System.currentTimeMillis() - t0
            val cards = ret.retrieve(query, clipVec, bgeVec)
            Log.i(TAG, "onVoiceResult: encode=${encMs}ms cards=${cards.size}")
            // Enrich each hit with speech that OVERLAPS its frame in time, so even an
            // image hit shows "what was said here" — and a long utterance surfaces on
            // every frame it spanned, not just the one nearest its midpoint.
            cards.map { card ->
                card.copy(frame = card.frame.copy(
                    nearbyTranscripts = db.transcriptsOverlappingFrame(card.frame.timestampMs)
                        .filter { it != card.frame.transcript }
                ))
            }
        } catch (e: Exception) {
            Log.e(TAG, "search failed for \"$query\"", e)
            null
        }
    }

    /** Runs on: main. Called by the VM's `onCleared`. */
    fun destroyAmbient() {
        ambient?.destroy()
    }
}
