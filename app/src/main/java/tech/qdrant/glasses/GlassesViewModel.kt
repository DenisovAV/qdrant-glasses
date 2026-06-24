package tech.qdrant.glasses

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import tech.qdrant.glasses.embedding.EncoderFactory
import tech.qdrant.glasses.embedding.TextEncoder
import tech.qdrant.glasses.embedding.VisionEncoder
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.VisionMemoryStore
import java.io.File
import java.io.FileOutputStream

sealed class AppState {
    object Loading : AppState()
    object Idle : AppState()
    data class Recording(val indexed: Long, val elapsedSeconds: Long) : AppState()
    data class Listening(val partial: String = "") : AppState()
    data class Processing(val query: String) : AppState()
    data class Results(val query: String, val cards: List<tech.qdrant.glasses.search.MomentCard>) : AppState()
}

class GlassesViewModel(app: Application) : AndroidViewModel(app) {

    companion object { private const val TAG = "GlassesVM" }

    private var visionEncoder: VisionEncoder? = null
    private var textEncoder: TextEncoder? = null
    private var bgeEncoder: tech.qdrant.glasses.embedding.BgeTextEncoder? = null
    private var retriever: tech.qdrant.glasses.search.MomentRetriever? = null
    private var store: VisionMemoryStore? = null

    private val _state = MutableStateFlow<AppState>(AppState.Loading)
    val state: StateFlow<AppState> = _state

    private val imagesDir = File(app.filesDir, "images").also { it.mkdirs() }
    private val thumbsDir = File(app.filesDir, "thumbnails").also { it.mkdirs() }
    private var recordingStartMs = 0L
    private var timerJob: Job? = null
    private var savedCount = 0L      // frames captured (internal log only)
    private var sessionIndexed = 0L  // HUD: memories actually INDEXED this session (frames + transcripts)
    private var encodeQueue = Channel<Pair<File, Bitmap>>(Channel.UNLIMITED)
    private var encodeWorker: Job? = null
    private val recentFrames = ArrayDeque<Pair<String, Long>>()  // (imagePath, t_ms), newest last
    private val recentFramesMax = 64
    // Reject transcript↔frame associations farther apart than this — a "nearest" frame
    // from minutes ago (camera stalled / session boundary) is worse than no frame.
    private val maxFrameAssocMs = 30_000L
    private var ambient: tech.qdrant.glasses.search.AmbientTranscriber? = null

    // TFLite Interpreter.run is NOT thread-safe, and EdgeShard's thread-safety is
    // unverified — serialize ALL inference + store work on one lane. A late ambient
    // segment encoding concurrently with a query encode is a real (demo-shaped) overlap.
    // This is ALSO the object-detection / tracker lane (ObjectTracker is not thread-safe:
    // update/confirmedUnembedded/markEmbedded must all run here, single-threaded).
    @OptIn(ExperimentalCoroutinesApi::class)
    private val inferLane = Dispatchers.Default.limitedParallelism(1)

    // ---- Object mode -------------------------------------------------------------------
    enum class AppMode { LEGACY, OBJECTS }
    private val appMode = AppMode.OBJECTS   // flip to LEGACY for the old whole-frame path

    private var detector: tech.qdrant.glasses.detect.ObjectDetector? = null
    private var tracker: tech.qdrant.glasses.detect.ObjectTracker? = null
    private var cropEncoder: tech.qdrant.glasses.embedding.CropEncoder? = null
    private var objectStore: tech.qdrant.glasses.storage.ObjectStore? = null
    // Crop embedding is a network call (Mac endpoint). It runs on its OWN single-thread lane
    // so a slow embed never blocks detection on inferLane. markEmbedded is marshalled BACK to
    // inferLane (the tracker's only legal thread) after a successful embed+upsert.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val cropLane = Dispatchers.IO.limitedParallelism(1)
    @Volatile private var streamer: tech.qdrant.glasses.stream.MjpegServer? = null  // set by MainActivity
    private val objectsDir by lazy {
        File(getApplication<Application>().filesDir, "object_thumbs").also { it.mkdirs() }
    }

    fun attachStreamer(s: tech.qdrant.glasses.stream.MjpegServer) { streamer = s }

    init {
        Log.i(TAG, "init: starting model + store loading")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "init: opening VisionMemoryStore")
                store = VisionMemoryStore(app)
                Log.d(TAG, "init: VisionMemoryStore OK, stored frames=${store?.count()}")
                store?.dumpAll()  // DIAG: log the whole base at startup
                retriever = store?.let { tech.qdrant.glasses.search.MomentRetriever(it) }

                // The whole-frame CLIP encoders (~945MB of on-device weights) are LEGACY-only:
                // in OBJECTS mode crop embedding runs on the Mac (SigLIP2), so these models are
                // never used — and are excluded from the APK via androidResources.ignoreAssetsPattern.
                // Loading must therefore be gated by mode too: touching a missing asset here would
                // throw and the init try/catch would never reach Idle.
                if (appMode == AppMode.LEGACY) {
                    Log.d(TAG, "init: loading vision encoder [${EncoderFactory.backend}]")
                    visionEncoder = EncoderFactory.createVision(app)
                    Log.d(TAG, "init: vision encoder OK")

                    Log.d(TAG, "init: loading text encoder [${EncoderFactory.backend}]")
                    textEncoder = EncoderFactory.createText(app)
                    Log.d(TAG, "init: text encoder OK")
                }

                bgeEncoder = tech.qdrant.glasses.embedding.BgeTextEncoder(app)
                Log.d(TAG, "init: bge encoder OK")

                if (appMode == AppMode.OBJECTS) {
                    detector = tech.qdrant.glasses.detect.DetectorFactory.create(app)
                    tracker = tech.qdrant.glasses.detect.ObjectTracker(confirmSightings = 3)
                    cropEncoder = tech.qdrant.glasses.embedding.CropEncoderFactory.create(app)
                    objectStore = tech.qdrant.glasses.storage.ObjectStore(app, dim = cropEncoder!!.dim)
                    Log.i(TAG, "object mode ready (detector+tracker+encoder+store), objects=${objectStore?.count()}")
                }

                // Pre-warm the ambient ASR model (~290MB) off the main thread so the
                // first recording doesn't block the UI loading it. ensureLoaded is
                // idempotent + @Synchronized, so AmbientTranscriber.start() becomes a
                // warm cache hit. Soft: a failure here just disables ambient transcription.
                tech.qdrant.glasses.search.SherpaVadAsr.ensureLoaded(app)

                Log.i(TAG, "init: all components ready → Idle")
                _state.value = AppState.Idle
            } catch (e: Exception) {
                Log.e(TAG, "init: FAILED", e)
            }
        }
    }

    fun startRecording() {
        // Only start from Idle — never while a query STT is active (Listening/Processing/
        // Results), or the ambient recognizer would fight the query recognizer for the mic.
        if (_state.value !is AppState.Idle) {
            Log.w(TAG, "startRecording ignored: not Idle (state=${_state.value::class.simpleName})")
            return
        }
        recordingStartMs = System.currentTimeMillis()
        savedCount = 0L
        sessionIndexed = 0L
        encodeQueue = Channel(Channel.UNLIMITED)
        // Fresh session — frames of the previous session must not become "nearest"
        // for this session's first transcripts.
        synchronized(recentFrames) { recentFrames.clear() }
        Log.i(TAG, "startRecording: indexed=${store?.count() ?: 0}")
        _state.value = AppState.Recording(0L, 0L)

        encodeWorker = viewModelScope.launch(inferLane) {
            for ((file, bitmap) in encodeQueue) {
                val enc = visionEncoder ?: continue
                val db  = store ?: continue
                try {
                    val timestampMs = file.nameWithoutExtension.removePrefix("frame_").toLongOrNull()
                        ?: System.currentTimeMillis()
                    val vector = enc.encode(bitmap)
                    db.storeImage(file.absolutePath, vector, timestampMs)
                    sessionIndexed++
                    val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
                    _state.update { if (it is AppState.Recording) AppState.Recording(sessionIndexed, elapsed) else it }
                    Log.d(TAG, "indexed frame ($sessionIndexed this session, total=${db.count()})")
                } catch (e: Exception) {
                    Log.e(TAG, "encode/store frame failed, dropping ${file.name}", e)
                }
            }
        }

        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (isActive) {
                delay(1000)
                val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
                _state.update { if (it is AppState.Recording) it.copy(elapsedSeconds = elapsed) else it }
            }
        }

        ambient = tech.qdrant.glasses.search.AmbientTranscriber(getApplication()) { text, tStart, tEnd ->
            viewModelScope.launch(inferLane) {
                val enc = textEncoder ?: run { Log.d(TAG, "ambient drop: textEncoder not ready"); return@launch }
                val db = store ?: run { Log.d(TAG, "ambient drop: store not ready"); return@launch }
                val mid = (tStart + tEnd) / 2
                // The speech is valuable on its own (the heard channel searches transcripts);
                // a frame is only an "episode cover". On a static scene the frame-dedup drops
                // near-identical frames, so recentFrames can be empty for this window — store
                // the transcript anyway with an empty image_path rather than losing the speech.
                val nearest = nearestFramePath(mid)
                if (nearest.isEmpty()) Log.d(TAG, "ambient: no nearby frame (deduped?), storing transcript without a cover")
                try {
                    val vec = enc.encode(text.take(300))  // CLIP truncates ~77 tokens; cap chars
                    val bge = bgeEncoder?.encode(text) ?: run {
                        Log.d(TAG, "ambient drop: bge not ready"); return@launch
                    }
                    db.storeTranscript(text, vec, bge, tStart, tEnd, nearest)
                    sessionIndexed++
                    val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
                    _state.update { if (it is AppState.Recording) AppState.Recording(sessionIndexed, elapsed) else it }
                    Log.d(TAG, "ambient segment stored: \"${text.take(40)}\"")
                } catch (e: Exception) {
                    // An encoder/FFI throw must not kill the recording session.
                    Log.e(TAG, "ambient segment failed, dropping \"${text.take(40)}\"", e)
                }
            }
        }.also { it.start() }  // offline Google engine has no model-load gate — always startable
    }

    fun stopRecording() {
        // Symmetric to the startRecording guard: never force-Idle a live query flow
        // (Listening/Processing/Results) whose recognizer still holds the mic.
        if (_state.value !is AppState.Recording) {
            Log.w(TAG, "stopRecording ignored: not Recording (state=${_state.value::class.simpleName})")
            return
        }
        timerJob?.cancel(); timerJob = null
        encodeQueue.close()
        val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
        Log.i(TAG, "stopRecording: ${elapsed}s indexed=$sessionIndexed (captured frames=$savedCount) total=${store?.count()}")
        ambient?.stop()
        ambient = null
        _state.value = AppState.Idle
    }

    fun onFrame(bitmap: Bitmap) {
        if (appMode == AppMode.OBJECTS) { onObjectFrame(bitmap); return }
        if (_state.value !is AppState.Recording) return
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
     * Object-mode per-frame processing: detect → track → stream boxes → embed confirmed crops.
     *
     * Called from the camera analyzer thread (FrameCaptureManager's single-thread executor),
     * NOT inferLane. ObjectTracker is NOT thread-safe, so the whole detect/track/confirm body
     * is dispatched onto inferLane (the tracker's only legal thread). The crop embed (a network
     * call to the Mac endpoint) runs on a SEPARATE cropLane so it can't block detection; on
     * success it marshals markEmbedded BACK to inferLane — the dedup flag is part of tracker
     * state and may only be mutated there.
     */
    fun onObjectFrame(bitmap: Bitmap) {
        if (_state.value !is AppState.Recording) return
        val det = detector ?: return
        val trk = tracker ?: return
        val store = objectStore ?: return
        val enc = cropEncoder ?: return
        // The camera RECYCLES `bitmap` the instant this callback returns (KEEP_ONLY_LATEST +
        // proxy.close() in the analyzer's finally). All real work runs async on inferLane, by
        // which time `bitmap` is dead → "Can't copy a recycled bitmap". Take an independent
        // snapshot synchronously here and own it ourselves; recycle the snapshot when done.
        val frame = try { bitmap.copy(Bitmap.Config.ARGB_8888, false) } catch (e: Throwable) {
            Log.w(TAG, "frame snapshot failed: ${e.message}"); return
        }
        viewModelScope.launch(inferLane) {
            try {
                val t0 = System.currentTimeMillis()
                val detections = try { det.detect(frame) } catch (e: Throwable) {
                    Log.e(TAG, "detect failed", e); return@launch
                }
                val detMs = System.currentTimeMillis() - t0
                val tracks = trk.update(detections)
                Log.d(TAG, "object frame: detect=${detMs}ms detections=${detections.size}")

                // 1) Stream the CLEAN frame (no baked boxes) + send box coordinates as an event.
                streamer?.let { s ->
                    try {
                        val baos = java.io.ByteArrayOutputStream()
                        frame.compress(Bitmap.CompressFormat.JPEG, 70, baos)
                        s.offerFrame(baos.toByteArray())
                        // score per track: a track's bbox == its matched detection's bbox this frame.
                        val scoreByTrack = HashMap<Int, Float>()
                        for (tk in tracks) {
                            detections.firstOrNull { it.label == tk.label && it.bbox == tk.bbox }?.let { scoreByTrack[tk.trackId] = it.score }
                        }
                        s.pushEvent(tech.qdrant.glasses.stream.HudEvents.boxesEvent(tracks, scoreByTrack, frame.width, frame.height))
                    } catch (e: Throwable) { Log.w(TAG, "stream/boxes failed: ${e.message}") }
                }

                // 2) Embed newly-confirmed objects on cropLane (network — must not block detection).
                //    confirmedUnembedded() reads tracker state, so it stays on inferLane here.
                //    cropFrom() copies pixels out of `frame` synchronously, so the snapshot can be
                //    safely recycled below even while these cropLane coroutines are still running.
                for (track in trk.confirmedUnembedded()) {
                    val crop = cropFrom(frame, track.bbox) ?: continue
                    val thumbFile = File(objectsDir, "obj_${track.trackId}_${System.currentTimeMillis()}.jpg")
                    try { FileOutputStream(thumbFile).use { crop.compress(Bitmap.CompressFormat.JPEG, 85, it) } }
                    catch (_: Throwable) {}
                    val bboxStr = "%.3f,%.3f,%.3f,%.3f".format(
                        track.bbox.left / frame.width, track.bbox.top / frame.height,
                        track.bbox.width() / frame.width, track.bbox.height() / frame.height)
                    val tid = track.trackId; val label = track.label
                    viewModelScope.launch(cropLane) {
                        val embedT0 = System.currentTimeMillis()
                        try {
                            val vec = enc.encode(crop)
                            val embedMs = System.currentTimeMillis() - embedT0
                            val storeT0 = System.currentTimeMillis()
                            store.upsert(vec, label, bboxStr, System.currentTimeMillis(), tid, thumbFile.absolutePath)
                            val storeMs = System.currentTimeMillis() - storeT0
                            // markEmbedded mutates tracker state → marshal back to inferLane (the
                            // tracker's only legal thread). Dedup applies only on a successful store.
                            withContext(inferLane) { trk.markEmbedded(tid) }
                            val count = store.count()
                            val key = thumbFile.nameWithoutExtension
                            streamer?.registerThumb(key, thumbFile.absolutePath)
                            streamer?.pushEvent(tech.qdrant.glasses.stream.HudEvents.storedEvent(key, label, count))
                            streamer?.pushEvent(tech.qdrant.glasses.stream.HudEvents.tickEvent(detMs, embedMs, storeMs, count))
                            Log.i(TAG, "object stored: $label (track $tid), total=$count")
                        } catch (e: Throwable) {
                            Log.w(TAG, "embed failed for $label (track $tid), will retry: ${e.message}")
                            // not marked embedded → retried on a later sighting
                        } finally {
                            crop.recycle()  // crop is our own pixels; release once encoded/stored
                        }
                    }
                }
            } finally {
                frame.recycle()  // our snapshot; crops were already copied out above
            }
        }
    }

    private fun cropFrom(frame: Bitmap, box: android.graphics.RectF): Bitmap? {
        val l = box.left.toInt().coerceIn(0, frame.width - 1)
        val t = box.top.toInt().coerceIn(0, frame.height - 1)
        val r = box.right.toInt().coerceIn(l + 1, frame.width)
        val b = box.bottom.toInt().coerceIn(t + 1, frame.height)
        return try { Bitmap.createBitmap(frame, l, t, r - l, b - t) } catch (_: Throwable) { null }
    }

    fun startListening() {
        Log.i(TAG, "startListening: waiting for voice input")
        _state.value = AppState.Listening()
    }

    fun onVoiceReady() {
        Log.i(TAG, "onVoiceReady: mic open")
    }

    fun onVoicePartial(text: String) {
        val s = _state.value
        if (s is AppState.Listening) _state.value = AppState.Listening(text)
        else if (s is AppState.Processing && s.query == "...") _state.value = AppState.Processing(text)
    }

    fun onVoiceStopped() {
        if (_state.value is AppState.Listening) _state.value = AppState.Processing("...")
    }

    fun onVoiceResult(text: String) {
        Log.i(TAG, "onVoiceResult: query=\"$text\"")
        // Don't search on silence: an empty/blank or 1-char STT result is noise, not a
        // query. Encoding "" still yields a vector that can scrape a stray frame past the
        // (low) vision gate, so guard at the source and just return to Idle.
        val query = text.trim()
        if (query.length < 2) { Log.i(TAG, "onVoiceResult: empty/too-short query, skipping search"); _state.value = AppState.Idle; return }
        _state.value = AppState.Processing(query)
        viewModelScope.launch(inferLane) {
            if (appMode == AppMode.OBJECTS) {
                val cropEnc = cropEncoder ?: run { _state.value = AppState.Idle; return@launch }
                val objStore = objectStore ?: run { _state.value = AppState.Idle; return@launch }
                val t0 = System.currentTimeMillis()
                val qvec = try { cropEnc.encodeText(query) } catch (e: Throwable) {
                    Log.e(TAG, "query embed failed", e); _state.value = AppState.Idle; return@launch
                }
                val encMs = System.currentTimeMillis() - t0
                val hits = objStore.search(qvec, topK = 5)
                Log.i(TAG, "onVoiceResult(objects): encode=${encMs}ms hits=${hits.size}")
                val cards = hits.map { h ->
                    tech.qdrant.glasses.search.MomentCard(
                        frame = MemoryFrame(
                            id = h.id, score = h.score, imagePath = h.thumbPath,
                            timestampMs = h.timestampMs, tEndMs = h.timestampMs,
                            type = "object", transcript = h.label,
                        ),
                        fromVision = true, fromHeard = false, strength = h.score,
                    )
                }
                _state.value = AppState.Results(query, cards)
                return@launch
            }
            val enc = textEncoder ?: return@launch
            val db  = store ?: return@launch
            try {
                val t0 = System.currentTimeMillis()
                val clipVec = enc.encode(query)
                val bgeVec = bgeEncoder?.encode(query)
                val ret = retriever
                if (bgeVec == null || ret == null) { Log.w(TAG, "retriever not ready"); _state.value = AppState.Idle; return@launch }
                val encMs = System.currentTimeMillis() - t0
                val cards = ret.retrieve(query, clipVec, bgeVec)
                Log.i(TAG, "onVoiceResult: encode=${encMs}ms cards=${cards.size}")
                // Enrich each hit with speech that OVERLAPS its frame in time, so even an
                // image hit shows "what was said here" — and a long utterance surfaces on
                // every frame it spanned, not just the one nearest its midpoint.
                val enriched = cards.map { c ->
                    c.copy(frame = c.frame.copy(
                        nearbyTranscripts = db.transcriptsOverlappingFrame(c.frame.timestampMs)
                            .filter { it != c.frame.transcript }
                    ))
                }
                _state.value = AppState.Results(query, enriched)
            } catch (e: Exception) {
                Log.e(TAG, "search failed for \"$text\"", e)
                _state.value = AppState.Idle
            }
        }
    }

    fun frameCount(): Long =
        if (appMode == AppMode.OBJECTS) objectStore?.count() ?: 0L
        else store?.count() ?: 0L

    fun onVoiceError(error: String) {
        Log.w(TAG, "onVoiceError: $error")
        _state.value = AppState.Idle
    }

    fun backToIdle() {
        Log.d(TAG, "backToIdle")
        _state.value = AppState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        Log.i(TAG, "onCleared: releasing resources")
        ambient?.destroy()
        // viewModelScope is already cancelled, but cancellation is cooperative — a worker
        // may still be INSIDE a native call (TFLite run / Qdrant FFI). Closing the
        // interpreter/shard under it is a native crash, not an exception. Give in-flight
        // work a bounded moment to drain before closing.
        runBlocking {
            withTimeoutOrNull(800) {
                viewModelScope.coroutineContext.job.children.forEach { it.join() }
            } ?: Log.w(TAG, "onCleared: in-flight work didn't drain in 800ms, closing anyway")
        }
        visionEncoder?.close()
        textEncoder?.close()
        bgeEncoder?.close()
        store?.close()
        detector?.close()
        cropEncoder?.close()
        objectStore?.close()
    }
}
