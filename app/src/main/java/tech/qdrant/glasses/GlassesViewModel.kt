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
    @OptIn(ExperimentalCoroutinesApi::class)
    private val inferLane = Dispatchers.Default.limitedParallelism(1)

    init {
        Log.i(TAG, "init: starting model + store loading")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "init: opening VisionMemoryStore")
                store = VisionMemoryStore(app)
                Log.d(TAG, "init: VisionMemoryStore OK, stored frames=${store?.count()}")
                store?.dumpAll()  // DIAG: log the whole base at startup
                retriever = store?.let { tech.qdrant.glasses.search.MomentRetriever(it) }

                Log.d(TAG, "init: loading vision encoder [${EncoderFactory.backend}]")
                visionEncoder = EncoderFactory.createVision(app)
                Log.d(TAG, "init: vision encoder OK")

                Log.d(TAG, "init: loading text encoder [${EncoderFactory.backend}]")
                textEncoder = EncoderFactory.createText(app)
                Log.d(TAG, "init: text encoder OK")

                bgeEncoder = tech.qdrant.glasses.embedding.BgeTextEncoder(app)
                Log.d(TAG, "init: bge encoder OK")

                // Pre-warm the ambient ASR model (~180MB) off the main thread so the
                // first recording doesn't block the UI loading it. ensureLoaded is
                // idempotent + @Synchronized, so AmbientTranscriber.start() becomes a
                // warm cache hit. Soft: a failure here just disables ambient transcription.
                tech.qdrant.glasses.search.SherpaStreamingAsr.ensureLoaded(app)

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
                val nearest = nearestFramePath(mid)
                if (nearest.isEmpty()) { Log.d(TAG, "ambient drop: no nearby frame for \"${text.take(40)}\""); return@launch }
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
        _state.value = AppState.Processing(text)
        viewModelScope.launch(inferLane) {
            val enc = textEncoder ?: return@launch
            val db  = store ?: return@launch
            try {
                val t0 = System.currentTimeMillis()
                val clipVec = enc.encode(text)
                val bgeVec = bgeEncoder?.encode(text)
                val ret = retriever
                if (bgeVec == null || ret == null) { Log.w(TAG, "retriever not ready"); _state.value = AppState.Idle; return@launch }
                val encMs = System.currentTimeMillis() - t0
                val cards = ret.retrieve(text, clipVec, bgeVec)
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
                _state.value = AppState.Results(text, enriched)
            } catch (e: Exception) {
                Log.e(TAG, "search failed for \"$text\"", e)
                _state.value = AppState.Idle
            }
        }
    }

    fun frameCount(): Long = store?.count() ?: 0L

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
    }
}
