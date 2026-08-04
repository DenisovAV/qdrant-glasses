package tech.qdrant.glasses

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import tech.qdrant.glasses.legacy.LegacyMomentPipeline
import tech.qdrant.glasses.pipeline.PerceptionPipeline
import java.io.File

class GlassesViewModel(app: Application) : AndroidViewModel(app) {

    companion object {
        private const val TAG = "GlassesVM"
    }

    // Sole owner of AppState + the recording session counter/clock (Task 4). GlassesViewModel
    // keeps the legal-transition GUARDS (only-from-Idle / only-from-Recording) at its entry
    // points; the holder itself enforces nothing.
    private val session = AppStateHolder()
    val state: StateFlow<AppState> = session.state

    private val imagesDir = File(app.filesDir, "images").also { it.mkdirs() }

    // TFLite Interpreter.run is NOT thread-safe, and EdgeShard's thread-safety is
    // unverified — serialize ALL inference + store work on one lane. A late ambient
    // segment encoding concurrently with a query encode is a real (demo-shaped) overlap.
    // This is ALSO the object-detection / tracker lane (ObjectTracker is not thread-safe:
    // update/confirmedUnembedded/markEmbedded must all run here, single-threaded).
    @OptIn(ExperimentalCoroutinesApi::class)
    private val inferLane = Dispatchers.Default.limitedParallelism(1)

    // ---- Object mode -------------------------------------------------------------------
    private val appMode = AppMode.OBJECTS   // flip to LEGACY for the old whole-frame path

    private val objectsDir by lazy {
        File(getApplication<Application>().filesDir, "object_thumbs").also { it.mkdirs() }
    }

    // When a HUD connects, hand it the objects already in memory so its rail isn't empty after a
    // restart. Read components?.objectStore lazily (it's created async, inside GlassesComponents);
    // a HUD that connects before components exists just gets an empty list and is refilled by
    // live `stored` events as usual.
    private val hud = tech.qdrant.glasses.stream.HudPublisher(railItems = {
        components?.objectStore?.all()?.map {
            tech.qdrant.glasses.stream.MjpegServer.RailItem(
                key = java.io.File(it.thumbPath).nameWithoutExtension,
                label = it.label,
                thumbPath = it.thumbPath,
            )
        } ?: emptyList()
    })

    fun attachStreamer(s: tech.qdrant.glasses.stream.FrameSink) = hud.attach(s)

    @Volatile private var components: GlassesComponents? = null

    // Quarantine for the dormant appMode == LEGACY path (Task 6). Constructed only when
    // appMode == LEGACY (mirrors GlassesComponents' "nullable by mode, never by timing" rule) —
    // in the shipped OBJECTS config this stays null and every legacy?.xxx() call below is a no-op.
    @Volatile private var legacy: LegacyMomentPipeline? = null

    // Object-mode hot path (detect→track→crop→embed→dedup→store), extracted verbatim in Task 7.
    // Constructed only when appMode == OBJECTS (once GlassesComponents has loaded the detector/
    // tracker/cropEncoder/objectStore); stays null in LEGACY and until load completes, so the
    // OBJECTS branch of onFrame is a no-op until it's ready — the same null-guard the old inline
    // `components?.detector ?: return` gave.
    @Volatile private var perception: PerceptionPipeline? = null

    // OBJECTS voice search, extracted verbatim in Task 8 (returns an Outcome; the VM maps it to
    // AppState). Constructed only when appMode == OBJECTS, alongside `perception` — same
    // "nullable by mode, never by timing" lifecycle.
    @Volatile private var searcher: tech.qdrant.glasses.search.ObjectSearcher? = null

    init {
        Log.i(TAG, "init: starting model + store loading")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Loading order (store → LEGACY encoders → bge → OBJECTS detector/tracker/crop/
                // objectStore/retriever → LEGACY ASR pre-warm) and per-mode nullability live in
                // GlassesComponents.load; it THROWS on failure (caught below, same as before).
                val c = GlassesComponents.load(app, appMode)
                components = c
                if (appMode == AppMode.OBJECTS) {
                    perception = PerceptionPipeline(
                        viewModelScope, inferLane, c.detector!!, c.tracker!!, c.cropEncoder!!,
                        c.objectStore!!, hud,
                        isRecording = { session.isRecording },
                        onMemoryIndexed = { session.onMemoryIndexed() },
                        objectThumbsDir = objectsDir,
                    )
                    searcher = tech.qdrant.glasses.search.ObjectSearcher(c.cropEncoder!!, c.objectStore!!, hud)
                    // The store is async (~10s); a HUD usually connected before now and got an
                    // empty rail. Now that objects are loadable, fill any already-connected HUDs'
                    // rails.
                    hud.broadcastRailSnapshot()
                }
                if (appMode == AppMode.LEGACY) {
                    legacy = LegacyMomentPipeline(
                        scope = viewModelScope,
                        inferLane = inferLane,
                        isRecording = { session.isRecording },
                        onMemoryIndexed = { session.onMemoryIndexed() },
                        imagesDir = imagesDir,
                        components = { components },
                        app = app,
                    )
                }

                Log.i(TAG, "init: all components ready → Idle")
                session.setIdle()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e   // normal scope teardown — never show it as a failure
            } catch (e: Throwable) {
                // Catch Throwable, not Exception: a missing HTP/vendor .so surfaces as an
                // UnsatisfiedLinkError (an Error, not an Exception) and would otherwise kill the
                // coroutine silently, leaving the app stuck on Loading forever with no signal.
                Log.e(TAG, "init: FAILED", e)
                session.setError(e.message ?: e.javaClass.simpleName)
            }
        }
    }

    fun startRecording() {
        // Only start from Idle — never while a query STT is active (Listening/Processing/
        // Results), or the ambient recognizer would fight the query recognizer for the mic.
        if (session.state.value !is AppState.Idle) {
            Log.w(TAG, "startRecording ignored: not Idle (state=${session.state.value::class.simpleName})")
            return
        }
        Log.i(TAG, "startRecording: indexed=${components?.store?.count() ?: 0}")
        session.beginRecording(viewModelScope)
        hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.modeEvent("recording"))
        legacy?.onRecordingStarted()
    }

    fun stopRecording() {
        // Symmetric to the startRecording guard: never force-Idle a live query flow
        // (Listening/Processing/Results) whose recognizer still holds the mic.
        val current = session.state.value
        if (current !is AppState.Recording) {
            Log.w(TAG, "stopRecording ignored: not Recording (state=${current::class.simpleName})")
            return
        }
        val elapsed = session.endRecording()
        Log.i(TAG, "stopRecording: ${elapsed}s indexed=${current.indexed} total=${components?.store?.count()}")
        legacy?.onRecordingStopped()
        hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.modeEvent("idle"))
    }

    /** Surface a fatal runtime failure (e.g. camera bind) as the error screen, same as an init failure. */
    fun reportFatal(reason: String) { session.setError(reason) }

    fun onFrame(bitmap: Bitmap) {
        // OBJECTS mode snapshots the frame into independent copies inside PerceptionPipeline.onFrame
        // and never touches `bitmap` after that returns; FrameCaptureManager does NOT recycle it (only
        // proxy.close()), so recycle it here or the ~2-3MB frame leaks to GC every call (~30fps).
        // LEGACY hands `bitmap` to an async encode queue below, so it must NOT be recycled here.
        if (appMode == AppMode.OBJECTS) {
            try { perception?.onFrame(bitmap) } finally { bitmap.recycle() }
            return
        }
        legacy?.onFrame(bitmap)
    }

    fun startListening() {
        Log.i(TAG, "startListening: waiting for voice input")
        session.startListening()
    }

    fun onVoiceReady() {
        Log.i(TAG, "onVoiceReady: mic open")
    }

    fun onVoicePartial(text: String) { session.onVoicePartial(text) }

    fun onVoiceStopped() { session.onVoiceStopped() }

    fun onVoiceResult(text: String) {
        Log.i(TAG, "onVoiceResult: query=\"$text\"")
        // Don't search on silence: an empty/blank or 1-char STT result is noise, not a
        // query. Encoding "" still yields a vector that can scrape a stray frame past the
        // (low) vision gate, so guard at the source and just return to Idle.
        val query = text.trim()
        if (query.length < 2) { Log.i(TAG, "onVoiceResult: empty/too-short query, skipping search"); session.setIdle(); return }
        session.setProcessing(query)
        hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.modeEvent("search", query))
        viewModelScope.launch(inferLane) {
            if (appMode == AppMode.OBJECTS) {
                when (val o = searcher!!.search(query)) {
                    is tech.qdrant.glasses.search.ObjectSearcher.Outcome.Success -> session.setResults(query, o.cards)
                    tech.qdrant.glasses.search.ObjectSearcher.Outcome.Unavailable -> session.setIdle()
                }
                return@launch
            }
            val cards = legacy?.search(query)
            if (cards != null) session.setResults(query, cards) else session.setIdle()
        }
    }

    fun frameCount(): Long =
        components?.let { if (appMode == AppMode.OBJECTS) it.objectStore?.count() else it.store.count() } ?: 0L

    fun onVoiceError(error: String) {
        Log.w(TAG, "onVoiceError: $error")
        // A missing/blocked recognizer (the no-response timeout) is indistinguishable from "nothing
        // found" if we just go Idle — that's exactly what confused a tester. Surface it briefly, then
        // auto-recover to Idle (the Error screen has no tap-out). A normal ASR error (e.g. no-match on
        // a mumble) stays silent so we don't nag on every failed utterance.
        if (error.contains("not responding")) {
            session.setError("Speech recognition unavailable — see setup")
            viewModelScope.launch { delay(3000); if (state.value is AppState.Error) session.setIdle() }
        } else {
            session.setIdle()
        }
    }

    fun backToIdle() {
        Log.d(TAG, "backToIdle")
        session.setIdle()
    }

    override fun onCleared() {
        super.onCleared()
        Log.i(TAG, "onCleared: releasing resources")
        legacy?.destroyAmbient()
        // viewModelScope is already cancelled, but cancellation is cooperative — a worker
        // may still be INSIDE a native call (TFLite run / Qdrant FFI). Closing the
        // interpreter/shard under it is a native crash, not an exception. Give in-flight
        // work a bounded moment to drain before closing.
        runBlocking {
            withTimeoutOrNull(800) {
                viewModelScope.coroutineContext.job.children.forEach { it.join() }
            } ?: Log.w(TAG, "onCleared: in-flight work didn't drain in 800ms, closing anyway")
        }
        components?.close()
    }
}
