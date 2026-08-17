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
    // update/confirmed must both run here, single-threaded).
    @OptIn(ExperimentalCoroutinesApi::class)
    private val inferLane = Dispatchers.Default.limitedParallelism(1)

    // MomentCapture's vision-encoder calls (frame confirms + region embeds, Spec §8.2) serialize
    // on this ONE single-thread lane: OrtSession.run's concurrency under the QNN EP isn't verified
    // safe. Task 2.4 retired PerceptionPipeline's own crop embed (it used to share this lane too,
    // via a private `cropLane` before that, then this shared instance) — MomentCapture is the only
    // consumer now. Still owned here, not by MomentCapture, so GlassesComponents.load can hand the
    // same instance to MomentCapture's constructor.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val embedLane = Dispatchers.IO.limitedParallelism(1)

    // ---- Object mode -------------------------------------------------------------------
    private val appMode = AppMode.OBJECTS   // flip to LEGACY for the old whole-frame path

    // The wired MjpegServer path used this to replay already-stored OBJECTS into a HUD that
    // connects/reconnects mid-session (crop-store path — retired, Task 2.4: no OBJECTS-mode
    // VectorStore exists to replay from anymore). Moments have their own backfill instead:
    // GlassesViewModel.init pushes momentStore.timeline() directly via HudEvents.momentEvent (see
    // below) rather than through this railItems/broadcastRailSnapshot seam, because storedEvent's
    // shape (`"t":"stored"`, label, count) is an OBJECT rail item — replaying moments through it
    // would land them in the dashboard's object rail under the wrong event type. Left as an empty
    // provider (not removed — HudPublisher's constructor still requires one for the wired path)
    // rather than wired to momentStore: a HUD reconnecting to the wired path after the one-time
    // init backfill below won't see a moment-rail replay — a known gap, not fixed by this task
    // (Spec §5 flags dashboard-side rendering as a separate, cross-repo change).
    private val hud = tech.qdrant.glasses.stream.HudPublisher(railItems = { emptyList() })

    fun attachStreamer(s: tech.qdrant.glasses.stream.FrameSink) = hud.attach(s)

    @Volatile private var components: GlassesComponents? = null

    // Quarantine for the dormant appMode == LEGACY path (Task 6). Constructed only when
    // appMode == LEGACY (mirrors GlassesComponents' "nullable by mode, never by timing" rule) —
    // in the shipped OBJECTS config this stays null and every legacy?.xxx() call below is a no-op.
    @Volatile private var legacy: LegacyMomentPipeline? = null

    // Object-mode hot path (detect→track→stream boxes→hand off to MomentCapture), extracted
    // verbatim in Task 7; Task 2.4 retired its crop-embed-and-store tail. Constructed only when
    // appMode == OBJECTS (once GlassesComponents has loaded the detector/tracker); stays null in
    // LEGACY and until load completes, so the OBJECTS branch of onFrame is a no-op until it's
    // ready — the same null-guard the old inline `components?.detector ?: return` gave.
    @Volatile private var perception: PerceptionPipeline? = null

    // OBJECTS voice search (Task 2.4: the ONLY OBJECTS-mode search path — the crop-based
    // ObjectSearcher/VectorStore it queried are retired; see onVoiceResult). Constructed only when
    // appMode == OBJECTS AND the sysprop doesn't disable the memory path (Config.MOMENT_MEMORY,
    // default ON), same "nullable by mode/opt-out, never by timing" rule as
    // `components.momentStore`/`momentCapture`. Null when disabled → onVoiceResult answers
    // Unavailable rather than crashing (no fallback memory path exists anymore).
    @Volatile private var momentSearcher: tech.qdrant.glasses.search.MomentSearcher? = null

    init {
        Log.i(TAG, "init: starting model + store loading")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Loading order (store → LEGACY encoders → bge → OBJECTS detector/tracker/crop/
                // retriever/moment memory → LEGACY ASR pre-warm) and per-mode nullability live in
                // GlassesComponents.load; it THROWS on failure (caught below, same as before).
                val c = GlassesComponents.load(
                    app, appMode,
                    scope = viewModelScope, embedLane = embedLane,
                    isRecording = { session.isRecording },
                    hud = hud,
                )
                components = c
                if (appMode == AppMode.OBJECTS) {
                    // Built into a LOCAL first, not straight into the `perception` field (Codex P2
                    // fix): the camera path only ever reads the @Volatile `perception` field below,
                    // so as long as regionsProvider is wired from this local BEFORE that field is
                    // published, a frame delivered the instant `perception` goes non-null can never
                    // observe MomentCapture's default empty provider.
                    val localPerception = PerceptionPipeline(
                        viewModelScope, inferLane, c.detector!!, c.tracker!!, hud,
                        isRecording = { session.isRecording },
                        momentCapture = c.momentCapture,
                    )
                    // Task 2.2: wire MomentCapture's region source to PerceptionPipeline's confirmed-
                    // tracks snapshot ONLY NOW — not inside GlassesComponents.load(), where
                    // momentCapture is actually constructed — because PerceptionPipeline doesn't
                    // exist until the line above: its OWN constructor needs `c.momentCapture` as an
                    // argument, so the two have a genuine circular dependency at wiring time (see
                    // MomentCapture.regionsProvider's KDoc). No-op when the sysprop is off /
                    // momentCapture is null. Closes over `localPerception` (not the field) and runs
                    // BEFORE `perception = localPerception` below, on purpose.
                    c.momentCapture?.regionsProvider = { localPerception.latestConfirmedRegions }
                    // Task 2.4 regression fix (Codex P2): retiring PerceptionPipeline's crop-embed
                    // tail deleted the only caller of session.onMemoryIndexed() (it used to run
                    // `withContext(inferLane) { onMemoryIndexed() }` right after a successful
                    // store.upsert — see that deleted block's history), so the on-glasses
                    // `indexed: N` counter stayed at 0 even though MomentCapture IS storing moments.
                    // Wrap the HUD-forward callback GlassesComponents.load already wired onto
                    // mc.onMoment rather than replace it — `session` is only reachable here, not
                    // inside GlassesComponents.load's companion-object scope. onMoment fires once per
                    // stored FRAME moment only — never per region (MomentCapture.confirmAndStore
                    // invokes it inside the frame-store block, BEFORE the additive region layer runs)
                    // — so counting it 1:1 already satisfies "one indexed item per moment, regions
                    // don't double-count" with no extra bookkeeping needed. Marshalled onto inferLane
                    // (session.onMemoryIndexed()'s hard requirement — see AppStateHolder's KDoc) even
                    // though onMoment itself fires on embedLane. Reassigning the var here, BEFORE
                    // `perception = localPerception` below publishes the field a camera frame could
                    // reach momentCapture through, is safe for the same happens-before reason
                    // regionsProvider's wiring above already documents — no @Volatile needed on
                    // onMoment either.
                    val hudOnMoment = c.momentCapture?.onMoment
                    c.momentCapture?.onMoment = { hit ->
                        // Schedule the counter update BEFORE invoking hudOnMoment (Codex P2 fix):
                        // if the HUD callback throws (e.g. ms.count() or an attached sink), the
                        // launch below would never be reached and MomentCapture's own catch would
                        // swallow the exception — the moment gets stored but `indexed` silently
                        // stays stale. Launching first makes the counter update immune to that.
                        viewModelScope.launch(inferLane) { session.onMemoryIndexed() }
                        hudOnMoment?.invoke(hit)
                    }
                    perception = localPerception
                    if (Config.MOMENT_MEMORY) {
                        momentSearcher = tech.qdrant.glasses.search.MomentSearcher(c.cropEncoder!!, c.momentStore!!, hud)
                    }
                    // momentStore is async-loaded (~10s); a HUD that connected before now got an
                    // empty timeline. Task 1.6: backfill it from MomentStore.timeline() (already
                    // oldest-first, its own contract) directly via HudEvents.momentEvent — Task 2.4
                    // retired the crop-store equivalent (hud.broadcastRailSnapshot()/railItems: no
                    // OBJECTS-mode VectorStore left to replay from). count is fetched once (not per
                    // item) — a single native round trip, not one per pushed event.
                    if (Config.MOMENT_MEMORY) {
                        val ms = c.momentStore
                        val moments = ms?.timeline() ?: emptyList()
                        val momentCount = ms?.count() ?: 0L
                        for (m in moments) {
                            val key = File(m.thumbPath).nameWithoutExtension
                            hud.registerThumb(key, m.thumbPath)
                            hud.pushEvent(tech.qdrant.glasses.stream.HudEvents.momentEvent(key, m.timestampMs, momentCount))
                        }
                    }
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
        // Task 1.5 P2 fix: MomentCapture is built once at load() and outlives every recording, so
        // without this a stop→start in the same process kept gating against the PRIOR session's
        // baseline and stamped the app-init episodeId onto a NEW session's frames — see its KDoc.
        // No-op when the sysprop is off / not OBJECTS mode (momentCapture stays null then).
        components?.momentCapture?.startSession()
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
                // Task 2.4: moments are the ONLY OBJECTS-mode search path now — the crop-based
                // ObjectSearcher/VectorStore it queried are retired. `momentSearcher` is null only
                // when the sysprop explicitly disables the memory path (debug.qdrant.memory=0, an
                // A/B/regression kill switch — Config.MOMENT_MEMORY), so Unavailable here is an
                // honest "no memory path", never a `!!` crash.
                val outcome = momentSearcher?.search(query) ?: tech.qdrant.glasses.search.ObjectSearcher.Outcome.Unavailable
                when (outcome) {
                    is tech.qdrant.glasses.search.ObjectSearcher.Outcome.Success -> session.setResults(query, outcome.cards)
                    tech.qdrant.glasses.search.ObjectSearcher.Outcome.Unavailable -> session.setIdle()
                }
                return@launch
            }
            val cards = legacy?.search(query)
            if (cards != null) session.setResults(query, cards) else session.setIdle()
        }
    }

    fun frameCount(): Long =
        components?.let { if (appMode == AppMode.OBJECTS) it.momentStore?.count() else it.store.count() } ?: 0L

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
