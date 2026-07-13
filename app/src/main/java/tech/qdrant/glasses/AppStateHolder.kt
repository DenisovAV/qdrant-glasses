package tech.qdrant.glasses

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/** Sole holder of [AppState] + the recording session counter/clock. NOT a state machine: the
 *  legal-transition guards ("record only from Idle") stay in GlassesViewModel's entry points. */
class AppStateHolder {
    private val _state = MutableStateFlow<AppState>(AppState.Loading)
    val state: StateFlow<AppState> = _state
    private var sessionIndexed = 0L
    private var recordingStartMs = 0L
    private var timerJob: Job? = null

    val isRecording: Boolean get() = _state.value is AppState.Recording   // Runs on: any thread

    fun setLoading() { _state.value = AppState.Loading }                  // Runs on: any thread
    fun setIdle() { _state.value = AppState.Idle }                        // Runs on: any thread
    fun setError(reason: String) { _state.value = AppState.Error(reason) }  // Runs on: any thread
    fun setProcessing(query: String) { _state.value = AppState.Processing(query) }  // Runs on: any thread
    /** Runs on: any thread. */
    fun setResults(query: String, cards: List<tech.qdrant.glasses.search.MomentCard>) {
        _state.value = AppState.Results(query, cards)
    }
    fun startListening() { _state.value = AppState.Listening() }          // Runs on: main
    fun onVoicePartial(text: String) {                                    // Runs on: main
        val s = _state.value
        if (s is AppState.Listening) _state.value = AppState.Listening(text)
        else if (s is AppState.Processing && s.query == "...") _state.value = AppState.Processing(text)
    }
    fun onVoiceStopped() { if (_state.value is AppState.Listening) _state.value = AppState.Processing("...") }  // Runs on: main

    private fun elapsedSeconds() = (System.currentTimeMillis() - recordingStartMs) / 1000

    /** Runs on: main. Reset counter+clock, → Recording(0,0), start the 1s ticker on [scope]. */
    fun beginRecording(scope: CoroutineScope) {
        recordingStartMs = System.currentTimeMillis()
        sessionIndexed = 0L
        _state.value = AppState.Recording(0L, 0L)
        timerJob?.cancel()
        timerJob = scope.launch {
            while (isActive) {
                delay(1000)
                _state.update { if (it is AppState.Recording) it.copy(elapsedSeconds = elapsedSeconds()) else it }
            }
        }
    }
    /** Runs on: main. Cancel ticker, → Idle, return elapsed seconds for the caller's log line. */
    fun endRecording(): Long {
        timerJob?.cancel(); timerJob = null
        val elapsed = elapsedSeconds()
        _state.value = AppState.Idle
        return elapsed
    }
    /** MUST run on inferLane (the VM's onMemoryIndexed seam invokes this inside withContext(inferLane)).
     *  Increments the counter inside the atomic update, guarded by `is Recording`. */
    fun onMemoryIndexed() {
        _state.update {
            if (it is AppState.Recording) { sessionIndexed++; AppState.Recording(sessionIndexed, elapsedSeconds()) }
            else it
        }
    }
    internal fun beginRecordingNoTicker() {   // test-only: recording state without a coroutine scope
        recordingStartMs = System.currentTimeMillis(); sessionIndexed = 0L
        _state.value = AppState.Recording(0L, 0L)
    }
}
