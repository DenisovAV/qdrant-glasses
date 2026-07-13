package tech.qdrant.glasses

sealed class AppState {
    object Loading : AppState()
    object Idle : AppState()
    data class Recording(val indexed: Long, val elapsedSeconds: Long) : AppState()
    data class Listening(val partial: String = "") : AppState()
    data class Processing(val query: String) : AppState()
    data class Results(val query: String, val cards: List<tech.qdrant.glasses.search.MomentCard>) : AppState()
    /** Startup failed (model/store/detector init). Shown on the lens so a failure is visible
     *  instead of a permanent Loading hang. `reason` aids on-device diagnosis. */
    data class Error(val reason: String) : AppState()
}

enum class AppMode { LEGACY, OBJECTS }
