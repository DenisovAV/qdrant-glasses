package tech.qdrant.glasses

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppStateHolderTest {
    @Test fun startsLoading() {
        assertTrue(AppStateHolder().state.value is AppState.Loading)
    }
    @Test fun onMemoryIndexedBumpsOnlyWhileRecording() {
        val h = AppStateHolder()
        h.onMemoryIndexed()                              // not recording → ignored
        assertTrue(h.state.value is AppState.Idle || h.state.value is AppState.Loading)
        h.setIdle(); h.beginRecordingNoTicker()          // test helper, see Step 3
        h.onMemoryIndexed(); h.onMemoryIndexed()
        assertEquals(2L, (h.state.value as AppState.Recording).indexed)
    }
    @Test fun onVoicePartialReplacesProcessingSentinel() {
        val h = AppStateHolder()
        h.startListening(); h.onVoiceStopped()           // → Processing("...")
        h.onVoicePartial("red cup")
        assertEquals("red cup", (h.state.value as AppState.Processing).query)
    }
}
