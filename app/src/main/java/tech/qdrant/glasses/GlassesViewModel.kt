package tech.qdrant.glasses

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import tech.qdrant.glasses.embedding.ClipTextEncoder
import tech.qdrant.glasses.embedding.ClipVisionEncoder
import tech.qdrant.glasses.storage.MemoryFrame
import tech.qdrant.glasses.storage.VisionMemoryStore
import java.io.File
import java.io.FileOutputStream

sealed class AppState {
    object Idle : AppState()
    data class Recording(val frameCount: Long, val elapsedSeconds: Long) : AppState()
    object Listening : AppState()
    data class Results(val query: String, val frames: List<MemoryFrame>) : AppState()
}

class GlassesViewModel(app: Application) : AndroidViewModel(app) {

    private val visionEncoder = ClipVisionEncoder(app)
    private val textEncoder   = ClipTextEncoder(app)
    val memoryStore           = VisionMemoryStore(app)

    private val _state = MutableStateFlow<AppState>(AppState.Idle)
    val state: StateFlow<AppState> = _state

    private val imagesDir = File(app.filesDir, "images").also { it.mkdirs() }
    private var recordingStartMs = 0L

    fun startRecording() {
        recordingStartMs = System.currentTimeMillis()
        _state.value = AppState.Recording(memoryStore.count(), 0L)
    }

    fun stopRecording() {
        _state.value = AppState.Idle
    }

    fun onFrame(bitmap: Bitmap) {
        if (_state.value !is AppState.Recording) return
        viewModelScope.launch(Dispatchers.Default) {
            val timestampMs = System.currentTimeMillis()
            val file = File(imagesDir, "frame_$timestampMs.jpg")
            FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 70, it) }
            val vector = visionEncoder.encode(bitmap)
            memoryStore.store(file.absolutePath, vector, timestampMs)
            val elapsed = (timestampMs - recordingStartMs) / 1000
            _state.value = AppState.Recording(memoryStore.count(), elapsed)
        }
    }

    fun startListening() {
        _state.value = AppState.Listening
    }

    fun onVoiceResult(text: String) {
        viewModelScope.launch(Dispatchers.Default) {
            val vector = textEncoder.encode(text)
            val results = memoryStore.search(vector, topK = 3)
            _state.value = AppState.Results(text, results)
        }
    }

    fun onVoiceError(error: String) {
        _state.value = AppState.Idle
    }

    fun backToIdle() {
        _state.value = AppState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        visionEncoder.close()
        textEncoder.close()
        memoryStore.close()
    }
}
