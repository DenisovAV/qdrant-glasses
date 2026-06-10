package tech.qdrant.glasses

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
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
    data class Recording(val saved: Long, val indexed: Long, val elapsedSeconds: Long) : AppState()
    data class Listening(val partial: String = "") : AppState()
    data class Processing(val query: String) : AppState()
    data class Results(val query: String, val frames: List<MemoryFrame>) : AppState()
}

class GlassesViewModel(app: Application) : AndroidViewModel(app) {

    companion object { private const val TAG = "GlassesVM" }

    private var visionEncoder: VisionEncoder? = null
    private var textEncoder: TextEncoder? = null
    private var store: VisionMemoryStore? = null

    private val _state = MutableStateFlow<AppState>(AppState.Loading)
    val state: StateFlow<AppState> = _state

    private val imagesDir = File(app.filesDir, "images").also { it.mkdirs() }
    private val thumbsDir = File(app.filesDir, "thumbnails").also { it.mkdirs() }
    private var recordingStartMs = 0L
    private var timerJob: Job? = null
    private var savedCount = 0L
    private var encodeQueue = Channel<Pair<File, Bitmap>>(Channel.UNLIMITED)
    private var encodeWorker: Job? = null

    init {
        Log.i(TAG, "init: starting model + store loading")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "init: opening VisionMemoryStore")
                store = VisionMemoryStore(app)
                Log.d(TAG, "init: VisionMemoryStore OK, stored frames=${store?.count()}")

                Log.d(TAG, "init: loading vision encoder [${EncoderFactory.backend}]")
                visionEncoder = EncoderFactory.createVision(app)
                Log.d(TAG, "init: vision encoder OK")

                Log.d(TAG, "init: loading text encoder [${EncoderFactory.backend}]")
                textEncoder = EncoderFactory.createText(app)
                Log.d(TAG, "init: text encoder OK")

                Log.i(TAG, "init: all components ready → Idle")
                _state.value = AppState.Idle
            } catch (e: Exception) {
                Log.e(TAG, "init: FAILED", e)
            }
        }
    }

    fun startRecording() {
        recordingStartMs = System.currentTimeMillis()
        savedCount = 0L
        encodeQueue = Channel(Channel.UNLIMITED)
        Log.i(TAG, "startRecording: indexed=${store?.count() ?: 0}")
        _state.value = AppState.Recording(0L, 0L, 0L)

        encodeWorker = viewModelScope.launch(Dispatchers.Default) {
            for ((file, bitmap) in encodeQueue) {
                val enc = visionEncoder ?: continue
                val db  = store ?: continue
                val timestampMs = file.nameWithoutExtension.removePrefix("frame_").toLongOrNull()
                    ?: System.currentTimeMillis()
                val vector = enc.encode(bitmap)
                db.storeImage(file.absolutePath, vector, timestampMs)
                val indexed = db.count()
                Log.d(TAG, "encoded+indexed: indexed=$indexed saved=$savedCount")
                if (_state.value is AppState.Recording) {
                    val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
                    _state.value = AppState.Recording(savedCount, indexed, elapsed)
                }
            }
        }

        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (isActive) {
                delay(1000)
                val current = _state.value
                if (current is AppState.Recording) {
                    val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
                    _state.value = current.copy(elapsedSeconds = elapsed)
                }
            }
        }
    }

    fun stopRecording() {
        timerJob?.cancel(); timerJob = null
        encodeQueue.close()
        val elapsed = (System.currentTimeMillis() - recordingStartMs) / 1000
        Log.i(TAG, "stopRecording: ${elapsed}s saved=$savedCount indexed=${store?.count()}")
        _state.value = AppState.Idle
    }

    fun onFrame(bitmap: Bitmap) {
        if (_state.value !is AppState.Recording) return
        val timestampMs = System.currentTimeMillis()
        val file = File(imagesDir, "frame_$timestampMs.jpg")
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 70, it) }
        savedCount++
        val elapsed = (timestampMs - recordingStartMs) / 1000
        val indexed = store?.count() ?: 0L
        Log.d(TAG, "frame saved: saved=$savedCount indexed=$indexed")
        if (_state.value is AppState.Recording) {
            _state.value = AppState.Recording(savedCount, indexed, elapsed)
        }
        encodeQueue.trySend(file to bitmap)
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
        viewModelScope.launch(Dispatchers.Default) {
            val enc = textEncoder ?: return@launch
            val db  = store ?: return@launch
            val t0 = System.currentTimeMillis()
            val vector = enc.encode(text)
            val encMs = System.currentTimeMillis() - t0
            val results = db.search(vector, topK = 3)
            val searchMs = System.currentTimeMillis() - t0 - encMs
            Log.i(TAG, "onVoiceResult: encode=${encMs}ms search=${searchMs}ms results=${results.size}")
            results.forEachIndexed { i, f ->
                Log.d(TAG, "  result[$i] score=%.3f path=${f.imagePath.substringAfterLast('/')}".format(f.score))
            }
            _state.value = AppState.Results(text, results)
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
        visionEncoder?.close()
        textEncoder?.close()
        store?.close()
    }
}
