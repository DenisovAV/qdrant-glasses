package tech.qdrant.glasses

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.FrameLayout
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import tech.qdrant.glasses.camera.FrameCaptureManager
import tech.qdrant.glasses.search.VoiceSearchManager
import tech.qdrant.glasses.ui.IdleView
import tech.qdrant.glasses.ui.ListeningView
import tech.qdrant.glasses.ui.RecordingView
import tech.qdrant.glasses.ui.SearchResultsView

class MainActivity : AppCompatActivity() {

    private val viewModel: GlassesViewModel by viewModels()
    private lateinit var root: FrameLayout
    private lateinit var cameraManager: FrameCaptureManager
    private lateinit var voiceManager: VoiceSearchManager
    private lateinit var gestureDetector: GestureDetector

    private var isRecording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        root = findViewById(R.id.root)

        requestMissingPermissions()
        setupCamera()
        setupVoice()
        setupGestures()
        observeState()
    }

    private fun requestMissingPermissions() {
        val perms = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        val missing = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) ActivityCompat.requestPermissions(this, missing.toTypedArray(), 1)
    }

    private fun setupCamera() {
        cameraManager = FrameCaptureManager(this) { bitmap -> viewModel.onFrame(bitmap) }
        cameraManager.start(this)
    }

    private fun setupVoice() {
        voiceManager = VoiceSearchManager(
            context = this,
            onResult = { text -> viewModel.onVoiceResult(text) },
            onError  = { error -> viewModel.onVoiceError(error) }
        )
    }

    private fun setupGestures() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                when (viewModel.state.value) {
                    is AppState.Idle -> {
                        viewModel.startListening()
                        voiceManager.startListening()
                    }
                    is AppState.Results -> viewModel.backToIdle()
                    else -> {}
                }
                return true
            }
        })

        root.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }

        root.setOnLongClickListener {
            if (!isRecording) {
                isRecording = true
                viewModel.startRecording()
            } else {
                isRecording = false
                viewModel.stopRecording()
            }
            true
        }
    }

    private fun observeState() {
        lifecycleScope.launch {
            viewModel.state.collect { state ->
                root.removeAllViews()
                when (state) {
                    is AppState.Idle -> {
                        root.addView(IdleView(this@MainActivity).also {
                            it.updateCount(viewModel.memoryStore.count())
                        })
                    }
                    is AppState.Recording -> {
                        root.addView(RecordingView(this@MainActivity).also {
                            it.update(state.frameCount, state.elapsedSeconds)
                        })
                    }
                    is AppState.Listening -> root.addView(ListeningView(this@MainActivity))
                    is AppState.Results -> {
                        root.addView(SearchResultsView(this@MainActivity).also {
                            it.showResults(state.query, state.frames)
                        })
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.stop()
        voiceManager.destroy()
    }
}
