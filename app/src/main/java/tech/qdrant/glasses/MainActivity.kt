package tech.qdrant.glasses

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
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
import tech.qdrant.glasses.ui.LoadingView
import tech.qdrant.glasses.ui.ProcessingView
import tech.qdrant.glasses.ui.RecordingView
import tech.qdrant.glasses.ui.SearchResultsView

class MainActivity : AppCompatActivity() {

    companion object { private const val TAG = "GlassesMain" }

    private val viewModel: GlassesViewModel by viewModels()
    private lateinit var root: LinearLayout
    private lateinit var eyeLeft: FrameLayout
    private lateinit var eyeRight: FrameLayout
    private lateinit var cameraManager: FrameCaptureManager
    private lateinit var voiceManager: VoiceSearchManager
    private lateinit var gestureDetector: GestureDetector

    private var isRecording = false
    private var longPressHandled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i(TAG, "onCreate")
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        root     = findViewById(R.id.root)
        eyeLeft  = findViewById(R.id.eye_left)
        eyeRight = findViewById(R.id.eye_right)

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
        if (missing.isNotEmpty()) {
            Log.w(TAG, "requesting permissions: $missing")
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 1)
        } else {
            Log.d(TAG, "all permissions already granted")
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        permissions.zip(grantResults.toList()).forEach { (perm, result) ->
            val status = if (result == PackageManager.PERMISSION_GRANTED) "GRANTED" else "DENIED"
            Log.i(TAG, "permission $perm: $status")
        }
    }

    private fun setupCamera() {
        Log.d(TAG, "setupCamera")
        cameraManager = FrameCaptureManager(this) { bitmap -> viewModel.onFrame(bitmap) }
        cameraManager.start(this)
    }

    private fun setupVoice() {
        Log.d(TAG, "setupVoice")
        voiceManager = VoiceSearchManager(
            context   = this,
            onResult  = { text  -> viewModel.onVoiceResult(text) },
            onError   = { error -> viewModel.onVoiceError(error) },
            onPartial = { text  -> viewModel.onVoicePartial(text) },
            onStopped = { runOnUiThread { viewModel.onVoiceStopped() } },
            onReady   = { runOnUiThread { viewModel.onVoiceReady() } }
        )
    }

    private fun setupGestures() {
        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                Log.d(TAG, "touch-tap confirmed")
                handleTap()
                return true
            }
            override fun onLongPress(e: MotionEvent) {
                Log.i(TAG, "touch-long-press")
                handleLongPress()
            }
        })
        root.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }
    }

    private fun handleTap() {
        val state = viewModel.state.value
        Log.d(TAG, "tap: state=$state")
        when (state) {
            is AppState.Idle -> {
                viewModel.startListening()
                voiceManager.startListening()
            }
            is AppState.Recording -> {
                Log.i(TAG, "tap: STOP recording")
                isRecording = false
                viewModel.stopRecording()
            }
            is AppState.Listening -> {
                Log.i(TAG, "tap: STOP listening")
                voiceManager.stopListening()
            }
            is AppState.Results -> viewModel.backToIdle()
            else -> {}
        }
    }

    private fun handleLongPress() {
        if (!isRecording) {
            Log.i(TAG, "long-press: START recording")
            isRecording = true
            viewModel.startRecording()
        } else {
            Log.i(TAG, "long-press: STOP recording")
            isRecording = false
            viewModel.stopRecording()
        }
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        Log.d(TAG, "dispatchKeyEvent: action=${event.action} keyCode=${event.keyCode} repeatCount=${event.repeatCount}")
        return super.dispatchKeyEvent(event)
    }

    override fun onKeyLongPress(keyCode: Int, event: KeyEvent): Boolean {
        Log.i(TAG, "onKeyLongPress: keyCode=$keyCode")
        if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER || keyCode == KeyEvent.KEYCODE_ENTER) {
            longPressHandled = true
            handleLongPress()
            return true
        }
        return super.onKeyLongPress(keyCode, event)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        Log.d(TAG, "onKeyDown: keyCode=$keyCode repeatCount=${event.repeatCount}")
        if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER || keyCode == KeyEvent.KEYCODE_ENTER) {
            event.startTracking()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        Log.d(TAG, "onKeyUp: keyCode=$keyCode longPressHandled=$longPressHandled")
        if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER || keyCode == KeyEvent.KEYCODE_ENTER) {
            if (!longPressHandled) handleTap()
            longPressHandled = false
            return true
        }
        return super.onKeyUp(keyCode, event)
    }

    private fun showInBothEyes(makeView: () -> android.view.View) {
        eyeLeft.removeAllViews()
        eyeRight.removeAllViews()
        eyeLeft.addView(makeView())
        eyeRight.addView(makeView())
    }

    private fun observeState() {
        lifecycleScope.launch {
            viewModel.state.collect { state ->
                Log.d(TAG, "state → $state")
                when (state) {
                    is AppState.Loading -> showInBothEyes { LoadingView(this@MainActivity) }
                    is AppState.Idle -> showInBothEyes {
                        IdleView(this@MainActivity).also { it.updateCount(viewModel.frameCount()) }
                    }
                    is AppState.Recording -> showInBothEyes {
                        RecordingView(this@MainActivity).also {
                            it.update(state.saved, state.indexed, state.elapsedSeconds)
                        }
                    }
                    is AppState.Listening -> {
                        val leftView = eyeLeft.getChildAt(0)
                        val rightView = eyeRight.getChildAt(0)
                        if (leftView is ListeningView && rightView is ListeningView) {
                            leftView.showPartial(state.partial)
                            rightView.showPartial(state.partial)
                        } else {
                            showInBothEyes { ListeningView(this@MainActivity) }
                        }
                    }
                    is AppState.Processing -> showInBothEyes { ProcessingView(this@MainActivity, state.query) }
                    is AppState.Results -> showInBothEyes {
                        SearchResultsView(this@MainActivity).also {
                            it.showResults(state.query, state.frames)
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "onDestroy")
        cameraManager.stop()
        voiceManager.destroy()
    }
}
