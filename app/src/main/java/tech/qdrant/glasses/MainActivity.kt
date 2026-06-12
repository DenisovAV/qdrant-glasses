package tech.qdrant.glasses

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import org.json.JSONObject
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

    private var buttonDownMs = 0L
    private var buttonLongFired = false

    private val actionButtonReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val extras = intent.extras ?: return
            val json = extras.keySet().mapNotNull { k ->
                extras.getString(k)?.let { k to it }
            }.firstOrNull { (_, v) -> v.contains("keyCode") }?.second ?: return
            try {
                val obj = JSONObject(json)
                val type = obj.getString("type")
                val keyCode = obj.getInt("keyCode")
                if (keyCode != 289) return
                Log.i(TAG, "actionButton: type=$type")
                when (type) {
                    "ACTION_DOWN" -> {
                        buttonDownMs = System.currentTimeMillis()
                        buttonLongFired = false
                        window.decorView.postDelayed({
                            if (!buttonLongFired) {
                                buttonLongFired = true
                                Log.i(TAG, "button: long press → recording toggle")
                                handleRecordingToggle()
                            }
                        }, 1200)
                    }
                    "ACTION_UP" -> {
                        if (!buttonLongFired) {
                            buttonLongFired = true  // cancel pending long press
                            Log.i(TAG, "button: short press → tap")
                            handleTap()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "actionButtonReceiver error: $e")
            }
        }
    }

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
        observeState()
        registerReceiver(actionButtonReceiver, IntentFilter("com.rayneo.key_pass_to_user"))
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
            onError   = { error -> runOnUiThread { viewModel.onVoiceError(error) } },
            onPartial = { text  -> viewModel.onVoicePartial(text) },
            onStopped = { runOnUiThread { viewModel.onVoiceStopped() } },
            onReady   = { runOnUiThread { viewModel.onVoiceReady() } }
        )
    }

    private fun handleTap() {
        val state = viewModel.state.value
        Log.d(TAG, "tap: state=$state")
        when (state) {
            is AppState.Idle -> {
                if (viewModel.frameCount() == 0L) {
                    Log.i(TAG, "tap: no frames indexed, ignoring")
                } else {
                    viewModel.startListening()
                    voiceManager.startListening()
                }
            }
            is AppState.Listening -> {
                Log.i(TAG, "tap: stop listening")
                voiceManager.stopListening()
            }
            is AppState.Processing -> {
                Log.i(TAG, "tap: cancel processing")
                voiceManager.stopListening()
                viewModel.backToIdle()
            }
            is AppState.Results -> {
                val left = eyeLeft.getChildAt(0) as? SearchResultsView
                val right = eyeRight.getChildAt(0) as? SearchResultsView
                val advanced = (left?.nextCard() ?: false)
                right?.nextCard()
                if (!advanced) { voiceManager.stopListening(); viewModel.backToIdle() }
            }
            else -> {}
        }
    }

    override fun onGenericMotionEvent(event: android.view.MotionEvent): Boolean {
        Log.d(TAG, "genericMotion: ${event.actionMasked} x=${event.x} y=${event.y} src=${event.source}")
        return super.onGenericMotionEvent(event)
    }

    private fun handleRecordingToggle() {
        // Derive the toggle from the actual app state — a local boolean desyncs when
        // startRecording() rejects (not Idle) and then force-Idles a live query flow.
        when (val s = viewModel.state.value) {
            is AppState.Recording -> {
                Log.i(TAG, "button: STOP recording")
                viewModel.stopRecording()
            }
            is AppState.Idle -> {
                Log.i(TAG, "button: START recording")
                viewModel.startRecording()
            }
            else -> Log.i(TAG, "button: long press ignored (state=${s::class.simpleName})")
        }
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
                            it.update(state.indexed, state.elapsedSeconds)
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
                            it.showCards(state.query, state.cards)
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "onDestroy")
        unregisterReceiver(actionButtonReceiver)
        cameraManager.stop()
        voiceManager.destroy()
    }
}
