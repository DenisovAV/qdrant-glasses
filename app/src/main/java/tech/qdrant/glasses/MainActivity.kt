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

    companion object {
        private const val TAG = "GlassesMain"
        // Button gesture tuning (RayNeo action button, keyCode 289).
        private const val ACTION_KEYCODE = 289
        private const val LONG_PRESS_MS = 1200L     // hold this long in Idle to start recording
        private const val START_GRACE_MS = 1000L    // ignore stop-releases for this long after start
        private const val DOWN_DEBOUNCE_MS = 250L   // drop a DOWN arriving this soon after the last
    }

    private val viewModel: GlassesViewModel by viewModels()
    private lateinit var root: LinearLayout
    private lateinit var eyeLeft: FrameLayout
    private lateinit var eyeRight: FrameLayout
    private lateinit var cameraManager: FrameCaptureManager
    private lateinit var voiceManager: VoiceSearchManager
    private var mjpeg: tech.qdrant.glasses.stream.MjpegServer? = null

    // Button gesture model (RayNeo action button, keyCode 289).
    //   IDLE:      long press — fires WHILE HELD at LONG_PRESS_MS (the original feel; you
    //              see recording begin and release) → start recording
    //              short press (released before that) → voice search
    //   RECORDING: ANY release (short or long)       → stop recording
    // Stop is decided on ACTION_UP so a stray brush doesn't end a session; a start-grace
    // window swallows the release of the very long-press that started the recording. A
    // DOWN-debounce drops button chatter.
    private var buttonDownMs = 0L              // wall-clock of the current press's DOWN (0 = up)
    private var lastDownMs = 0L                // for DOWN debounce
    private var longFired = false              // long-press already started recording this press
    private var recordingStartedAtMs = 0L      // when the current recording began (for grace)
    private val longPressRunnable = Runnable {
        // Fired while the button is still held — the original start feel.
        if (buttonDownMs != 0L && !longFired && viewModel.state.value is AppState.Idle) {
            longFired = true
            recordingStartedAtMs = System.currentTimeMillis()
            Log.i(TAG, "button: long press (held) → start recording")
            // Drop the dedup baseline so the first frame of THIS session is always saved —
            // the camera ran before recording, so otherwise the opening frame is deduped
            // against the pre-recording scene and early speech gets no cover frame.
            cameraManager.resetForNewSession()
            viewModel.startRecording()
        }
    }

    private val actionButtonReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val extras = intent.extras ?: return
            val json = extras.keySet().mapNotNull { k ->
                extras.getString(k)?.let { k to it }
            }.firstOrNull { (_, v) -> v.contains("keyCode") }?.second ?: return
            try {
                val obj = JSONObject(json)
                val type = obj.getString("type")
                if (obj.getInt("keyCode") != ACTION_KEYCODE) return
                val now = System.currentTimeMillis()
                when (type) {
                    "ACTION_DOWN" -> {
                        if (now - lastDownMs < DOWN_DEBOUNCE_MS) return  // bounce
                        lastDownMs = now
                        buttonDownMs = now
                        longFired = false
                        // Start fires while held (Idle only); harmless if not Idle.
                        window.decorView.postDelayed(longPressRunnable, LONG_PRESS_MS)
                    }
                    "ACTION_UP" -> {
                        val downAt = buttonDownMs
                        buttonDownMs = 0L
                        window.decorView.removeCallbacks(longPressRunnable)
                        if (downAt == 0L) return                         // UP with no matching DOWN
                        onButtonRelease(now - downAt, longFired)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "actionButtonReceiver error: $e")
            }
        }
    }

    /** A press was released after [heldMs]; [longAlreadyStarted] = the long-press timer fired. */
    private fun onButtonRelease(heldMs: Long, longAlreadyStarted: Boolean) {
        // The long-press that just started a recording already did its job; its release is
        // swallowed by the start grace below — do not treat it as a stop or a tap.
        val state = viewModel.state.value
        if (state is AppState.Recording) {
            if (System.currentTimeMillis() - recordingStartedAtMs < START_GRACE_MS) {
                Log.i(TAG, "button: release ignored (start grace, held=${heldMs}ms)")
                return
            }
            Log.i(TAG, "button: STOP recording (held=${heldMs}ms)")
            viewModel.stopRecording()
            return
        }
        // Not recording: a long-press that already fired its start consumed this press.
        if (longAlreadyStarted) return
        Log.i(TAG, "button: short press (held=${heldMs}ms) → tap")
        handleTap()
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
        startStreamer()
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
        // Object mode is the active mode: deliver every analyzed frame (passthrough) so the
        // detector/tracker get continuous frames. The legacy CLIP dedup gate is bypassed.
        cameraManager = FrameCaptureManager(this, passthrough = true) { bitmap -> viewModel.onFrame(bitmap) }
        cameraManager.start(this)
    }

    private fun startStreamer() {
        try {
            mjpeg = tech.qdrant.glasses.stream.MjpegServer(8080, assets).also { it.start(10_000, false) }
            mjpeg?.setPlaceholder(buildPlaceholderJpeg())
            viewModel.attachStreamer(mjpeg!!)
            Log.i(TAG, "MJPEG on :8080 — desktop: adb forward tcp:8080 tcp:8080 ; open http://localhost:8080/stream")
            Log.i(TAG, "embed endpoint expected on :9000 — desktop: adb reverse tcp:9000 tcp:9000")
        } catch (e: Exception) {
            Log.e(TAG, "MJPEG start failed", e)
        }
    }

    /** A standby frame so the browser shows something (not blank white) before recording starts. */
    private fun buildPlaceholderJpeg(): ByteArray {
        val w = 640; val h = 480
        val bmp = android.graphics.Bitmap.createBitmap(w, h, android.graphics.Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(bmp)
        c.drawColor(android.graphics.Color.rgb(18, 18, 22))
        val title = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE; textSize = 34f; isAntiAlias = true
            textAlign = android.graphics.Paint.Align.CENTER
        }
        val sub = android.graphics.Paint().apply {
            color = android.graphics.Color.rgb(150, 150, 160); textSize = 22f; isAntiAlias = true
            textAlign = android.graphics.Paint.Align.CENTER
        }
        c.drawText("Qdrant Glasses", w / 2f, h / 2f - 18f, title)
        c.drawText("Waiting for recording — hold the button to start", w / 2f, h / 2f + 24f, sub)
        val baos = java.io.ByteArrayOutputStream()
        bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 80, baos)
        bmp.recycle()
        return baos.toByteArray()
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
        mjpeg?.stop()
    }
}
