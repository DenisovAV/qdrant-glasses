package tech.qdrant.glasses.search

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.util.Log
import tech.qdrant.glasses.BuildConfig
import java.io.ByteArrayOutputStream
import java.util.concurrent.ArrayBlockingQueue

class VoiceSearchManager(
    private val context: Context,
    private val onResult: (String) -> Unit,
    private val onError: (String) -> Unit,
    private val onPartial: (String) -> Unit = {},
    private val onStopped: () -> Unit = {},
    private val onReady: () -> Unit = {}
) {
    companion object {
        private const val TAG = "VoiceSearch"
        private const val SAMPLE_RATE = 16000
        private const val CHUNK_MS = 20
        private const val SILENCE_RMS_THRESHOLD = 50
        private const val SPEECH_TIMEOUT_MS = 1000L
    }

    private val vosk = VoskSpeechRecognizer(context)
    private val googleApiKey = BuildConfig.GOOGLE_STT_API_KEY  // from local.properties
    private val googleStt: GoogleSpeechRecognizer? =
        if (googleApiKey.isNotEmpty()) GoogleSpeechRecognizer(googleApiKey) else null

    // True only when a key is configured AND the device currently has internet.
    // Wrapped in try/catch so a connectivity check failure can never break listening —
    // worst case we fall back to VOSK.
    private fun useGoogle(): Boolean {
        if (googleStt == null) return false
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val net = cm.activeNetwork ?: return false
            val caps = cm.getNetworkCapabilities(net) ?: return false
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        } catch (e: Exception) {
            Log.w(TAG, "useGoogle check failed, falling back to VOSK: ${e.message}")
            false
        }
    }

    private var audioRecord: AudioRecord? = null
    private var readerThread: Thread? = null
    private var voskThread: Thread? = null
    @Volatile private var isListening = false
    @Volatile private var wasStopped = false  // true = finished normally; false = cancelled
    private var listenStartMs = 0L
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var audioFocusRequest: android.media.AudioFocusRequest? = null

    fun startListening() {
        if (isListening) return

        val google = useGoogle()
        if (!google && !vosk.isReady) { onError("VOSK model not ready"); return }

        Log.i(TAG, "startListening [backend=${if (google) "google" else "vosk"}]")

        requestAudioFocus()

        if (google) googleStt!!.startListening(onPartial, onResult, onError)
        else vosk.startListening(onPartial, onResult, onError)

        isListening = true
        wasStopped = false
        listenStartMs = System.currentTimeMillis()

        val chunkSamples = SAMPLE_RATE * CHUNK_MS / 1000
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)

        val sources = listOf(
            MediaRecorder.AudioSource.MIC,
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.UNPROCESSED
        )
        val rec = sources.firstNotNullOfOrNull { src ->
            try {
                AudioRecord(src, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, chunkSamples * 2 * 32))
                    .takeIf { it.state == AudioRecord.STATE_INITIALIZED }
                    ?.also { Log.i(TAG, "AudioRecord initialized with source=$src") }
            } catch (e: Exception) { Log.w(TAG, "source=$src failed: ${e.message}"); null }
        } ?: run { onError("No audio source available"); return }

        audioRecord = rec
        rec.startRecording()
        Log.i(TAG, "recording started")
        onReady()

        // Queue between reader and VOSK threads: reader writes raw chunks, VOSK consumes them.
        // Capacity 500 = ~10 seconds of 20ms chunks — enough headroom for VOSK to catch up.
        val audioQueue = ArrayBlockingQueue<ByteArray>(500)
        // Sentinel to signal VOSK thread that recording is done
        val POISON = ByteArray(0)

        // Thread 1: reads AudioRecord, does VAD/auto-stop, pushes chunks to queue
        readerThread = Thread {
            val buf = ShortArray(chunkSamples)
            val diagBuffer = ByteArrayOutputStream()
            var hadSpeech = false
            var lastSpeechMs = System.currentTimeMillis()
            var logCount = 0
            var maxRms = 0.0

            while (isListening) {
                val read = rec.read(buf, 0, buf.size)
                if (read > 0) {
                    val rms = rms(buf, read)
                    if (rms > maxRms) maxRms = rms
                    if (logCount++ < 100) {
                        val peak = buf.take(read).maxOf { kotlin.math.abs(it.toInt()) }
                        Log.v(TAG, "rms=${"%.0f".format(rms)} peak=$peak")
                    }
                    val bytes = shortsToBytes(buf, read)
                    diagBuffer.write(bytes)
                    audioQueue.offer(bytes)

                    val isSpeech = rms > SILENCE_RMS_THRESHOLD
                    if (isSpeech) {
                        hadSpeech = true
                        lastSpeechMs = System.currentTimeMillis()
                    } else if (hadSpeech) {
                        val silenceMs = System.currentTimeMillis() - lastSpeechMs
                        if (silenceMs > SPEECH_TIMEOUT_MS) {
                            Log.d(TAG, "auto-stop: ${silenceMs}ms silence after speech (maxRms=${"%.0f".format(maxRms)})")
                            wasStopped = true
                            isListening = false
                            onStopped()
                        }
                    }
                }
            }

            rec.stop()
            rec.release()
            releaseAudioFocus()

            val pcm = diagBuffer.toByteArray()
            Log.i(TAG, "recording done (${pcm.size} bytes, maxRms=${"%.0f".format(maxRms)})")

            if (google) {
                // Google STT is batch: send the whole utterance once recording stops.
                // wasStopped distinguishes auto-stop / user-stop (recognize) from cancel.
                if (wasStopped) googleStt!!.recognize(pcm)
                else { onError("Cancelled") }
            } else {
                audioQueue.offer(POISON)
            }
        }.also { it.priority = Thread.MAX_PRIORITY; it.start() }

        // Thread 2 (VOSK only): drains queue, feeds VOSK — decoupled so VAD is not delayed.
        // Google takes the batch path in the reader thread above, so no consumer is needed.
        if (!google) {
            voskThread = Thread {
                while (true) {
                    val chunk = audioQueue.take()
                    if (chunk === POISON) break
                    if (!isListening) { audioQueue.clear(); break }
                    vosk.acceptChunk(chunk, onPartial) { text ->
                        if (isListening) onResult(text)
                    }
                }
                if (wasStopped) vosk.finalize(onResult, onError)
                else vosk.finalize({ }, { })
            }.also { it.start() }
        }
    }

    fun stopListening() {
        val elapsed = System.currentTimeMillis() - listenStartMs
        if (elapsed < 1000) { Log.d(TAG, "stopListening ignored — too soon"); return }
        Log.d(TAG, "stopListening after ${elapsed}ms")
        wasStopped = true
        isListening = false
        onStopped()
    }

    fun destroy() {
        isListening = false
        releaseAudioFocus()
        audioRecord?.release()
        audioRecord = null
        vosk.destroy()
        googleStt?.destroy()
    }

    private fun requestAudioFocus() {
        val result = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val req = android.media.AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
                .setAudioAttributes(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAcceptsDelayedFocusGain(false)
                .setOnAudioFocusChangeListener { }
                .build()
            audioFocusRequest = req
            audioManager.requestAudioFocus(req)
        } else {
            @Suppress("DEPRECATION")
            audioManager.requestAudioFocus(null, AudioManager.STREAM_VOICE_CALL, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
        }
        Log.i(TAG, "audioFocus result=$result")
    }

    private fun releaseAudioFocus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            audioFocusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
            audioFocusRequest = null
        } else {
            @Suppress("DEPRECATION")
            audioManager.abandonAudioFocus(null)
        }
    }

    private fun saveWav(file: java.io.File, pcm: ByteArray) {
        val byteRate = SAMPLE_RATE * 2
        java.io.FileOutputStream(file).use { fos ->
            fun writeInt(v: Int) = fos.write(byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte()))
            fun writeShort(v: Int) = fos.write(byteArrayOf(v.toByte(), (v shr 8).toByte()))
            fos.write("RIFF".toByteArray()); writeInt(36 + pcm.size)
            fos.write("WAVE".toByteArray())
            fos.write("fmt ".toByteArray()); writeInt(16); writeShort(1); writeShort(1)
            writeInt(SAMPLE_RATE); writeInt(byteRate); writeShort(2); writeShort(16)
            fos.write("data".toByteArray()); writeInt(pcm.size)
            fos.write(pcm)
        }
    }

    private fun rms(shorts: ShortArray, count: Int): Double {
        var sum = 0.0
        for (i in 0 until count) sum += shorts[i].toLong() * shorts[i]
        return Math.sqrt(sum / count)
    }

    private fun shortsToBytes(shorts: ShortArray, count: Int): ByteArray {
        val bytes = ByteArray(count * 2)
        for (i in 0 until count) {
            bytes[i * 2]     = (shorts[i].toInt() and 0xFF).toByte()
            bytes[i * 2 + 1] = (shorts[i].toInt() shr 8 and 0xFF).toByte()
        }
        return bytes
    }
}
