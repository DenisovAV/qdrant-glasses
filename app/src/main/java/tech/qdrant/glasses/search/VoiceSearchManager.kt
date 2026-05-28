package tech.qdrant.glasses.search

import android.content.Context
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.ByteArrayOutputStream

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

    // Switch between backends here — set to false once Google STT key is available
    private val useGoogleStt = false
    private val googleApiKey = ""  // TODO: set key to enable Google STT

    private val vosk = VoskSpeechRecognizer(context)
    private val googleStt: GoogleSpeechRecognizer? =
        if (useGoogleStt && googleApiKey.isNotEmpty()) GoogleSpeechRecognizer(googleApiKey) else null

    private var audioRecord: AudioRecord? = null
    private var readerThread: Thread? = null
    @Volatile private var isListening = false
    private var listenStartMs = 0L
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var audioFocusRequest: android.media.AudioFocusRequest? = null

    fun startListening() {
        if (isListening) return
        if (useGoogleStt && googleStt == null) { onError("Google STT: no API key set"); return }
        if (!useGoogleStt && !vosk.isReady) { onError("VOSK model not ready"); return }

        Log.i(TAG, "startListening [backend=${if (useGoogleStt) "google" else "vosk"}]")

        // Init STT and start recording immediately — before audio focus IPC (~700ms)
        if (useGoogleStt) googleStt!!.startListening(onPartial, onResult, onError)
        else vosk.startListening(onPartial, onResult, onError)

        isListening = true
        listenStartMs = System.currentTimeMillis()

        val chunkSamples = SAMPLE_RATE * CHUNK_MS / 1000
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, chunkSamples * 2 * 32)
        )
        audioRecord = rec
        rec.startRecording()
        Log.i(TAG, "recording started")
        onReady()

        // Request audio focus after recording started — async so system can't silence us mid-phrase
        Thread { requestAudioFocus() }.start()

        if (useGoogleStt) {
            startRecordingThread(rec, chunkSamples, onChunk = null, onStop = { pcm ->
                Log.i(TAG, "google stt: sending ${pcm.size} bytes")
                googleStt?.recognize(pcm)
            })
        } else {
            startRecordingThread(rec, chunkSamples, onChunk = { bytes ->
                vosk.acceptChunk(bytes, onPartial) { text ->
                    isListening = false
                    onResult(text)
                }
            }, onStop = { _ ->
                vosk.finalize(onResult, onError)
            })
        }
    }

    private fun startRecordingThread(
        rec: AudioRecord,
        chunkSamples: Int,
        onChunk: ((ByteArray) -> Unit)?,
        onStop: (ByteArray) -> Unit
    ) {
        readerThread = Thread {
            val buf = ShortArray(chunkSamples)
            val pcmBuffer = if (useGoogleStt) ByteArrayOutputStream() else null
            val diagBuffer = ByteArrayOutputStream()  // record everything for diagnostics
            var hadSpeech = false
            var lastSpeechMs = System.currentTimeMillis()
            var logCount = 0

            while (isListening) {
                val read = rec.read(buf, 0, buf.size)
                if (read > 0) {
                    val rms = rms(buf, read)
                    if (logCount++ < 50) Log.v(TAG, "rms=${"%.0f".format(rms)}")
                    val isSpeech = rms > SILENCE_RMS_THRESHOLD
                    val bytes = shortsToBytes(buf, read)

                    // For VOSK: send all chunks (VOSK has its own VAD), use RMS only for auto-stop timing
                    // For Google STT: only buffer speech chunks to save bandwidth
                    if (isSpeech) {
                        hadSpeech = true
                        lastSpeechMs = System.currentTimeMillis()
                        pcmBuffer?.write(bytes)
                        onChunk?.invoke(bytes)
                    } else if (hadSpeech) {
                        pcmBuffer?.write(bytes)
                        onChunk?.invoke(bytes)
                        val silenceMs = System.currentTimeMillis() - lastSpeechMs
                        if (silenceMs > SPEECH_TIMEOUT_MS) {
                            Log.d(TAG, "auto-stop: ${silenceMs}ms silence after speech")
                            isListening = false
                            onStopped()
                        }
                    } else {
                        // Pre-speech silence: send to VOSK anyway (it handles silence itself)
                        onChunk?.invoke(bytes)
                    }
                }
            }

            rec.stop()
            rec.release()
            releaseAudioFocus()
            onStop(pcmBuffer?.toByteArray() ?: ByteArray(0))
        }.also { it.priority = Thread.MAX_PRIORITY; it.start() }
    }

    fun stopListening() {
        val elapsed = System.currentTimeMillis() - listenStartMs
        if (elapsed < 2000) { Log.d(TAG, "stopListening ignored — too soon"); return }
        Log.d(TAG, "stopListening after ${elapsed}ms")
        isListening = false
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
