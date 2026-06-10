package tech.qdrant.glasses.embedding

import android.content.Context
import android.util.Log
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

private const val TAG = "LiteRtEncoder"

/** Which hardware accelerator the LiteRT interpreter runs on. */
enum class Accelerator { CPU, GPU, NPU }

/** Memory-maps a .tflite asset (zero-copy; requires noCompress += "tflite"). */
internal fun mapAsset(context: Context, assetPath: String): MappedByteBuffer {
    val afd = context.assets.openFd(assetPath)
    FileInputStream(afd.fileDescriptor).channel.use { ch ->
        return ch.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }
}

/**
 * A LiteRT Interpreter backed by either the Adreno GPU or the Hexagon NPU (QNN HTP).
 * Fail-fast: if the chosen delegate can't init it THROWS — no silent CPU fallback
 * (switch accelerator or EncoderFactory.backend to ONNX instead). Caller closes this.
 */
internal class LiteRtSession(
    context: Context,
    assetPath: String,
    accelerator: Accelerator,
) : AutoCloseable {
    private val delegate: Delegate?       // null for CPU (XNNPACK, no delegate)
    val interpreter: Interpreter

    init {
        delegate = when (accelerator) {
            Accelerator.CPU -> null
            Accelerator.GPU -> createGpuDelegate(assetPath)
            Accelerator.NPU -> createNpuDelegate(context, assetPath)
        }
        val options = Interpreter.Options().apply {
            if (delegate != null) {
                addDelegate(delegate)
            } else {
                // CPU path: XNNPACK is on by default; use all but one core.
                setNumThreads(maxOf(1, Runtime.getRuntime().availableProcessors() - 1))
                setUseXNNPACK(true)
            }
        }
        interpreter = Interpreter(mapAsset(context, assetPath), options)
        Log.i(TAG, "LiteRT interpreter created for $assetPath on $accelerator")
    }

    private fun createGpuDelegate(assetPath: String): GpuDelegate {
        // CompatibilityList is a conservative whitelist; Adreno 621 reports unsupported
        // even though the delegate works. Try directly, fail-fast if it can't init.
        val compat = CompatibilityList()
        val opts = if (compat.isDelegateSupportedOnThisDevice) compat.bestOptionsForThisDevice
                   else GpuDelegate.Options()
        Log.i(TAG, "GPU whitelist=${compat.isDelegateSupportedOnThisDevice}, attempting GPU for $assetPath")
        return GpuDelegate(opts)
    }

    private fun createNpuDelegate(context: Context, assetPath: String): QnnDelegate {
        // checkCapability is conservative (like GPU's CompatibilityList, which reported
        // false yet the delegate worked). Try creating the HTP delegate directly; the
        // constructor throws if it genuinely can't init, and we fail-fast on that.
        val fp16Ok = QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
        Log.i(TAG, "QNN HTP fp16Capability=$fp16Ok version=${QnnDelegate.getVersion()?.joinToString(".")} — attempting HTP anyway")
        // fp32 .tflite run as fp16 on the Hexagon HTP — no INT8 quantization needed.
        val opts = QnnDelegate.Options().apply {
            setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
            setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
            setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_BURST)
            setHtpUseConvHmx(QnnDelegate.Options.HtpUseConvHmx.HTP_CONV_HMX_ON)
            setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
            setCacheDir(context.cacheDir.absolutePath)        // cache compiled HTP graph
            setModelToken(assetPath.substringAfterLast('/'))
            setLogLevel(QnnDelegate.Options.LogLevel.LOG_LEVEL_WARN)
        }
        return QnnDelegate(opts)
    }

    override fun close() {
        interpreter.close()
        (delegate as? AutoCloseable)?.close()
    }
}
