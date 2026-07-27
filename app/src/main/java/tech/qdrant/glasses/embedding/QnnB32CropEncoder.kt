package tech.qdrant.glasses.embedding

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import tech.qdrant.glasses.Config

/**
 * On-device crop/text embedding on the **Hexagon NPU** — the W8A16 CLIP ViT-B/32 alternative to
 * [OnDeviceCropEncoder] (which runs TinyCLIP-40M on the CPU at ~200ms idle / ~870ms under load).
 *
 * Both towers are the SAME CLIP ViT-B/32 model → one shared 512-dim space, so a crop vector and a
 * text-query vector land near each other (cosine 0.995 image-to-float despite the W8A16 quant):
 *  - crop → [QnnClipVisionEncoder]: 8-bit weights / 16-bit activations, static-quant QDQ-ONNX on
 *    the AR1 HTP via ORT's QNN EP. The AR1 has no fp16 and plain int8 collapses CLIP (cosine ~0.45)
 *    — W8A16 is the escape hatch that keeps vectors correct. ~28ms/crop isolated on the NPU.
 *  - text → [B32ClipTextEncoder]: the cold, once-per-query side. Times GPU(float) vs CPU(int8) and
 *    serves the CPU-int8 vector.
 *
 * This is a DIFFERENT vector space from the TinyCLIP `ondevice` index → own namespace ("qnnb32"),
 * so it never searches ViT-B/32 vectors against a TinyCLIP-built collection.
 *
 * Must be called OFF the main thread (ORT inference), like every [CropEncoder].
 */
class QnnB32CropEncoder(context: Context) : CropEncoder {
    override val dim: Int = 512
    // ViT-B/32 text→image cosine is NOT yet calibrated on real glasses crops (TinyCLIP's 0.20 gate
    // does not transfer — different model, different modality-gap scale). Placeholder from the
    // W8A16 verification runs; re-measure against real objects before trusting search precision.
    override val visionMinScore: Float = 0.22f

    private val vision = QnnClipVisionEncoder(context)
    private val text = B32ClipTextEncoder(context)

    init {
        // Optional in-app micro-benchmark (no camera / no recording needed):
        //   adb shell setprop debug.qdrant.clipbench 1  → runs at load, writes the latency CSVs.
        // Measures the REAL ORT-QNN-EP latency of the partitioned W8A16 graph on THIS device —
        // the number the native qnn-net-run path (28.6ms, all-on-HTP) can't tell us, since ORT
        // QNN EP rejects LayerNorm/GELU-Div (error 3110) and falls those ops back to the CPU.
        if (Config.sysprop("qdrant.clipbench") == "1") runBenchmark()
    }

    override fun encode(crop: Bitmap): FloatArray = vision.encode(crop)

    override fun encodeText(query: String): FloatArray = text.encode(query)

    override fun close() {
        vision.close()
        text.close()
    }

    private fun runBenchmark() {
        val img = syntheticCrop()
        Log.i(TAG, "clipbench: START (vision ${BENCH_N}× on ${img.width}×${img.height}, text ${BENCH_TEXT_N}×)")
        // Vision: warm up (first run pays the ORT graph-partition + HTP context load), then time.
        repeat(BENCH_WARMUP) { vision.encode(img) }
        val vms = LongArray(BENCH_N)
        for (i in 0 until BENCH_N) {
            val t0 = System.currentTimeMillis()
            vision.encode(img)                       // also appends its own line to clip_npu_latency.csv
            vms[i] = System.currentTimeMillis() - t0
        }
        logStats("vision(NPU W8A16)", vms)
        // Text: exercises BOTH sessions (GPU float + CPU int8); B32ClipTextEncoder logs each.
        repeat(BENCH_WARMUP) { text.encode(BENCH_QUERY) }
        val tms = LongArray(BENCH_TEXT_N)
        for (i in 0 until BENCH_TEXT_N) {
            val t0 = System.currentTimeMillis()
            text.encode(BENCH_QUERY)
            tms[i] = System.currentTimeMillis() - t0
        }
        logStats("text(GPU+CPU total)", tms)
        img.recycle()
        Log.i(TAG, "clipbench: DONE")
    }

    /** A fixed 224×224 crop — content is irrelevant to latency, only shape/dtype are. */
    private fun syntheticCrop(): Bitmap {
        val b = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        for (y in 0 until 224) for (x in 0 until 224) {
            b.setPixel(x, y, Color.rgb((x * 7) and 0xFF, (y * 5) and 0xFF, ((x + y) * 3) and 0xFF))
        }
        return b
    }

    private fun logStats(label: String, ms: LongArray) {
        val sorted = ms.sorted()
        val mean = ms.average()
        val median = sorted[sorted.size / 2]
        val p90 = sorted[(sorted.size * 9) / 10]
        Log.i(TAG, "clipbench: $label n=${ms.size} mean=%.1fms median=${median}ms p90=${p90}ms min=${sorted.first()}ms max=${sorted.last()}ms".format(mean))
    }

    companion object {
        private const val TAG = "ClipEncoder"
        private const val BENCH_WARMUP = 3
        private const val BENCH_N = 20
        private const val BENCH_TEXT_N = 5
        private const val BENCH_QUERY = "a photo of a coffee cup"
    }
}
