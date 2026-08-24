package tech.qdrant.glasses.embedding

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

/**
 * SIGLIP_NPU gate calibration + on-device A/B — CLIP-ViT-B/32-on-NPU vs SigLIP2-FULLY-on-device, on
 * the glasses, over the SAME 14 labeled crops the host-side fp32 A/B used (`app/src/androidTest/
 * assets/ab/`). Both arms run on the device: [QnnB32CropEncoder] = CLIP ViT-B/32 W8A16 on the Hexagon
 * NPU (what currently ships); [SiglipCropEncoder] = SigLIP2-base vision W8A16 on the NPU + text int8
 * on the CPU (the adoption target — NO relay). Two calibrations per arm:
 *  - runArm: query→crop retrieval — present-object top-1 cosines vs the absent-object floor. This is
 *    the searchGate (voice-search hit) band ([CropEncoderFactory.searchGate]).
 *  - verifyBands: crop↔its-own-label vs crop↔wrong-labels — the label-verify band, which calibrates
 *    [CropEncoderFactory.verifyGate] (MomentCapture's capture-time label verify).
 * Set each SIGLIP_NPU gate at the measured present/absent gap.
 *
 * Read the result: `adb logcat -d -s AB_ENC:I`. The test asserts only that both on-device arms
 * produced finite vectors — the calibration is the logged tables, not a pass/fail.
 */
@RunWith(AndroidJUnit4::class)
class CropEncoderAbTest {

    private val TAG = "AB_ENC"

    // (crop file in assets/ab, its object label) — identical to the host-side manifest.
    private val images = listOf(
        "o0001" to "plant", "o0002" to "framed_picture", "o0003" to "leather_chair",
        "o0006" to "table_lamp", "o0010" to "table_lamp", "o0009" to "bar_stools",
        "o0078" to "bar_stools", "o0012" to "mirror", "o0023" to "sofa",
        "o0089" to "serving_tray", "o0100" to "wine_bottles", "o0133" to "bathroom_sink",
        "o0155" to "dining_set", "o0166" to "outdoor_furniture",
    )
    // (query, relevant labels) — empty relevant = absent-object distractor.
    private val queries = listOf(
        "a potted plant" to setOf("plant"),
        "a framed picture on the wall" to setOf("framed_picture"),
        "a leather armchair" to setOf("leather_chair"),
        "a table lamp" to setOf("table_lamp"),
        "bar stools at a kitchen counter" to setOf("bar_stools"),
        "a sofa" to setOf("sofa"),
        "a mirror" to setOf("mirror"),
        "a serving tray with bottles" to setOf("serving_tray"),
        "green wine bottles" to setOf("wine_bottles"),
        "a bathroom sink" to setOf("bathroom_sink"),
        "a dining table with chairs" to setOf("dining_set"),
        "outdoor patio furniture" to setOf("outdoor_furniture"),
        "a laptop computer" to emptySet(),
        "a dog" to emptySet(),
    )

    @Test fun calibrateSiglipNpuGates() {
        val targetCtx = InstrumentationRegistry.getInstrumentation().targetContext
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val bitmaps = images.associate { (f, _) ->
            f to testAssets.open("ab/$f.jpg").use { BitmapFactory.decodeStream(it) }
        }

        // CLIP on the NPU (what currently ships) — the comparison arm.
        val clip = QnnB32CropEncoder(targetCtx)
        val clipOk = runArm("CLIP-ViT-B/32 (W8A16 NPU)", clip, bitmaps)
        verifyBands("CLIP-ViT-B/32 (W8A16 NPU)", clip, bitmaps)
        clip.close()
        assertTrue("on-device CLIP arm must produce finite vectors", clipOk)

        // SigLIP2 FULLY ON-DEVICE (vision W8A16 on the Hexagon NPU + text int8 on the CPU) — the
        // adoption target. No relay. This is the arm we calibrate the SIGLIP_NPU gates from.
        val siglip = SiglipCropEncoder(targetCtx)
        val sigOk = runArm("SigLIP2-base (W8A16 NPU + int8 CPU)", siglip, bitmaps)
        verifyBands("SigLIP2-base (W8A16 NPU + int8 CPU)", siglip, bitmaps)
        siglip.close()
        assertTrue("on-device SigLIP arm must produce finite vectors", sigOk)
    }

    /** Embeds all images + queries, L2-normalizes, logs the metric table. Returns false if the
     *  encoder produced a non-finite/empty image vector (a dead arm). */
    private fun runArm(name: String, enc: CropEncoder, bitmaps: Map<String, Bitmap>): Boolean {
        val imgVecs = images.associate { (f, _) -> f to l2(enc.encode(bitmaps.getValue(f))) }
        if (imgVecs.values.any { it.isEmpty() || it.any { x -> !x.isFinite() } }) {
            Log.e(TAG, "$name: produced non-finite image vectors"); return false
        }
        val labelOf = images.toMap()

        var top1Hits = 0; var recall3Hits = 0; var nPresent = 0
        val relMeans = ArrayList<Float>(); val irrMeans = ArrayList<Float>()
        val top1Cos = ArrayList<Float>(); val absentMax = ArrayList<Float>()
        val lines = ArrayList<String>()

        for ((q, relevant) in queries) {
            val qv = l2(enc.encodeText(q))
            val ranked = imgVecs.entries
                .map { (f, v) -> cos(qv, v) to f }
                .sortedByDescending { it.first }
            val (topScore, topFile) = ranked.first()
            if (relevant.isEmpty()) {
                absentMax += topScore
                lines += "  ABSENT  \"$q\" -> ${labelOf[topFile]} (%.3f)".format(topScore)
                continue
            }
            nPresent++
            val hit1 = labelOf[topFile] in relevant
            val hit3 = ranked.take(3).any { labelOf[it.second] in relevant }
            if (hit1) top1Hits++; if (hit3) recall3Hits++
            top1Cos += topScore
            relMeans += ranked.filter { labelOf[it.second] in relevant }.map { it.first }.average().toFloat()
            irrMeans += ranked.filter { labelOf[it.second] !in relevant }.map { it.first }.average().toFloat()
            lines += "  ${if (hit1) "OK " else "X  "}\"$q\" -> ${labelOf[topFile]} (%.3f)".format(topScore)
        }

        val margin = relMeans.average() - irrMeans.average()
        Log.i(TAG, "==== $name  dim=${enc.dim}  gate=${enc.visionMinScore} ====")
        lines.forEach { Log.i(TAG, it) }
        Log.i(TAG, "$name SUMMARY: top1=%.3f recall@3=%.3f margin=%.4f meanRel=%.4f meanIrr=%.4f meanTop1=%.4f absentFloor=%.4f".format(
            top1Hits.toFloat() / nPresent, recall3Hits.toFloat() / nPresent, margin,
            relMeans.average(), irrMeans.average(), top1Cos.average(),
            if (absentMax.isEmpty()) Float.NaN else absentMax.average()))
        return true
    }

    /** verifyGate calibration — the region-label-verify direction (mirrors MomentCapture's
     *  `cache.verify(regionVec, region.label)`): for each crop, cosine to its OWN label text (should
     *  KEEP the label) vs the MAX cosine to any OTHER label (should DROP it). The right verifyGate sits
     *  in the gap [max-wrong .. min-correct]. Labels are terse (underscore→space) to match YOLO tags.
     *  Read with `adb logcat -d -s AB_ENC:I`. */
    private fun verifyBands(name: String, enc: CropEncoder, bitmaps: Map<String, Bitmap>) {
        val labels = images.map { it.second }.distinct()
        val labelVecs = labels.associateWith { l2(enc.encodeText(it.replace('_', ' '))) }
        val imgVecs = images.associate { (f, _) -> f to l2(enc.encode(bitmaps.getValue(f))) }
        val correct = ArrayList<Float>(); val wrongMax = ArrayList<Float>()
        val lines = ArrayList<String>()
        for ((f, lbl) in images) {
            val v = imgVecs.getValue(f)
            val cCos = cos(v, labelVecs.getValue(lbl))
            val wMax = labels.filter { it != lbl }.maxOf { cos(v, labelVecs.getValue(it)) }
            correct += cCos; wrongMax += wMax
            lines += "  verify %-16s[%s] correct=%.3f wrongMax=%.3f".format(f, lbl, cCos, wMax)
        }
        Log.i(TAG, "==== $name VERIFY-BANDS (calibrates CropEncoderFactory.verifyGate) ====")
        lines.forEach { Log.i(TAG, it) }
        Log.i(TAG, ("$name VERIFY-SUMMARY: correct[min=%.3f mean=%.3f] wrongMax[max=%.3f mean=%.3f]" +
            " -> gate in gap [%.3f .. %.3f], midpoint~%.3f").format(
            correct.min(), correct.average(), wrongMax.max(), wrongMax.average(),
            wrongMax.max(), correct.min(), (wrongMax.max() + correct.min()) / 2f))
    }

    /**
     * Scene-dedup calibration (MomentCapture.CONFIRM_COSINE) for SigLIP. Encodes the REAL stored
     * moment keyframes (`assets/scenes/`, pulled from the device — each is a DIFFERENT scene, they
     * passed the 0.85 dedup) with on-device SigLIP whole-frame vision, then:
     *  - DIFFERENT-scene band = pairwise cosine across the distinct keyframes (should be LOW).
     *  - SAME-scene band = each keyframe vs a mild perturbation (crop/brightness) of itself (HIGH).
     * CONFIRM_COSINE belongs between different-max and the real same-scene floor (~0.85 from live
     * logs). If different-max reaches ~0.85, SigLIP whole-frames don't separate scenes → lower it.
     * Read: `adb logcat -d -s AB_ENC:I`.
     */
    @Test fun calibrateSceneDedup() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val a = InstrumentationRegistry.getInstrumentation().context.assets
        val scenes = (a.list("scenes") ?: arrayOf()).filter { it.endsWith(".jpg") }.sorted()
        val enc = SiglipCropEncoder(ctx)
        val bmps = scenes.associateWith { a.open("scenes/$it").use { s -> BitmapFactory.decodeStream(s) } }
        val vecs = scenes.map { it to l2(enc.encode(bmps.getValue(it))) }
        Log.i(TAG, "==== SCENE-DEDUP (SigLIP whole-frame, ${scenes.size} distinct stored moments) ====")
        val diff = ArrayList<Float>()
        for (i in vecs.indices) for (j in i + 1 until vecs.size) {
            val c = cos(vecs[i].second, vecs[j].second); diff += c
            Log.i(TAG, "  DIFF %s..%s = %.3f".format(vecs[i].first.drop(7).take(6), vecs[j].first.drop(7).take(6), c))
        }
        val same = ArrayList<Float>()
        for ((s, v0) in vecs) for (kind in listOf("crop", "bright")) {
            val c = cos(v0, l2(enc.encode(perturb(bmps.getValue(s), kind))))
            same += c; Log.i(TAG, "  SAME %s/%s = %.3f".format(s.drop(7).take(6), kind, c))
        }
        enc.close()
        Log.i(TAG, "SCENE-DEDUP SUMMARY: DIFFERENT[min=%.3f mean=%.3f max=%.3f] SAME-perturb[min=%.3f mean=%.3f] realSame~0.85-0.94(logs). CONFIRM_COSINE now=0.85".format(
            diff.min(), diff.average(), diff.max(), same.min(), same.average()))
        assertTrue("scene-dedup produced cosines", diff.isNotEmpty())
    }

    private fun perturb(b: Bitmap, kind: String): Bitmap = if (kind == "crop") {
        val mx = (b.width * 0.06f).toInt(); val my = (b.height * 0.06f).toInt()
        Bitmap.createBitmap(b, mx, my, b.width - 2 * mx, b.height - 2 * my)
    } else {
        val out = b.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(out.width * out.height); out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
        for (i in px.indices) {
            val p = px[i]
            val r = (((p shr 16) and 0xFF) * 1.15f).toInt().coerceAtMost(255)
            val g = (((p shr 8) and 0xFF) * 1.15f).toInt().coerceAtMost(255)
            val bl = ((p and 0xFF) * 1.15f).toInt().coerceAtMost(255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or bl
        }
        out.setPixels(px, 0, out.width, 0, 0, out.width, out.height); out
    }

    private fun l2(v: FloatArray): FloatArray {
        var n = 0.0; for (x in v) n += (x * x).toDouble()
        val inv = if (n > 0) (1.0 / sqrt(n)).toFloat() else 0f
        return FloatArray(v.size) { v[it] * inv }
    }

    private fun cos(a: FloatArray, b: FloatArray): Float {
        var s = 0f; val n = minOf(a.size, b.size); for (i in 0 until n) s += a[i] * b[i]; return s
    }
}
