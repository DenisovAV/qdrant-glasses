package tech.qdrant.glasses.embedding

import android.graphics.BitmapFactory
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Diagnose the SIGLIP_NPU startup OOM/reboot: measure how much RESIDENT memory each SigLIP encoder
 * actually costs, loaded together in ONE process but WITHOUT the camera/store pipeline. If this
 * survives, the two encoders' steady-state fits and the reboot needs the camera on top (→ staggering
 * the loads could work). If this itself reboots, the two encoders alone exceed budget (→ mmap/shrink
 * is required regardless). VmRSS (from /proc/self/status) includes native ORT allocations.
 * Read: `adb logcat -d -s SiglipMem:I`.
 */
@RunWith(AndroidJUnit4::class)
class SiglipMemoryTest {

    private fun vmRssMb(): Long {
        val m = Regex("VmRSS:\\s+(\\d+) kB").find(File("/proc/self/status").readText())
        return (m?.groupValues?.get(1)?.toLong() ?: -1L) / 1024
    }

    private fun log(tag: String) {
        Log.i("SiglipMem", "%-22s VmRSS=%4dMB nativeHeap=%4dMB".format(
            tag, vmRssMb(), Debug.getNativeHeapAllocatedSize() / 1024 / 1024))
    }

    @Test fun measureEncoderFootprint() {
        val targetCtx = InstrumentationRegistry.getInstrumentation().targetContext
        val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
        val bmp = testAssets.open("ab/o0001.jpg").use { BitmapFactory.decodeStream(it) }

        log("baseline")
        val vision = SiglipVisionEncoder(targetCtx)
        log("vision ctor")
        vision.encode(bmp)
        log("vision +encode")
        val text = SiglipTextEncoder(targetCtx)
        log("text ctor")
        text.encode("a potted plant")
        log("text +encode")
        log("BOTH resident")
        vision.close(); text.close()
        log("after close")
    }
}
