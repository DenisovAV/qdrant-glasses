package tech.qdrant.glasses.storage

import android.app.Application

/**
 * Flavor seam for [GlassesComponents]' optional in-app vector-DB benchmark trigger. [GlassesComponents]
 * (a `src/main` file compiled into BOTH flavors) cannot reference [VectorStoreBenchmark] directly —
 * it is `benchmark`-flavor only (it exercises [Backend.OBJECTBOX]/[Backend.SQLITE_VEC] alongside the
 * default) — so it calls THIS object instead, and each flavor supplies its own behavior (flavor
 * source sets replace, not merge, a file at the same path; see the `demo` copy for the no-op side).
 *
 *   adb shell setprop debug.qdrant.dbbench 1   → runs the full matrix at next launch.
 *
 * Uses a DEDICATED bench namespace + wipes itself, so it never touches demo memory.
 */
object DbBenchRunner {
    fun runIfEnabled(app: Application) {
        if (tech.qdrant.glasses.Config.sysprop("qdrant.dbbench") == "1") {
            VectorStoreBenchmark(app).launch()
        }
    }
}
