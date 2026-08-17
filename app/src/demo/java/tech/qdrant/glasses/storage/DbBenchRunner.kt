package tech.qdrant.glasses.storage

import android.app.Application
import android.util.Log

/**
 * Flavor seam for [GlassesComponents]' optional in-app vector-DB benchmark trigger — see the
 * `benchmark`-flavor copy of this file for what it actually does. [VectorStoreBenchmark] (which
 * exercises [Backend.OBJECTBOX]/[Backend.SQLITE_VEC] alongside the default) is `benchmark`-flavor
 * only, so [GlassesComponents] (a `src/main` file compiled into BOTH flavors) cannot reference it
 * directly — it calls THIS object instead, and each flavor supplies its own behavior (flavor source
 * sets replace, not merge, a file at the same path).
 *
 * THIS (demo) copy never runs the benchmark — [VectorStoreBenchmark] and the alt engines it
 * compares (ObjectBox, sqlite-vec) are not compiled into this flavor at all — but it DOES read the
 * same `debug.qdrant.dbbench` sysprop the benchmark copy does, so an operator who sets it on the
 * wrong (demo) build gets a log line instead of silent nothing.
 */
object DbBenchRunner {
    private const val TAG = "dbbench"

    fun runIfEnabled(app: Application) {
        if (tech.qdrant.glasses.Config.sysprop("qdrant.dbbench") == "1") {
            Log.i(TAG, "debug.qdrant.dbbench=1 but this is the demo flavor — the benchmark is not " +
                "compiled in here; build :app:assembleBenchmarkDebug to run it")
        }
    }
}
