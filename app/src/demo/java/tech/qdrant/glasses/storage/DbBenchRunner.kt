package tech.qdrant.glasses.storage

import android.app.Application

/**
 * Flavor seam for [GlassesComponents]' optional in-app vector-DB benchmark trigger — see the
 * `benchmark`-flavor copy of this file for what it actually does. [VectorStoreBenchmark] (which
 * exercises [Backend.OBJECTBOX]/[Backend.SQLITE_VEC] alongside the default) is `benchmark`-flavor
 * only, so [GlassesComponents] (a `src/main` file compiled into BOTH flavors) cannot reference it
 * directly — it calls THIS object instead, and each flavor supplies its own behavior (flavor source
 * sets replace, not merge, a file at the same path).
 *
 * THIS (demo) copy is a no-op: the shipping build never runs the benchmark.
 */
object DbBenchRunner {
    fun runIfEnabled(app: Application) {
        // No-op in the demo flavor — VectorStoreBenchmark and the alt engines it compares
        // (ObjectBox, sqlite-vec) are not compiled into this flavor at all.
    }
}
