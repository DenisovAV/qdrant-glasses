package tech.qdrant.glasses.storage

import android.content.Context

/**
 * Selects the on-device vector engine — the single build-time switch behind [VectorStore], exactly
 * mirroring [tech.qdrant.glasses.embedding.CropEncoderFactory] (MAC/on-device).
 *
 * THIS is the `demo`-flavor copy (`src/demo/.../VectorStoreFactory.kt`) — the shipping build.
 * It wires ONLY [Backend.QDRANT_EDGE]: [Backend.OBJECTBOX] and [Backend.SQLITE_VEC] (and their
 * `benchmarkImplementation`-only deps + `libvec.so`) live in `src/benchmark`, which this flavor
 * never compiles, so this `create()` cannot reference [ObjectBoxStore]/[SqliteVecStore] — they
 * don't exist in this source set. The `benchmark` flavor has its OWN copy of this file at
 * `src/benchmark/.../VectorStoreFactory.kt` that wires all three (flavor source sets REPLACE, not
 * merge, a file at the same path; this class lives in neither `src/main` nor both flavors at once,
 * or it would collide).
 *
 * [Backend.QDRANT_EDGE] is the product default + baseline, and the ONLY backend this flavor can
 * select — [backend] is fixed, not meant to be flipped here (flip it in the benchmark copy to
 * measure a different engine via `VectorStoreBenchmark`).
 */
object VectorStoreFactory {
    enum class Backend { QDRANT_EDGE, OBJECTBOX, SQLITE_VEC, USEARCH, FAISS }

    /** ← fixed to the shipping default in this (demo) flavor. */
    val backend = Backend.QDRANT_EDGE

    fun create(context: Context, dim: Int, namespace: String): VectorStore = when (backend) {
        Backend.QDRANT_EDGE -> QdrantEdgeStore(context, dim, namespace)
        Backend.OBJECTBOX -> TODO("benchmark-flavor-only backend — not compiled into the demo flavor")
        Backend.SQLITE_VEC -> TODO("benchmark-flavor-only backend — not compiled into the demo flavor")
        Backend.USEARCH -> TODO("phase 4 — USearch adapter")
        Backend.FAISS -> TODO("phase 5 — FAISS adapter")
    }
}
