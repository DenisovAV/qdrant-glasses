package tech.qdrant.glasses.storage

import android.content.Context

/**
 * Selects the on-device vector engine — the single build-time switch behind [VectorStore], exactly
 * mirroring [tech.qdrant.glasses.embedding.CropEncoderFactory] (MAC/on-device).
 *
 * THIS is the `benchmark`-flavor copy (`src/benchmark/.../VectorStoreFactory.kt`) — the `demo`
 * flavor has its OWN copy at `src/demo/.../VectorStoreFactory.kt` that wires only [Backend.QDRANT_EDGE]
 * (flavor source sets REPLACE, not merge, a file at the same path; this class lives in neither
 * `src/main` nor both flavors at once, or it would collide). [Backend.OBJECTBOX], [Backend.SQLITE_VEC]
 * and [Backend.CHROMA] only compile in the benchmark flavor — that's where their engines
 * ([ObjectBoxStore], [SqliteVecStore], [ChromaStore]) and native libs live — so only THIS copy can
 * reference them. The demo flavor pulls none of it: no ObjectBox/sqlite-vec/Chroma deps, no
 * `libvec.so`/`libchroma_jni.so`, no extra APK size.
 *
 * [Backend.QDRANT_EDGE] is the product default + baseline. [Backend.OBJECTBOX], [Backend.SQLITE_VEC]
 * and [Backend.CHROMA] are also fully wired here — the benchmark's HNSW, brute-force-in-SQLite and
 * embedded-Rust comparison points ([ComparisonBenchmarkTest] builds several side by side). Only
 * [Backend.USEARCH] and [Backend.FAISS] remain `TODO`, until their phase lands. Flipping [backend] is
 * the ONLY change needed to build+measure a different engine via [VectorStoreBenchmark] — never a
 * runtime switch.
 */
object VectorStoreFactory {
    enum class Backend { QDRANT_EDGE, OBJECTBOX, SQLITE_VEC, CHROMA, USEARCH, FAISS }

    /** ← the single build-time switch (like [tech.qdrant.glasses.embedding.CropEncoderFactory.backend]). */
    val backend = Backend.QDRANT_EDGE

    fun create(context: Context, dim: Int, namespace: String): VectorStore = when (backend) {
        Backend.QDRANT_EDGE -> QdrantEdgeStore(context, dim, namespace)
        Backend.OBJECTBOX -> ObjectBoxStore(context, dim, namespace)
        Backend.SQLITE_VEC -> SqliteVecStore(context, dim, namespace)
        Backend.CHROMA -> ChromaStore(context, dim, namespace)
        Backend.USEARCH -> TODO("phase 4 — USearch adapter")
        Backend.FAISS -> TODO("phase 5 — FAISS adapter")
    }
}
