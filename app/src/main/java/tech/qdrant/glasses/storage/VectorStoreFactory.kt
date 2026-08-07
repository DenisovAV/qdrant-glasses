package tech.qdrant.glasses.storage

import android.content.Context

/**
 * Selects the on-device vector engine — the single build-time switch behind [VectorStore], exactly
 * mirroring [tech.qdrant.glasses.embedding.CropEncoderFactory] (MAC/on-device). Exactly ONE engine
 * is compiled into any given build, so the benchmark ([VectorStoreBenchmark]) is honest by
 * construction (only the active engine's native libs and symbols live in-process) and the shipping
 * demo build stays on the default with no extra dependencies.
 *
 * Only [Backend.QDRANT_EDGE] is wired today (the product default + baseline); the alternatives are
 * the phased benchmark integrations (design §4) and each `TODO`s until its phase lands. Flipping
 * [backend] is the ONLY change needed to build+measure a different engine — never a runtime switch.
 */
object VectorStoreFactory {
    enum class Backend { QDRANT_EDGE, OBJECTBOX, SQLITE_VEC, USEARCH, FAISS }

    /** ← the single build-time switch (like [tech.qdrant.glasses.embedding.CropEncoderFactory.backend]). */
    val backend = Backend.QDRANT_EDGE

    fun create(context: Context, dim: Int, namespace: String): VectorStore = when (backend) {
        Backend.QDRANT_EDGE -> QdrantEdgeStore(context, dim, namespace)
        Backend.OBJECTBOX -> ObjectBoxStore(context, dim, namespace)
        Backend.SQLITE_VEC -> SqliteVecStore(context, dim, namespace)
        Backend.USEARCH -> TODO("phase 4 — USearch adapter")
        Backend.FAISS -> TODO("phase 5 — FAISS adapter")
    }
}
