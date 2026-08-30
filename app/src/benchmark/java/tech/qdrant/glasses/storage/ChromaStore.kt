package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import com.trychroma.android.Collection
import com.trychroma.android.DocumentResult
import com.trychroma.android.PersistentClient
import com.trychroma.android.Query
import java.io.File
import java.util.UUID

/**
 * ChromaDB implementation of [VectorStore] (the benchmark's 4th engine) — an embedded Rust/JNI
 * vector DB reached through the official `com.trychroma.android` prebuilt AAR (beta v0.0.1, no
 * Maven artifact yet; see NOTICE). Its index is HNSW/SPANN → approximate ANN, same family as
 * [ObjectBoxStore] and Qdrant-HNSW mode, so recall@k here is expected to be < 1.0. Only
 * instantiated when [VectorStoreFactory.backend] == CHROMA (a benchmark build).
 *
 * Beta-API caveats that shape this class (all confirmed against the decompiled AAR + upstream
 * source, not assumed):
 *  - **NO index configuration at all — this is the beta AAR's hard ceiling, verified in its source
 *    (`chroma-core/chroma-android` @ v0.0.1), not assumed.** `clientCreateCollection(ptr, name)`
 *    takes only a name; `clientUpdateCollection(ptr, name, newName)` is a RENAME (not a config
 *    setter); `clientQueryCollection(...)` has no query-time `ef`; and `rust/src/chromadb.rs`'s
 *    `create_collection` hardcodes `hnsw: None`, so every collection runs on Chroma's DEFAULT HNSW
 *    (documented `hnsw:search_ef = 10`, `space = l2`, `M = 16`, `ef_construction = 100`). Two
 *    consequences pull in OPPOSITE directions, so keep them separate:
 *      · **Distance space does NOT hurt recall.** Every benchmark vector is unit-normalized (see
 *        `VectorStoreBenchmark.randomUnitVector`); for unit vectors squared-L2, cosine-distance and
 *        IP-distance are all strictly decreasing in cosine similarity → identical top-k SET. See
 *        [scoreOf] for the (metric-agnostic) score this store reports.
 *      · **The fixed `search_ef = 10` DOES crater recall at scale, and there is no knob to raise
 *        it.** With a fixed tiny candidate list, recall@k falls monotonically as N grows
 *        (measured on-device: recall@5 ≈ 1.0 @1k → 0.46 @10k → 0.04 @100k on random near-orthogonal
 *        512-d vectors — an adversarial worst case for ANY HNSW; real clustered embeddings score
 *        higher). This is a property of the shipped SDK (server Chroma exposes `ef_search`; this
 *        Android beta does not), NOT of this class's usage — do not "fix" it here, there is nothing
 *        to configure. It IS a fairness caveat for the comparison report.
 *  - **`createCollection` throws if the name already exists** — the ONLY way to reopen a
 *    persisted collection (e.g. the benchmark's cold-load/reopen step) is `getCollection`, so
 *    [openOrCreateCollection] tries create-then-get.
 *  - **No list-all / scroll primitive** (unlike the Python client's `collection.get()`) — only
 *    `query()` (a kNN search) and `count()` are exposed. [all] is a documented best-effort
 *    approximation, not oldest-first as the interface asks (see its KDoc) — nothing in
 *    [VectorStoreBenchmark] calls it; only the demo's HUD rail would, and CHROMA is never the demo
 *    backend.
 *  - **Metadata filter pushdown exists** (`where` JSON, same `$gte`/`$lte` operators as the
 *    Python client) — [searchFiltered] uses it, storing `timestamp_ms` as per-doc metadata
 *    alongside the payload JSON (stored verbatim in Chroma's `documents` text field, the same
 *    "one opaque payload column" shape [SqliteVecStore] and [ObjectBoxStore] use).
 *  - Every public [Client]/[Collection] call is a blocking FFI round-trip into Rust
 *    (`.block_on(...)` on the Rust side) — synchronous from Kotlin's perspective, but this store
 *    still serializes through [lock] (matching [QdrantEdgeStore]'s caution): [deleteAll]
 *    reassigns [collection] (a `var`), and the native client's thread-safety is unverified.
 */
class ChromaStore(context: Context, dim: Int, namespace: String) : VectorStore {

    companion object {
        private const val TAG = "ChromaStore"
        private const val COLLECTION = "objects"
    }

    override val name: String = "chroma"

    private val vectorDim = dim
    private val lock = Any()
    private val dir: File = File(context.filesDir, "chroma_$namespace").also { it.mkdirs() }

    private var client: PersistentClient = PersistentClient(dir.absolutePath)
    private var collection: Collection = openOrCreateCollection()

    init {
        Log.i(TAG, "chroma opened (${dir.absolutePath}), collection=$COLLECTION count=${collection.count()}")
    }

    /** create-then-get: [Collection] creation throws if the name is already persisted on disk
     *  (a REOPEN of an existing store, e.g. the benchmark's cold-load step) — there is no
     *  "create if absent" call in this beta API. The catch is deliberately broad because the beta
     *  API doesn't type an "already exists" exception, but a create failure is NOT assumed benign:
     *  it's logged, and if the get ALSO fails the ORIGINAL create error is chained in — so a real
     *  JNI/storage fault surfaces instead of being masked as a routine reopen. */
    private fun openOrCreateCollection(): Collection = try {
        client.createCollection(COLLECTION)
    } catch (createEx: Throwable) {
        Log.w(TAG, "createCollection($COLLECTION) failed; assuming reopen, trying getCollection", createEx)
        try {
            client.getCollection(COLLECTION)
        } catch (getEx: Throwable) {
            getEx.addSuppressed(createEx)
            throw IllegalStateException("Chroma collection $COLLECTION could not be created or opened", getEx)
        }
    }

    override fun upsert(vector: FloatArray, payload: ObjectPayload): String =
        upsertBatch(listOf(vector to payload)).first()

    override fun upsertBatch(items: List<Pair<FloatArray, ObjectPayload>>): List<String> = synchronized(lock) {
        val ids = Array(items.size) { UUID.randomUUID().toString() }
        val vectors = Array(items.size) { i ->
            val v = items[i].first
            require(v.size == vectorDim) { "dim ${v.size} != $vectorDim" }
            v
        }
        val documents = Array(items.size) { items[it].second.toJson() }
        // timestamp_ms as per-doc metadata (a raw JSON number, not a string) so searchFiltered can
        // push a $gte/$lte range down into the native query instead of over-fetching + filtering
        // in Kotlin.
        val metadatas = Array(items.size) { "{\"timestamp_ms\":${items[it].second.timestampMs}}" }
        val added = collection.add(ids, vectors, documents, null, metadatas)
        check(added == items.size) { "chroma add() returned $added, expected ${items.size}" }
        ids.toList()
    }

    override fun search(vector: FloatArray, topK: Int): List<ObjectHit> = synchronized(lock) {
        require(vector.size == vectorDim) { "dim ${vector.size} != $vectorDim" }
        val results = collection.query(
            Query.Builder()
                .queryEmbeddings(arrayOf(vector))
                .nResults(topK)
                .include(arrayOf("documents", "metadatas", "distances"))
                .build()
        )
        val hits = results.map { toHit(it) }
        Log.i(TAG, "search: topK=$topK returned=${hits.size} " +
            hits.take(3).joinToString { "%.3f \"%s\"".format(it.score, it.label.take(20)) })
        hits
    }

    override fun searchFiltered(
        vector: FloatArray,
        topK: Int,
        sinceMs: Long?,
        untilMs: Long?,
    ): List<ObjectHit> = synchronized(lock) {
        if (sinceMs == null && untilMs == null) return search(vector, topK)
        require(vector.size == vectorDim) { "dim ${vector.size} != $vectorDim" }
        val bounds = buildList {
            if (sinceMs != null) add("\"\$gte\":$sinceMs")
            if (untilMs != null) add("\"\$lte\":$untilMs")
        }.joinToString(",")
        val where = "{\"timestamp_ms\":{$bounds}}"
        val results = collection.query(
            Query.Builder()
                .queryEmbeddings(arrayOf(vector))
                .nResults(topK)
                .where(where)
                .include(arrayOf("documents", "metadatas", "distances"))
                .build()
        )
        val hits = results.map { toHit(it) }
        Log.i(TAG, "searchFiltered: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        hits
    }

    /**
     * Best-effort ONLY: the beta API has no scroll/list-all primitive, so this approximates it with
     * a kNN [Query] against a fixed unit probe vector (`e_0`) and returns whatever order comes
     * back — NOT oldest-first like every other [VectorStore]'s [all], and capped by whatever the
     * ANN index actually surfaces near that probe, not a true full scan. Documented deviation, not
     * a bug: [VectorStoreBenchmark] never calls [all] (only the demo's HUD rail would, and CHROMA
     * is never the demo backend).
     */
    override fun all(limit: Int): List<ObjectHit> = synchronized(lock) {
        val probe = FloatArray(vectorDim).also { it[0] = 1f }
        val results = collection.query(
            Query.Builder()
                .queryEmbeddings(arrayOf(probe))
                .nResults(limit)
                .include(arrayOf("documents", "metadatas", "distances"))
                .build()
        )
        results.map { toHit(it) }.also { Log.i(TAG, "all(): ${it.size} stored objects (approximate, not oldest-first)") }
    }

    override fun count(): Long = synchronized(lock) { collection.count().toLong() }

    override fun deleteAll(): Unit = synchronized(lock) {
        val before = runCatching { collection.count().toLong() }.getOrDefault(-1L)
        check(client.reset()) { "chroma reset() returned false" }
        collection = client.createCollection(COLLECTION)
        Log.i(TAG, "deleteAll: dropped $before points")
    }

    override fun close() = synchronized(lock) { client.close() }

    // ---- helpers ----

    private fun toHit(d: DocumentResult): ObjectHit {
        val p = ObjectPayload.fromJson(d.data ?: "{}")
        return ObjectHit(
            id = d.id,
            score = scoreOf(d.distance),
            label = p.label, bbox = p.bbox, timestampMs = p.timestampMs, thumbPath = p.thumbPath,
        )
    }

    /**
     * See the class doc's "no distance-metric control" note. On-device (glasses, 512-dim random
     * unit vectors) the observed [DocumentResult.distance] for a topK=5 search over near-orthogonal
     * fillers clusters around 1.6-1.8 — consistent with squared L2 (2 − 2·cos, ≈2 for cos≈0) and NOT
     * with a cosine-distance space (1 − cos, which would cluster near 1.0). So this converts
     * assuming squared L2: cos = 1 − d/2. Still just a proxy, not a guarantee — the beta API can't
     * confirm the space, and a future AAR version may pick a different default — but it now matches
     * the other engines' cosine-similarity SCALE, not just their rank order.
     */
    private fun scoreOf(distance: Float): Float = 1f - distance / 2f
}
