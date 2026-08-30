package tech.qdrant.glasses.storage

import android.content.Context
import android.util.Log
import androidx.sqlite.SQLiteConnection
import androidx.sqlite.SQLiteStatement
import androidx.sqlite.driver.bundled.BundledSQLiteDriver
import androidx.sqlite.execSQL
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * sqlite-vec implementation of [VectorStore] (the benchmark's Phase 3 engine) — brute-force f32
 * cosine via the `vec0` virtual table (no binary/quantized path in THIS class — that's a separate
 * capability of Qdrant's), the same index family Qdrant Edge wins with at edge scale, but inside
 * SQLite.
 *
 * The `vec0` virtual table does the KNN; we reach it through **androidx.sqlite**'s
 * [BundledSQLiteDriver] (it ships its own SQLite with extension-loading enabled) plus
 * [BundledSQLiteDriver.addExtension] pointed at the loadable extension shipped as
 * `jniLibs/arm64-v8a/libvec.so` (renamed from the release's `vec0.so` so Android extracts it and
 * SQLite derives the entry point `sqlite3_vec_init` from the lib name). Cosine is native
 * (`distance_metric=cosine`); vec0 returns cosine *distance* (0 = identical), flipped to a
 * higher-is-better similarity so [ObjectHit.score] matches the other engines.
 *
 * Only instantiated when [VectorStoreFactory.backend] == SQLITE_VEC (a benchmark build).
 */
class SqliteVecStore(
    context: Context,
    dim: Int,
    namespace: String,
    // Binary mode: store 1-bit codes (packed in Kotlin, bit = value>0) + the f32 vector, retrieve an
    // oversampled candidate set by Hamming brute-force, then rescore those candidates by exact cosine.
    // sqlite-vec has the primitives (vec_distance_hamming/vec_bit/vec_distance_cosine) but no managed
    // quantize+rescore pipeline like Qdrant's — so we assemble it in SQL here. This is honest binary
    // brute-force; the f32 originals are kept (like Qdrant's rescore path), so disk is NOT reduced.
    private val binary: Boolean = false,
) : VectorStore {

    companion object {
        private const val TAG = "SqliteVecStore"
        private const val TABLE = "vec_items"
        private const val BINARY_OVERSAMPLE = 32L   // MUST equal QdrantEdgeStore.BINARY_OVERSAMPLE for a like-for-like binary comparison (same candidate count reranked)
    }

    override val name: String = if (binary) "sqlite-binary" else "sqlite-vec"

    private val vectorDim = dim
    private val lock = Any()
    private val conn: SQLiteConnection

    init {
        val soPath = File(context.applicationInfo.nativeLibraryDir, "libvec.so").absolutePath
        val dbPath = File(context.filesDir, "sqlitevec_$namespace.db").absolutePath
        val driver = BundledSQLiteDriver().apply { addExtension(soPath) }
        conn = driver.open(dbPath)
        if (binary) {
            // Plain table: 1-bit codes (`bits`) + f32 (`embedding`), both as blobs. sqlite-vec v0.1.9's
            // vec0 KNN `k=?` only works for a float column at the top level — it does NOT accept a bit
            // column KNN composed with a cosine rescore (errors "no such column: k"). So binary here is a
            // per-row Hamming scan (`ORDER BY vec_distance_hamming`) over the oversampled set + rescore.
            // Correct (recall 1.0) but slower than the native f32 vec0 KNN — the cost is SQL per-row
            // function-call overhead, NOT Hamming being slow; a fair fast-binary needs native bit-KNN,
            // which this vec0 version can't compose with rescore.
            conn.execSQL(
                "CREATE TABLE IF NOT EXISTS $TABLE (" +
                    "bits blob not null, embedding blob not null, timestamp_ms integer, payload text)"
            )
        } else {
            // vec0: `embedding` is the indexed vector (cosine); `timestamp_ms` a filterable metadata
            // column (for searchFiltered); `+payload` an auxiliary (stored, not indexed) column.
            conn.execSQL(
                "CREATE VIRTUAL TABLE IF NOT EXISTS $TABLE USING vec0(" +
                    "embedding float[$vectorDim] distance_metric=cosine, " +
                    "timestamp_ms integer, " +
                    "+payload text)"
            )
        }
        Log.i(TAG, "sqlite-vec opened ($dbPath, binary=$binary), vec_version=${vecVersion()}, count=${count()}")
    }

    private fun vecVersion(): String = conn.prepare("SELECT vec_version()").use { st ->
        if (st.step()) st.getText(0) else "?"
    }

    override fun upsert(vector: FloatArray, payload: ObjectPayload): String = synchronized(lock) {
        insertOne(vector, payload).toString()
    }

    override fun upsertBatch(items: List<Pair<FloatArray, ObjectPayload>>): List<String> = synchronized(lock) {
        val ids = ArrayList<String>(items.size)
        conn.execSQL("BEGIN")
        try {
            for ((vector, payload) in items) ids.add(insertOne(vector, payload).toString())
            conn.execSQL("COMMIT")
        } catch (t: Throwable) {
            conn.execSQL("ROLLBACK"); throw t
        }
        ids
    }

    private fun insertOne(vector: FloatArray, payload: ObjectPayload): Long {
        require(vector.size == vectorDim) { "dim ${vector.size} != $vectorDim" }
        if (binary) {
            conn.prepare("INSERT INTO $TABLE(bits, embedding, timestamp_ms, payload) VALUES (?, ?, ?, ?)").use { st ->
                st.bindBlob(1, packBits(vector))
                st.bindBlob(2, blob(vector))
                st.bindLong(3, payload.timestampMs)
                st.bindText(4, payload.toJson())
                st.step()
            }
        } else {
            conn.prepare("INSERT INTO $TABLE(embedding, timestamp_ms, payload) VALUES (?, ?, ?)").use { st ->
                st.bindBlob(1, blob(vector))
                st.bindLong(2, payload.timestampMs)
                st.bindText(3, payload.toJson())
                st.step()
            }
        }
        return conn.prepare("SELECT last_insert_rowid()").use { st -> st.step(); st.getLong(0) }
    }

    override fun search(vector: FloatArray, topK: Int): List<ObjectHit> = synchronized(lock) {
        val hits = if (binary) {
            // Per-row Hamming scan → oversampled candidates → exact-cosine rescore → top-k.
            val sql = "SELECT rowid, vec_distance_cosine(vec_f32(embedding), vec_f32(?)) AS distance, payload FROM (" +
                "SELECT rowid, embedding, payload FROM $TABLE " +
                "ORDER BY vec_distance_hamming(vec_bit(bits), vec_bit(?)) LIMIT ?" +
                ") ORDER BY distance LIMIT ?"
            conn.prepare(sql).use { st ->
                st.bindBlob(1, blob(vector)); st.bindBlob(2, packBits(vector))
                st.bindLong(3, topK * BINARY_OVERSAMPLE); st.bindLong(4, topK.toLong()); readHits(st)
            }
        } else {
            val sql = "SELECT rowid, distance, payload FROM $TABLE WHERE embedding MATCH ? AND k = ? ORDER BY distance"
            conn.prepare(sql).use { st ->
                st.bindBlob(1, blob(vector)); st.bindLong(2, topK.toLong()); readHits(st)
            }
        }
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
        val from = sinceMs ?: Long.MIN_VALUE
        val to = untilMs ?: Long.MAX_VALUE
        val hits = if (binary) {
            // Filter the timestamp window FIRST, then Hamming-oversample within it, then cosine rescore.
            val sql = "SELECT rowid, vec_distance_cosine(vec_f32(embedding), vec_f32(?)) AS distance, payload FROM (" +
                "SELECT rowid, embedding, payload FROM $TABLE WHERE timestamp_ms BETWEEN ? AND ? " +
                "ORDER BY vec_distance_hamming(vec_bit(bits), vec_bit(?)) LIMIT ?" +
                ") ORDER BY distance LIMIT ?"
            conn.prepare(sql).use { st ->
                st.bindBlob(1, blob(vector)); st.bindLong(2, from); st.bindLong(3, to)
                st.bindBlob(4, packBits(vector)); st.bindLong(5, topK * BINARY_OVERSAMPLE); st.bindLong(6, topK.toLong())
                readHits(st)
            }
        } else {
            // vec0 applies the metadata predicate during the KNN scan (filter-during-scan).
            val sql = "SELECT rowid, distance, payload FROM $TABLE " +
                "WHERE embedding MATCH ? AND k = ? AND timestamp_ms BETWEEN ? AND ? ORDER BY distance"
            conn.prepare(sql).use { st ->
                st.bindBlob(1, blob(vector)); st.bindLong(2, topK.toLong())
                st.bindLong(3, from); st.bindLong(4, to); readHits(st)
            }
        }
        Log.i(TAG, "searchFiltered: topK=$topK since=$sinceMs until=$untilMs returned=${hits.size}")
        hits
    }

    override fun all(limit: Int): List<ObjectHit> = synchronized(lock) {
        conn.prepare("SELECT rowid, payload FROM $TABLE ORDER BY timestamp_ms LIMIT ?").use { st ->
            st.bindLong(1, limit.toLong())
            val out = ArrayList<ObjectHit>()
            while (st.step()) out.add(toHit(st.getLong(0), st.getText(1), distance = null))
            out
        }.also { Log.i(TAG, "all(): ${it.size} stored objects") }
    }

    override fun count(): Long = synchronized(lock) {
        conn.prepare("SELECT count(*) FROM $TABLE").use { st -> st.step(); st.getLong(0) }
    }

    override fun deleteAll(): Unit = synchronized(lock) {
        val before = count()
        conn.execSQL("DELETE FROM $TABLE")
        Log.i(TAG, "deleteAll: dropped $before points")
    }

    override fun close() = synchronized(lock) { conn.close() }

    // ---- helpers ----

    /** rowid, distance, payload result rows → hits (score = 1 − cosine distance). */
    private fun readHits(st: SQLiteStatement): List<ObjectHit> {
        val out = ArrayList<ObjectHit>()
        while (st.step()) out.add(toHit(st.getLong(0), st.getText(2), distance = st.getDouble(1)))
        return out
    }

    private fun toHit(rowid: Long, payloadJson: String, distance: Double?): ObjectHit {
        val p = ObjectPayload.fromJson(payloadJson)
        return ObjectHit(
            id = rowid.toString(),
            score = if (distance == null) 0f else (1.0 - distance).toFloat(),
            label = p.label, bbox = p.bbox, timestampMs = p.timestampMs, thumbPath = p.thumbPath,
        )
    }

    /** sqlite-vec accepts a vector as a little-endian float32 blob. */
    private fun blob(v: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (x in v) bb.putFloat(x)
        return bb.array()
    }

    /** 1-bit quantization: dim/8 bytes, bit i set iff v[i] > 0 (sign quantization). The exact
     *  bit-order is irrelevant to Hamming distance as long as stored codes and query codes are packed
     *  the SAME way — which they are, since both go through this one function. `vec_bit()` reads the
     *  blob as a bit vector (8 bits/byte) for `vec_distance_hamming`. */
    private fun packBits(v: FloatArray): ByteArray {
        val bytes = ByteArray((v.size + 7) / 8)
        for (i in v.indices) if (v[i] > 0f) bytes[i ushr 3] = (bytes[i ushr 3].toInt() or (1 shl (i and 7))).toByte()
        return bytes
    }
}
