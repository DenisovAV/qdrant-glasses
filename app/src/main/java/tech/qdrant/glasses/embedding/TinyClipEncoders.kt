package tech.qdrant.glasses.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * TinyCLIP-ViT-8M-16 (int8, ~24MB) — a single combined ONNX graph holding BOTH towers.
 * Unlike the split ViT-B/32 files, every run takes all three inputs (pixel_values,
 * input_ids, attention_mask) and emits four outputs; the towers are independent, so the
 * unused branch is fed a cheap dummy and we read the wanted embedding BY NAME
 * ("image_embeds"/"text_embeds" — output[0] is logits, NOT the embedding).
 *
 * Same 512-dim space shape as ViT-B/32 (projection_dim=512), same CLIP mean/std
 * preprocessing, same CLIP BPE tokenizer — but a DIFFERENT model → different embedding
 * space → switching backends requires re-indexing (pm clear).
 */
private const val TINYCLIP_ASSET = "tinyclip-int8.onnx"
private const val IMG = 224

class TinyClipVisionEncoder(context: Context) : VisionEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val std  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    // The text branch is independent of the image branch (verified: image_embeds is
    // bit-identical regardless of text input) — feed a minimal 1-token dummy.
    private val dummyIds = OnnxTensor.createTensor(
        env, LongBuffer.wrap(longArrayOf(Tokenizer.EOT_TOKEN.toLong())), longArrayOf(1, 1)
    )
    private val dummyMask = OnnxTensor.createTensor(
        env, LongBuffer.wrap(longArrayOf(1L)), longArrayOf(1, 1)
    )

    init {
        val modelFile = extractAsset(context, TINYCLIP_ASSET)
        session = createAcceleratedSession(env, modelFile.absolutePath)
    }

    override fun encode(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG, IMG, true)
        val tensor = OnnxTensor.createTensor(env, bitmapToTensor(resized), longArrayOf(1, 3, IMG.toLong(), IMG.toLong()))
        if (resized !== bitmap) resized.recycle()   // pixels copied into the tensor; free the scaled copy
        val t1 = System.currentTimeMillis()
        val results = tensor.use {
            session.run(mapOf("pixel_values" to it, "input_ids" to dummyIds, "attention_mask" to dummyMask))
        }
        Log.i("ClipEncoder", "TinyCLIP vision run=${System.currentTimeMillis() - t1}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("image_embeds").get().value as Array<FloatArray>)[0]
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(IMG * IMG).also { bitmap.getPixels(it, 0, IMG, 0, 0, IMG, IMG) }
        val buf = FloatBuffer.allocate(3 * IMG * IMG)
        val stride = IMG * IMG
        for (i in pixels.indices) {
            val px = pixels[i]
            buf.put(i,              ((px shr 16 and 0xFF) / 255f - mean[0]) / std[0])
            buf.put(i + stride,     ((px shr  8 and 0xFF) / 255f - mean[1]) / std[1])
            buf.put(i + stride * 2, ((px        and 0xFF) / 255f - mean[2]) / std[2])
        }
        return buf.apply { rewind() }
    }

    override fun close() {
        dummyIds.close()
        dummyMask.close()
        session.close()
        env.close()
    }
}

class TinyClipTextEncoder(context: Context) : TextEncoder {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer = TokenizerFactory.create(context)

    // The image branch ignores the text branch — feed a zero image once.
    private val dummyPixels = OnnxTensor.createTensor(
        env, FloatBuffer.allocate(3 * IMG * IMG), longArrayOf(1, 3, IMG.toLong(), IMG.toLong())
    )

    init {
        val modelFile = extractAsset(context, TINYCLIP_ASSET)
        session = createAcceleratedSession(env, modelFile.absolutePath)
    }

    override fun encode(text: String): FloatArray {
        val ids = tokenizer.encodeToIds(text)  // [SOT, tok…, EOT, 0-pad], length 77
        val maxLen = Tokenizer.MAX_LENGTH
        val inputIds = LongBuffer.allocate(maxLen)
        val attnMask = LongBuffer.allocate(maxLen)
        for (i in 0 until maxLen) {
            val id = ids[i]
            inputIds.put(id.toLong())
            attnMask.put(if (id != 0 || i == 0) 1L else 0L)
        }
        inputIds.rewind(); attnMask.rewind()

        val shape = longArrayOf(1, maxLen.toLong())
        val idsTensor = OnnxTensor.createTensor(env, inputIds, shape)
        val maskTensor = OnnxTensor.createTensor(env, attnMask, shape)
        val t1 = System.currentTimeMillis()
        val results = idsTensor.use { i -> maskTensor.use { m ->
            session.run(mapOf("input_ids" to i, "attention_mask" to m, "pixel_values" to dummyPixels))
        }}
        Log.i("ClipEncoder", "TinyCLIP text run=${System.currentTimeMillis() - t1}ms")
        return results.use { r ->
            @Suppress("UNCHECKED_CAST")
            (r.get("text_embeds").get().value as Array<FloatArray>)[0]
        }
    }

    override fun close() {
        dummyPixels.close()
        session.close()
        env.close()
    }
}
