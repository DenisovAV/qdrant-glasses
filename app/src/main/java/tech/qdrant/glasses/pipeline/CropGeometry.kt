package tech.qdrant.glasses.pipeline

import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF

/** Grow [box] by [padding] of its own size on each side, clamped to [w]x[h]; null if degenerate. */
fun paddedCropRect(box: RectF, padding: Float, w: Int, h: Int): Rect? {
    val padX = box.width() * padding
    val padY = box.height() * padding
    val l = (box.left - padX).toInt().coerceIn(0, w - 1)
    val t = (box.top - padY).toInt().coerceIn(0, h - 1)
    val r = (box.right + padX).toInt().coerceIn(l + 1, w)
    val b = (box.bottom + padY).toInt().coerceIn(t + 1, h)
    return if (r > l && b > t) Rect(l, t, r, b) else null
}

// Fraction of the bbox size to add as context padding on EACH side when cropping an object for
// embedding (0.20 = grow the box 20% left/right/top/bottom). Enough context to disambiguate the
// object and give the embedder scene cues, without letting the background dominate the crop.
// Lives HERE (not PerceptionPipeline's companion, where it started) because MomentCapture's
// region layer (plan Task 2.2) needs the identical crop geometry against a DIFFERENT keyframe
// bitmap — one shared constant, not two copies to keep in sync.
const val CROP_PADDING = 0.20f

/**
 * Crop [box] out of [frame] with [padding] context on each side (defaults to [CROP_PADDING]),
 * returning pixels the CALLER owns — never an alias of [frame]. Top-level here (not folded back
 * into its one caller) because it was ORIGINALLY shared between two callers — PerceptionPipeline's
 * object-crop path and [tech.qdrant.glasses.pipeline.MomentCapture]'s region-embed path (Task 2.2)
 * — but the object-crop path was retired (`4272653`), leaving MomentCapture as the ONLY caller now.
 * Left as a standalone top-level fn rather than moved back into MomentCapture: the padded-rect
 * computation and the `Bitmap.createBitmap` aliasing guard below are self-contained and Bitmap/
 * RectF-only (no MomentCapture state), so there's nothing to gain from re-inlining it.
 */
fun cropFrom(frame: Bitmap, box: RectF, padding: Float = CROP_PADDING): Bitmap? =
    paddedCropRect(box, padding, frame.width, frame.height)?.let { r ->
        // Defensive, matching every other crop call site in this codebase: createBitmap ALLOCATES
        // and can throw OutOfMemoryError even with valid coords, and this codebase has a history
        // of OOM crashes — swallow it → null → the caller skips this box.
        try {
            val b = Bitmap.createBitmap(frame, r.left, r.top, r.width(), r.height())
            // `Bitmap.createBitmap(src, …)` is documented to return THE SOURCE ITSELF when the
            // requested subset is the whole bitmap ("the new bitmap may be the same object as
            // source"). That can happen even at this file's modest [CROP_PADDING] (20% context per
            // side): once [box] already covers most of [frame], padding it out clamps to the full
            // frame on one or more edges (see [paddedCropRect]'s `coerceIn`). A caller then recycles
            // `frame` as soon as its own scope ends — while a crop derived from it may still be in
            // use elsewhere — and an aliased crop dies with it ("Can't compress a recycled bitmap").
            // Force a copy so a crop NEVER aliases the frame it was cut from.
            if (b === frame) b.copy(frame.config ?: Bitmap.Config.ARGB_8888, false) else b
        } catch (_: Throwable) { null }
    }
