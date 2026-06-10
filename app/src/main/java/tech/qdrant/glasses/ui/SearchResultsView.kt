package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import tech.qdrant.glasses.storage.MemoryFrame

class SearchResultsView(context: Context) : FrameLayout(context) {

    private val image: ImageView
    private val overlay: LinearLayout
    private val queryText: TextView
    private val transcriptText: TextView
    private val timeText: TextView

    init {
        setBackgroundColor(Color.BLACK)

        image = ImageView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
            scaleType = ImageView.ScaleType.CENTER_CROP
        }
        addView(image)

        overlay = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(12, 8, 12, 8)
            setBackgroundColor(0xCC000000.toInt())
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).also {
                it.gravity = android.view.Gravity.BOTTOM
            }
        }

        queryText = TextView(context).apply {
            textSize = 18f
            setTextColor(Color.WHITE)
        }
        overlay.addView(queryText)

        // What was HEARD at that moment — shown only for transcript (type=text) hits.
        transcriptText = TextView(context).apply {
            textSize = 14f
            setTextColor(Color.WHITE)
            setTypeface(null, android.graphics.Typeface.ITALIC)
            visibility = GONE
        }
        overlay.addView(transcriptText)

        timeText = TextView(context).apply {
            textSize = 13f
            setTextColor(Color.LTGRAY)
        }
        overlay.addView(timeText)

        overlay.addView(TextView(context).apply {
            text = "Tap to dismiss"
            textSize = 10f
            setTextColor(Color.DKGRAY)
        })

        addView(overlay)
    }

    fun showResults(query: String, results: List<MemoryFrame>) {
        queryText.text = "\"$query\""

        if (results.isEmpty()) {
            image.setImageDrawable(null)
            transcriptText.visibility = GONE
            timeText.text = "Nothing found"
            return
        }

        val best = results.first()
        try {
            BitmapFactory.decodeFile(best.imagePath)?.let { image.setImageBitmap(it) }
        } catch (_: Exception) {}

        val elapsed = formatElapsed(System.currentTimeMillis() - best.timestampMs)
        val isTextHit = best.type == "text" && !best.transcript.isNullOrEmpty()

        // Compose what to show: the hit's own line (for a text hit) plus any speech
        // spoken near this frame (for any hit). An image hit thus still surfaces
        // "what was said here", not just the picture.
        val lines = buildList {
            if (isTextHit) add("“${best.transcript}”")
            best.nearbyTranscripts.forEach { add("• $it") }
        }
        if (lines.isNotEmpty()) {
            transcriptText.text = lines.joinToString("\n")
            transcriptText.visibility = VISIBLE
        } else {
            transcriptText.visibility = GONE
        }
        timeText.text = if (isTextHit) "Heard $elapsed" else "Seen $elapsed"
    }

    private fun formatElapsed(ms: Long): String {
        val secs = ms / 1000
        return when {
            secs < 60   -> "${secs}s ago"
            secs < 3600 -> "${secs / 60} min ago"
            else        -> "${secs / 3600} hrs ago"
        }
    }
}
