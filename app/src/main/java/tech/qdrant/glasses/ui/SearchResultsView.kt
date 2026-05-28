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
            timeText.text = "Nothing found"
            return
        }

        val best = results.first()
        try {
            BitmapFactory.decodeFile(best.imagePath)?.let { image.setImageBitmap(it) }
        } catch (_: Exception) {}

        val elapsed = formatElapsed(System.currentTimeMillis() - best.timestampMs)
        timeText.text = "Seen $elapsed"
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
