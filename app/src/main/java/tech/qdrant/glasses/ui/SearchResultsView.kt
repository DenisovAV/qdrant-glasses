package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import tech.qdrant.glasses.storage.MemoryFrame

class SearchResultsView(context: Context) : LinearLayout(context) {
    private val queryText: TextView
    private val resultsContainer: LinearLayout

    init {
        orientation = VERTICAL
        setPadding(16, 16, 16, 16)
        setBackgroundColor(Color.BLACK)

        queryText = TextView(context).apply {
            textSize = 16f
            setTextColor(Color.WHITE)
        }
        addView(queryText)

        resultsContainer = LinearLayout(context).apply { orientation = VERTICAL }
        addView(resultsContainer)

        addView(TextView(context).apply {
            text = "\nTap to search again"
            textSize = 11f
            setTextColor(Color.DKGRAY)
        })
    }

    fun showResults(query: String, results: List<MemoryFrame>) {
        queryText.text = "\"$query\""
        resultsContainer.removeAllViews()

        if (results.isEmpty()) {
            resultsContainer.addView(TextView(context).apply {
                text = "No results found"
                setTextColor(Color.GRAY)
                textSize = 14f
            })
            return
        }

        for (frame in results) {
            val row = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, 8, 0, 8)
            }

            val thumb = ImageView(context).apply {
                layoutParams = LayoutParams(80, 60)
                scaleType = ImageView.ScaleType.CENTER_CROP
            }
            try {
                BitmapFactory.decodeFile(frame.imagePath)?.let { bmp ->
                    thumb.setImageBitmap(Bitmap.createScaledBitmap(bmp, 80, 60, true))
                }
            } catch (_: Exception) {}
            row.addView(thumb)

            val elapsedMs = System.currentTimeMillis() - frame.timestampMs
            row.addView(TextView(context).apply {
                text = "  ${formatElapsed(elapsedMs)}    ${"%.2f".format(frame.score)}"
                textSize = 13f
                setTextColor(Color.WHITE)
            })

            resultsContainer.addView(row)
        }
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
