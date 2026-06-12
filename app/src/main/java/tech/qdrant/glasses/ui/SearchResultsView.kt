package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import tech.qdrant.glasses.search.MomentCard

class SearchResultsView(context: Context) : FrameLayout(context) {

    private val image: ImageView
    private val overlay: LinearLayout
    private val queryText: TextView
    private val transcriptText: TextView
    private val timeText: TextView
    private val pagerText: TextView

    private var cards: List<MomentCard> = emptyList()
    private var index = 0

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

        pagerText = TextView(context).apply {
            textSize = 11f
            setTextColor(Color.LTGRAY)
        }
        overlay.addView(pagerText)

        overlay.addView(TextView(context).apply {
            text = "Tap: next / dismiss"
            textSize = 10f
            setTextColor(Color.DKGRAY)
        })

        addView(overlay)
    }

    fun showCards(query: String, cards: List<MomentCard>) {
        this.cards = cards; this.index = 0
        queryText.text = "\"$query\""
        render()
    }

    /** Advance to next card. Returns false if there is no next (caller should dismiss). */
    fun nextCard(): Boolean {
        if (index + 1 >= cards.size) return false
        index++; render(); return true
    }

    private fun render() {
        val card = cards.getOrNull(index)
        if (card == null) {
            image.setImageDrawable(null); transcriptText.visibility = GONE
            timeText.text = "Nothing found"; pagerText.text = ""
            return
        }
        val f = card.frame
        try { BitmapFactory.decodeFile(f.imagePath)?.let { image.setImageBitmap(it) } } catch (_: Exception) {}
        val lines = buildList {
            f.transcript?.let { add("“$it”") }
            f.nearbyTranscripts.forEach { add("• $it") }
        }
        transcriptText.text = lines.joinToString("\n")
        transcriptText.visibility = if (lines.isEmpty()) GONE else VISIBLE
        val badges = (if (card.fromVision) "👁" else "") + (if (card.fromHeard) "🎙" else "")
        val verb = if (card.fromHeard && !card.fromVision) "Heard" else "Seen"
        timeText.text = "$badges  $verb ${formatElapsed(System.currentTimeMillis() - f.timestampMs)}"
        pagerText.text = "${index + 1}/${cards.size}"
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
