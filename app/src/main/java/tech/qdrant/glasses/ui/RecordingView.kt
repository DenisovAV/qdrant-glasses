package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView

class RecordingView(context: Context) : LinearLayout(context) {
    private val statusText: TextView
    private val countText: TextView

    init {
        orientation = VERTICAL
        gravity = Gravity.TOP or Gravity.START
        setPadding(20, 20, 20, 20)
        setBackgroundColor(Color.BLACK)

        statusText = TextView(context).apply {
            text = "● REC"
            textSize = 18f
            setTextColor(Color.RED)
        }
        addView(statusText)

        countText = TextView(context).apply {
            text = "0 frames"
            textSize = 14f
            setTextColor(Color.WHITE)
        }
        addView(countText)

        addView(TextView(context).apply {
            text = "Tap to stop"
            textSize = 11f
            setTextColor(Color.GRAY)
        })
    }

    fun update(frames: Long, speech: Long, elapsedSeconds: Long) {
        val mins = elapsedSeconds / 60
        val secs = elapsedSeconds % 60
        statusText.text = "● REC  %02d:%02d".format(mins, secs)
        countText.text = "frames: $frames  speech: $speech"
    }
}
