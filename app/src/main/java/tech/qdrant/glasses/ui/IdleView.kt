package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView

class IdleView(context: Context) : LinearLayout(context) {
    private val countText: TextView

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(Color.BLACK)

        addView(TextView(context).apply {
            text = "Qdrant Glasses"
            textSize = 24f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
        })

        countText = TextView(context).apply {
            text = "0 frames in memory"
            textSize = 14f
            setTextColor(Color.GRAY)
            gravity = Gravity.CENTER
        }
        addView(countText)

        addView(TextView(context).apply {
            text = "\nHold button to record  ·  Press to search"
            textSize = 12f
            setTextColor(Color.DKGRAY)
            gravity = Gravity.CENTER
        })
    }

    fun updateCount(count: Long) {
        countText.text = "$count frames in memory"
    }
}
