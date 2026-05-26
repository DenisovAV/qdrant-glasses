package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView

class ListeningView(context: Context) : LinearLayout(context) {
    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(Color.BLACK)

        addView(TextView(context).apply {
            text = "..."
            textSize = 48f
            gravity = Gravity.CENTER
        })
        addView(TextView(context).apply {
            text = "Listening..."
            textSize = 20f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
        })
        addView(TextView(context).apply {
            text = "Say what you're looking for"
            textSize = 13f
            setTextColor(Color.GRAY)
            gravity = Gravity.CENTER
        })
    }
}
