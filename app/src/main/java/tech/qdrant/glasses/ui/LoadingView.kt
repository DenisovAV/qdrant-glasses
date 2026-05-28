package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView

class LoadingView(context: Context) : LinearLayout(context) {
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
        addView(TextView(context).apply {
            text = "Loading models..."
            textSize = 14f
            setTextColor(Color.GRAY)
            gravity = Gravity.CENTER
        })
    }
}
