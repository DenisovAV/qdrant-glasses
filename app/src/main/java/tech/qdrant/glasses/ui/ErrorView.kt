package tech.qdrant.glasses.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Shown when startup (model / store / detector init) fails. Without it, an init failure left the
 * app on the "Loading models…" screen forever with no signal (see GlassesViewModel.init). The
 * reason string is rendered so the failure can be diagnosed on-device (e.g. a Hexagon HTP init
 * error), matching the hand-built style of the other lens views.
 */
class ErrorView(context: Context, reason: String) : LinearLayout(context) {
    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(Color.BLACK)
        setPadding(24, 24, 24, 24)

        addView(TextView(context).apply {
            text = "Startup failed"
            textSize = 22f
            setTextColor(Color.rgb(255, 90, 90))
            gravity = Gravity.CENTER
        })
        addView(TextView(context).apply {
            text = reason
            textSize = 13f
            setTextColor(Color.GRAY)
            gravity = Gravity.CENTER
        })
    }
}
