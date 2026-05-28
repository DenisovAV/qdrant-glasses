package tech.qdrant.glasses.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.LinearLayout
import android.widget.TextView

class ListeningView(context: Context) : LinearLayout(context) {

    private val dot: TextView
    private val partial: TextView
    private val animator: ValueAnimator

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(Color.BLACK)

        dot = TextView(context).apply {
            text = "●"
            textSize = 56f
            setTextColor(Color.RED)
            gravity = Gravity.CENTER
        }
        addView(dot)

        addView(TextView(context).apply {
            text = "Listening..."
            textSize = 20f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
        })

        partial = TextView(context).apply {
            text = ""
            textSize = 14f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
            setPadding(16, 8, 16, 0)
        }
        addView(partial)

        addView(TextView(context).apply {
            text = "Tap to stop"
            textSize = 11f
            setTextColor(Color.DKGRAY)
            gravity = Gravity.CENTER
        })

        animator = ValueAnimator.ofFloat(0.3f, 1f).apply {
            duration = 600
            repeatMode = ValueAnimator.REVERSE
            repeatCount = ValueAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { dot.alpha = it.animatedValue as Float }
            start()
        }
    }

    fun showPartial(text: String) {
        if (text.isNotEmpty()) partial.text = "\"$text\""
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator.cancel()
    }
}
