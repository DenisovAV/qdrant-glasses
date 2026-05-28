package tech.qdrant.glasses.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.animation.LinearInterpolator
import android.widget.LinearLayout
import android.widget.TextView

class ProcessingView(context: Context, query: String) : LinearLayout(context) {

    private val dotsText: TextView
    private val animator: ValueAnimator

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(Color.BLACK)

        addView(TextView(context).apply {
            text = "\"$query\""
            textSize = 22f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(16, 0, 16, 16)
        })

        addView(TextView(context).apply {
            text = "Searching..."
            textSize = 16f
            setTextColor(Color.LTGRAY)
            gravity = Gravity.CENTER
        })

        dotsText = TextView(context).apply {
            text = "●○○"
            textSize = 24f
            setTextColor(Color.DKGRAY)
            gravity = Gravity.CENTER
            setPadding(0, 12, 0, 0)
        }
        addView(dotsText)

        val frames = arrayOf("●○○", "○●○", "○○●")
        animator = ValueAnimator.ofInt(0, 2).apply {
            duration = 900
            repeatMode = ValueAnimator.RESTART
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
            addUpdateListener { dotsText.text = frames[it.animatedValue as Int] }
            start()
        }
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator.cancel()
    }
}
