package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import yolov8tflite.R

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var regionPaint = Paint()
    private var depthMap: Array<FloatArray>? = null // 2D array from MiDaS (height x width)

    private var bounds = Rect()

    // Region thresholds (match Detector.kt)
    private val aboveHeightNormalized = 0.15f
    private val belowHeightNormalized = 0.15f
    private val regionWidth = 1f / 3f
    private val leftWidthNormalized = regionWidth
    private val rightWidthNormalized = regionWidth

    init {
        initPaints()
    }

    fun setDepthMap(depth: Array<FloatArray>) {
        this.depthMap = depth
        // just added for the pipeline - triggers onDraw to redraw with new data
        invalidate()
    }

    private fun initPaints() {
        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        regionPaint.style = Paint.Style.STROKE
        regionPaint.strokeWidth = 4F
        regionPaint.alpha = 150 // Semi-transparent
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        drawRegionBorders(canvas)
        drawDetectionResults(canvas)
    }



    private fun drawRegionBorders(canvas: Canvas) {
//        regionPaint.color = Color.GREEN
        val aboveY = height * aboveHeightNormalized
//        canvas.drawLine(0f, aboveY, width.toFloat(), aboveY, regionPaint)

//        regionPaint.color = Color.RED
        val belowY = height * (1 - belowHeightNormalized)
//        canvas.drawLine(0f, belowY, width.toFloat(), belowY, regionPaint)

//        regionPaint.color = Color.BLUE
        val leftX = width * leftWidthNormalized
//        canvas.drawLine(leftX, 0f, leftX, height.toFloat(), regionPaint)

//        regionPaint.color = Color.MAGENTA
        val rightX = width * (1 - rightWidthNormalized)
//        canvas.drawLine(rightX, 0f, rightX, height.toFloat(), regionPaint)
    }

    private fun drawDetectionResults(canvas: Canvas) {
        results.forEach { box ->
            val left = box.x1 * width
            val top = box.y1 * height
            val right = box.x2 * width
            val bottom = box.y2 * height

            canvas.drawRect(left, top, right, bottom, boxPaint)

            var depthLabel = "N/A"
            try {
                val depthArray = depthMap
                if (depthArray != null) {
                    val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
                    val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
                    if (centerY in depthArray.indices && centerX in depthArray[0].indices) {
                        val rawDepthValue = depthArray[centerY][centerX]
                        val scaleFactor = 0.0025f  // Adjust this based on your model
                        val meters = 1.0f / (rawDepthValue * scaleFactor)
                        depthLabel = if (meters in 0.5f..5.0f) {
                            String.format("%.1f m (raw %.1f)", meters, rawDepthValue)
                        } else {
                            "out of range (raw %.1f)".format(rawDepthValue)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("DEPTH_DEBUG", "Depth calc error", e)
            }

            val label = "${box.clsName} (${String.format("%.2f", box.cnf)}) - $depthLabel"

            textBackgroundPaint.getTextBounds(label, 0, label.length, bounds)
            canvas.drawRect(
                left,
                top,
                left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
                top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            canvas.drawText(label, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    fun clear() {
        results = listOf()
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
