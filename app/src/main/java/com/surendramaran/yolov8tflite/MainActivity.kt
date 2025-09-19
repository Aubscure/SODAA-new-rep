package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PorterDuff
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import yolov8tflite.R
import yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.util.DisplayMetrics
import android.view.View
import android.view.WindowManager
import android.view.animation.AlphaAnimation
import androidx.camera.core.ImageProxy
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
private const val DEPTH_SCALE_FACTOR = 0.0025f

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var ttsReady = false
    private var detectorReady = false
    private var depthReady = false
    private var cameraReady = false

    private val DEPTH_MODEL_PATH = "Midas-V2_w8a8.tflite"

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var lastSpokenGuidance: String? = null

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var detectionExecutor: ExecutorService
    private lateinit var depthExecutor: ExecutorService
    private lateinit var tts: TextToSpeech

    private var bitmapBuffer: Bitmap? = null
    private var depthMap: Array<FloatArray>? = null

    private var frameStep = 0;
    private val totalPipelineSteps = 3;

    private lateinit var loadingOverlay: View

    private var depthFrameCounter = 0
    private val DEPTH_SKIP_INTERVAL = 1 // Safe default, adjust as needed

    // HUD/debug metrics
    private var lastDetectionInferenceTime: Long = 0L
    private var lastDepthInferenceTime: Long = 0L
    private var lastLagMs: Long = 0L
    private var frameTimestampMs: Long = 0L
    private var depthSourceTimestampMs: Long = 0L
    private var lastDetectionFrameTimestampMs: Long = 0L

    

    // Data class for region analysis
    private data class RegionAnalysis(
        val leftOccupied: Boolean,
        val centerOccupied: Boolean,
        val rightOccupied: Boolean,
        val aboveOccupied: Boolean,
        val belowOccupied: Boolean
    )
    private fun getDistanceDescription(meters: Float): String {
        return when {
            meters >= 2.5f -> "more than 2 meters ahead"
            else -> String.format("%.1f meters ahead", meters)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadingOverlay = binding.loadingOverlay

        cameraExecutor = Executors.newSingleThreadExecutor()
        detectionExecutor = Executors.newSingleThreadExecutor()
        depthExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            // Initialize TTS
            tts = TextToSpeech(this, OnInitListener { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts.language = Locale.US // Set the language for TTS
                    tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            isSpeaking = true
                        }
                        override fun onDone(utteranceId: String?) {
                            isSpeaking = false
                            lastSpokenTime = System.currentTimeMillis()
                            runOnUiThread {
                                speechQueue.removeFirstOrNull()?.let { next ->
                                    speakGuidance(next)
                                }
                            }
                        }
                        override fun onError(utteranceId: String?) {
                            isSpeaking = false
                        }
                    })
                    ttsReady = true
                    checkAllReady()
                    // Flush any queued speech
                    runOnUiThread {
                        speechQueue.removeFirstOrNull()?.let { next ->
                            speakGuidance(next)
                        }
                    }
                }
            })

            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this) {
                toast(it)
            }
            detectorReady = true
            checkAllReady()

            depthEstimator = DepthEstimator(baseContext, DEPTH_MODEL_PATH)
            depthReady = true
            checkAllReady()
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val metrics = DisplayMetrics().also { binding.viewFinder.display.getRealMetrics(it) }
        val width = metrics.widthPixels
        val height = metrics.heightPixels

        if (bitmapBuffer == null || bitmapBuffer?.width != width || bitmapBuffer?.height != height) {
            bitmapBuffer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        val screenWidth = binding.viewFinder.width
        val screenHeight = binding.viewFinder.height

        preview = Preview.Builder()
            .setTargetResolution(Size(width, height))
//            .setTargetResolution(analysisSize)  // Based on the throttle
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(Size(screenWidth, screenHeight))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            // Record a frame timestamp in ms for consistent lag computation
            frameTimestampMs = System.currentTimeMillis()
            // 1) Ensure buffer is initialized and matches frame size
            if (bitmapBuffer == null ||
                bitmapBuffer!!.width  != imageProxy.width ||
                bitmapBuffer!!.height != imageProxy.height) {
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            }

            val bmp = bitmapBuffer!!
//            if (bmp.width  != imageProxy.width ||
//                bmp.height != imageProxy.height) {
//                // Re-create if resolution changed
//                bitmapBuffer = Bitmap.createBitmap(
//                    imageProxy.width,
//                    imageProxy.height,
//                    Bitmap.Config.ARGB_8888
//                )
//            }

            // 2) Copy raw camera bytes into the reused bitmap
            imageProxy.planes[0].buffer.rewind()  // reset position
            bmp.copyPixelsFromBuffer(imageProxy.planes[0].buffer)

            // 3) Rotate/mirror into a new Bitmap only if needed by your model
            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) postScale(-1f, 1f, bmp.width / 2f, bmp.height / 2f)
            }
            val rotatedBitmap = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)

            // detector?.detect(rotatedBitmap, screenWidth, screenHeight)
//            detectionExecutor.execute {
//                detector?.detect(rotatedBitmap, screenWidth, screenHeight)
//            }
//
//            depthExecutor.execute {
//                val depthResult = depthEstimator?.estimateDepth(rotatedBitmap)
//                runOnUiThread {
//                    depthResult?.let { result ->
//                        result.depthBitmap?.let { bitmap ->
//                            binding.depthView.setImageBitmap(bitmap)
//                            binding.overlay.setDepthMap(result.rawDepthArray)
//                            depthMap = result.rawDepthArray
//                        }
//                    }
//                }
//            }

            // Implementing a pipeline
            when (frameStep % totalPipelineSteps) {
                0 -> {
                    detectionExecutor.execute {
                        lastDetectionFrameTimestampMs = frameTimestampMs
                        detector?.detect(rotatedBitmap, screenWidth, screenHeight);
                    }
                }
                1 -> {
                    if (depthFrameCounter % DEPTH_SKIP_INTERVAL == 0) {
                        depthExecutor.execute {
                            val startTime = System.currentTimeMillis()
                            val sourceTs = frameTimestampMs
                            val depthResult = depthEstimator?.estimateDepth(rotatedBitmap)
                            val elapsed = System.currentTimeMillis() - startTime
                            val depthTimestamp = System.currentTimeMillis()
                            val lag = depthTimestamp - frameTimestampMs
                            lastDepthInferenceTime = elapsed
                            lastLagMs = lag
                            runOnUiThread {
                                val raw = depthResult?.rawDepthArray
                                if (raw != null) {
                                    binding.overlay.setDepthMap(raw)
                                    depthMap = raw
                                    depthSourceTimestampMs = sourceTs
                                }
                                updateHud()
                            }
                            Log.d("DEPTH_DEBUG", "Depth inference: ${elapsed}ms, lag from frame: ${lag}ms")
                        }
                    }
                    depthFrameCounter++
                }
                2 -> {
                    runOnUiThread {
                        lastSpokenGuidance?.let { guidance ->
                            speakGuidance(guidance)
                        }
                    }
                }
            }
            frameStep++
            imageProxy.close()
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.surfaceProvider = binding.viewFinder.surfaceProvider
            cameraReady = true
            checkAllReady()
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true) {
            startCamera()
        }
    }

    private fun speakGuidance(guidance: String, objectId: String? = null) {
        val now = System.currentTimeMillis()
        // Per-object cooldown
        if (objectId != null) {
            val lastObjectSpoken = spokenObjectTimestamps[objectId] ?: 0
            if (now - lastObjectSpoken < OBJECT_ALERT_COOLDOWN_MS) return
            spokenObjectTimestamps[objectId] = now
        }
        // Global cooldown and queue
        if (!ttsReady) {
            // TTS not ready, queue the guidance
            if (!speechQueue.contains(guidance)) {
                speechQueue.addLast(guidance)
            }
            return
        }
        if (isSpeaking || now - lastSpokenTime < SPEECH_COOLDOWN_MS) {
            if (!speechQueue.contains(guidance)) {
                speechQueue.addLast(guidance)
            }
            return
        }
        isSpeaking = true
        lastSpokenTime = now
        tts.speak(guidance, TextToSpeech.QUEUE_FLUSH, null, "GUIDANCE")
    }

    private fun toast(message: String) {
        runOnUiThread {
            Toast.makeText(baseContext, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun checkAllReady() {
        if (ttsReady && detectorReady && depthReady && cameraReady) {
            runOnUiThread {
                loadingOverlay.visibility = View.GONE // Hide loading spinner/overlay
                // You can add other actions here if needed, e.g.:
                // toast("System ready!")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        detector?.close()
        cameraExecutor.shutdown()
        detectionExecutor.shutdown()
        depthExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf(
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
            if (lastSpokenGuidance != "Path clear, move forward") {
                lastSpokenGuidance = "Path clear, move forward"
                speakGuidance(lastSpokenGuidance!!)
            }
        }
    }

    // Add these fields to MainActivity (top of class, after other vars)
    private val objectHistory = mutableMapOf<String, ObjectTrack>()
    private val OBJECT_PERSISTENCE_FRAMES = 10
    private val MOVEMENT_THRESHOLD = 0.04f // normalized movement threshold
    private val DEPTH_THRESHOLD = 0.5f // meters


    data class ObjectTrack(
        val id: String,
        var lastFrame: Int,
        var lastX: Float,
        var lastY: Float,
        var lastDepth: Float?
    )

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            lastDetectionInferenceTime = inferenceTime
            updateHud()
            // Filter boxes by distance (0.5m to 5.0m)
            val filteredBoxes = boundingBoxes
            binding.overlay.apply {
                setResults(filteredBoxes)
                invalidate()
            }

            if (filteredBoxes.isNotEmpty()) {
                val regionAnalysis = analyzeRegions(filteredBoxes)
                val guidance = generateNavigationGuidance(
                    regionAnalysis.leftOccupied,
                    regionAnalysis.centerOccupied,
                    regionAnalysis.rightOccupied,
                    regionAnalysis.aboveOccupied,
                    regionAnalysis.belowOccupied,
                    filteredBoxes
                )

                // Object persistence check
                val shouldSpeak = filteredBoxes.any { box ->
                    val objectId = box.clsName // Use class+region as ID, or add more info if needed
                    val centerX = (box.x1 + box.x2) / 2f
                    val centerY = (box.y1 + box.y2) / 2f
                    val depth = depthMap?.let { depthArray ->
                        val x = (centerX * (depthArray[0].size - 1)).toInt()
                        val y = (centerY * (depthArray.size - 1)).toInt()
                        if (y in depthArray.indices && x in depthArray[0].indices) {
                            RawDepth(depthArray[y][x]).toMeters()
                        } else null
                    }

                    val track = objectHistory[objectId]
                    val moved = track == null ||
                        (Math.abs(centerX - track.lastX) > MOVEMENT_THRESHOLD) ||
                        (Math.abs(centerY - track.lastY) > MOVEMENT_THRESHOLD) ||
                        (depth != null && track.lastDepth != null && Math.abs(depth - track.lastDepth!!) > DEPTH_THRESHOLD) ||
                        (frameStep - track.lastFrame > OBJECT_PERSISTENCE_FRAMES)

                    // Update history
                    objectHistory[objectId] = ObjectTrack(
                        id = objectId,
                        lastFrame = frameStep,
                        lastX = centerX,
                        lastY = centerY,
                        lastDepth = depth
                    )

                    moved
                }

                // Only speak if guidance is not null, different, and object is new/moved
                if (guidance != null && guidance != lastSpokenGuidance && shouldSpeak) {
                    lastSpokenGuidance = guidance
                    // Use the primary object's ID for debouncing
                    val primaryObjectId = filteredBoxes.firstOrNull()?.clsName ?: "general"
                    speakGuidance(guidance, primaryObjectId)
                }
            } else {
                if (lastSpokenGuidance != "Path clear, move forward") {
                    lastSpokenGuidance = "Path clear, move forward"
                    speakGuidance(lastSpokenGuidance!!)
                }
            }
        }
    }

    private fun analyzeRegions(boundingBoxes: List<BoundingBox>): RegionAnalysis {
        var leftOccupied = false
        var centerOccupied = false
        var rightOccupied = false
        var aboveOccupied = false
        var belowOccupied = false

        boundingBoxes.forEach { box ->
            when {
                box.clsName.endsWith("-left") -> leftOccupied = true
                box.clsName.endsWith("-center") -> centerOccupied = true
                box.clsName.endsWith("-right") -> rightOccupied = true
                box.clsName.endsWith("-above") -> aboveOccupied = true
                box.clsName.endsWith("-below") -> belowOccupied = true
            }
        }

        return RegionAnalysis(leftOccupied, centerOccupied, rightOccupied, aboveOccupied, belowOccupied)
    }

    private fun generateNavigationGuidance(
        leftOccupied: Boolean,
        centerOccupied: Boolean,
        rightOccupied: Boolean,
        aboveOccupied: Boolean,
        belowOccupied: Boolean,
        boxes: List<BoundingBox>
    ): String? {
        // Centralized, optimized grouping by depth (single pass, sorted)
        fun clusterByDepthFast(
            boxes: List<BoundingBox>,
            depthMap: Array<FloatArray>?,
            threshold: Float
        ): List<List<BoundingBox>> {
            if (depthMap == null) return listOf(boxes)
            val boxesWithDepth = boxes.mapNotNull { box ->
                val depth = getBoxDepth(box, depthMap)
                if (depth != null) Pair(box, depth) else null
            }.sortedBy { it.second }
            val clusters = mutableListOf<MutableList<BoundingBox>>()
            var currentCluster = mutableListOf<BoundingBox>()
            var lastDepth: Float? = null
            for ((box, depth) in boxesWithDepth) {
                if (lastDepth == null || kotlin.math.abs(depth - lastDepth) < threshold) {
                    currentCluster.add(box)
                } else {
                    clusters.add(currentCluster)
                    currentCluster = mutableListOf(box)
                }
                lastDepth = depth
            }
            if (currentCluster.isNotEmpty()) clusters.add(currentCluster)
            return clusters
        }

        // Group people by depth
        val personBoxes = boxes.filter { it.clsName.startsWith("person") }
        val personGroups = clusterByDepthFast(personBoxes, depthMap, 2.0f) // 1.0m threshold

        if (personGroups.isNotEmpty()) {
            // Find the largest group and its region
            val largestGroup = personGroups.maxByOrNull { it.size }
            val groupRegion = largestGroup?.let { group ->
                val regionCounts = group.groupingBy { it.clsName.substringAfter("-") }.eachCount()
                regionCounts.maxByOrNull { it.value }?.key ?: "ahead"
            }
            val groupDistance = largestGroup?.let { group ->
                group.mapNotNull { box ->
                    getBoxDepth(box, depthMap)
                }.minOrNull()
            }

            if (largestGroup != null && largestGroup.size > 1 && groupDistance != null && groupDistance in 0.5f..5.0f) {
                val distanceDescription = String.format("%.1f meters", groupDistance)
                return "people $groupRegion $distanceDescription ahead"
            }
        }

        // Single object guidance (most prominent)
        val primaryObject = boxes.maxByOrNull { it.cnf * it.w * it.h }
        val objectName = primaryObject?.clsName?.substringBefore("-") ?: "object"
        val region = primaryObject?.clsName?.substringAfter("-") ?: "ahead"
        val distance = primaryObject?.let { box ->
            getBoxDepth(box, depthMap)
        }
        if (distance != null && (distance < 0.5f || distance > 5.0f)) {
            return null
        }
        val distanceDescription = distance?.let { String.format("%.1f meters", it) } ?: ""

        // Priority checks (safety first)
        when {
            belowOccupied -> return "$objectName below, stop immediately"
            aboveOccupied && !(leftOccupied || centerOccupied || rightOccupied) ->
                return "$objectName above, lower your head"
        }

        // After the person group logic, always check for the closest object:
        val closestBox = boxes.minByOrNull { getBoxDepth(it, depthMap) ?: Float.MAX_VALUE }
        val closestDistance = closestBox?.let { getBoxDepth(it, depthMap) }
        if (closestBox != null && closestDistance != null && closestDistance > 0f && closestDistance < 0.5f) {
            val objectName = closestBox.clsName.substringBefore("-")
            return "$objectName very 4close, stop!"
        }

        return when {
            // Object on left - move right (and keep moving right if path isn't clear)
            leftOccupied && !rightOccupied ->
                if (!centerOccupied) "$objectName left $distanceDescription ahead, move right"
                else "$objectName left and center $distanceDescription ahead, move further right"

            // Object on right - move left (and keep moving left if path isn't clear)
            rightOccupied && !leftOccupied ->
                if (!centerOccupied) "$objectName right $distanceDescription ahead, move left"
                else "$objectName right and center $distanceDescription ahead, move further left"

            // Object in center - check sides and move to clearest path
            centerOccupied -> {
                when {
                    !leftOccupied && !rightOccupied -> "$objectName center $distanceDescription ahead, move left or right"
                    !leftOccupied -> "$objectName center $distanceDescription ahead, move left"
                    !rightOccupied -> "$objectName center $distanceDescription ahead, move right"
                    else -> "$objectName $distanceDescription ahead blocking path, stop"
                }
            }

            // Objects on both sides but center is clear
            leftOccupied && rightOccupied && !centerOccupied ->
                "Objects on both sides $distanceDescription ahead, proceed carefully forward"

            // No objects detected
            !leftOccupied && !centerOccupied && !rightOccupied && !aboveOccupied && !belowOccupied ->
                "Path clear, move forward"

            // Default case when no specific guidance applies
            else -> {
                // Suggest moving to the most open side
                val openSide = when {
                    !leftOccupied && !rightOccupied -> "forward"
                    !leftOccupied -> "left"
                    !rightOccupied -> "right"
                    else -> "carefully forward"
                }
                "Path clear, move $openSide"
            }
        }
    }

    // Helper function for getting box depth (already present in your code)
    private fun getBoxDepth(box: BoundingBox, depthMap: Array<FloatArray>?): Float? {
        return depthMap?.let { depthArray ->
            val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
            val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
            val medianRaw = getMedianDepthPatch(depthArray, centerX, centerY, patchSize = 2)
            medianRaw?.let { RawDepth(it).toMeters() }
        }
    }

    private val spokenObjectTimestamps = mutableMapOf<String, Long>()
    private val OBJECT_ALERT_COOLDOWN_MS = 1000L // 5 seconds per object
    private var lastSpokenTime: Long = 0
    private val SPEECH_COOLDOWN_MS = 1000L // 3.5 seconds global debounce

    private val speechQueue: ArrayDeque<String> = ArrayDeque()
    private var isSpeaking: Boolean = false

    // Add to MainActivity.kt (inside the class)
    private fun getMedianDepthPatch(
        depthArray: Array<FloatArray>,
        centerX: Int,
        centerY: Int,
        patchSize: Int = 2
    ): Float? {
        val depths = mutableListOf<Float>()
        for (dy in -patchSize..patchSize) {
            for (dx in -patchSize..patchSize) {
                val px = centerX + dx
                val py = centerY + dy
                if (py in depthArray.indices && px in depthArray[0].indices) {
                    depths.add(depthArray[py][px])
                }
            }
        }
        if (depths.isEmpty()) return null
        return depths.sorted()[depths.size / 2]
    }



    @JvmInline
    private value class RawDepth(val value: Float)

    private inline fun RawDepth.toMeters(): Float = 1.0f / (value * DEPTH_SCALE_FACTOR)

    private fun updateHud() {
        val det = lastDetectionInferenceTime
        val dep = lastDepthInferenceTime
        val lag = lastLagMs
        val depthAgeMs = if (depthSourceTimestampMs > 0L) System.currentTimeMillis() - depthSourceTimestampMs else -1L

        fun speedLabel(ms: Long, fast: Long, slow: Long): String {
            return when {
                ms <= fast -> "fast"
                ms <= slow -> "ok"
                else -> "slow"
            }
        }

        val detLabel = speedLabel(det, fast = 30, slow = 60)
        val depLabel = speedLabel(dep, fast = 40, slow = 100)
        val lagLabel = speedLabel(lag, fast = 80, slow = 150)
        val ageLabel = if (depthAgeMs >= 0) speedLabel(depthAgeMs, fast = 80, slow = 160) else "n/a"

        val statusPriority = mapOf("fast" to 0, "ok" to 1, "slow" to 2)
        val worstLabel = listOf(detLabel, depLabel, lagLabel, ageLabel).maxByOrNull { statusPriority[it] ?: 0 } ?: "ok"
        val color = when (worstLabel) {
            "fast" -> Color.GREEN
            "ok" -> Color.YELLOW
            else -> Color.RED
        }

        binding.inferenceTime.setTextColor(color)
        val ageText = if (depthAgeMs >= 0) "\nDepthAge: ${depthAgeMs}ms (${ageLabel})" else ""
        binding.inferenceTime.isSingleLine = false
        binding.inferenceTime.maxLines = 4
        binding.inferenceTime.textAlignment = View.TEXT_ALIGNMENT_VIEW_START
        binding.inferenceTime.text = "Det: ${det}ms (${detLabel})\nDepth: ${dep}ms (${depLabel})\nLag: ${lag}ms (${lagLabel})${ageText}"
    }
}

