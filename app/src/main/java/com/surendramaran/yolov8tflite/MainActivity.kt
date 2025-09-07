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

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var ttsReady = false
    private var detectorReady = false
    private var depthReady = false
    private var cameraReady = false

    private val DEPTH_MODEL_PATH = "depth_model.tflite"

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
                        detector?.detect(rotatedBitmap, screenWidth, screenHeight);
                    }
                }
                1 -> {
                    depthExecutor.execute {
                        val depthResult = depthEstimator?.estimateDepth(rotatedBitmap)
                        runOnUiThread {
                            depthResult?.let { result ->
                                binding.overlay.setDepthMap(result.rawDepthArray)
                                depthMap = result.rawDepthArray
                            }
                        }
                    }
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
    private val MOVEMENT_THRESHOLD = 0.08f // normalized movement threshold
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
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            if (boundingBoxes.isNotEmpty()) {
                val regionAnalysis = analyzeRegions(boundingBoxes)
                val guidance = generateNavigationGuidance(
                    regionAnalysis.leftOccupied,
                    regionAnalysis.centerOccupied,
                    regionAnalysis.rightOccupied,
                    regionAnalysis.aboveOccupied,
                    regionAnalysis.belowOccupied,
                    boundingBoxes
                )

                // Object persistence check
                val shouldSpeak = boundingBoxes.any { box ->
                    val objectId = box.clsName // Use class+region as ID, or add more info if needed
                    val centerX = (box.x1 + box.x2) / 2f
                    val centerY = (box.y1 + box.y2) / 2f
                    val depth = depthMap?.let { depthArray ->
                        val x = (centerX * (depthArray[0].size - 1)).toInt()
                        val y = (centerY * (depthArray.size - 1)).toInt()
                        if (y in depthArray.indices && x in depthArray[0].indices) {
                            depthArray[y][x] * 0.0025f
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
                    val primaryObjectId = boundingBoxes.firstOrNull()?.clsName ?: "general"
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
        // Cluster people (or other classes) by proximity/depth
        val personBoxes = boxes.filter { it.clsName.startsWith("person") }
        val personGroups = clusterByDepth(personBoxes, depthMap, 1.0f) // 1.0m threshold for grouping

        if (personGroups.isNotEmpty()) {
            // Find the largest group and its region
            val largestGroup = personGroups.maxByOrNull { it.size }
            val groupRegion = largestGroup?.let { group ->
                val regionCounts = group.groupingBy { it.clsName.substringAfter("-") }.eachCount()
                regionCounts.maxByOrNull { it.value }?.key ?: "ahead"
            }
            val groupDistance = largestGroup?.let { group ->
                group.mapNotNull { box ->
                    depthMap?.let { depthArray ->
                        val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
                        val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
                        if (centerY in depthArray.indices && centerX in depthArray[0].indices) {
                            val rawDepthValue = depthArray[centerY][centerX]
                            val scaleFactor = 0.0025f
                            rawDepthValue * scaleFactor
                        } else null
                    }
                }.minOrNull()
            }

            if (largestGroup != null && largestGroup.size > 1 && groupDistance != null && groupDistance in 0.5f..5.0f) {
                val distanceDescription = getDistanceDescription(groupDistance)
                return "${largestGroup.size} people $groupRegion, $distanceDescription"
            }
        }

        // ...existing code for single object guidance...
        val primaryObject = boxes.maxByOrNull { it.cnf * it.w * it.h }
        val objectName = primaryObject?.clsName?.substringBefore("-") ?: "object"

        // Get distance if available
        val distance = primaryObject?.let { box ->
            depthMap?.let { depthArray ->
                val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
                val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
                if (centerY in depthArray.indices && centerX in depthArray[0].indices) {
                    val rawDepthValue = depthArray[centerY][centerX]
                    val scaleFactor = 0.0025f
                    rawDepthValue * scaleFactor
                } else null
            }
        }

        // Only speak if object is within 0.5m–5m
        if (distance != null && (distance < 0.5f || distance > 5.0f)) {
            return null
        }

        val distanceDescription = distance?.let { getDistanceDescription(it) } ?: "ahead"

        // Priority checks (safety first)
        when {
            belowOccupied -> return "$objectName below, stop immediately"
            aboveOccupied && !(leftOccupied || centerOccupied || rightOccupied) ->
                return "$objectName above, lower your head"
        }

        return when {
            // Object on left - move right (and keep moving right if path isn't clear)
            leftOccupied && !rightOccupied ->
                if (!centerOccupied) "$objectName left $distanceDescription, move right"
                else "$objectName left and center $distanceDescription, move further right"

            // Object on right - move left (and keep moving left if path isn't clear)
            rightOccupied && !leftOccupied ->
                if (!centerOccupied) "$objectName right $distanceDescription, move left"
                else "$objectName right and center $distanceDescription, move further left"

            // Object in center - check sides and move to clearest path
            centerOccupied -> {
                when {
                    !leftOccupied && !rightOccupied -> "$objectName center $distanceDescription, move left or right"
                    !leftOccupied -> "$objectName center $distanceDescription, move left"
                    !rightOccupied -> "$objectName center $distanceDescription, move right"
                    else -> "$objectName $distanceDescription blocking path, stop"
                }
            }

            // Objects on both sides but center is clear
            leftOccupied && rightOccupied && !centerOccupied ->
                "Objects on both sides $distanceDescription, proceed carefully forward"

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

    // Helper function for clustering by depth
    private fun clusterByDepth(
        boxes: List<BoundingBox>,
        depthMap: Array<FloatArray>?,
        threshold: Float
    ): List<List<BoundingBox>> {
        val clusters = mutableListOf<MutableList<BoundingBox>>()
        val visited = mutableSetOf<Int>()

        for (i in boxes.indices) {
            if (i in visited) continue
            val boxA = boxes[i]
            val depthA = getBoxDepth(boxA, depthMap)
            val cluster = mutableListOf(boxA)
            visited.add(i)
            for (j in boxes.indices) {
                if (j == i || j in visited) continue
                val boxB = boxes[j]
                val depthB = getBoxDepth(boxB, depthMap)
                if (depthA != null && depthB != null && Math.abs(depthA - depthB) < threshold) {
                    cluster.add(boxB)
                    visited.add(j)
                }
            }
            clusters.add(cluster)
        }
        return clusters
    }

    private fun getBoxDepth(box: BoundingBox, depthMap: Array<FloatArray>?): Float? {
        return depthMap?.let { depthArray ->
            val centerX = ((box.x1 + box.x2) / 2 * (depthArray[0].size - 1)).toInt()
            val centerY = ((box.y1 + box.y2) / 2 * (depthArray.size - 1)).toInt()
            if (centerY in depthArray.indices && centerX in depthArray[0].indices) {
                val rawDepthValue = depthArray[centerY][centerX]
                val scaleFactor = 0.0025f
                rawDepthValue * scaleFactor
            } else null
        }
    }

    private val spokenObjectTimestamps = mutableMapOf<String, Long>()
    private val OBJECT_ALERT_COOLDOWN_MS = 2000L // 5 seconds per object
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
}

