package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import android.util.Size
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
// -----------------------------------
import java.nio.ByteBuffer
import java.nio.ByteOrder



class DepthEstimator(
    context: Context,
    modelPath: String
) {
    private var interpreter: Interpreter
    private var inputWidth: Int = 0
    private var inputHeight: Int = 0
    private var inputDataType: DataType? = null
    private var outputDataType: DataType? = null

    init {
        try {
            val assetFileDescriptor = context.assets.openFd(modelPath)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options()
            var delegateAdded = false

            // Try NNAPI (NPU) first
            try {
                val nnApiDelegate = NnApiDelegate()
                options.addDelegate(nnApiDelegate)
                delegateAdded = true
                Log.d("DepthEstimator", "Using NNAPI delegate (NPU) for depth estimation")
            } catch (e: Exception) {
                Log.d("DepthEstimator", "NNAPI delegate not available: ${e.message}")
            }

            // If NNAPI not available, try GPU
            if (!delegateAdded) {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    options.addDelegate(GpuDelegate(delegateOptions))
                    delegateAdded = true
                    Log.d("DepthEstimator", "Using GPU delegate for depth estimation")
                }
            }

            // If neither, use CPU
            if (!delegateAdded) {
                options.setNumThreads(4)
                Log.d("DepthEstimator", "Using CPU for depth estimation")
            }

            interpreter = Interpreter(modelBuffer, options)

            // now your debug logs
            val inT = interpreter.getInputTensor(0)
            val outT = interpreter.getOutputTensor(0)
            inputDataType = inT.dataType()
            outputDataType = outT.dataType()
            Log.d("FastDepth", "Input 0 shape=${inT.shape().toList()}  dtype=${inputDataType} q=(scale=${inT.quantizationParams().scale}, zp=${inT.quantizationParams().zeroPoint})")
            Log.d("FastDepth", "Output 0 shape=${outT.shape().toList()} dtype=${outputDataType} q=(scale=${outT.quantizationParams().scale}, zp=${outT.quantizationParams().zeroPoint})")

            // Derive input spatial dimensions dynamically from model
            // Expecting NHWC: [1, height, width, channels] or [1, width, height, channels] depending on model
            val inShape = inT.shape()
            if (inShape.size >= 3) {
                // Most TFLite vision models are NHWC: [1, H, W, C]
                inputHeight = inShape[1]
                inputWidth = inShape[2]
            }
            // Enforce 256x256 if the model reports dynamic or unexpected dims (fallback)
            if (inputWidth <= 0 || inputHeight <= 0) {
                inputWidth = 256
                inputHeight = 256
            }
        }
        catch (e: Exception) {
            Log.e("FastDepth", "Failed to load/interrogate model", e)
            // re-throw if you want the crash to still bubble up:
            throw e
        }
    }
    data class DepthResult(
        val rawDepthArray: Array<FloatArray>
    )

    fun estimateDepth(inputBitmap: Bitmap): DepthResult {
        // Enforce 256x256 resize as requested (and as model expects)
        val targetWidth = if (inputWidth > 0) inputWidth else 256
        val targetHeight = if (inputHeight > 0) inputHeight else 256
        val modelInputSize = Size(targetWidth, targetHeight)
        val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, modelInputSize.width, modelInputSize.height, true)

        val inTensor = interpreter.getInputTensor(0)
        val outTensor = interpreter.getOutputTensor(0)
        val inQuant = inTensor.quantizationParams()
        val outQuant = outTensor.quantizationParams()
        val inType = inputDataType ?: inTensor.dataType()
        val outType = outputDataType ?: outTensor.dataType()

        // Build input according to data type
        val inputObject: Any = if (inType == DataType.FLOAT32) {
            val input = Array(1) {
                Array(modelInputSize.height) {
                    Array(modelInputSize.width) {
                        FloatArray(3)
                    }
                }
            }
            for (y in 0 until modelInputSize.height) {
                for (x in 0 until modelInputSize.width) {
                    val pixel = resizedBitmap.getPixel(x, y)
                    input[0][y][x][0] = Color.red(pixel) / 255.0f
                    input[0][y][x][1] = Color.green(pixel) / 255.0f
                    input[0][y][x][2] = Color.blue(pixel) / 255.0f
                }
            }
            input
        } else {
            // Quantized input: UINT8 or INT8
            val channels = 3
            val buffer = ByteBuffer.allocateDirect(modelInputSize.width * modelInputSize.height * channels)
            buffer.order(ByteOrder.nativeOrder())
            val scale = inQuant.scale
            val zeroPoint = inQuant.zeroPoint
            for (y in 0 until modelInputSize.height) {
                for (x in 0 until modelInputSize.width) {
                    val pixel = resizedBitmap.getPixel(x, y)
                    val r = (Color.red(pixel) / 255.0f)
                    val g = (Color.green(pixel) / 255.0f)
                    val b = (Color.blue(pixel) / 255.0f)
                    // Quantize per tensor (NHWC, RGB)
                    // q = round(real/scale + zp)
                    fun q(real: Float): Int {
                        return Math.round(real / scale + zeroPoint)
                    }
                    if (inType == DataType.UINT8) {
                        buffer.put(q(r).coerceIn(0, 255).toByte())
                        buffer.put(q(g).coerceIn(0, 255).toByte())
                        buffer.put(q(b).coerceIn(0, 255).toByte())
                    } else { // INT8
                        buffer.put(q(r).coerceIn(-128, 127).toByte())
                        buffer.put(q(g).coerceIn(-128, 127).toByte())
                        buffer.put(q(b).coerceIn(-128, 127).toByte())
                    }
                }
            }
            buffer.rewind()
            buffer
        }

        // Prepare output container according to data type
        val rawDepth: Array<FloatArray>
        if (outType == DataType.FLOAT32) {
            val output = Array(1) {
                Array(modelInputSize.height) {
                    Array(modelInputSize.width) {
                        FloatArray(1)
                    }
                }
            }
            interpreter.run(inputObject, output)
            rawDepth = Array(modelInputSize.height) { y ->
                FloatArray(modelInputSize.width) { x ->
                    output[0][y][x][0]
                }
            }
        } else {
            val elements = modelInputSize.width * modelInputSize.height
            val outBuffer = ByteBuffer.allocateDirect(elements)
            outBuffer.order(ByteOrder.nativeOrder())
            interpreter.run(inputObject, outBuffer)
            outBuffer.rewind()
            val scale = outQuant.scale
            val zeroPoint = outQuant.zeroPoint
            rawDepth = Array(modelInputSize.height) { FloatArray(modelInputSize.width) }
            for (y in 0 until modelInputSize.height) {
                for (x in 0 until modelInputSize.width) {
                    val q = outBuffer.get().toInt()
                    val qVal = if (outType == DataType.UINT8) {
                        q and 0xFF
                    } else {
                        // INT8 already signed
                        q.toByte().toInt()
                    }
                    val real = scale * (qVal - zeroPoint)
                    rawDepth[y][x] = real
                }
            }
        }

        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        for (y in 0 until rawDepth.size) {
            for (x in 0 until rawDepth[0].size) {
                val v = rawDepth[y][x]
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
            }
        }
        Log.d("DepthEstimator", "Depth range (dequantized): min=$minVal, max=$maxVal")

        return DepthResult(
            rawDepthArray = rawDepth
        )
    }

    fun close() {
        interpreter.close()
    }
}