package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import android.util.Size
import org.tensorflow.lite.Interpreter
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

    init {
        try {
            // — your existing asset-loading code —
            val assetFileDescriptor = context.assets.openFd(modelPath)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            // instantiate the interpreter
            interpreter = Interpreter(modelBuffer)

            // now your debug logs
            val inT = interpreter.getInputTensor(0)
            Log.d("FastDepth", "Input 0 shape=${inT.shape().toList()}  dtype=${inT.dataType()}")
            val outT = interpreter.getOutputTensor(0)
            Log.d("FastDepth", "Output 0 shape=${outT.shape().toList()} dtype=${outT.dataType()}")
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
        // Preprocess input img: resize bitmap to expected size
        val modelInputSize = Size(256, 256)  // 256, 256 change width and height for midas - 224, 224 fastDepth
        val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, modelInputSize.width, modelInputSize.height, true)

        // Prepare input & Output array (adjust dimension based on model requirement)
        val input = Array(1) {
            Array(modelInputSize.height) {
                Array(modelInputSize.width) {
                    FloatArray(3)
                }
            }
        }

        // Example normalization: convert bitmap pixels to float values [0,1]
        for (y in 0 until modelInputSize.height) {
            for (x in 0 until modelInputSize.width) {
                val pixel = resizedBitmap.getPixel(x, y)
                input[0][y][x][0] = Color.red(pixel) / 255.0f
                input[0][y][x][1] = Color.green(pixel) / 255.0f
                input[0][y][x][2] = Color.blue(pixel) / 255.0f
            }
        }

        // Output depth map, assume model output is [1, H, W, 1]
        val output = Array(1) {
            Array(modelInputSize.height) {
                Array(modelInputSize.width) {
                    FloatArray(1)
                }
            }
        }
        interpreter.run(input, output)

        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE

        val rawDepth = Array(modelInputSize.height) { FloatArray(modelInputSize.width) }

        for (y in 0 until modelInputSize.height) {
            for (x in 0 until modelInputSize.width) {
                val v = output[0][y][x][0]
                rawDepth[y][x] = v
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
            }
        }
        Log.d("DepthEstimator", "minVal=$minVal, maxVal=$maxVal")

        return DepthResult(

            rawDepthArray = rawDepth
        )
    }

    fun close() {
        interpreter.close()
    }
}