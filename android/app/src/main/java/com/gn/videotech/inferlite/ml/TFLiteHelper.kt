/*
 * MIT License
 *
 * Copyright (c) 2025 Fabricio Batista Narcizo, Elizabete Munzlinger, Sai Narsi Reddy Donthi Reddy,
 * and Shan Ahmed Shaffi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package com.gn.videotech.inferlite.ml

import android.app.Application
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.gn.videotech.inferlite.data.DetectionResult
import com.gn.videotech.inferlite.data.Keypoint
import com.gn.videotech.inferlite.data.classNameMapping
import com.gn.videotech.inferlite.utils.BitmapUtility
import com.gn.videotech.inferlite.utils.resized
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate

/**
 * A helper class for managing TFLite-based object detection inference.
 *
 * This class initializes and handles a TensorFlow Lite model for object detection, loads the model
 * from assets, prepares input tensors, and manages bitmap preprocessing.
 *
 * @param application The application context used for accessing assets and other system resources.
 */
class TFLiteHelper(private val application: Application) {

    /**
     * The TensorFlow Lite interpreter for the model. Null until initialized.
     */
    private var interpreter: Interpreter? = null

    /**
     * The selected model type for inference.
     */
    private var selectedModel = ModelType.default

    /**
     * The dimensions (height, width, channels) of the input tensor.
     */
    private var inputTensorHWC = IntArray(0)

    /**
     * The input tensor used to store preprocessed image data.
     */
    private var inputTensor: TensorImage? = null

    /**
     * The input buffer for the model.
     */
    private var inputBuffer: TensorBuffer? = null

    /**
     * The output tensor for bounding box predictions.
     */
    private var outputTensors: Map<String, TensorBuffer> = emptyMap()

    /**
     * Utility object for bitmap-to-buffer preprocessing operations.
     */
    private val bitmapUtility = BitmapUtility()

    companion object {

        /**
         * Names of the output layers.
         */
        private val OUTPUT_LAYERS = arrayOf(
            arrayOf("output0"),
            arrayOf("output0"),
            arrayOf("output0")
        )

    }

    /**
     * Loads the selected TensorFlow Lite model and prepares the interpreter for inference.
     *
     * This method configures the [Interpreter.Options] based on the provided runtime character,
     * loads the model from the assets using memory-mapped I/O, and initializes the input tensor
     * and image buffer. If the model cannot be loaded or the input tensor is invalid, it returns
     * `false`.
     *
     * @param runtimeChar A character indicating the desired runtime:
     *  - `'G'` for GPU
     *  - `'N'` for NNAPI
     *  - any other character defaults to CPU
     * @param selectedModel The [ModelType] to be loaded, including its file path and metadata.
     *
     * @return `true` if the model was successfully loaded and prepared; `false` otherwise.
     */
    fun loadModel(runtimeChar: Char, selectedModel: ModelType): Boolean {
        this.selectedModel = selectedModel
        disposeInterpreter()

        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            configureDelegate(this, runtimeChar, compatList)
        }

        val model = loadModelFromAssets(options) ?: return false

        // Input tensor configuration.
        val inputTensor = model.getInputTensor(0)
        val inputShape = inputTensor.shape() ?: return false
        val inputType = inputTensor.dataType() ?: return false

        inputTensorHWC = inputShape
        this.inputTensor = TensorImage(inputType)
        inputBuffer = TensorBuffer.createFixedSize(inputShape, inputType)

        // Output tensors configuration (assumes 2 outputs).
        outputTensors = (0 until model.outputTensorCount).associate { i ->
            val outputTensor = model.getOutputTensor(i)
            val name = outputTensor.name() ?: return false
            val shape = outputTensor.shape() ?: return false
            val type = outputTensor.dataType() ?: return false
            name to TensorBuffer.createFixedSize(shape, type)
        }

        interpreter = model

        return true
    }

    /**
     * Configures the TensorFlow Lite [Interpreter.Options] with the appropriate delegate or CPU
     * fallback.
     *
     * This method checks whether the device supports hardware acceleration via [CompatibilityList].
     * If supported, it adds the selected delegate (GPU or NNAPI) based on the [runtimeChar].
     * Otherwise, it configures the interpreter to run on CPU with 4 threads.
     *
     * @param options The [Interpreter.Options] instance to configure.
     * @param runtimeChar The desired runtime:
     *  - `'G'` for GPU delegate
     *  - `'N'` for NNAPI delegate
     *  - Any other character defaults to CPU with 4 threads
     * @param compatList The device compatibility list used to determine delegate support.
     */
    private fun configureDelegate(
        options: Interpreter.Options,
        runtimeChar: Char,
        compatList: CompatibilityList
    ) {
        if (compatList.isDelegateSupportedOnThisDevice)
            when (runtimeChar) {
                'G' -> options.addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
                'N' -> options.addDelegate(NnApiDelegate())
                else -> options.setNumThreads(4)
            }
        else
            options.setNumThreads(4)
    }

    /**
     * Loads a TensorFlow Lite model from the app's assets and returns an [Interpreter] instance.
     *
     * The method uses memory mapping for efficient model loading and configures the interpreter
     * with the provided [options]. If the model cannot be loaded, it logs the error and returns
     * `null`.
     *
     * @param options The [Interpreter.Options] used to configure the TensorFlow Lite interpreter
     * (e.g., delegates, threads).
     *
     * @return A [Interpreter] instance ready for inference, or `null` if loading fails.
     */
    private fun loadModelFromAssets(options: Interpreter.Options): Interpreter? = try {
        val modelPath = selectedModel.filePath
        application.assets.openFd(modelPath).use { assetFd ->
            val modelBuffer = assetFd.createInputStream().channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
            Interpreter(modelBuffer, options)
        }
    } catch (e: Exception) {
        Log.e("TFLiteHelper", "Failed to load model from assets: ${selectedModel.filePath}", e)
        null
    }

    /**
     * Runs object detection inference on the given [Bitmap] and returns the filtered results.
     *
     * This method preprocesses the image, runs the model, extracts bounding boxes and class scores,
     * determines the best class for each detection, scales bounding boxes to the image dimensions,
     * and applies Non-Maximum Suppression (NMS) to remove redundant results.
     *
     * @param bitmap The input image on which to perform inference.
     * @param threshold The minimum confidence score required to keep a detection (default is 0.5).
     *
     * @return A list of [DetectionResult]s with label, confidence, and bounding box.
     */
    fun inference(bitmap: Bitmap, threshold: Float = 0.5f): List<DetectionResult> {
        runModel(bitmap) ?: return emptyList()

        val outputLayers = getOutputLayers()
        val scaleX = bitmap.width.toFloat() / getInputWidth()
        val scaleY = bitmap.height.toFloat() / getInputHeight()
        val rectFormat = selectedModel.rectFormat
        val classNameMap = getClassNameMapping()

        val results = mutableListOf<DetectionResult>()

        if (outputLayers.size == 1) {
            val output = outputTensors[outputLayers[0]] ?: return emptyList()
            val data = output.floatArray
            val shape = output.shape

            // Case 1: Sequential output [1, num_detections, 6] (e.g., [x1, y1, x2, y2, score, class])
            if (shape.size == 3 && shape[2] >= 6 && shape[1] > shape[2]) {
                val numDetections = shape[1]
                val numValues = shape[2]

                for (i in 0 until numDetections) {
                    val offset = i * numValues
                    // We assume the format is [box(4), score, class] or [box(4), class, score]
                    // Common TFLite models use [box(4), score, class]
                    val score = data[offset + 4]

                    if (score >= threshold) {
                        val classIndex = data[offset + 5].toInt()
                        val boxData = floatArrayOf(
                            data[offset],
                            data[offset + 1],
                            data[offset + 2],
                            data[offset + 3]
                        )
                        Log.d("TFLiteHelper", "Detection $i: score=$score, class=$classIndex, box=${boxData.contentToString()}")
                        val box = createRectF(boxData, 0, scaleX, scaleY, rectFormat)
                        val label = classNameMap[classIndex] ?: "Unknown"

                        val keypoints = mutableListOf<Keypoint>()
                        if (numValues > 6) {
                            val numKeypoints = (numValues - 6) / 3
                            for (k in 0 until numKeypoints) {
                                val kOffset = offset + 6 + k * 3
                                keypoints.add(
                                    Keypoint(
                                        data[kOffset] * scaleX,
                                        data[kOffset + 1] * scaleY,
                                        data[kOffset + 2]
                                    )
                                )
                            }
                        }

                        results.add(DetectionResult(label, score, box, keypoints))
                    }
                }
            } 
            // Case 2: Transposed output [1, num_values, num_anchors] (e.g., YOLOv8 style [1, 14, 8400])
            else if (shape.size == 3 && shape[1] < shape[2]) {
                val numValues = shape[1]
                val numAnchors = shape[2]

                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    var bestClass = -1
                    for (c in 4 until numValues) {
                        val score = data[c * numAnchors + i]
                        if (score > maxScore) {
                            maxScore = score
                            bestClass = c - 4
                        }
                    }

                    if (maxScore >= threshold) {
                        val boxData = floatArrayOf(
                            data[0 * numAnchors + i],
                            data[1 * numAnchors + i],
                            data[2 * numAnchors + i],
                            data[3 * numAnchors + i]
                        )
                        val box = createRectF(boxData, 0, scaleX, scaleY, rectFormat)
                        val label = classNameMap[bestClass] ?: "Unknown"
                        results.add(DetectionResult(label, maxScore, box))
                    }
                }
            }
        }
        // For other output tensors bboxes, scores, as shown by fabricio.
        else {
            val boxes = outputTensors[outputLayers[0]] ?: return emptyList()
            val classes = outputTensors[outputLayers[1]] ?: return emptyList()

            val boxArray = boxes.floatArray
            val classArray = classes.floatArray

            val numDetections = classes.shape[1]
            val numClasses = classes.shape[2]
            val numCorners = boxes.shape[2]

            for (i in 0 until numDetections) {
                val boxOffset = i * numCorners
                val classOffset = i * numClasses

                val box = createRectF(boxArray, boxOffset, scaleX, scaleY, rectFormat)
                val (bestIndex, maxScore) = getBestClass(classArray, classOffset, numClasses)

                if (maxScore >= threshold) {
                    val label = classNameMap[bestIndex] ?: "Unknown"
                    results.add(DetectionResult(label, maxScore, box))
                }
            }
        }

        return applyNMS(results)
    }

    /**
     * Preprocesses the input [Bitmap] and runs inference using the loaded TFLite model.
     *
     * The image is resized to match the model’s input dimensions, converted to a float buffer,
     * and fed into the interpreter. Output tensors are collected into a map by index. If the frame
     * is black, or prerequisites are missing, it returns `null`.
     *
     * @param bitmap The input image to process.
     *
     * @return A map of output tensor indices to their corresponding buffers, or `null` if
     *      inference fails.
     */
    private fun runModel(bitmap: Bitmap): Map<Int, Any>? {
        val resized = bitmap.resized(getInputWidth())
        if (
            resized.width != getInputWidth() || resized.height != getInputHeight() ||
            inputTensor == null || interpreter == null
        ) return null

        return runCatching {
            Log.d("TFLiteHelper", "Input shape: ${inputTensorHWC.contentToString()}")
            bitmapUtility.convertBitmapToBuffer(resized)
            val inputFloats = if (inputTensorHWC[1] == 3) {
                bitmapUtility.bufferToFloatsNCHW()
            } else {
                bitmapUtility.bufferToFloatsRGB()
            }

            // Skip black frames.
            if (bitmapUtility.isBufferBlack()) return null

            inputBuffer?.loadArray(inputFloats)
            val inputs = arrayOf(inputBuffer?.buffer)
            val outputs = outputTensors.entries.withIndex().associate { (index, entry) ->
                index to entry.value.buffer
            }

            interpreter?.runForMultipleInputsOutputs(inputs, outputs)

            outputs
        }.onFailure {
            Log.e("TFLiteHelper", "Inference error", it)
        }.getOrNull()
    }

    /**
     * Creates a [RectF] representing a bounding box from a float array, applying scaling and format conversion.
     *
     * Supports two formats:
     * - `"center"`: (cx, cy, width, height) — the box is defined by center coordinates and size.
     * - `"corner"`: (left, top, right, bottom) — the box is defined directly by corner coordinates.
     *
     * The resulting rectangle is scaled using [scaleX] and [scaleY] to match image dimensions.
     *
     * @param boxArray The array containing bounding box values.
     * @param boxOffset The starting index of the box in the array.
     * @param scaleX The horizontal scaling factor.
     * @param scaleY The vertical scaling factor.
     * @param rectFormat The box format: either `"center"` (default) or `"corner"`.
     *
     * @return A scaled [RectF] representing the bounding box.
     *
     * @throws IllegalArgumentException if the format is unsupported.
     */
    private fun createRectF(
        boxArray: FloatArray,
        boxOffset: Int,
        scaleX: Float,
        scaleY: Float,
        rectFormat: String = "center" // "center" or "corner"
    ): RectF {
        return when {
            rectFormat.equals("center", ignoreCase = true) -> {
                val cx = boxArray[boxOffset]
                val cy = boxArray[boxOffset + 1]
                val w = boxArray[boxOffset + 2]
                val h = boxArray[boxOffset + 3]

                RectF(
                    (cx - w / 2f) * scaleX,
                    (cy - h / 2f) * scaleY,
                    (cx + w / 2f) * scaleX,
                    (cy + h / 2f) * scaleY
                )
            }

            rectFormat.equals("corner", ignoreCase = true) -> {
                val x1 = boxArray[boxOffset]
                val y1 = boxArray[boxOffset + 1]
                val x2 = boxArray[boxOffset + 2]
                val y2 = boxArray[boxOffset + 3]

                RectF(x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY)
            }

            else -> error("Unsupported format: '$rectFormat'. Use 'center' or 'corner'.")
        }
    }

    /**
     * Finds the class with the highest confidence score starting from the given offset.
     *
     * This method scans a segment of the class score array and identifies the index (class ID)
     * with the maximum value, which represents the most likely predicted class.
     *
     * @param data The full array of class confidence scores output by the model.
     * @param offset The starting index of the class scores for a specific detection.
     * @param numClasses The number of classes in the model's output.
     *
     * @return A [Pair] where the first element is the index of the best class, and the second is
     *         its corresponding confidence score.
     */
    private fun getBestClass(data: FloatArray, offset: Int, numClasses: Int): Pair<Int, Float> {
        var best = -1
        var max = -Float.MAX_VALUE
        for (i in 0 until numClasses) {
            val score = data[offset + i]
            if (score > max) {
                best = i
                max = score
            }
        }
        return best to max
    }

    fun floatArrayToTensorImage(
        floatArray: FloatArray,
        height: Int,
        width: Int,
        channels: Int
    ): TensorImage {
        // Create a TensorBuffer with the same shape
        val shape = intArrayOf(1, height, width, channels)
        val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.FLOAT32)
        tensorBuffer.loadArray(floatArray)

        // Load the TensorBuffer into a TensorImage
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(tensorBuffer)

        return tensorImage
    }

    /**
     * Applies Non-Maximum Suppression (NMS) to remove redundant overlapping detections.
     *
     * Detections are first grouped by label. Within each group, they are sorted by confidence, and
     * overlapping boxes (based on Intersection-over-Union) are removed to keep only the most
     * confident prediction per overlapping region.
     *
     * @param detections A list of [DetectionResult]s returned from raw model output.
     * @param iouThreshold The IoU threshold above which overlapping boxes are suppressed (default
     *                     is 0.2).
     *
     * @return A filtered list of [DetectionResult]s with reduced redundancy.
     */
    private fun applyNMS(detections: List<DetectionResult>,
                         iouThreshold: Float = 0.2f): List<DetectionResult> {
        return detections.groupBy { it.label }.flatMap { (_, group) ->
            val sorted = group.sortedByDescending { it.confidence }.toMutableList()
            val final = mutableListOf<DetectionResult>()
            while (sorted.isNotEmpty()) {
                val top = sorted.removeAt(0)
                final.add(top)
                sorted.removeAll { computeIoU(top.boundingBox, it.boundingBox) > iouThreshold }
            }
            final
        }
    }

    /**
     * Computes the Intersection-over-Union (IoU) between two bounding boxes.
     *
     * IoU is a metric used to evaluate the overlap between two rectangular regions. It is defined
     * as the area of their intersection divided by the area of their union. A higher IoU indicates
     * a greater overlap.
     *
     * @param a The first bounding box.
     * @param b The second bounding box.
     *
     * @return The IoU score between the two rectangles, ranging from 0.0 to 1.0.
     */
    private fun computeIoU(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    /**
     * Retrieves the width of the model's expected input tensor.
     *
     * @return The width dimension, or `0` if the tensor shape is not properly initialized.
     */
    private fun getInputWidth(): Int {
        if (inputTensorHWC.size < 4) return 0
        return if (inputTensorHWC[1] == 3) inputTensorHWC[3] else inputTensorHWC[2]
    }

    /**
     * Retrieves the height of the model's expected input tensor.
     *
     * @return The height dimension, or `0` if the tensor shape is not properly initialized.
     */
    private fun getInputHeight(): Int {
        if (inputTensorHWC.size < 4) return 0
        return if (inputTensorHWC[1] == 3) inputTensorHWC[2] else inputTensorHWC[1]
    }

    /**
     * Returns the output layer names associated with the currently selected model.
     *
     * This uses the [selectedModel]'s [ModelType.index] property to select the correct set of
     * output layer names from the predefined [OUTPUT_LAYERS] list.
     *
     * @return An array of output layer names for the selected model.
     */
    private fun getOutputLayers() = OUTPUT_LAYERS[selectedModel.index]

    /**
     * Returns the class index-to-label mapping for the currently selected model.
     *
     * This uses the [selectedModel]'s [ModelType.index] to access the corresponding class name map
     * from [classNameMapping], enabling label decoding after inference.
     *
     * @return A map of class indices to human-readable class labels.
     */
    private fun getClassNameMapping() = classNameMapping[selectedModel.index]

    /**
     * Releases resources associated with the current TensorFlow Lite model.
     *
     * This method safely releases the underlying native resources and clears all references to the
     * model, input tensor, and input map to avoid memory leaks.
     */
    private fun disposeInterpreter() {
        interpreter?.close()
        interpreter = null
        inputTensorHWC = IntArray(0)
        inputTensor = null
        inputBuffer = null
        outputTensors = emptyMap()
    }

}
