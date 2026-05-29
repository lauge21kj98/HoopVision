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

/**
 * Enum representing supported TFLite model variants, including their file paths and classification
 * indices.
 *
 * Each model type includes:
 * - [displayName]: A human-readable label for display in UI.
 * - [filePath]: The file name of the model in the assets directory.
index]: An integer index used to select label maps or output formats (e.g., 0 = COCO,
 *            1 = hadRID).
 *
 * This enum supports multiple precision formats (FP32, INT8, INT8+HTP) for both YOLO-NAS and
 * YOLO-hagRID models.
 *
 * @property displayName The name shown to users (e.g., in dropdowns or settings).
 * @property filePath The relative file path to the model within the assets directory.
 * @property rectFormat The format used to represent bounding boxes (e.g., "center" or "corner").
 * @property index The model group index used to look up class mappings or rect formats.
 */
enum class ModelType(
    val displayName: String,
    val filePath: String,
    val rectFormat: String,
    val index: Int
) {
    //FP32("FP32", "player-yolo26s-B-best_float32.tflite", "corner", 1),
    //FP16("FP16", "player-yolo26s-B-best_float16.tflite", "corner", 1),
    //INT8("INT8", "player-yolo26s-B-best_int8.tflite", "corner", 1),

    PLAYER_DETECTOR_YOLO26N_A_32("PLAYER YOLO26N A FP32", "player-detector-yolo26n-A-fp32.tflite", "corner", 0),
    PLAYER_DETECTOR_YOLO26N_A_8("PLAYER YOLO26N A INT8", "player-detector-yolo26n-A-int8.tflite", "corner", 0),

    PLAYER_DETECTOR_YOLO26N_B_32("PLAYER YOLO26N B FP32", "player-detector-yolo26n-B-fp32.tflite", "corner", 1),
    PLAYER_DETECTOR_YOLO26N_B_8("PLAYER YOLO26N B INT8", "player-detector-yolo26n-B-int8.tflite", "corner", 1),
    PLAYER_DETECTOR_YOLO26S_B_32("PLAYER YOLO26S B FP32", "player-detector-yolo26s-B-fp32.tflite", "corner", 1),
    PLAYER_DETECTOR_YOLO26S_B_8("PLAYER YOLO26S B INT8", "player-detector-yolo26s-B-int8.tflite", "corner", 1),

    COURT_DETECTOR_YOLO26N_C_32("COURT YOLO26N (FP32)", "court-detector-yolo26n-C-fp32.tflite", "corner", 2),
    COURT_DETECTOR_YOLO26N_C_8("COURT YOLO26N (INT8)", "court-detector-yolo26n-C-int8.tflite", "corner", 2),
    COURT_DETECTOR_YOLO26S_C_32("COURT YOLO26S (FP32)", "court-detector-yolo26s-C-fp32.tflite", "corner", 2),
    COURT_DETECTOR_YOLO26S_C_8("COURT YOLO26S (INT8)", "court-detector-yolo26s-C-int8.tflite", "corner", 2);

    companion object {

        /**
         * The default model to be used when no specific selection is made.
         */
        val default = PLAYER_DETECTOR_YOLO26N_A_32

        /**
         * Returns a [ModelType] based on a case-insensitive display name match.
         *
         * @param name The display name to search for.
         *
         * @return A matching [ModelType], or `null` if not found.
         */
        fun fromDisplayName(name: String): ModelType? =
            entries.find { it.displayName.equals(name, ignoreCase = true) }

    }

}
