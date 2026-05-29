# HoopVision

https://github.com/user-attachments/assets/5e135c05-0dc7-463e-88a3-29c716be1c28

## Getting Started

### Model Training and Inference (Google Colab)

The `notebooks/` directory contains Jupyter notebooks for training and running models:

- **Detection Model**: Train or run a basketball and player detection model on GPU
- **Pose Estimation Model**: Train or run a pose estimation model for player keypoint detection
- **Export Models**: Upload models to Qualcomm AI Hub and export to .tflite, profile, quantize to int8. Furthermore, export for iOS apps.
- **Inference on GPU**: Run pre-trained models on GPU for efficient batch processing

All notebooks are designed to run in Google Colab with GPU support for fast model training and inference.

### Android App

The `android/` project is an Android application that runs YOLO detection models in real-time inference using `.tflite` (TensorFlow Lite) model files. The app provides:

- Real-time object detection on mobile devices
- Support for multiple YOLO model variants
- Efficient on-device inference

To use the app import .tflite model file compiled in Qualcomm AI Hub.

**This project is provided by Fabricio Narcizo Batista and modified for YOLO26 model outputs**
Link: https://github.com/fabricionarcizo/InferLite

### iOS App

The `ios/` project is an iOS application built with Swift that runs YOLO models using `.mlpackage` files exported with YOLOv8 ultralytics. Features include:

- Real-time inference using Core ML
- Support for multiple YOLO tasks (detection, segmentation, pose estimation, etc.)
- SwiftUI interface for seamless user experience

Open `ios/ExampleApps/YOLORealTimeSwiftUI/` in Xcode and sign project with your apple account to run project.

**This project is provided by YOLO Ultralytics and modified for this project**

Link: https://github.com/ultralytics/yolo-ios-app/tree/main

# Videos
## Detections
Basketball object detection:

https://github.com/user-attachments/assets/3d936aaf-e3d2-4db4-936a-56fafbda9771

## Tracking
Bytetrack player tracking:
IDs are shown as labels. 

https://github.com/user-attachments/assets/628ed3e0-64bd-4b4f-99fa-accc932b1e90


SAM 2 player tracking:
IDs are shown as unique color.

https://github.com/user-attachments/assets/b95ac741-2f61-43d8-a444-b9f11732ff6a

## Court Detection
Court Detections (Raw detections vs Smoothing)

https://github.com/user-attachments/assets/9160912e-7a48-4099-ae71-89340d8fd7b8

## Team Assignment
ByteTrack team assignment:

https://github.com/user-attachments/assets/0a49c28a-7959-4bb4-9947-7b3f92eaf070

SAM 2 team assignment:

https://github.com/user-attachments/assets/c04c2fb0-5554-4182-a3c9-51f8d715eb06


## Pipeline
Full end-to-end pipeline for multi-object tracking, court registration, action recognition, and shot detection map.

https://github.com/user-attachments/assets/5e135c05-0dc7-463e-88a3-29c716be1c28