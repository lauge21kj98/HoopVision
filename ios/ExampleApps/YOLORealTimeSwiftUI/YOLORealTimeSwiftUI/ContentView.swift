// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Example Apps of Ultralytics YOLO Package, providing a SwiftUI example for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ContentView demonstrates how to implement real-time object detection using the YOLOCamera
//  SwiftUI component. It shows how to create a full-screen camera view that performs continuous
//  object detection with a specified YOLO model. This example specifically uses the oriented
//  bounding box (OBB) model variant, but can be easily modified to use other model types like
//  detection, segmentation, or pose estimation by changing the task parameter and model name.
//  The view ignores safe areas to provide a full-screen camera experience.

import SwiftUI
import YOLO

/// A SwiftUI view that demonstrates real-time object detection using the YOLOCamera component.
struct ContentView: View {
  @State private var selectedModel = "yolo26n-posefp32"
  
  private var selectedTask: YOLOTask {
    if selectedModel.starts(with: "yolo26n-pose") {
      return .pose
    } else {
      return .detect
    }
  }
  
  var body: some View {
    ZStack {
      YOLOCamera(
        modelPathOrName: selectedModel,
        task: selectedTask,
        cameraPosition: .back
      )
      .id(selectedModel)
      .ignoresSafeArea()
      
      VStack(spacing: 0) {
        // Model selector at the top
        Picker("Model", selection: $selectedModel) {
          Text("yolo26n").tag("yolo26n")
          Text("yolo26x").tag("yolo26x")
          Text("yolo26n-posefp").tag("yolo26n-pose")
          Text("yolo26n-posefp32").tag("yolo26n-posefp32")
        }
        .pickerStyle(.segmented)
        .padding()
        .background(Color.black.opacity(0.7))
        
        Spacer()
      }
    }
  }
}

#Preview {
  ContentView()
}
