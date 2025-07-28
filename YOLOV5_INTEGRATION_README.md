# YOLOv5 Integration for Qt BRBooth

This document explains how to use the **C++ YOLOv5 detector** integration in your Qt BRBooth project for real-time person detection.

## 🎯 Overview

The project now includes a **C++ YOLOv5 detector** that can detect persons in real-time using OpenCV DNN. This replaces the Python-based YOLO detection and provides:

- ✅ **No Python Dependencies** - Eliminates all DLL/OpenCV Python issues
- ✅ **Better Performance** - Faster than Python, no process overhead  
- ✅ **More Reliable** - Direct OpenCV integration, no external processes
- ✅ **Easier Deployment** - Single executable, no Python installation required

## 📁 Files Added

1. **`yolov5_detector.h`** - Header file for the YOLOv5 detector class
2. **`yolov5_detector.cpp`** - Implementation of the YOLOv5 detector
3. **`download_yolo_model.py`** - Script to download the YOLOv5n ONNX model
4. **`YOLOV5_INTEGRATION_README.md`** - This documentation

## 🚀 Quick Start

### Step 1: Download the YOLOv5n Model

**Manual Download (Recommended):**
1. Go to: https://github.com/ultralytics/assets/releases
2. Download `yolov5n.pt` from the latest release
3. Create a `models/` directory in your project
4. Place the file in `models/` and rename it to `yolov5n.onnx`

**Or use PowerShell:**
```powershell
mkdir models
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.pt" -OutFile "models/yolov5n.onnx"
```

### Step 2: Build the Project

```bash
qmake
make
```

### Step 3: Run the Application

The application will automatically:
1. Load the YOLOv5 model from `models/yolov5n.onnx`
2. Use C++ YOLOv5 detection for person detection
3. Show a warning if the model is not found

## 🔧 How It Works

### C++ YOLOv5 Detection (Primary)

When the ONNX model is available, the application uses the C++ YOLOv5 detector:

1. **Model Loading**: Loads `yolov5n.onnx` using OpenCV DNN
2. **Frame Processing**: Each camera frame goes through YOLOv5 network
3. **Person Detection**: Detects persons with confidence > 30%
4. **Signal Emission**: Emits `personDetectedInFrame()` when person found

### Pure C++ Implementation

This implementation uses only C++ and OpenCV DNN. No Python dependencies are required.

## ⚙️ Configuration

### Detection Parameters

You can adjust the detection parameters in `capture.cpp`:

```cpp
yoloDetector = new YoloV5Detector(modelPath.toStdString(), 0.3f, 0.4f);
//                                                                    ^  ^
//                                                                    |  |
//                                                              NMS threshold
//                                                      Confidence threshold
```

- **Confidence threshold**: 0.3f (30%) - Minimum confidence for detection
- **NMS threshold**: 0.4f (40%) - Non-maximum suppression threshold

### Model Location

The model is expected at: `[application_directory]/models/yolov5n.onnx`

## 📊 Performance

### Advantages of C++ YOLOv5

| Aspect | C++ YOLOv5 | Python YOLOv5 |
|--------|-------------|---------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Reliability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Memory Usage** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Expected Performance

- **Detection Speed**: ~30-60 FPS depending on hardware
- **Memory Usage**: ~100-200MB for the model
- **CPU Usage**: Moderate (depends on model size and hardware)

## 🛠️ API Reference

### YoloV5Detector Class

```cpp
class YoloV5Detector {
public:
    // Constructor with model path and thresholds
    YoloV5Detector(const std::string& modelPath, float confThreshold = 0.4, float nmsThreshold = 0.5);
    
    // Detect all objects in image
    std::vector<Detection> detect(const cv::Mat& image);
    
    // Detect person specifically (optimized for photo booth)
    bool detectPerson(const cv::Mat& image, float minConfidence = 0.3f);
    
    // Draw detection boxes on image
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
    
    // Check if model is loaded
    bool isModelLoaded() const;
};
```

### Detection Structure

```cpp
struct Detection {
    cv::Rect box;           // Bounding box coordinates
    float confidence;        // Detection confidence (0-1)
    int class_id;           // Class ID from COCO dataset
    std::string class_name; // Class name (e.g., "person")
};
```

## 🔍 Usage Examples

### Basic Person Detection

```cpp
// Initialize detector
YoloV5Detector detector("models/yolov5n.onnx");

// Detect person in frame
cv::Mat frame = // ... your camera frame
bool personDetected = detector.detectPerson(frame, 0.3f);

if (personDetected) {
    // Enable capture button or trigger photo
    ui->capture->setEnabled(true);
}
```

### Full Object Detection

```cpp
// Get all detections
std::vector<Detection> detections = detector.detect(frame);

// Process each detection
for (const auto& det : detections) {
    if (det.class_name == "person") {
        // Handle person detection
        qDebug() << "Person detected with confidence:" << det.confidence;
    }
}
```

## 🐛 Troubleshooting

### Model Not Found

**Error**: "YOLOv5 model not found at: [path]"

**Solution**:
1. Run `python download_yolo_model.py`
2. Ensure the model is in `models/yolov5n.onnx`
3. Check file permissions

### OpenCV DNN Issues

**Error**: "OpenCV error loading model"

**Solution**:
1. Ensure your OpenCV build includes DNN module
2. Check that the ONNX model file is not corrupted
3. Try re-downloading the model

### Model Requirements

The application requires:
- `yolov5n.onnx` file in the `models/` directory
- OpenCV with DNN module enabled

## 🎨 Customization

### Training Your Own Model

To train a custom YOLOv5 model for person detection:

1. **Prepare Dataset**: Collect images with person annotations
2. **Train YOLOv5**: Use the official YOLOv5 repository
3. **Export to ONNX**: Use `python export.py --weights best.pt --include onnx`
4. **Replace Model**: Put your `best.onnx` in the `models/` directory

### Adding More Classes

To detect additional objects:

1. Modify `classNames` in `yolov5_detector.cpp`
2. Update the detection logic in `capture.cpp`
3. Train a model with your desired classes

### Performance Optimization

For better performance:

```cpp
// Process every 3rd frame for better performance
if (frameCount % 3 == 0) {
    detectPersonInImage(tempImagePath);
}

// Use lower confidence threshold for more sensitive detection
yoloDetector = new YoloV5Detector(modelPath.toStdString(), 0.2f, 0.3f);
```

## 📈 Integration with Photo Booth

### Person Detection Workflow

1. **Camera Feed**: Real-time camera feed is processed
2. **Person Detection**: YOLOv5 detects persons in each frame
3. **UI Updates**: When person detected, enable capture button
4. **Photo Capture**: User can take photo when person is detected

### Signal/Slot Integration

```cpp
// In capture.cpp
if (personDetected) {
    emit personDetectedInFrame();  // Signal for UI updates
    ui->capture->setEnabled(true); // Enable capture button
}
```

## 📄 License

The YOLOv5 model is provided under the same license as the original YOLOv5 project. Please refer to the Ultralytics repository for licensing details.

## 🤝 Support

If you encounter any issues:

1. Check the console output for error messages
2. Ensure the model file is properly downloaded
3. Verify OpenCV DNN module is available
4. Check that your Qt project builds successfully

---

**🎉 Congratulations!** Your Qt BRBooth project now has reliable, fast person detection using C++ YOLOv5! 