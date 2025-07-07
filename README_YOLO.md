# YOLOv5n Object Detection Integration

This document provides setup and usage instructions for the YOLOv5n object detection integration in the Qt BR Booth application.

## Overview

The YOLOv5n integration adds real-time object detection capabilities to the Qt BR Booth application using ONNX Runtime. It supports:

- Pre-trained YOLOv5n models (80 COCO classes)
- Real-time inference with configurable thresholds
- Qt-friendly API (QImage input/output)
- Bounding box visualization
- Thread-safe operation

## Prerequisites

### 1. ONNX Runtime

Download and install ONNX Runtime from the official releases:
- **URL**: https://github.com/microsoft/onnxruntime/releases
- **Recommended version**: 1.15.0 or later

#### Windows Installation:
1. Download the Windows release (e.g., `onnxruntime-win-x64-1.15.0.zip`)
2. Extract to `C:\onnxruntime` (or update the path in `qt-brbooth.pro`)
3. Ensure the following directory structure:
   ```
   C:\onnxruntime\
   ├── include\
   │   ├── onnxruntime_cxx_api.h
   │   └── ...
   └── lib\
       ├── onnxruntime.lib
       └── onnxruntime.dll
   ```

#### Linux Installation:
1. Download the Linux release (e.g., `onnxruntime-linux-x64-1.15.0.tgz`)
2. Extract to `/usr/local/onnxruntime` (or update the path in `qt-brbooth.pro`)
3. Add to library path: `export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH`

### 2. YOLOv5n Model

Download a pre-trained YOLOv5n ONNX model:

#### Option 1: Download from Ultralytics
```bash
# Install ultralytics
pip install ultralytics

# Export YOLOv5n to ONNX format
python -c "from ultralytics import YOLO; YOLO('yolov5n.pt').export(format='onnx')"
```

#### Option 2: Download pre-converted model
- Download `yolov5n.onnx` from the [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases)
- Or use any compatible YOLOv5n ONNX model with input size 640x640

### 3. OpenCV

OpenCV is already configured in the project. Ensure you have OpenCV 4.x installed with the following modules:
- core
- imgproc
- dnn (for NMS support)
- imgcodecs
- highgui

## Building

### 1. Configure Project

Update the paths in `qt-brbooth.pro` if needed:

```qmake
# Update ONNX Runtime path
ONNXRUNTIME_ROOT_PATH = /path/to/your/onnxruntime

# Update OpenCV paths if needed
OPENCV_INSTALL_DIR = /path/to/your/opencv
```

### 2. Build Main Application

```bash
qmake qt-brbooth.pro
make
```

### 3. Build Example Application

```bash
cd examples
qmake yolo_example.pro
make
```

## Usage

### Basic Usage in Code

```cpp
#include "src/yolo/yolov5detector.h"

// Create detector
YOLOv5Detector detector;

// Initialize with model
if (detector.initialize("path/to/yolov5n.onnx")) {
    // Set parameters
    detector.setConfidenceThreshold(0.5f);
    detector.setNmsThreshold(0.4f);
    
    // Load image
    QImage image("test_image.jpg");
    
    // Detect objects
    QVector<Detection> detections = detector.detectObjects(image);
    
    // Draw bounding boxes
    QImage result = detector.drawBoundingBoxes(image, detections);
    
    // Process results
    for (const Detection& detection : detections) {
        qDebug() << "Detected:" << detection.className 
                 << "confidence:" << detection.confidence;
    }
}
```

### Example Application

The example application provides a complete GUI for testing object detection:

1. **Run the example**: `./yolo_example`
2. **Load model**: Click "Load YOLO Model" and select your `.onnx` file
3. **Load image**: Click "Load Image" and select a test image
4. **Adjust parameters**: Use sliders to adjust confidence and NMS thresholds
5. **Detect objects**: Click "Detect Objects" to run inference

### Integration with BR Booth

To integrate object detection into the main BR Booth application:

1. **Include the header**:
   ```cpp
   #include "src/yolo/yolov5detector.h"
   ```

2. **Add detector to your class**:
   ```cpp
   private:
       YOLOv5Detector* m_objectDetector;
   ```

3. **Initialize in constructor**:
   ```cpp
   m_objectDetector = new YOLOv5Detector(this);
   m_objectDetector->initialize("models/yolov5n.onnx");
   ```

4. **Use in capture pipeline**:
   ```cpp
   // In your image processing code
   QVector<Detection> objects = m_objectDetector->detectObjects(capturedImage);
   if (!objects.isEmpty()) {
       // Add object detection overlay
       QImage imageWithDetections = m_objectDetector->drawBoundingBoxes(capturedImage, objects);
       // Display or save imageWithDetections
   }
   ```

## Configuration Options

### Detection Parameters

```cpp
// Confidence threshold (0.0 - 1.0)
detector.setConfidenceThreshold(0.5f);  // Default: 0.5

// NMS threshold (0.0 - 1.0) 
detector.setNmsThreshold(0.4f);         // Default: 0.4
```

### Model Information

```cpp
// Get model input size
QSize inputSize = detector.getModelInputSize();  // Usually 640x640

// Check if detector is ready
bool ready = detector.isInitialized();

// Get supported classes
QVector<QString> classes = YOLOv5Detector::getCocoClassNames();
```

### Thread Safety

The detector is thread-safe and can be used from multiple threads:

```cpp
// Connect to signals for async operation
connect(&detector, &YOLOv5Detector::detectionCompleted,
        this, &MyClass::onDetectionCompleted);
connect(&detector, &YOLOv5Detector::errorOccurred,
        this, &MyClass::onDetectionError);
```

## Supported Object Classes

The YOLOv5n model supports 80 COCO object classes:

**People & Animals**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Objects**: chair, couch, potted plant, bed, dining table, toilet, tv, laptop, cell phone, book, clock, etc.

See `YOLOv5Detector::getCocoClassNames()` for the complete list.

## Performance Tips

1. **Model Selection**:
   - YOLOv5n: Fastest, lowest accuracy
   - YOLOv5s: Balanced speed/accuracy
   - YOLOv5m/l/x: Higher accuracy, slower

2. **Input Resolution**:
   - 640x640: Standard, good balance
   - 416x416: Faster inference
   - 1280x1280: Higher accuracy, slower

3. **Optimization**:
   - Use GPU inference if available (requires ONNX Runtime GPU package)
   - Adjust confidence threshold based on use case
   - Consider model quantization for faster inference

## Troubleshooting

### Common Issues

1. **"ONNX Runtime not available"**:
   - Ensure ONNX Runtime is installed
   - Check path in `qt-brbooth.pro`
   - Verify `ONNXRUNTIME_AVAILABLE` is defined

2. **Model loading fails**:
   - Verify model file exists and is valid ONNX format
   - Check model input shape is compatible (expects [1, 3, 640, 640])
   - Ensure model is YOLOv5 format, not YOLOv8 or other variants

3. **No detections**:
   - Lower confidence threshold
   - Check input image format and size
   - Verify model is trained on similar data

4. **Poor performance**:
   - Check if running on CPU vs GPU
   - Reduce input resolution
   - Use YOLOv5n instead of larger models

### Debug Information

Enable debug output:
```cpp
// The detector automatically logs debug information
// Check console output for model loading and inference details
```

### Memory Usage

Monitor memory usage for long-running applications:
```cpp
// The detector manages memory automatically
// However, for video processing, consider frame rate limits
```

## License

This integration uses:
- **ONNX Runtime**: MIT License
- **YOLOv5**: GPL-3.0 License
- **OpenCV**: Apache 2.0 License

Ensure compliance with all relevant licenses in your application.

## Contributing

To contribute improvements:

1. Follow existing code style
2. Add unit tests for new features
3. Update documentation
4. Ensure cross-platform compatibility

## Support

For issues related to:
- **ONNX Runtime**: Check [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- **YOLOv5 Models**: Check [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **Integration**: Check project issues or create new issue

## Version History

- **v1.0**: Initial YOLOv5n integration
  - Basic object detection
  - Qt integration
  - Example application
  - Documentation