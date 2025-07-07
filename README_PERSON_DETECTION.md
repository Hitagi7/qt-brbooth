# YOLOv5n Person Detection for Qt Booth Application

This document provides comprehensive setup and usage instructions for integrating YOLOv5n object detection into the Qt booth application, specifically configured for **person detection only**.

## Overview

The PersonDetector class provides real-time person detection capabilities using YOLOv5n model with ONNX Runtime. It filters all detections to return only people (COCO class ID 0), making it perfect for booth applications that need to:

- Count people in front of the booth
- Detect when someone approaches
- Track person movement  
- Provide analytics for booth traffic
- Security/monitoring applications

## Features

- **Person-only detection** - Filters out all other 79 COCO classes
- **Real-time inference** - Optimized for live camera feeds
- **Qt-friendly API** - Works seamlessly with QImage and OpenCV Mat
- **Configurable thresholds** - Adjustable confidence and NMS thresholds
- **Bounding box visualization** - Built-in drawing functions
- **Person counting** - Simple API for counting detected people
- **Booth-optimized** - Designed for typical booth/camera scenarios

## Requirements

### System Dependencies

1. **Qt6** (Core, Widgets, Multimedia)
2. **OpenCV 4.x** with DNN module
3. **ONNX Runtime** (C++ API)
4. **CMake 3.16+**

### Installing ONNX Runtime

#### Ubuntu/Debian
```bash
# Install from package manager (if available)
sudo apt update
sudo apt install libonnxruntime-dev

# OR download prebuilt binaries
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
sudo cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/
sudo ldconfig
```

#### Windows
```bash
# Download from GitHub releases
# Extract to C:/onnxruntime/
# Update CMakeLists.txt paths accordingly
```

#### macOS
```bash
# Using Homebrew (if available)
brew install onnxruntime

# OR download prebuilt binaries from GitHub releases
```

### YOLOv5n Model

Download the pre-trained YOLOv5n model in ONNX format:

```bash
# Download YOLOv5n ONNX model
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx

# Place in your project directory or models folder
mkdir -p models
mv yolov5n.onnx models/
```

## Building the Project

### Using CMake

```bash
# Clone the repository
git clone https://github.com/Hitagi7/qt-brbooth.git
cd qt-brbooth

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Run the example
./person_detection_example
```

### Build Options

If ONNX Runtime is installed in a custom location:

```bash
cmake -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime ..
```

## Usage

### Basic Integration

```cpp
#include "persondetector.h"

// Initialize detector
PersonDetector detector("path/to/yolov5n.onnx", 0.5f, 0.4f);

if (detector.isInitialized()) {
    // Detect people in image
    QVector<PersonDetection> people = detector.detectPersons(qimage);
    
    // Count people
    int personCount = detector.countPersons(qimage);
    
    // Draw detections
    QImage result = detector.drawDetections(qimage, people);
}
```

### Real-time Camera Integration

```cpp
// In your camera capture loop
cv::Mat frame;
if (cap.read(frame)) {
    // Flip for mirror effect (typical for booths)
    cv::flip(frame, frame, 1);
    
    // Detect people
    QVector<PersonDetection> people = detector.detectPersons(frame);
    
    // Draw bounding boxes
    detector.drawDetections(frame, people);
    
    // Display or process results
    qDebug() << "People detected:" << people.size();
}
```

### Booth Application Integration

Add to your existing booth capture system:

```cpp
// In capture.h
#include "persondetector.h"

class Capture : public QWidget {
    // ... existing code ...
private:
    PersonDetector *m_personDetector;
    int m_currentPersonCount;
    
public slots:
    void updatePersonCount();
    
signals:
    void personDetected(int count);
    void personEntered();
    void personLeft();
};

// In capture.cpp
Capture::Capture(QWidget *parent) : QWidget(parent) {
    // ... existing initialization ...
    
    // Initialize person detector
    m_personDetector = new PersonDetector("models/yolov5n.onnx");
    m_currentPersonCount = 0;
}

void Capture::updateCameraFeed() {
    // ... existing camera code ...
    
    if (cap.read(frame)) {
        cv::flip(frame, frame, 1);
        
        // Person detection
        int newPersonCount = m_personDetector->countPersons(frame);
        
        if (newPersonCount != m_currentPersonCount) {
            if (newPersonCount > m_currentPersonCount) {
                emit personEntered();
            } else {
                emit personLeft();
            }
            m_currentPersonCount = newPersonCount;
            emit personDetected(newPersonCount);
        }
        
        // ... rest of camera processing ...
    }
}
```

## API Reference

### PersonDetector Class

#### Constructor
```cpp
PersonDetector(const QString& modelPath = "", 
               float confidenceThreshold = 0.5f, 
               float nmsThreshold = 0.4f);
```

#### Main Methods
```cpp
// Initialization
bool initialize(const QString& modelPath);
bool isInitialized() const;

// Detection
QVector<PersonDetection> detectPersons(const QImage& image);
QVector<PersonDetection> detectPersons(const cv::Mat& image);

// Counting
int countPersons(const QImage& image);
int countPersons(const cv::Mat& image);

// Visualization
QImage drawDetections(const QImage& image, 
                     const QVector<PersonDetection>& detections,
                     bool drawConfidence = true);

void drawDetections(cv::Mat& image, 
                   const QVector<PersonDetection>& detections,
                   bool drawConfidence = true);

// Configuration
void setConfidenceThreshold(float threshold);
void setNmsThreshold(float threshold);
QSize getInputSize() const; // Returns QSize(640, 640)
```

### PersonDetection Structure
```cpp
struct PersonDetection {
    QRect boundingBox;      // Bounding box coordinates
    float confidence;       // Detection confidence (0.0-1.0)
    int classId;           // Always 0 for person
};
```

## Configuration

### Confidence Threshold
- **Default**: 0.5 (50%)
- **Range**: 0.0 - 1.0
- **Booth recommendation**: 0.3-0.7 depending on lighting conditions
- **Higher values**: Fewer false positives, may miss some people
- **Lower values**: More detections, may include false positives

### NMS Threshold
- **Default**: 0.4
- **Range**: 0.0 - 1.0  
- **Purpose**: Removes duplicate detections of the same person
- **Booth recommendation**: 0.4-0.6

### Model Input Size
- **Fixed**: 640x640 pixels
- **Format**: RGB
- **Preprocessing**: Automatic resize with padding to maintain aspect ratio

## Performance Optimization

### For Booth Applications

1. **Frame Rate**: Limit detection to 10-15 FPS for smooth operation
```cpp
// Use timer to limit detection frequency
QTimer *detectionTimer = new QTimer();
detectionTimer->setInterval(66); // ~15 FPS
```

2. **Resolution**: Use moderate camera resolution (640x480 or 1280x720)

3. **Threading**: Run detection in separate thread to avoid UI blocking
```cpp
// Run detection in worker thread
QtConcurrent::run([this, frame]() {
    auto detections = m_detector->detectPersons(frame);
    emit detectionsReady(detections);
});
```

4. **Memory Management**: Reuse detection objects when possible

### Expected Performance
- **YOLOv5n on CPU**: 50-200ms per frame (depending on hardware)
- **YOLOv5n on GPU**: 10-30ms per frame (with CUDA-enabled ONNX Runtime)
- **Memory usage**: ~100MB for model + runtime

## Troubleshooting

### Common Issues

1. **"ONNX Runtime not available"**
   - Install ONNX Runtime development libraries
   - Check CMake configuration
   - Verify library paths

2. **"Failed to load model"**
   - Check model file path
   - Ensure model is YOLOv5n ONNX format
   - Verify file permissions

3. **Poor detection accuracy**
   - Adjust confidence threshold
   - Check lighting conditions
   - Ensure camera is focused properly
   - Verify model is trained for your use case

4. **Slow performance**
   - Reduce camera resolution
   - Lower detection frame rate
   - Use GPU acceleration if available
   - Consider YOLOv5s model for better speed/accuracy trade-off

### Debug Information

Enable debug output:
```cpp
#include <QLoggingCategory>
Q_LOGGING_CATEGORY(personDetector, "PersonDetector")

// In your code
qDebug() << "Detections:" << detections.size();
```

## Example Applications

### 1. People Counter
Simple application that counts people entering/leaving booth area.

### 2. Booth Activation
Automatically start booth sequence when person detected.

### 3. Analytics Dashboard
Track booth traffic patterns throughout the day.

### 4. Security Monitor
Alert when multiple people or no people detected for extended periods.

## Model Information

### YOLOv5n Specifications
- **Input**: 640x640x3 RGB
- **Output**: Detections with 85 values each (4 bbox + 1 objectness + 80 classes)
- **Person Class**: Index 0 in COCO dataset
- **Model Size**: ~14MB
- **Speed**: ~45 FPS on modern CPU

### Supported Formats
- **Input**: ONNX format only
- **Alternative models**: YOLOv5s, YOLOv5m, YOLOv5l (larger, more accurate)

## License and Attribution

This implementation uses:
- **YOLOv5**: Ultralytics (GPL-3.0 License)
- **ONNX Runtime**: Microsoft (MIT License)
- **OpenCV**: Intel/OpenCV Foundation (Apache 2.0 License)

## Support

For issues and questions:
1. Check this documentation
2. Review example code
3. Check GitHub issues
4. Create new issue with detailed description

## Future Enhancements

Planned features:
- [ ] Person tracking across frames
- [ ] Age/gender estimation
- [ ] Pose detection integration
- [ ] Multiple camera support
- [ ] Cloud model deployment
- [ ] Real-time analytics dashboard