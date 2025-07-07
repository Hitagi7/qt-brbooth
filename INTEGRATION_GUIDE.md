# YOLOv5 Integration Quick Start Guide

This guide shows how to quickly integrate the YOLOv5 object detector into your existing Qt BR Booth application.

## Basic Integration Steps

### 1. Include the Header

Add to your source file where you want to use object detection:

```cpp
#include "src/yolo/yolov5detector.h"
```

### 2. Add to Your Class

In your class header (e.g., `capture.h`):

```cpp
private:
    YOLOv5Detector* m_objectDetector;
    bool m_detectionEnabled;
```

### 3. Initialize in Constructor

In your class constructor (e.g., `capture.cpp`):

```cpp
Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , m_objectDetector(new YOLOv5Detector(this))
    , m_detectionEnabled(false)
{
    // ... existing code ...
    
    // Connect YOLO signals
    connect(m_objectDetector, &YOLOv5Detector::detectionCompleted,
            this, &Capture::onDetectionCompleted);
    connect(m_objectDetector, &YOLOv5Detector::errorOccurred,
            this, &Capture::onDetectionError);
    
    // Try to initialize with default model
    if (QFile::exists("models/yolov5n.onnx")) {
        if (m_objectDetector->initialize("models/yolov5n.onnx")) {
            m_detectionEnabled = true;
            qDebug() << "YOLO detector initialized successfully";
        }
    }
}
```

### 4. Modify Image Capture

Update your image capture method to include object detection:

```cpp
void Capture::performImageCapture()
{
    cv::Mat frameToCapture;
    
    if (cap.read(frameToCapture)) {
        cv::flip(frameToCapture, frameToCapture, 1);
        QImage capturedImageQ = cvMatToQImage(frameToCapture);
        
        if (!capturedImageQ.isNull()) {
            m_capturedImage = QPixmap::fromImage(capturedImageQ);
            
            // Add object detection if enabled
            if (m_detectionEnabled && m_objectDetector->isInitialized()) {
                detectObjectsInImage(capturedImageQ);
            }
            
            emit imageCaptured(m_capturedImage);
        }
    }
    
    emit showFinalOutputPage();
}
```

### 5. Add Detection Method

Add a method to handle object detection:

```cpp
void Capture::detectObjectsInImage(const QImage& image)
{
    // Run detection asynchronously
    QVector<Detection> detections = m_objectDetector->detectObjects(image);
    
    if (!detections.isEmpty()) {
        // Create image with bounding boxes
        QImage imageWithBoxes = m_objectDetector->drawBoundingBoxes(image, detections);
        QPixmap pixmapWithBoxes = QPixmap::fromImage(imageWithBoxes);
        
        // You can choose to replace the original or keep both
        m_capturedImage = pixmapWithBoxes;  // Replace with detection overlay
        
        // Log detections
        qDebug() << QString("Detected %1 objects:").arg(detections.size());
        for (const Detection& detection : detections) {
            qDebug() << QString("  - %1: %2%").arg(detection.className).arg(detection.confidence * 100, 0, 'f', 1);
        }
    }
}
```

### 6. Add Signal Handlers

Add these methods to handle YOLO signals:

```cpp
void Capture::onDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs)
{
    qDebug() << QString("Object detection completed in %1ms, found %2 objects")
                .arg(processingTimeMs).arg(detections.size());
}

void Capture::onDetectionError(const QString& error)
{
    qWarning() << "Object detection error:" << error;
    m_detectionEnabled = false;  // Disable on error
}
```

### 7. Update Header File

Add these declarations to your header file:

```cpp
private slots:
    void onDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs);
    void onDetectionError(const QString& error);

private:
    void detectObjectsInImage(const QImage& image);
    YOLOv5Detector* m_objectDetector;
    bool m_detectionEnabled;
```

## Optional: Add UI Controls

### Toggle Detection

Add a checkbox or button to enable/disable detection:

```cpp
// In your UI setup
QPushButton* toggleDetectionButton = new QPushButton("Toggle Object Detection");
connect(toggleDetectionButton, &QPushButton::clicked, [this]() {
    m_detectionEnabled = !m_detectionEnabled;
    qDebug() << "Object detection" << (m_detectionEnabled ? "enabled" : "disabled");
});
```

### Adjust Parameters

Add sliders for confidence and NMS thresholds:

```cpp
// Confidence threshold slider
QSlider* confidenceSlider = new QSlider(Qt::Horizontal);
confidenceSlider->setRange(10, 90);
confidenceSlider->setValue(50);
connect(confidenceSlider, &QSlider::valueChanged, [this](int value) {
    float threshold = value / 100.0f;
    m_objectDetector->setConfidenceThreshold(threshold);
    qDebug() << "Confidence threshold:" << threshold;
});

// NMS threshold slider
QSlider* nmsSlider = new QSlider(Qt::Horizontal);
nmsSlider->setRange(10, 80);
nmsSlider->setValue(40);
connect(nmsSlider, &QSlider::valueChanged, [this](int value) {
    float threshold = value / 100.0f;
    m_objectDetector->setNmsThreshold(threshold);
    qDebug() << "NMS threshold:" << threshold;
});
```

## Model Setup

1. **Download Model**: Get `yolov5n.onnx` from [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases)
2. **Place Model**: Put it in a `models/` directory in your project
3. **Update Path**: Change the path in initialization code if needed

## Testing

1. **Build**: Make sure your project builds without errors
2. **Run**: Start the application
3. **Check Console**: Look for YOLO initialization messages
4. **Capture**: Take a photo and check if objects are detected
5. **Logs**: Monitor debug output for detection results

## Troubleshooting

### No Detection Results
- Check model file exists and is valid ONNX format
- Lower confidence threshold (try 0.3 or 0.2)
- Verify image contains recognizable objects

### Poor Performance
- Use YOLOv5n (smallest, fastest model)
- Reduce image resolution before detection
- Run detection in separate thread for real-time use

### Build Errors
- Ensure ONNX Runtime is properly installed
- Check paths in `qt-brbooth.pro`
- Verify `ONNXRUNTIME_AVAILABLE` is defined

This minimal integration adds object detection to your existing capture workflow while preserving all current functionality.