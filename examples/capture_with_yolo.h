/*
 * YOLO Integration Example for Qt BR Booth Application
 * 
 * This file demonstrates how to integrate the YOLOv5Detector
 * into the existing capture workflow with minimal changes.
 */

#ifndef CAPTURE_WITH_YOLO_H
#define CAPTURE_WITH_YOLO_H

#include "capture.h"
#include "src/yolo/yolov5detector.h"

/**
 * @brief Extended Capture class with optional YOLO object detection
 * 
 * This demonstrates how to add object detection to the existing capture system
 * without breaking existing functionality.
 */
class CaptureWithYOLO : public Capture
{
    Q_OBJECT

public:
    explicit CaptureWithYOLO(QWidget *parent = nullptr);
    ~CaptureWithYOLO();

    /**
     * @brief Enable or disable object detection
     * @param enabled Whether to enable object detection
     */
    void setObjectDetectionEnabled(bool enabled);

    /**
     * @brief Check if object detection is enabled
     * @return true if enabled, false otherwise
     */
    bool isObjectDetectionEnabled() const;

    /**
     * @brief Initialize YOLO detector with model file
     * @param modelPath Path to YOLO ONNX model
     * @return true if successful, false otherwise
     */
    bool initializeYOLO(const QString& modelPath);

    /**
     * @brief Set object detection parameters
     * @param confidence Confidence threshold [0.0-1.0]
     * @param nms NMS threshold [0.0-1.0]
     */
    void setDetectionParameters(float confidence, float nms);

protected:
    /**
     * @brief Override to add object detection to image capture
     */
    void performImageCaptureWithDetection();

private slots:
    /**
     * @brief Handle YOLO detection completion
     * @param detections Vector of detected objects
     * @param processingTimeMs Processing time in milliseconds
     */
    void onObjectDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs);

    /**
     * @brief Handle YOLO detector errors
     * @param error Error message
     */
    void onDetectorError(const QString& error);

signals:
    /**
     * @brief Emitted when objects are detected in captured image
     * @param detections Vector of detected objects
     * @param imageWithDetections Image with bounding boxes drawn
     */
    void objectsDetected(const QVector<Detection>& detections, const QPixmap& imageWithDetections);

private:
    YOLOv5Detector* m_yoloDetector;
    bool m_detectionEnabled;
    QPixmap m_lastCapturedImage;
    
    void setupYOLODetector();
};

/**
 * Example usage in main application:
 * 
 * // In BRBooth constructor, replace:
 * // capturePage = new Capture;
 * // with:
 * capturePage = new CaptureWithYOLO;
 * 
 * // Initialize YOLO (optional)
 * if (QFile::exists("models/yolov5n.onnx")) {
 *     capturePage->initializeYOLO("models/yolov5n.onnx");
 *     capturePage->setObjectDetectionEnabled(true);
 *     capturePage->setDetectionParameters(0.5f, 0.4f);
 * }
 * 
 * // Connect to object detection signals
 * connect(capturePage, &CaptureWithYOLO::objectsDetected,
 *         this, &BRBooth::onObjectsDetected);
 */

#endif // CAPTURE_WITH_YOLO_H

/*
 * Implementation Notes:
 * 
 * 1. This approach extends the existing Capture class rather than modifying it,
 *    preserving backward compatibility.
 * 
 * 2. Object detection is optional and can be enabled/disabled at runtime.
 * 
 * 3. The detection runs asynchronously to avoid blocking the UI.
 * 
 * 4. Results are provided through Qt signals for easy integration.
 * 
 * 5. Existing capture functionality remains unchanged when detection is disabled.
 * 
 * To implement this integration:
 * 
 * 1. Create the CaptureWithYOLO class implementation
 * 2. Update BRBooth to use CaptureWithYOLO instead of Capture
 * 3. Add UI controls for enabling/configuring object detection
 * 4. Handle the objectsDetected signal to display or save results
 * 
 * This design allows the YOLO integration to be added incrementally without
 * breaking existing functionality.
 */