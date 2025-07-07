#ifndef YOLOV5DETECTOR_H
#define YOLOV5DETECTOR_H

#include <QObject>
#include <QImage>
#include <QPixmap>
#include <QString>
#include <QVector>
#include <QRectF>
#include <QMutex>
#include <memory>

#include <opencv2/opencv.hpp>

// Forward declarations for ONNX Runtime
namespace Ort {
    class Env;
    class Session;
    class Value;
    class MemoryInfo;
}

/**
 * @brief Structure representing a detected object
 */
struct Detection {
    QRectF boundingBox;     // Bounding box in original image coordinates
    float confidence;       // Detection confidence [0.0-1.0]
    int classId;           // Class ID (0-79 for COCO classes)
    QString className;     // Human-readable class name
    
    Detection() : confidence(0.0f), classId(-1) {}
    Detection(const QRectF& bbox, float conf, int id, const QString& name)
        : boundingBox(bbox), confidence(conf), classId(id), className(name) {}
};

/**
 * @brief YOLOv5n Object Detector using ONNX Runtime
 * 
 * This class provides real-time object detection capabilities using a pre-trained
 * YOLOv5n model. It supports Qt-friendly input/output with QImage and provides
 * configurable confidence thresholds and bounding box visualization.
 */
class YOLOv5Detector : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param parent Parent QObject
     */
    explicit YOLOv5Detector(QObject *parent = nullptr);
    
    /**
     * @brief Destructor
     */
    ~YOLOv5Detector();

    /**
     * @brief Initialize the detector with a model file
     * @param modelPath Path to the ONNX model file
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const QString& modelPath);

    /**
     * @brief Check if the detector is initialized and ready for inference
     * @return true if ready, false otherwise
     */
    bool isInitialized() const;

    /**
     * @brief Set the confidence threshold for detections
     * @param threshold Confidence threshold [0.0-1.0], default is 0.5
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief Get the current confidence threshold
     * @return Current confidence threshold
     */
    float getConfidenceThreshold() const;

    /**
     * @brief Set the NMS (Non-Maximum Suppression) threshold
     * @param threshold NMS threshold [0.0-1.0], default is 0.4
     */
    void setNmsThreshold(float threshold);

    /**
     * @brief Get the current NMS threshold
     * @return Current NMS threshold
     */
    float getNmsThreshold() const;

    /**
     * @brief Detect objects in a QImage
     * @param image Input image
     * @return Vector of detected objects
     */
    QVector<Detection> detectObjects(const QImage& image);

    /**
     * @brief Detect objects in a cv::Mat
     * @param image Input image (BGR format)
     * @return Vector of detected objects
     */
    QVector<Detection> detectObjects(const cv::Mat& image);

    /**
     * @brief Draw bounding boxes on an image
     * @param image Input image
     * @param detections Vector of detections to draw
     * @return Image with bounding boxes drawn
     */
    QImage drawBoundingBoxes(const QImage& image, const QVector<Detection>& detections);

    /**
     * @brief Draw bounding boxes on a cv::Mat
     * @param image Input image (will be modified in-place)
     * @param detections Vector of detections to draw
     */
    void drawBoundingBoxes(cv::Mat& image, const QVector<Detection>& detections);

    /**
     * @brief Get the list of COCO class names
     * @return Vector of class names (80 COCO classes)
     */
    static QVector<QString> getCocoClassNames();

    /**
     * @brief Get input image size expected by the model
     * @return QSize representing model input dimensions (default 640x640)
     */
    QSize getModelInputSize() const;

signals:
    /**
     * @brief Emitted when detection is completed
     * @param detections Vector of detected objects
     * @param processingTimeMs Processing time in milliseconds
     */
    void detectionCompleted(const QVector<Detection>& detections, int processingTimeMs);

    /**
     * @brief Emitted when an error occurs
     * @param error Error message
     */
    void errorOccurred(const QString& error);

private slots:
    void onDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs);

private:
    // Private member variables
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;
    
    bool m_initialized;
    float m_confidenceThreshold;
    float m_nmsThreshold;
    QSize m_inputSize;
    
    mutable QMutex m_mutex;  // Thread safety
    
    // Input/output tensor names
    QString m_inputName;
    QString m_outputName;
    
    // COCO class names
    static QVector<QString> s_cocoClassNames;
    
    // Private helper methods
    /**
     * @brief Preprocess image for YOLO input
     * @param image Input image
     * @return Preprocessed cv::Mat ready for inference
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief Post-process YOLO output to extract detections
     * @param output Raw model output
     * @param originalSize Original image size for coordinate conversion
     * @return Vector of detections
     */
    QVector<Detection> postprocessOutput(const cv::Mat& output, const QSize& originalSize);
    
    /**
     * @brief Apply Non-Maximum Suppression to filter overlapping detections
     * @param detections Input detections
     * @return Filtered detections
     */
    QVector<Detection> applyNMS(const QVector<Detection>& detections);
    
    /**
     * @brief Convert QImage to cv::Mat
     * @param qimage Input QImage
     * @return Converted cv::Mat
     */
    cv::Mat qImageToCvMat(const QImage& qimage);
    
    /**
     * @brief Convert cv::Mat to QImage
     * @param mat Input cv::Mat
     * @return Converted QImage
     */
    QImage cvMatToQImage(const cv::Mat& mat);
    
    /**
     * @brief Initialize COCO class names
     */
    static void initializeCocoClassNames();
};

#endif // YOLOV5DETECTOR_H