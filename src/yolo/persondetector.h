#ifndef PERSONDETECTOR_H
#define PERSONDETECTOR_H

#include <QString>
#include <QImage>
#include <QRect>
#include <QVector>
#include <opencv2/opencv.hpp>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

/**
 * @brief Detection result for a person
 */
struct PersonDetection {
    QRect boundingBox;      // Bounding box in original image coordinates
    float confidence;       // Detection confidence score (0.0 - 1.0)
    int classId;           // Should always be 0 for person
    
    PersonDetection(const QRect& box = QRect(), float conf = 0.0f, int id = 0)
        : boundingBox(box), confidence(conf), classId(id) {}
};

/**
 * @brief YOLOv5n Person Detector Class
 * 
 * This class provides person detection functionality using YOLOv5n model
 * with ONNX Runtime. It filters detections to only return people (COCO class 0).
 */
class PersonDetector
{
public:
    /**
     * @brief Constructor
     * @param modelPath Path to the YOLOv5n ONNX model file
     * @param confidenceThreshold Minimum confidence threshold for detections (default: 0.5)
     * @param nmsThreshold Non-maximum suppression threshold (default: 0.4)
     */
    PersonDetector(const QString& modelPath = "", 
                   float confidenceThreshold = 0.5f, 
                   float nmsThreshold = 0.4f);
    
    /**
     * @brief Destructor
     */
    ~PersonDetector();
    
    /**
     * @brief Initialize the detector with a model file
     * @param modelPath Path to the YOLOv5n ONNX model file
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const QString& modelPath);
    
    /**
     * @brief Check if the detector is initialized and ready
     * @return true if ready, false otherwise
     */
    bool isInitialized() const;
    
    /**
     * @brief Detect people in an image
     * @param image Input QImage to process
     * @return Vector of PersonDetection results
     */
    QVector<PersonDetection> detectPersons(const QImage& image);
    
    /**
     * @brief Detect people in an OpenCV Mat
     * @param image Input cv::Mat to process
     * @return Vector of PersonDetection results
     */
    QVector<PersonDetection> detectPersons(const cv::Mat& image);
    
    /**
     * @brief Count the number of people detected in an image
     * @param image Input QImage to process
     * @return Number of people detected
     */
    int countPersons(const QImage& image);
    
    /**
     * @brief Count the number of people detected in an OpenCV Mat
     * @param image Input cv::Mat to process
     * @return Number of people detected
     */
    int countPersons(const cv::Mat& image);
    
    /**
     * @brief Draw bounding boxes on image for detected people
     * @param image Input/output QImage to draw on
     * @param detections Vector of PersonDetection results
     * @param drawConfidence Whether to draw confidence scores (default: true)
     * @return Modified QImage with bounding boxes drawn
     */
    QImage drawDetections(const QImage& image, 
                         const QVector<PersonDetection>& detections,
                         bool drawConfidence = true);
    
    /**
     * @brief Draw bounding boxes on OpenCV Mat for detected people
     * @param image Input/output cv::Mat to draw on
     * @param detections Vector of PersonDetection results
     * @param drawConfidence Whether to draw confidence scores (default: true)
     */
    void drawDetections(cv::Mat& image, 
                       const QVector<PersonDetection>& detections,
                       bool drawConfidence = true);
    
    // Getters and setters
    float getConfidenceThreshold() const { return m_confidenceThreshold; }
    void setConfidenceThreshold(float threshold) { m_confidenceThreshold = threshold; }
    
    float getNmsThreshold() const { return m_nmsThreshold; }
    void setNmsThreshold(float threshold) { m_nmsThreshold = threshold; }
    
    QString getModelPath() const { return m_modelPath; }
    
    /**
     * @brief Get input size used by the model (always 640x640 for YOLOv5n)
     * @return QSize representing model input dimensions
     */
    QSize getInputSize() const { return QSize(640, 640); }

private:
    QString m_modelPath;
    float m_confidenceThreshold;
    float m_nmsThreshold;
    bool m_initialized;
    
#ifdef HAVE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::SessionOptions> m_sessionOptions;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<int64_t> m_inputShape;
    std::vector<int64_t> m_outputShape;
#endif
    
    /**
     * @brief Preprocess image for YOLO input
     * @param image Input cv::Mat
     * @return Preprocessed cv::Mat ready for inference
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief Post-process YOLO output to extract person detections
     * @param output Raw model output
     * @param originalSize Original image size for coordinate scaling
     * @return Vector of PersonDetection results
     */
    QVector<PersonDetection> postprocessOutput(const cv::Mat& output, const QSize& originalSize);
    
    /**
     * @brief Apply Non-Maximum Suppression to remove duplicate detections
     * @param detections Input detections
     * @return Filtered detections after NMS
     */
    QVector<PersonDetection> applyNMS(const QVector<PersonDetection>& detections);
    
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
};

#endif // PERSONDETECTOR_H