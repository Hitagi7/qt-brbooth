#ifndef OPTIMIZED_DETECTOR_H
#define OPTIMIZED_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <QList>
#include <QMutex>
#include <QDebug>
#include <QThread>
#include <chrono>
#include "common_types.h"

// Include common types for shared structs

class OptimizedPersonDetector : public QObject {
    Q_OBJECT
    
public:
    enum ModelType {
        YOLO_DETECTION,     // Standard detection (fast)
        YOLO_SEGMENTATION,  // Segmentation (slightly slower but gives masks)
        TENSORRT_OPTIMIZED  // TensorRT optimized (fastest)
    };
    
    enum PerformanceMode {
        REAL_TIME,      // Maximum speed, good quality
        BALANCED,       // Balance of speed and quality  
        HIGH_QUALITY    // Best quality, acceptable speed
    };
    
    OptimizedPersonDetector(QObject* parent = nullptr);
    ~OptimizedPersonDetector();
    
    // Initialize with specific model and mode
    bool initialize(ModelType modelType = YOLO_SEGMENTATION, 
                   PerformanceMode perfMode = REAL_TIME);
    
    // Main detection function - returns both boxes and masks
    QList<OptimizedDetection> detectPersons(const cv::Mat& image, 
                                           double confThreshold = 0.5,
                                           double nmsThreshold = 0.4);
    
    // Async detection for real-time processing
    void detectPersonsAsync(const cv::Mat& image, 
                           double confThreshold = 0.5,
                           double nmsThreshold = 0.4);
    
    // Check if detector is ready
    bool isInitialized() const { return m_initialized; }
    bool isProcessing() const { return m_processing; }
    
    // Performance monitoring
    double getAverageInferenceTime() const { return m_avgInferenceTime; }
    int getCurrentFPS() const { return m_currentFPS; }
    
    // Configuration
    void setInputSize(int width, int height);
    void setConfidenceThreshold(double threshold) { m_defaultConfThreshold = threshold; }
    void setNMSThreshold(double threshold) { m_defaultNMSThreshold = threshold; }
    
signals:
    void detectionsReady(const QList<OptimizedDetection>& detections);
    void processingFinished();
    void errorOccurred(const QString& error);
    
// Removed slots that were used for async processing
    
private:
    // Core detection methods
    QList<OptimizedDetection> runYOLODetection(const cv::Mat& image, 
                                              double confThreshold, 
                                              double nmsThreshold);
    QList<OptimizedDetection> runYOLOSegmentation(const cv::Mat& image, 
                                                 double confThreshold, 
                                                 double nmsThreshold);
    
    // Preprocessing and postprocessing
    cv::Mat preprocessImage(const cv::Mat& image);
    QList<OptimizedDetection> postprocessDetections(const std::vector<cv::Mat>& outputs,
                                                   const cv::Mat& originalImage,
                                                   double confThreshold,
                                                   double nmsThreshold);
    QList<OptimizedDetection> postprocessSegmentation(const std::vector<cv::Mat>& outputs,
                                                     const cv::Mat& originalImage,
                                                     double confThreshold,
                                                     double nmsThreshold);
    
    // Utility methods
    void warmupModel();
    void updatePerformanceStats(double inferenceTime);
    cv::Mat extractMask(const cv::Mat& maskProtos, const cv::Mat& maskCoeffs, 
                       const cv::Rect& bbox, const cv::Size& originalSize);
    
    // Members
    cv::dnn::Net m_net;
    ModelType m_modelType;
    PerformanceMode m_perfMode;
    bool m_initialized;
    bool m_processing;
    
    // Model configuration
    cv::Size m_inputSize;
    std::vector<std::string> m_outputNames;
    std::vector<int> m_classIds;
    
    // Performance settings
    double m_defaultConfThreshold;
    double m_defaultNMSThreshold;
    double m_scaleX, m_scaleY;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point m_lastProcessTime;
    QList<double> m_inferenceTimes;
    double m_avgInferenceTime;
    int m_currentFPS;
    int m_frameCount;
    
    // Threading (simplified for synchronous processing)
    mutable QMutex m_mutex;
    
    // Constants
    static constexpr int MAX_TIMING_SAMPLES = 30;
    static constexpr double MIN_CONFIDENCE = 0.1;
    static constexpr double MIN_BOX_AREA = 400.0;  // Minimum bounding box area
};

#endif // OPTIMIZED_DETECTOR_H