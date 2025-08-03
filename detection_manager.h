#ifndef DETECTION_MANAGER_H
#define DETECTION_MANAGER_H

#include <QObject>
#include <QProcess>
#include <QTimer>
#include <QMutex>
#include <QList>
#include <opencv2/opencv.hpp>
#include "common_types.h"

// Forward declarations 
class Capture;
class SimplePersonDetector;
class OptimizedPersonDetector;

/**
 * @brief Manages all detection processing, reducing complexity in Capture class
 * 
 * This class centralizes:
 * - YOLO detection (Python subprocess)
 * - Simple person detection (HOG)
 * - Optimized detection (C++ ONNX)
 * - Detection result management
 * - Performance monitoring
 */
class DetectionManager : public QObject {
    Q_OBJECT

public:
    enum DetectionMode {
        YOLO_PYTHON,        // Original Python YOLO subprocess
        SIMPLE_CPP,         // HOG-based C++ detector
        OPTIMIZED_ONNX      // Optimized C++ ONNX detector
    };

    explicit DetectionManager(QObject* parent = nullptr);
    ~DetectionManager();

    // Configuration
    void setDetectionMode(DetectionMode mode);
    void setUseCppDetector(bool use);
    void setUseOptimizedDetector(bool use);
    void setShowBoundingBoxes(bool show);
    
    // Detection methods
    void detectPersonInImage(const QString& imagePath);
    void detectPersonsInFrame(const cv::Mat& frame);
    void testDetection();
    
    // Results access
    QList<BoundingBox> getCurrentDetections() const;
    QList<OptimizedDetection> getCurrentOptimizedDetections() const;
    
    // Performance
    double getAverageDetectionTime() const;
    int getCurrentFPS() const;
    void printPerformanceStats();

signals:
    void detectionsReady(const QList<BoundingBox>& detections);
    void optimizedDetectionsReady(const QList<OptimizedDetection>& detections);
    void detectionProcessingFinished();
    void detectionError(const QString& error);

private slots:
    void handleYoloOutput();
    void handleYoloError();
    void handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleYoloErrorOccurred(QProcess::ProcessError error);
    void onOptimizedDetectionsReady(const QList<OptimizedDetection>& detections);
    void onOptimizedProcessingFinished();

private:
    // Detection processors
    QProcess* m_yoloProcess;
    SimplePersonDetector* m_simpleDetector;
    OptimizedPersonDetector* m_optimizedDetector;
    
    // Settings
    DetectionMode m_currentMode;
    bool m_useCppDetector;
    bool m_useOptimizedDetector;
    bool m_showBoundingBoxes;
    bool m_isProcessingFrame;
    
    // Results storage
    mutable QMutex m_detectionMutex;
    mutable QMutex m_optimizedDetectionMutex;
    QList<BoundingBox> m_currentDetections;
    QList<OptimizedDetection> m_currentOptimizedDetections;
    
    // Performance tracking
    QList<double> m_detectionTimes;
    double m_averageDetectionTime;
    int m_frameCount;
    QTimer m_performanceTimer;
    
    // Internal methods
    void initializeDetectors();
    void processYoloOutput(const QString& output);
    void updatePerformanceStats(double detectionTime);
    void updateDetectionResults(const QList<BoundingBox>& detections);
    void updateOptimizedDetectionResults(const QList<OptimizedDetection>& detections);
    bool isValidForDetection(const cv::Mat& frame) const;
    void clearResults();
};

#endif // DETECTION_MANAGER_H