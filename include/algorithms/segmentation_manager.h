#ifndef SEGMENTATION_MANAGER_H
#define SEGMENTATION_MANAGER_H

#include <QObject>
#include <QMutex>
#include <QList>
#include <opencv2/opencv.hpp>
#include "algorithms/personsegmentation.h"
#include "core/common_types.h"

// Forward declarations
class Capture;

/**
 * @brief Manages all segmentation processing, reducing complexity in Capture class
 * 
 * This class centralizes:
 * - Person segmentation logic
 * - Performance monitoring
 * - Result caching and management
 */
class SegmentationManager : public QObject {
    Q_OBJECT

public:
    explicit SegmentationManager(QObject* parent = nullptr);
    ~SegmentationManager();

    // Configuration
    void setShowSegmentation(bool show);
    void setConfidenceThreshold(double threshold);
    void setSegmentationMethod(PersonSegmentationProcessor::PerformanceMode mode);
    
    // Processing methods
    void processPersonSegmentation(const cv::Mat& frame, const QList<BoundingBox>& detections);
    void processOptimizedSegmentation(const cv::Mat& frame, const QList<OptimizedDetection>& detections);
    
    // Results access
    QList<SegmentationResult> getCurrentSegmentationResults() const;
    
    // Performance
    double getAverageProcessingTime() const;
    int getCurrentFPS() const;
    
    // Frame management
    void applySegmentationToFrame(cv::Mat& frame) const;
    void saveSegmentedFrame(const QString& filename);

signals:
    void segmentationCompleted(const QList<SegmentationResult>& results);
    void segmentationError(const QString& error);

private slots:
    void onSegmentationProcessingFinished();

private:
    // Processors
    PersonSegmentationProcessor* m_segmentationProcessor;
    
    // Settings
    bool m_showSegmentation;
    double m_confidenceThreshold;
    
    // Results storage
    mutable QMutex m_segmentationMutex;
    QList<SegmentationResult> m_currentSegmentations;
    
    // Performance tracking
    QList<double> m_processingTimes;
    double m_averageProcessingTime;
    int m_frameCount;
    
    // Internal methods
    void updatePerformanceStats(double processingTime);
    void clearResults();
    bool isValidForSegmentation(const cv::Mat& frame) const;
};

#endif // SEGMENTATION_MANAGER_H