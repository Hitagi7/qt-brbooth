#ifndef PERSONSEGMENTATION_H
#define PERSONSEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <QList>
#include <QDebug>

// Forward declaration for BoundingBox
struct BoundingBox;

struct SegmentationResult {
    cv::Mat mask;           // Binary mask where 255 = person, 0 = background
    cv::Mat segmentedImage; // Image with transparent background
    double confidence;      // Confidence of the detection used for segmentation
    cv::Rect boundingBox;   // Original bounding box
    bool isValid;           // Whether segmentation was successful
    
    SegmentationResult() : confidence(0.0), isValid(false) {}
};

class PersonSegmentationProcessor {
public:
    // Performance modes
    enum PerformanceMode {
        HighQuality,    // GrabCut with full iterations (slower, best quality)
        Balanced,       // Reduced GrabCut iterations (medium speed/quality)
        HighSpeed,      // Fast edge-based segmentation (fastest)
        Adaptive        // Automatically switches based on performance
    };

    PersonSegmentationProcessor();
    ~PersonSegmentationProcessor();
    
    // Main segmentation function
    QList<SegmentationResult> segmentPersons(const cv::Mat& image, 
                                            const QList<BoundingBox>& detections,
                                            double minConfidence = 0.7);
    
    // Real-time optimized segmentation
    QList<SegmentationResult> segmentPersonsFast(const cv::Mat& image, 
                                                const QList<BoundingBox>& detections,
                                                double minConfidence = 0.7);
    
    // Create a single segmented image with transparent background
    cv::Mat createTransparentBackground(const cv::Mat& originalImage, 
                                       const SegmentationResult& result);
    
    // Combine multiple segmentations into one image
    cv::Mat combineSegmentations(const cv::Mat& originalImage,
                                const QList<SegmentationResult>& results);
    
    // Performance settings
    void setPerformanceMode(PerformanceMode mode) { m_performanceMode = mode; }
    PerformanceMode getPerformanceMode() const { return m_performanceMode; }
    void setGrabCutIterations(int iterations) { m_grabCutIterations = iterations; }
    void setMorphologyKernelSize(int size) { m_morphKernelSize = size; }
    void setBlurKernelSize(int size) { m_blurKernelSize = size; }
    void setMaxProcessingTime(int milliseconds) { m_maxProcessingTime = milliseconds; }
    
    // Performance monitoring
    double getLastProcessingTime() const { return m_lastProcessingTime; }
    double getAverageProcessingTime() const { return m_averageProcessingTime; }
    
    // Debug functions
    cv::Mat getDebugMask(const SegmentationResult& result);
    void saveDebugImages(const cv::Mat& original, const SegmentationResult& result, const QString& prefix);

private:
    // Core segmentation algorithms
    SegmentationResult performGrabCutSegmentation(const cv::Mat& image, const cv::Rect& bbox, double confidence);
    SegmentationResult performColorBasedSegmentation(const cv::Mat& image, const cv::Rect& bbox, double confidence);
    SegmentationResult performFastEdgeSegmentation(const cv::Mat& image, const cv::Rect& bbox, double confidence);
    SegmentationResult performAdaptiveSegmentation(const cv::Mat& image, const cv::Rect& bbox, double confidence);
    
    // Post-processing methods
    cv::Mat refineMask(const cv::Mat& mask);
    cv::Mat refineMaskFast(const cv::Mat& mask);
    cv::Mat applyMorphology(const cv::Mat& mask);
    cv::Mat smoothMaskEdges(const cv::Mat& mask);
    
    // Validation methods
    bool isValidBoundingBox(const cv::Rect& bbox, const cv::Size& imageSize);
    bool isMaskValid(const cv::Mat& mask, double minAreaRatio = 0.01);
    
    // Performance monitoring
    void startTiming();
    void endTiming();
    void updatePerformanceStats();
    
    // Parameters
    PerformanceMode m_performanceMode;
    int m_grabCutIterations;
    int m_morphKernelSize;
    int m_blurKernelSize;
    double m_minMaskArea;
    int m_maxProcessingTime; // Maximum time per frame in milliseconds
    
    // Performance tracking
    double m_lastProcessingTime;
    double m_averageProcessingTime;
    std::chrono::steady_clock::time_point m_timingStart;
    QList<double> m_processingTimes;
    static const int MAX_TIMING_SAMPLES = 30;
    
    // GrabCut working matrices
    cv::Mat m_backgroundModel;
    cv::Mat m_foregroundModel;
    
    // Fast segmentation caches
    cv::Mat m_previousFrame;
    cv::Mat m_backgroundSubtractor;
};

#endif // PERSONSEGMENTATION_H