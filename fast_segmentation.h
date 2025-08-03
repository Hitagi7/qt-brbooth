#ifndef FAST_SEGMENTATION_H
#define FAST_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <QList>
#include <QDebug>
#include <chrono>

// Forward declaration
struct BoundingBox;
struct OptimizedDetection;

struct FastSegmentationResult {
    cv::Mat mask;           // Binary mask where 255 = person, 0 = background
    cv::Mat segmentedImage; // Image with transparent background
    cv::Rect boundingBox;   // Original bounding box
    double confidence;      // Confidence of the detection
    bool isValid;           // Whether segmentation was successful
    
    FastSegmentationResult() : confidence(0.0), isValid(false) {}
};

class FastSegmentationProcessor {
public:
    enum SegmentationMethod {
        WATERSHED_FAST,      // Watershed + morphology (fastest)
        SUPERPIXEL_SLIC,     // SLIC superpixels (balanced)
        EDGE_BASED,          // Edge detection + flood fill (very fast)
        YOLO_MASK_DIRECT     // Use YOLO segmentation masks directly (fastest when available)
    };
    
    FastSegmentationProcessor();
    ~FastSegmentationProcessor();
    
    // Main segmentation function for real-time performance
    QList<FastSegmentationResult> segmentPersonsFast(const cv::Mat& image, 
                                                    const QList<BoundingBox>& detections,
                                                    double minConfidence = 0.5);
    
    // Segmentation using optimized detections with masks
    QList<FastSegmentationResult> segmentFromOptimizedDetections(const cv::Mat& image,
                                                                const QList<OptimizedDetection>& detections);
    
    // Create combined segmented image with transparent background
    cv::Mat createCombinedSegmentation(const cv::Mat& originalImage, 
                                     const QList<FastSegmentationResult>& results);
    
    // Configuration
    void setSegmentationMethod(SegmentationMethod method) { m_segmentationMethod = method; }
    void setMaxProcessingTime(double maxTime) { m_maxProcessingTime = maxTime; }
    
    // Performance monitoring
    double getAverageProcessingTime() const { return m_avgProcessingTime; }
    int getCurrentFPS() const { return m_currentFPS; }
    
private:
    // Core segmentation methods
    FastSegmentationResult performWatershedSegmentation(const cv::Mat& image, 
                                                      const cv::Rect& bbox, 
                                                      double confidence);
    
    FastSegmentationResult performEdgeBasedSegmentation(const cv::Mat& image, 
                                                       const cv::Rect& bbox, 
                                                       double confidence);
    
    FastSegmentationResult performSuperpixelSegmentation(const cv::Mat& image, 
                                                        const cv::Rect& bbox, 
                                                        double confidence);
    
    FastSegmentationResult processYOLOMask(const cv::Mat& image,
                                          const cv::Mat& mask,
                                          const cv::Rect& bbox,
                                          double confidence);
    
    // Utility methods
    cv::Mat preprocessROI(const cv::Mat& image, const cv::Rect& bbox);
    cv::Mat postprocessMask(const cv::Mat& mask, const cv::Size& originalSize);
    cv::Mat createTransparentBackground(const cv::Mat& originalImage, const cv::Mat& mask);
    bool isValidMask(const cv::Mat& mask, double minAreaRatio = 0.01);
    
    // Performance optimization
    void startTiming();
    void endTiming();
    void updatePerformanceStats();
    
    // Morphological operations for cleanup
    cv::Mat cleanupMask(const cv::Mat& mask);
    cv::Mat fillHoles(const cv::Mat& mask);
    cv::Mat smoothEdges(const cv::Mat& mask, int kernelSize = 3);
    
    // Members
    SegmentationMethod m_segmentationMethod;
    double m_maxProcessingTime;  // Target max processing time in ms
    
    // Performance monitoring
    std::chrono::steady_clock::time_point m_timingStart;
    QList<double> m_processingTimes;
    double m_avgProcessingTime;
    int m_currentFPS;
    int m_frameCount;
    
    // OpenCV optimization - using basic watershed instead of ximgproc
    cv::Mat m_watershedKernel;
    
    // Constants
    static constexpr int MAX_TIMING_SAMPLES = 30;
    static constexpr double MIN_MASK_AREA_RATIO = 0.005;  // Minimum 0.5% of bounding box
    static constexpr int WATERSHED_KERNEL_SIZE = 3;
    static constexpr int MORPH_KERNEL_SIZE = 2;
};

#endif // FAST_SEGMENTATION_H