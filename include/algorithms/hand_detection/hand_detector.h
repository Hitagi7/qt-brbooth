#ifndef HAND_DETECTOR_H
#define HAND_DETECTOR_H

#include <QObject>
#include <QList>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>

// Consolidated hand detection result
struct HandDetection {
    cv::Rect boundingBox;
    double confidence;
    QString handType;  // "left", "right", "unknown"
    std::vector<cv::Point> landmarks;
    bool isRaised;
    cv::Point palmCenter;
    std::vector<cv::Point> fingerTips;
    cv::Point wristPoint;
    cv::Point middleFingerTip;
    cv::Point indexFingerTip;
    cv::Point thumbTip;
    bool isOpen;
    bool isClosed;
};

class HandDetector : public QObject
{
    Q_OBJECT

public:
    explicit HandDetector(QObject *parent = nullptr);
    ~HandDetector();

    // Initialization
    bool initialize();
    bool isInitialized() const;
    void reset();

    // Main detection
    QList<HandDetection> detect(const cv::Mat& image);
    
    // Configuration
    void setConfidenceThreshold(double threshold);
    double getConfidenceThreshold() const;
    void setShowBoundingBox(bool show);
    bool getShowBoundingBox() const;
    void setPerformanceMode(int mode); // 0=Fast, 1=Balanced, 2=Accurate
    int getPerformanceMode() const;
    
    // Gesture detection
    bool isHandClosed(const std::vector<cv::Point>& contour);
    bool isHandOpen(const std::vector<cv::Point>& contour);
    double calculateHandClosureRatio(const std::vector<cv::Point>& contour);
    bool shouldTriggerCapture();
    void resetGestureState();
    void updateHandState(bool isClosed); // New method for tracking hand state over time
    bool isHandClosedTimerValid() const;
    bool isHandClosedFast(const std::vector<cv::Point>& contour); // Fast detection for poor camera quality

    // ROI tracking (for continuous hand tracking)
    bool hasLock() const;
    cv::Rect getRoi() const;
    void updateRoi(const cv::Mat& frame);

    // Performance monitoring
    double getHandDetectionProcessingTime() const;
    bool isCudaAvailable() const;
    QString getDetectorType() const;
    double getAverageProcessingTime() const;
    double getCurrentFPS() const;
    int getTotalFramesProcessed() const;

signals:
    void detectionCompleted(const QList<HandDetection>& detections);
    void processingTimeUpdated(double milliseconds);
    void cudaError(const QString& error);
    void detectorTypeChanged(const QString& type);

private:
    // Detection methods
    QList<HandDetection> detectHandsByShape(const cv::Mat& image);
    QList<HandDetection> detectHandsByMotion(const cv::Mat& image);
    QList<HandDetection> detectHandsByKeypoints(const cv::Mat& image);
    
    // Fast motion methods (legacy)
    QList<HandDetection> detectHandsByMotionFast(const cv::Mat& image);
    cv::Mat createFastMotionMask(const cv::Mat& gray);
    bool analyzeGestureClosedFast(const cv::Mat& gray, const cv::Rect& roi) const;
    bool analyzeGestureOpenFast(const cv::Mat& gray, const cv::Rect& roi) const;
    void updateRoiFast(const cv::Mat& frame);
    bool acquireRoiFromMotionFast(const cv::Mat& motionMask);
    void trackRoiSimple(const cv::Mat& motionMask);
    
    // Hand gesture detection
    QList<HandDetection> detectHandGestures(const cv::Mat& image);
    cv::Mat createSkinMask(const cv::Mat& image);
    bool isHandShape(const std::vector<cv::Point>& contour, const cv::Mat& image);
    double calculateHandConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image);
    
    // Stricter hand detection methods
    bool isHandShapeStrict(const std::vector<cv::Point>& contour, const cv::Mat& image);
    bool isHandOpenStrict(const std::vector<cv::Point>& contour);
    bool isHandClosedStrict(const std::vector<cv::Point>& contour);
    double calculateHandConfidenceStrict(const std::vector<cv::Point>& contour, const cv::Mat& image);
    
    // Shape analysis
    std::vector<cv::Point> findFingerTips(const std::vector<cv::Point>& contour);
    cv::Point findPalmCenter(const std::vector<cv::Point>& contour);
    
    // Motion detection and tracking
    bool detectMotion(const cv::Mat& image);
    bool acquireRoiFromMotion(const cv::Mat& gray, const cv::Mat& motionMask);
    void trackRoiLK(const cv::Mat& grayPrev, const cv::Mat& grayCurr);
    void updateMotionHistory(const cv::Mat& motionMask);
    bool isMotionStable();
    
    // Gesture analysis
    bool analyzeGestureClosed(const cv::Mat& gray, const cv::Rect& roi) const;
    bool analyzeGestureOpen(const cv::Mat& gray, const cv::Rect& roi) const;
    
    // Image processing
    cv::Mat createEnhancedSkinMask(const cv::Mat& image);
    cv::Mat createMotionMask(const cv::Mat& image);
    void optimizeForPerformance(cv::Mat& image);
    void updateBackgroundModel(const cv::Mat& frame);
    cv::Mat getBackgroundModel();

    // CUDA detection methods
    QList<HandDetection> detectHandsByCudaShape(const cv::cuda::GpuMat& gpuImage);
    QList<HandDetection> detectHandsByCudaMotion(const cv::cuda::GpuMat& gpuImage);
    QList<HandDetection> detectHandsByCudaKeypoints(const cv::cuda::GpuMat& gpuImage);
    
    // CUDA image processing
    cv::cuda::GpuMat createCudaSkinMask(const cv::cuda::GpuMat& gpuImage);
    cv::cuda::GpuMat createCudaMotionMask(const cv::cuda::GpuMat& gpuImage);
    std::vector<std::vector<cv::Point>> findCudaContours(const cv::cuda::GpuMat& gpuMask);
    
    // Hand gesture detection with CUDA
    QList<HandDetection> detectCudaHandGestures(const cv::Mat& image);
    QList<HandDetection> detectCudaHandGesturesOptimized(const cv::Mat& image);
    QList<HandDetection> detectHandGesturesOptimized(const cv::Mat& image);
    bool isCudaHandShape(const std::vector<cv::Point>& contour, const cv::cuda::GpuMat& gpuImage);
    double calculateCudaHandConfidence(const std::vector<cv::Point>& contour, const cv::cuda::GpuMat& gpuImage);
    
    // CUDA shape analysis
    std::vector<cv::Point> findCudaFingerTips(const std::vector<cv::Point>& contour);
    cv::Point findCudaPalmCenter(const std::vector<cv::Point>& contour);
    
    // CUDA motion detection and tracking
    bool detectCudaMotion(const cv::cuda::GpuMat& gpuImage);
    bool acquireCudaRoiFromMotion(const cv::cuda::GpuMat& gpuGray, const cv::cuda::GpuMat& gpuMotionMask);
    void trackCudaRoiLK(const cv::cuda::GpuMat& gpuGrayPrev, const cv::cuda::GpuMat& gpuGrayCurr);
    void updateCudaMotionHistory(const cv::cuda::GpuMat& gpuMotionMask);
    bool isCudaMotionStable();
    
    // CUDA gesture analysis
    bool analyzeCudaGestureClosed(const cv::cuda::GpuMat& gpuGray, const cv::Rect& roi) const;
    bool analyzeCudaGestureOpen(const cv::cuda::GpuMat& gpuGray, const cv::Rect& roi) const;
    
    // CUDA image processing utilities
    cv::cuda::GpuMat convertToCuda(const cv::Mat& cpuImage);
    cv::Mat convertFromCuda(const cv::cuda::GpuMat& gpuImage);
    void applyCudaGaussianBlur(cv::cuda::GpuMat& gpuImage, int kernelSize = 5);
    void applyCudaMorphology(cv::cuda::GpuMat& gpuImage, int operation = cv::MORPH_CLOSE);
    
    // Optimized processing methods
    cv::cuda::GpuMat createCudaSkinMaskOptimized(const cv::cuda::GpuMat& gpuImage);
    std::vector<std::vector<cv::Point>> findCudaContoursOptimized(const cv::cuda::GpuMat& gpuMask);
    bool isCudaHandShapeFast(const std::vector<cv::Point>& contour, const cv::cuda::GpuMat& gpuImage);
    double calculateCudaHandConfidenceFast(const std::vector<cv::Point>& contour, const cv::cuda::GpuMat& gpuImage);
    std::vector<cv::Point> findCudaFingerTipsFast(const std::vector<cv::Point>& contour);
    bool isHandOpenFast(const std::vector<cv::Point>& contour);
    cv::Mat createSkinMaskOptimized(const cv::Mat& image);
    bool isHandShapeFast(const std::vector<cv::Point>& contour);
    double calculateHandConfidenceFast(const std::vector<cv::Point>& contour);
    cv::Point findPalmCenterFast(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> findFingerTipsFast(const std::vector<cv::Point>& contour);
    
    // Performance optimization
    void preallocateCudaMemory(int width, int height);
    void releaseCudaMemory();
    
    // State management
    void updatePerformanceStats(double processingTime);

    // Member variables
    bool m_initialized;
    bool m_cudaAvailable;
    int m_cudaDeviceId;
    QString m_detectorType;
    double m_confidenceThreshold;
    bool m_showBoundingBox;
    int m_performanceMode;
    
    // Gesture state
    bool m_wasOpen;
    bool m_wasClosed;
    int m_stableFrames;
    bool m_triggered;
    
    // Hand state tracking for delayed trigger
    bool m_handClosed;
    QElapsedTimer m_handClosedTimer;
    int m_requiredClosedFrames; // Number of frames hand must be closed before trigger
    int m_closedFrameCount;
    
    // Tracking state
    bool m_hasLock;
    cv::Rect m_roi;
    cv::Mat m_prevGray;
    std::vector<cv::Point2f> m_prevPts;
    QElapsedTimer m_timer;
    
    // Background model
    cv::Mat m_bgFloat;
    bool m_bgInit;
    
    // Performance
    QMutex m_mutex;
    QTimer m_updateTimer;
    QElapsedTimer m_performanceTimer;
    
    // Parameters
    int m_frameWidth;
    int m_frameHeight;
    int m_frameCount;
    int m_motionThreshold;
    int m_minMotionArea;
    int m_redetectInterval;
    int m_minRoiSize;
    int m_maxRoiSize;
    int m_requiredStableFrames;
    int m_motionHistory;
    int m_noMotionFrames;

    // CUDA memory pools for performance
    cv::cuda::GpuMat m_gpuGray;
    cv::cuda::GpuMat m_gpuPrevGray;
    cv::cuda::GpuMat m_gpuMotionMask;
    cv::cuda::GpuMat m_gpuSkinMask;
    cv::cuda::GpuMat m_gpuTemp1;
    cv::cuda::GpuMat m_gpuTemp2;
    
    // CUDA filters and processors
    cv::Ptr<cv::cuda::Filter> m_gaussianFilter;
    cv::Ptr<cv::cuda::Filter> m_morphFilter;
    cv::Ptr<cv::cuda::CannyEdgeDetector> m_cannyDetector;
    
    // Performance monitoring
    QElapsedTimer m_processingTimer;
    double m_averageProcessingTime;
    double m_currentFPS;
    int m_totalFramesProcessed;
    QList<double> m_processingTimes;
    QList<HandDetection> m_lastDetections;
};

#endif // HAND_DETECTOR_H
