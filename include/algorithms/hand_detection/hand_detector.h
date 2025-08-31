#ifndef HAND_DETECTOR_H
#define HAND_DETECTOR_H

#include <QObject>
#include <QList>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

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
    bool isOpenGLAvailable() const; // Returns OpenCL/OpenGL availability
    QString getDetectorType() const;
    double getAverageProcessingTime() const;
    double getCurrentFPS() const;
    int getTotalFramesProcessed() const;

signals:
    void detectionCompleted(const QList<HandDetection>& detections);
    void processingTimeUpdated(double milliseconds);
    void openclError(const QString& error);
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

    // Hand gesture detection with OpenCL
    QList<HandDetection> detectOpenGLHandGestures(const cv::Mat& image);
    QList<HandDetection> detectOpenGLHandGesturesOptimized(const cv::Mat& image);
    QList<HandDetection> detectHandGesturesOptimized(const cv::Mat& image);
    
    // OpenCL image processing utilities
    cv::UMat convertToOpenCL(const cv::Mat& cpuImage);
    cv::Mat convertFromOpenCL(const cv::UMat& openclImage);
    void applyOpenCLGaussianBlur(cv::UMat& openclImage, int kernelSize = 5);
    void applyOpenCLMorphology(cv::UMat& openclImage, int operation = cv::MORPH_CLOSE);
    
    // OpenCL detection methods
    QList<HandDetection> detectHandsByOpenCLShape(const cv::UMat& openclImage);
    QList<HandDetection> detectHandsByOpenCLMotion(const cv::UMat& openclImage);
    QList<HandDetection> detectHandsByOpenCLKeypoints(const cv::UMat& openclImage);
    
    // OpenCL image processing
    cv::UMat createOpenCLSkinMask(const cv::UMat& openclImage);
    cv::UMat createOpenCLMotionMask(const cv::UMat& openclImage);
    std::vector<std::vector<cv::Point>> findOpenCLContours(const cv::UMat& openclMask);
    
    // Hand gesture detection with OpenCL
    QList<HandDetection> detectOpenCLHandGestures(const cv::Mat& image);
    QList<HandDetection> detectOpenCLHandGesturesOptimized(const cv::Mat& image);
    bool isOpenCLHandShape(const std::vector<cv::Point>& contour, const cv::UMat& openclImage);
    double calculateOpenCLHandConfidence(const std::vector<cv::Point>& contour, const cv::UMat& openclImage);
    
    // OpenCL shape analysis
    std::vector<cv::Point> findOpenCLFingerTips(const std::vector<cv::Point>& contour);
    cv::Point findOpenCLPalmCenter(const std::vector<cv::Point>& contour);
    
    // OpenCL motion detection and tracking
    bool detectOpenCLMotion(const cv::UMat& openclImage);
    bool acquireOpenCLRoiFromMotion(const cv::UMat& openclGray, const cv::UMat& openclMotionMask);
    void trackOpenCLRoiLK(const cv::UMat& openclGrayPrev, const cv::UMat& openclGrayCurr);
    void updateOpenCLMotionHistory(const cv::UMat& openclMotionMask);
    bool isOpenCLMotionStable();
    
    // OpenCL gesture analysis
    bool analyzeOpenCLGestureClosed(const cv::UMat& openclGray, const cv::Rect& roi) const;
    bool analyzeOpenCLGestureOpen(const cv::UMat& openclGray, const cv::Rect& roi) const;
    
    // Optimized processing methods
    cv::UMat createOpenCLSkinMaskOptimized(const cv::UMat& openclImage);
    std::vector<std::vector<cv::Point>> findOpenCLContoursOptimized(const cv::UMat& openclMask);
    bool isOpenCLHandShapeFast(const std::vector<cv::Point>& contour, const cv::UMat& openclImage);
    double calculateOpenCLHandConfidenceFast(const std::vector<cv::Point>& contour, const cv::UMat& openclImage);
    std::vector<cv::Point> findOpenCLFingerTipsFast(const std::vector<cv::Point>& contour);
    bool isHandOpenFast(const std::vector<cv::Point>& contour);
    cv::Mat createSkinMaskOptimized(const cv::Mat& image);
    bool isHandShapeFast(const std::vector<cv::Point>& contour);
    double calculateHandConfidenceFast(const std::vector<cv::Point>& contour);
    cv::Point findPalmCenterFast(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> findFingerTipsFast(const std::vector<cv::Point>& contour);
    
    // Performance optimization
    void preallocateOpenCLMemory(int width, int height);
    void releaseOpenCLMemory();
    
    // Enhanced segmentation methods
    QList<HandDetection> detectEnhancedSegmentation(const cv::Mat& image);
    cv::Mat enhanceMorphologicalProcessing(const cv::Mat& mask);
    std::vector<std::vector<cv::Point>> findMultiScaleContours(const cv::Mat& mask);
    bool isEnhancedHandShape(const std::vector<cv::Point>& contour, const cv::Mat& image);
    double calculateEnhancedConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image);
    cv::Point findEnhancedPalmCenter(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> findEnhancedFingerTips(const std::vector<cv::Point>& contour);
    bool isEnhancedHandOpen(const std::vector<cv::Point>& contour);
    bool isEnhancedHandClosed(const std::vector<cv::Point>& contour);
    QList<HandDetection> applyTemporalConsistency(const QList<HandDetection>& detections);
    
    // State management
    void updatePerformanceStats(double processingTime);

    // Member variables
    bool m_initialized;
    bool m_openclAvailable;
    bool m_openglAvailable;
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

    // OpenCL memory pools for performance
    cv::UMat m_openclGray;
    cv::UMat m_openclPrevGray;
    cv::UMat m_openclMotionMask;
    cv::UMat m_openclSkinMask;
    cv::UMat m_openclTemp1;
    cv::UMat m_openclTemp2;
    
    // OpenCL filters and processors (using OpenCV's UMat)
    cv::Mat m_gaussianKernel; // For GaussianBlur
    cv::Mat m_morphKernel;    // For morphological operations
    
    // Performance monitoring
    QElapsedTimer m_processingTimer;
    double m_averageProcessingTime;
    double m_currentFPS;
    int m_totalFramesProcessed;
    QList<double> m_processingTimes;
    QList<HandDetection> m_lastDetections;
};

#endif // HAND_DETECTOR_H
