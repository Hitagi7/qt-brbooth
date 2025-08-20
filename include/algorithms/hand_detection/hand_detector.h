#ifndef HAND_DETECTOR_H
#define HAND_DETECTOR_H

#include <QObject>
#include <QList>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>

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

    // ROI tracking (for continuous hand tracking)
    bool hasLock() const;
    cv::Rect getRoi() const;
    void updateRoi(const cv::Mat& frame);

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

    // Member variables
    bool m_initialized;
    double m_confidenceThreshold;
    bool m_showBoundingBox;
    int m_performanceMode;
    
    // Gesture state
    bool m_wasOpen;
    bool m_wasClosed;
    int m_stableFrames;
    bool m_triggered;
    
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
};

#endif // HAND_DETECTOR_H
