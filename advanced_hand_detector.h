#ifndef ADVANCED_HAND_DETECTOR_H
#define ADVANCED_HAND_DETECTOR_H

#include <QObject>
#include <QList>
#include <QMutex>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>

struct AdvancedHandDetection {
    cv::Rect boundingBox;
    double confidence;
    QString handType;
    std::vector<cv::Point> landmarks;
    bool isRaised;
    cv::Point palmCenter;
    std::vector<cv::Point> fingerTips;
    std::vector<cv::Point> palmKeypoints;
    cv::Point wristPoint;
    cv::Point middleFingerTip;
    cv::Point indexFingerTip;
    cv::Point thumbTip;
};

class AdvancedHandDetector : public QObject
{
    Q_OBJECT

public:
    explicit AdvancedHandDetector(QObject *parent = nullptr);
    ~AdvancedHandDetector();

    bool initialize();
    bool isInitialized() const;
    QList<AdvancedHandDetection> detect(const cv::Mat& image);
    
    // Configuration
    void setConfidenceThreshold(double threshold);
    double getConfidenceThreshold() const;
    void setShowBoundingBox(bool show);
    bool getShowBoundingBox() const;
    void setPerformanceMode(int mode); // 0=Fast, 1=Balanced, 2=Accurate
    int getPerformanceMode() const;
    
    // Hand gesture detection methods (public for capture.cpp access)
    bool isHandClosed(const std::vector<cv::Point>& contour);
    bool isHandOpen(const std::vector<cv::Point>& contour);
    double calculateHandClosureRatio(const std::vector<cv::Point>& contour);
    bool shouldTriggerCapture();
    void resetGestureState(); // Reset gesture tracking state

private:
    // Main detection methods - MUCH MORE ACCURATE
    QList<AdvancedHandDetection> detectHandsPalmBased(const cv::Mat& image);
    QList<AdvancedHandDetection> detectHandsByKeypoints(const cv::Mat& image);
    QList<AdvancedHandDetection> detectHandsByMotion(const cv::Mat& image);
    
    // Advanced hand shape detection methods (NOT SKIN COLOR)
    QList<AdvancedHandDetection> detectHandsByShape(const cv::Mat& image);
    QList<AdvancedHandDetection> detectHandsByEdges(const cv::Mat& image);
    bool isHandShapeAdvanced(const std::vector<cv::Point>& contour, const cv::Mat& image);
    bool validateAdvancedHandShape(const std::vector<cv::Point>& contour, int fingerCount);
    bool isHandEdgePattern(const std::vector<cv::Point>& contour, const cv::Mat& image);
    double calculateHandShapeConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image);
    double calculateEdgeHandConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image);
    std::vector<cv::Point> findFingerTipsFromContour(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> extractPalmKeypoints(const std::vector<cv::Point>& contour);
    

    
    // Legacy methods (kept for compatibility)
    cv::Point findPalmCenter(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> findFingerTips(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter);
    bool validateHandStructure(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter);
    std::vector<cv::Point> detectPalmKeypoints(const cv::Mat& skinMask);
    
    // Enhanced skin detection
    cv::Mat createEnhancedSkinMask(const cv::Mat& image);
    cv::Mat createMotionMask(const cv::Mat& image);
    
    // Hand validation and confidence calculation
    bool isHandShape(const std::vector<cv::Point>& contour, const cv::Mat& image);
    double calculateHandConfidence(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter, const cv::Mat& image);
    bool isRaisedHand(const cv::Point& palmCenter, const cv::Mat& image);
    
    // Performance optimization
    void updateBackgroundModel(const cv::Mat& frame);
    cv::Mat getBackgroundModel();
    void optimizeForPerformance(cv::Mat& image);
    
    // Motion detection
    bool detectMotion(const cv::Mat& image);
    void updateMotionHistory(const cv::Mat& motionMask);
    bool isMotionStable();
    
    // Member variables
    bool m_initialized;
    double m_confidenceThreshold;
    bool m_showBoundingBox;
    int m_performanceMode;
    mutable QMutex m_mutex;
    QElapsedTimer m_processingTimer;
    
    // Motion detection variables
    cv::Mat m_backgroundModel;
    bool m_backgroundInitialized;
    int m_frameCount;
    std::vector<cv::Mat> m_motionHistory;
    std::vector<cv::Point> m_lastMotionCenters;
    int m_motionStabilityCount;
    bool m_motionDetected;
    
    // Performance tracking
    double m_lastProcessingTime;
    int m_detectionFPS;
    
    // Hand tracking for stability
    std::vector<cv::Point> m_lastPalmKeypoints;
    cv::Point m_lastPalmCenter;
    int m_trackingFrames;
    bool m_hasStableTracking;
    QList<AdvancedHandDetection> m_lastHandDetections; // For tracking stability
    
    // Gesture state tracking
    bool m_wasHandClosed;
    bool m_wasHandOpen;
    int m_gestureStableFrames;
    
    // Configuration - MOTION-BASED HAND DETECTION
    static const int PROCESSING_INTERVAL = 1; // Process every frame for 60 FPS hand detection
    static const int MAX_DETECTIONS = 2;
    static const int MIN_HAND_AREA = 150;
    static const int MAX_HAND_AREA = 25000;
    static const int PALM_KEYPOINTS_COUNT = 7; // Palm center + 6 keypoints around palm
    static const int MIN_PALM_CONFIDENCE_INT = 40; // 0.4 * 100 for integer storage
    static const int TRACKING_FRAMES_THRESHOLD = 3;
    
    // MOTION DETECTION CONSTANTS
    static const int MOTION_THRESHOLD = 30; // Minimum motion to consider
    static const int MOTION_HISTORY_FRAMES = 5; // Frames to track motion
    static const int MIN_MOTION_AREA = 500; // Minimum area with motion
    static const int MOTION_STABILITY_FRAMES = 3; // Frames of consistent motion
    
    // HAND SHAPE DETECTION CONSTANTS (NOT SKIN COLOR)
    static const int MIN_FINGER_COUNT = 3;
    static const int MAX_FINGER_COUNT = 5;
    static const int MIN_HAND_ASPECT_RATIO_INT = 8; // 0.8 * 10
    static const int MAX_HAND_ASPECT_RATIO_INT = 25; // 2.5 * 10
    static const int MIN_HAND_CIRCULARITY_INT = 1; // 0.1 * 10
    static const int MAX_HAND_CIRCULARITY_INT = 6; // 0.6 * 10
    static const int MIN_CONVEXITY_DEFECTS = 2;
    static const int MAX_CONVEXITY_DEFECTS = 8;
};

#endif // ADVANCED_HAND_DETECTOR_H
