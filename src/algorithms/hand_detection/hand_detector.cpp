#include "algorithms/hand_detection/hand_detector.h"
#include <QDebug>
#include <QThread>
#include <opencv2/imgproc.hpp>

HandDetector::HandDetector(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_confidenceThreshold(0.5)
    , m_showBoundingBox(true)
    , m_performanceMode(1)
    , m_wasOpen(false)
    , m_wasClosed(false)
    , m_stableFrames(0)
    , m_triggered(false)
    , m_hasLock(false)
    , m_bgInit(false)
    , m_frameWidth(0)
    , m_frameHeight(0)
    , m_frameCount(0)
    , m_motionThreshold(15)
    , m_minMotionArea(200)
    , m_redetectInterval(10)
    , m_minRoiSize(30)
    , m_maxRoiSize(150)
    , m_requiredStableFrames(3) // Reduced for faster response
    , m_motionHistory(0)
    , m_noMotionFrames(0)
{
    qDebug() << "HandDetector: Specialized hand gesture detector constructor called";
}

HandDetector::~HandDetector()
{
    qDebug() << "HandDetector: Destructor called";
}

bool HandDetector::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "HandDetector: Initializing specialized hand gesture detection...";
    
    try {
        m_prevGray = cv::Mat();
        m_roi = cv::Rect();
        m_hasLock = false;
        resetGestureState();
        m_motionHistory = 0;
        m_noMotionFrames = 0;
        
        m_initialized = true;
        qDebug() << "HandDetector: Specialized hand gesture detection initialized successfully";
        qDebug() << "🎯 Hand detector ready - looking for hand gestures!";
        return true;
    }
    catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Initialization failed:" << e.what();
        return false;
    }
}

bool HandDetector::isInitialized() const
{
    return m_initialized;
}

void HandDetector::reset()
{
    m_hasLock = false;
    m_roi = cv::Rect();
    m_prevGray = cv::Mat();
    resetGestureState();
    m_frameCount = 0;
    m_motionHistory = 0;
    m_noMotionFrames = 0;
}

QList<HandDetection> HandDetector::detect(const cv::Mat& image)
{
    if (!m_initialized || image.empty()) {
        return QList<HandDetection>();
    }

    // Update frame dimensions if needed
    if (m_frameWidth != image.cols || m_frameHeight != image.rows) {
        m_frameWidth = image.cols;
        m_frameHeight = image.rows;
    }

    m_frameCount++;

    // Use specialized hand gesture detection
    QList<HandDetection> detections = detectHandGestures(image);
    
    // Check for closed hand to trigger capture
    bool hasClosedHand = false;
    for (const auto& detection : detections) {
        if (detection.isClosed && detection.confidence > 0.3) {
            hasClosedHand = true;
            break;
        }
    }
    
    // Update gesture state
    if (hasClosedHand) {
        m_noMotionFrames++;
        qDebug() << "📊 Hand closed - Stable frames:" << m_noMotionFrames << "/" << m_requiredStableFrames;
        if (m_noMotionFrames >= m_requiredStableFrames && !m_triggered) {
            m_triggered = true;
            qDebug() << "🎯 CAPTURE TRIGGERED! Closed hand gesture detected!";
        }
    } else {
        if (m_noMotionFrames > 0) {
            qDebug() << "🔄 Hand opened - Resetting stable frames";
        }
        m_noMotionFrames = 0;
        m_triggered = false;
    }

    return detections;
}

void HandDetector::setConfidenceThreshold(double threshold)
{
    m_confidenceThreshold = qBound(0.0, threshold, 1.0);
}

double HandDetector::getConfidenceThreshold() const
{
    return m_confidenceThreshold;
}

void HandDetector::setShowBoundingBox(bool show)
{
    m_showBoundingBox = show;
}

bool HandDetector::getShowBoundingBox() const
{
    return m_showBoundingBox;
}

void HandDetector::setPerformanceMode(int mode)
{
    m_performanceMode = qBound(0, mode, 2);
}

int HandDetector::getPerformanceMode() const
{
    return m_performanceMode;
}

bool HandDetector::isHandClosed(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 10) return false;
    
    // Use convexity defects to detect hand closing
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    if (hull.size() < 3) return false;
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Count significant defects (finger gaps)
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0; // Depth of the defect
        if (depth > 10.0) { // Significant defect threshold
            significantDefects++;
        }
    }
    
    // Closed hand (fist) has fewer defects than open hand
    return significantDefects <= 1; // Fist has 0-1 defects, open hand has 4-5
}

bool HandDetector::isHandOpen(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 10) return false;
    
    // Use convexity defects to detect hand opening
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    if (hull.size() < 3) return false;
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Count significant defects (finger gaps)
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0; // Depth of the defect
        if (depth > 10.0) { // Significant defect threshold
            significantDefects++;
        }
    }
    
    // Open hand has more defects than closed hand
    return significantDefects >= 3; // Open hand has 3-5 defects
}

double HandDetector::calculateHandClosureRatio(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 10) return 0.0;
    
    // Calculate closure ratio based on convexity defects
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 10.0) {
            significantDefects++;
        }
    }
    
    // Convert defect count to closure ratio (0 = open, 1 = closed)
    double closureRatio = 1.0 - (significantDefects / 5.0); // Normalize to 0-1
    return qBound(0.0, closureRatio, 1.0);
}

bool HandDetector::shouldTriggerCapture()
{
    if (m_triggered) {
        m_triggered = false;
        m_noMotionFrames = 0;
        return true;
    }
    return false;
}

void HandDetector::resetGestureState()
{
    m_wasOpen = false;
    m_wasClosed = false;
    m_stableFrames = 0;
    m_triggered = false;
    m_noMotionFrames = 0;
}

bool HandDetector::hasLock() const
{
    return m_hasLock;
}

cv::Rect HandDetector::getRoi() const
{
    return m_roi;
}

void HandDetector::updateRoi(const cv::Mat& /*frame*/)
{
    // Not needed for gesture detection
}

// SPECIALIZED HAND GESTURE DETECTION USING CONVEXITY DEFECTS

QList<HandDetection> HandDetector::detectHandGestures(const cv::Mat& image)
{
    QList<HandDetection> detections;
    
    // Resize image for faster processing
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(320, 240), 0, 0, cv::INTER_LINEAR);
    
    // Create skin color mask
    cv::Mat skinMask = createSkinMask(resized);
    
    // Debug: Show skin mask statistics
    int skinPixels = cv::countNonZero(skinMask);
    double skinRatio = (double)skinPixels / (skinMask.rows * skinMask.cols);
    qDebug() << "🔍 Skin detection - Pixels:" << skinPixels << "Ratio:" << skinRatio;
    
    // Apply morphological operations to clean up the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(skinMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Debug: Show detection count
    if (!contours.empty()) {
        qDebug() << "🔍 Found" << contours.size() << "potential hand(s)";
    }
    
    // Process contours for hand gesture detection
    int contourCount = 0;
    for (const auto& contour : contours) {
        if (contourCount >= 5) break; // Process first 5 contours
        contourCount++;
        
        if (contour.size() < 10) continue;
        
        double area = cv::contourArea(contour);
        if (area < 500 || area > 30000) continue; // Filter by size
        
        // Check if this looks like a hand using aspect ratio and position
        if (isHandShape(contour, resized)) {
            cv::Rect boundingRect = cv::boundingRect(contour);
            
            // Scale back to original image size
            double scaleX = (double)image.cols / resized.cols;
            double scaleY = (double)image.rows / resized.rows;
            
            cv::Rect scaledRect(
                boundingRect.x * scaleX,
                boundingRect.y * scaleY,
                boundingRect.width * scaleX,
                boundingRect.height * scaleY
            );
            
            // Analyze hand gesture using convexity defects
            bool isOpen = isHandOpen(contour);
            bool isClosed = isHandClosed(contour);
            double confidence = calculateHandConfidence(contour, resized);
            
            qDebug() << "🔍 Analyzing detection - Confidence:" << confidence << "| Closed:" << isClosed << "| Open:" << isOpen;
            
            if (confidence > 0.3) {
                HandDetection detection;
                detection.boundingBox = scaledRect;
                detection.confidence = confidence;
                detection.handType = "unknown";
                detection.landmarks = contour;
                detection.isOpen = isOpen;
                detection.isClosed = isClosed;
                detection.palmCenter = findPalmCenter(contour);
                
                detections.append(detection);
                
                if (isClosed && confidence > 0.3) {
                    qDebug() << "🎯 HAND CLOSED DETECTED! Confidence:" << confidence << "| Gesture: Closed hand";
                } else if (isOpen && confidence > 0.3) {
                    qDebug() << "✋ HAND OPEN DETECTED! Confidence:" << confidence << "| Gesture: Open hand";
                } else if (confidence > 0.1) {
                    qDebug() << "🤔 Hand-like object detected - Confidence:" << confidence << "| Closed:" << isClosed << "| Open:" << isOpen;
                }
            }
        }
    }
    
    return detections;
}

cv::Mat HandDetector::createSkinMask(const cv::Mat& image)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    // Define skin color ranges (multiple ranges for different lighting conditions)
    cv::Mat skinMask1, skinMask2, skinMask3;
    
    // Range 1: Light skin (more permissive)
    cv::inRange(hsv, cv::Scalar(0, 15, 50), cv::Scalar(25, 255, 255), skinMask1);
    
    // Range 2: Darker skin (more permissive)
    cv::inRange(hsv, cv::Scalar(0, 20, 40), cv::Scalar(30, 255, 255), skinMask2);
    
    // Range 3: Reddish skin (for different lighting)
    cv::inRange(hsv, cv::Scalar(160, 15, 50), cv::Scalar(180, 255, 255), skinMask3);
    
    // Combine all skin masks
    cv::Mat skinMask = skinMask1 | skinMask2 | skinMask3;
    
    return skinMask;
}

bool HandDetector::isHandShape(const std::vector<cv::Point>& contour, const cv::Mat& image)
{
    if (contour.size() < 10) return false;
    
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter == 0) return false;
    
    // Check aspect ratio (hands are usually taller than wide)
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = (double)boundingRect.width / boundingRect.height;
    
    // Hands typically have aspect ratio between 0.5 and 2.0
    if (aspectRatio < 0.5 || aspectRatio > 2.0) return false;
    
    // Check if it's in the lower part of the image (hands are usually lower than faces)
    cv::Point center = findPalmCenter(contour);
    if (center.y < image.rows * 0.2) return false; // Too high (likely a face)
    
    return true;
}

double HandDetector::calculateHandConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image)
{
    if (contour.size() < 10) return 0.0;
    
    double confidence = 0.0;
    
    // Area-based confidence
    double area = cv::contourArea(contour);
    double normalizedArea = area / (image.rows * image.cols);
    confidence += qMin(normalizedArea * 1000, 0.3);
    
    // Convexity-based confidence
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    if (hullArea > 0) {
        double solidity = area / hullArea;
        confidence += solidity * 0.4;
    }
    
    // Defect-based confidence (more defects = more likely to be a hand)
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 5.0) {
            significantDefects++;
        }
    }
    
    // Hands typically have 0-5 defects
    if (significantDefects >= 0 && significantDefects <= 5) {
        confidence += 0.3;
    }
    
    return qBound(0.0, confidence, 1.0);
}

cv::Point HandDetector::findPalmCenter(const std::vector<cv::Point>& contour)
{
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 == 0) {
        return cv::Point(0, 0);
    }
    
    int centerX = moments.m10 / moments.m00;
    int centerY = moments.m01 / moments.m00;
    
    return cv::Point(centerX, centerY);
}

// Legacy methods (simplified)

QList<HandDetection> HandDetector::detectHandsByShape(const cv::Mat& image)
{
    return detectHandGestures(image);
}

QList<HandDetection> HandDetector::detectHandsByMotion(const cv::Mat& /*image*/)
{
    return QList<HandDetection>();
}

QList<HandDetection> HandDetector::detectHandsByKeypoints(const cv::Mat& /*image*/)
{
    return QList<HandDetection>();
}

QList<HandDetection> HandDetector::detectHandsByMotionFast(const cv::Mat& /*image*/)
{
    return QList<HandDetection>();
}

cv::Mat HandDetector::createFastMotionMask(const cv::Mat& /*gray*/)
{
    return cv::Mat();
}

bool HandDetector::analyzeGestureClosedFast(const cv::Mat& /*gray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

bool HandDetector::analyzeGestureOpenFast(const cv::Mat& /*gray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

void HandDetector::updateRoiFast(const cv::Mat& /*frame*/)
{
}

bool HandDetector::acquireRoiFromMotionFast(const cv::Mat& /*motionMask*/)
{
    return false;
}

void HandDetector::trackRoiSimple(const cv::Mat& /*motionMask*/)
{
}

std::vector<cv::Point> HandDetector::findFingerTips(const std::vector<cv::Point>& /*contour*/)
{
    return std::vector<cv::Point>();
}

bool HandDetector::detectMotion(const cv::Mat& /*image*/)
{
    return false;
}

bool HandDetector::acquireRoiFromMotion(const cv::Mat& /*gray*/, const cv::Mat& /*motionMask*/)
{
    return false;
}

void HandDetector::trackRoiLK(const cv::Mat& /*grayPrev*/, const cv::Mat& /*grayCurr*/)
{
}

void HandDetector::updateMotionHistory(const cv::Mat& /*motionMask*/)
{
}

bool HandDetector::isMotionStable()
{
    return m_stableFrames >= m_requiredStableFrames;
}

bool HandDetector::analyzeGestureClosed(const cv::Mat& /*gray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

bool HandDetector::analyzeGestureOpen(const cv::Mat& /*gray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

cv::Mat HandDetector::createEnhancedSkinMask(const cv::Mat& image)
{
    return createSkinMask(image);
}

cv::Mat HandDetector::createMotionMask(const cv::Mat& /*gray*/)
{
    return cv::Mat();
}

void HandDetector::optimizeForPerformance(cv::Mat& /*image*/)
{
}

void HandDetector::updateBackgroundModel(const cv::Mat& /*frame*/)
{
}

cv::Mat HandDetector::getBackgroundModel()
{
    return cv::Mat();
}
