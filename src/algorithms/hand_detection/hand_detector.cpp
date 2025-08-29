#include "algorithms/hand_detection/hand_detector.h"
#include "core/amd_gpu_verifier.h"
#include <QDebug>
#include <QThread>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>

HandDetector::HandDetector(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_openclAvailable(false)
    , m_openglAvailable(false)
    , m_detectorType("CPU")
    , m_confidenceThreshold(0.5)
    , m_showBoundingBox(true)
    , m_performanceMode(1)
    , m_wasOpen(false)
    , m_wasClosed(false)
    , m_stableFrames(0)
    , m_triggered(false)
    , m_handClosed(false)
    , m_requiredClosedFrames(30) // 30 frames at ~20fps = 1.5 seconds
    , m_closedFrameCount(0)
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
    , m_averageProcessingTime(0.0)
    , m_currentFPS(0.0)
    , m_totalFramesProcessed(0)
{
    qDebug() << "HandDetector: Specialized hand gesture detector constructor called";
}

HandDetector::~HandDetector()
{
    releaseOpenCLMemory();
    qDebug() << "HandDetector: Destructor called";
}

bool HandDetector::initialize()
{
    if (m_initialized) {
        return true;
    }

    qDebug() << "HandDetector: Initializing hand gesture detection...";
    
    try {
        // Initialize AMD GPU verification
        bool amdGPUAvailable = AMDGPUVerifier::initialize();
        
        if (amdGPUAvailable) {
            // Check OpenCL availability
            if (cv::ocl::useOpenCL()) {
                cv::ocl::setUseOpenCL(true);
                m_openclAvailable = true;
                m_detectorType = "OpenCL";
                
                qDebug() << "HandDetector: Using AMD GPU with OpenCL acceleration";
                qDebug() << "HandDetector: OpenCL device available for image processing";
                
                // Get GPU info
                AMDGPUVerifier::GPUInfo gpuInfo = AMDGPUVerifier::getGPUInfo();
                qDebug() << "HandDetector: GPU Name:" << gpuInfo.name;
                qDebug() << "HandDetector: GPU Memory:" << gpuInfo.totalMemory / (1024*1024) << "MB";
                qDebug() << "HandDetector: Compute Units:" << gpuInfo.computeUnits;
                
                qDebug() << "🚀 OpenCL-accelerated hand detection enabled!";
            } else {
                qWarning() << "HandDetector: OpenCL not available, falling back to CPU";
                m_openclAvailable = false;
                m_detectorType = "CPU";
            }
            
            // Check OpenGL availability
            if (AMDGPUVerifier::testOpenGLAcceleration()) {
                m_openglAvailable = true;
                qDebug() << "HandDetector: OpenGL acceleration available";
            } else {
                m_openglAvailable = false;
                qDebug() << "HandDetector: OpenGL acceleration not available";
            }
        } else {
            qDebug() << "HandDetector: No AMD GPU available, using CPU";
            m_openclAvailable = false;
            m_openglAvailable = false;
            m_detectorType = "CPU";
        }

        m_prevGray = cv::Mat();
        m_roi = cv::Rect();
        m_hasLock = false;
        resetGestureState();
        m_motionHistory = 0;
        m_noMotionFrames = 0;
        
        m_initialized = true;
        emit detectorTypeChanged(m_detectorType);
        qDebug() << "HandDetector: Hand gesture detection initialized successfully";
        qDebug() << "🎯 Hand detector ready - using" << m_detectorType << "processing!";
        return true;
    }
    catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Initialization failed:" << e.what();
        m_openclAvailable = false;
        m_openglAvailable = false;
        m_detectorType = "CPU";
        m_initialized = true;
        emit detectorTypeChanged(m_detectorType);
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
    m_totalFramesProcessed = 0;
    m_processingTimes.clear();
}

QList<HandDetection> HandDetector::detect(const cv::Mat& image)
{
    if (!m_initialized || image.empty()) {
        return QList<HandDetection>();
    }

    m_processingTimer.start();

    // Update frame dimensions if needed
    if (m_frameWidth != image.cols || m_frameHeight != image.rows) {
        m_frameWidth = image.cols;
        m_frameHeight = image.rows;
        if (m_openclAvailable) {
            preallocateOpenCLMemory(m_frameWidth, m_frameHeight);
        }
    }

    m_frameCount++;

    QList<HandDetection> detections;
    
    try {
        // Always use the new strict hand detection method
        detections = detectHandGestures(image);
        
        // Cache detections for frame skipping
        m_lastDetections = detections;
        
        // Update performance stats
        double processingTime = m_processingTimer.elapsed();
        updatePerformanceStats(processingTime);
        
        // Optimized gesture detection with reduced logging
        bool hasClosedHand = false;
        for (const auto& detection : detections) {
            if (detection.isClosed && detection.confidence > 0.3) {
                hasClosedHand = true;
                break;
            }
        }
        
        // Update gesture state with reduced logging
        if (hasClosedHand) {
            m_noMotionFrames++;
            if (m_noMotionFrames >= m_requiredStableFrames && !m_triggered) {
                m_triggered = true;
                qDebug() << "🎯 CAPTURE TRIGGERED! Closed hand gesture detected!";
            }
        } else {
            m_noMotionFrames = 0;
            m_triggered = false;
        }
        
        emit detectionCompleted(detections);
        emit processingTimeUpdated(processingTime);
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Detection error:" << e.what();
        emit openclError(QString("Detection error: %1").arg(e.what()));
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

// Stricter hand detection methods
bool HandDetector::isHandShapeStrict(const std::vector<cv::Point>& contour, const cv::Mat& image)
{
    if (contour.size() < 10) return false; // Reduced from 20
    
    double area = cv::contourArea(contour);
    if (area < 500 || area > 30000) return false; // More lenient size range
    
    // Calculate perimeter and circularity
    double perimeter = cv::arcLength(contour, true);
    double circularity = 4 * M_PI * area / (perimeter * perimeter);
    
    // Hands are not very circular (circularity < 0.8) - more lenient
    if (circularity > 0.8) return false;
    
    // Calculate aspect ratio
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = (double)boundingRect.width / boundingRect.height;
    
    // Hands typically have aspect ratio between 0.3 and 3.0 - more lenient
    if (aspectRatio < 0.3 || aspectRatio > 3.0) return false;
    
    // Check for convexity defects (fingers)
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    if (hull.size() < 3) return false; // Reduced from 5
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Hands should have some convexity defects (fingers) - more lenient
    if (defects.size() < 1) return false; // Reduced from 2
    
    // Check if the shape is in the lower part of the image (more likely to be a hand)
    cv::Point center = findPalmCenter(contour);
    if (center.y < image.rows * 0.2) return false; // More lenient - not in upper 20%
    
    return true;
}

bool HandDetector::isHandOpenStrict(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 20) return false;
    
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Count significant defects (fingers)
    int fingerCount = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 10.0) { // Significant depth
            fingerCount++;
        }
    }
    
    // Open hand should have 3-5 fingers
    return fingerCount >= 3 && fingerCount <= 5;
}

bool HandDetector::isHandClosedStrict(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 10) return false; // Reduced from 20
    
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Count significant defects (fingers)
    int fingerCount = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 5.0) { // More lenient depth threshold
            fingerCount++;
        }
    }
    
    // Closed hand should have 0-3 fingers visible - more lenient
    return fingerCount <= 3;
}

double HandDetector::calculateHandConfidenceStrict(const std::vector<cv::Point>& contour, const cv::Mat& image)
{
    if (!isHandShapeStrict(contour, image)) return 0.0;
    
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    double circularity = 4 * M_PI * area / (perimeter * perimeter);
    
    // Calculate aspect ratio
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = (double)boundingRect.width / boundingRect.height;
    
    // Count convexity defects
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
    
    // Calculate confidence based on multiple factors
    double areaScore = qMin(1.0, area / 5000.0); // Normalize area
    double circularityScore = 1.0 - circularity; // Lower circularity is better for hands
    double aspectRatioScore = (aspectRatio >= 0.7 && aspectRatio <= 1.5) ? 1.0 : 0.5;
    double defectScore = qMin(1.0, significantDefects / 4.0); // Normalize defect count
    
    // Weighted average
    double confidence = (areaScore * 0.3 + circularityScore * 0.3 + aspectRatioScore * 0.2 + defectScore * 0.2);
    
    return qBound(0.0, confidence, 1.0);
}

bool HandDetector::shouldTriggerCapture()
{
    // QUICK: Trigger after hand has been closed for 1 second
    if (m_handClosed && !m_triggered && m_handClosedTimer.elapsed() >= 1000) { // 1 second
        m_triggered = true;
        qDebug() << "🎯 TRIGGER READY! Hand closed for 1+ seconds (QUICK MODE)";
        return true;
    }
    
    // Debug output (reduced frequency for performance)
    static int debugCounter = 0;
    if (++debugCounter % 60 == 0) { // Every 60 frames for more frequent updates
        if (m_handClosed) {
            qDebug() << "🔍 Trigger check - m_handClosed:" << m_handClosed 
                     << "m_triggered:" << m_triggered
                     << "Elapsed:" << m_handClosedTimer.elapsed() << "ms"
                     << "Required: 1000ms"
                     << "Closed frames:" << m_closedFrameCount;
        }
    }
    
    return false;
}

bool HandDetector::isHandClosedTimerValid() const
{
    return m_handClosedTimer.isValid();
}

void HandDetector::resetGestureState()
{
    m_wasOpen = false;
    m_wasClosed = false;
    m_stableFrames = 0;
    m_triggered = false;
    m_noMotionFrames = 0;
    m_handClosed = false;
    m_closedFrameCount = 0;
    m_handClosedTimer.invalidate();
    qDebug() << "🔄 Gesture state reset - ready for new detection";
}

void HandDetector::updateHandState(bool isClosed)
{
    qDebug() << "🖐️ updateHandState called - isClosed:" << isClosed << "m_handClosed:" << m_handClosed;
    
    if (isClosed) {
        if (!m_handClosed) {
            // Hand just closed, start timer
            m_handClosed = true;
            m_handClosedTimer.start();
            m_closedFrameCount = 0;
            qDebug() << "🖐️ Hand closed - Starting trigger timer...";
        }
        m_closedFrameCount++;
        qDebug() << "🖐️ Hand still closed - Frame count:" << m_closedFrameCount;
    } else {
        if (m_handClosed) {
            // Hand opened, reset timer
            m_handClosed = false;
            m_handClosedTimer.invalidate();
            m_closedFrameCount = 0;
            m_triggered = false; // Reset trigger state when hand opens
            qDebug() << "🖐️ Hand opened - Resetting trigger timer and trigger state";
        }
    }
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
    
    // Create skin color mask (SIMPLE AND WORKING)
    cv::Mat skinMask = createSkinMask(resized);
    
    // Debug: Show skin mask statistics
    int skinPixels = cv::countNonZero(skinMask);
    double skinRatio = (double)skinPixels / (skinMask.rows * skinMask.cols);
    qDebug() << "🔍 Skin detection - Pixels:" << skinPixels << "Ratio:" << skinRatio;
    
    // If no skin detected, try more lenient detection
    if (skinPixels < 100) {
        qDebug() << "⚠️ Very few skin pixels detected - trying alternative detection";
        // Try a more lenient skin detection for poor lighting
        cv::Mat hsv;
        cv::cvtColor(resized, hsv, cv::COLOR_BGR2HSV);
        
        // More lenient skin color ranges for poor lighting
        cv::Mat skinMask2;
        cv::inRange(hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), skinMask2);
        
        // Combine masks
        cv::bitwise_or(skinMask, skinMask2, skinMask);
        
        int newSkinPixels = cv::countNonZero(skinMask);
        qDebug() << "🔍 After lenient detection - Pixels:" << newSkinPixels << "Ratio:" << (double)newSkinPixels / (skinMask.rows * skinMask.cols);
    }
    
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
    
    // Process contours for hand gesture detection (SIMPLE)
    int contourCount = 0;
    for (const auto& contour : contours) {
        if (contourCount >= 5) break; // Process first 5 contours
        contourCount++;
        
        if (contour.size() < 5) continue; // Very lenient - just need 5 points
        
        double area = cv::contourArea(contour);
        if (area < 200 || area > 50000) continue; // Very lenient size filter for poor lighting
        
        // Check if this looks like a hand using simple criteria
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
            
            // Analyze hand gesture using simple methods
            bool isOpen = isHandOpen(contour);
            bool isClosed = isHandClosed(contour);
            double confidence = calculateHandConfidence(contour, resized);
            
            qDebug() << "🔍 Analyzing detection - Confidence:" << confidence << "| Closed:" << isClosed << "| Open:" << isOpen;
            
            if (confidence > 0.3) { // Lower confidence threshold for better detection
                HandDetection detection;
                detection.boundingBox = scaledRect;
                detection.confidence = confidence;
                detection.handType = "hand";
                detection.landmarks = contour;
                detection.isOpen = isOpen;
                detection.isClosed = isClosed;
                detection.palmCenter = findPalmCenter(contour);
                
                detections.append(detection);
                
                if (isClosed && confidence > 0.3) {
                    qDebug() << "🎯 HAND CLOSED DETECTED! Confidence:" << confidence << "| Gesture: Closed hand";
                } else if (isOpen && confidence > 0.3) {
                    qDebug() << "✋ HAND OPEN DETECTED! Confidence:" << confidence << "| Gesture: Open hand";
                }
            } else {
                qDebug() << "🤔 Object detected but not confident enough - Confidence:" << confidence << " (need > 0.3)";
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
    if (contour.size() < 5) return false; // Very lenient for poor lighting
    
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter == 0) return false;
    
    // Check aspect ratio (hands are usually taller than wide) - very lenient
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = (double)boundingRect.width / boundingRect.height;
    
    // Very lenient aspect ratio for poor lighting
    if (aspectRatio < 0.2 || aspectRatio > 5.0) return false;
    
    // Check if it's in the lower part of the image (hands are usually lower than faces) - very lenient
    cv::Point center = findPalmCenter(contour);
    if (center.y < image.rows * 0.1) return false; // Very lenient - only reject if in top 10%
    
    return true;
}

double HandDetector::calculateHandConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image)
{
    if (contour.size() < 5) return 0.0; // Very lenient for poor lighting
    
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

double HandDetector::getHandDetectionProcessingTime() const
{
    return m_averageProcessingTime;
}

bool HandDetector::isOpenGLAvailable() const
{
    return m_openclAvailable || m_openglAvailable;
}

QString HandDetector::getDetectorType() const
{
    if (m_openclAvailable) return "OpenCL";
    if (m_openglAvailable) return "OpenGL";
    return "CPU";
}

double HandDetector::getAverageProcessingTime() const
{
    return m_averageProcessingTime;
}

double HandDetector::getCurrentFPS() const
{
    return m_currentFPS;
}

int HandDetector::getTotalFramesProcessed() const
{
    return m_totalFramesProcessed;
}

// OpenCL detection methods
QList<HandDetection> HandDetector::detectOpenGLHandGestures(const cv::Mat& image)
{
    QList<HandDetection> detections;
    
    try {
        // Convert to GPU memory
        cv::UMat gpuImage = convertToOpenCL(image);
        
        // Create skin mask on GPU
        cv::UMat gpuSkinMask = createOpenCLSkinMask(gpuImage);
        
        // Find contours on GPU
        std::vector<std::vector<cv::Point>> contours = findOpenCLContours(gpuSkinMask);
        
        // Process each contour
        for (const auto& contour : contours) {
            if (contour.size() < 50) continue; // Skip small contours
            
            // Check if it's a hand shape
            if (isOpenCLHandShape(contour, gpuImage)) {
                double confidence = calculateOpenCLHandConfidence(contour, gpuImage);
                
                if (confidence >= m_confidenceThreshold) {
                    HandDetection detection;
                    detection.boundingBox = cv::boundingRect(contour);
                    detection.confidence = confidence;
                    detection.handType = "unknown"; // Could be enhanced with left/right detection
                    detection.landmarks = contour;
                    detection.palmCenter = findOpenCLPalmCenter(contour);
                    detection.fingerTips = findOpenCLFingerTips(contour);
                    detection.isOpen = isHandOpen(contour);
                    detection.isClosed = isHandClosed(contour);
                    detection.isRaised = detection.boundingBox.y < m_frameHeight / 2;
                    
                    detections.append(detection);
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: OpenCL detection error:" << e.what();
    }
    
    return detections;
}

QList<HandDetection> HandDetector::detectOpenGLHandGesturesOptimized(const cv::Mat& image)
{
    QList<HandDetection> detections;
    
    try {
        // Convert to GPU memory
        cv::UMat gpuImage = convertToOpenCL(image);
        
        // Create skin mask on GPU with optimized parameters
        cv::UMat gpuSkinMask = createOpenCLSkinMaskOptimized(gpuImage);
        
        // Find contours on GPU with reduced filtering
        std::vector<std::vector<cv::Point>> contours = findOpenCLContoursOptimized(gpuSkinMask);
        
        // Process each contour with faster criteria
        for (const auto& contour : contours) {
            if (contour.size() < 30) continue; // Reduced minimum size for speed
            
            // Quick hand shape check
            if (isOpenCLHandShapeFast(contour, gpuImage)) {
                double confidence = calculateOpenCLHandConfidenceFast(contour, gpuImage);
                
                if (confidence >= m_confidenceThreshold * 0.8) { // Slightly lower threshold for speed
                    HandDetection detection;
                    detection.boundingBox = cv::boundingRect(contour);
                    detection.confidence = confidence;
                    detection.handType = "unknown";
                    detection.landmarks = contour;
                    detection.palmCenter = findOpenCLPalmCenter(contour);
                    detection.fingerTips = findOpenCLFingerTipsFast(contour);
                    detection.isOpen = isHandOpenFast(contour);
                    detection.isClosed = isHandClosedFast(contour);
                    detection.isRaised = detection.boundingBox.y < m_frameHeight / 2;
                    
                    detections.append(detection);
                    
                    // Limit to 2 detections for speed
                    if (detections.size() >= 2) break;
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: OpenCL optimized detection error:" << e.what();
    }
    
    return detections;
}

QList<HandDetection> HandDetector::detectHandGesturesOptimized(const cv::Mat& image)
{
    QList<HandDetection> detections;
    
    try {
        // Use optimized CPU detection with reduced processing
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(), 0.5, 0.5); // Scale down for speed
        
        cv::Mat skinMask = createSkinMaskOptimized(resizedImage);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Process contours with faster criteria
        for (const auto& contour : contours) {
            if (contour.size() < 20) continue; // Reduced minimum size
            
            double area = cv::contourArea(contour);
            if (area < m_minMotionArea * 0.5) continue; // Reduced area threshold
            
            // Quick hand shape check
            if (isHandShapeFast(contour)) {
                double confidence = calculateHandConfidenceFast(contour);
                
                if (confidence >= m_confidenceThreshold * 0.8) {
                    HandDetection detection;
                    detection.boundingBox = cv::boundingRect(contour);
                    // Scale bounding box back to original size
                    detection.boundingBox.x *= 2;
                    detection.boundingBox.y *= 2;
                    detection.boundingBox.width *= 2;
                    detection.boundingBox.height *= 2;
                    
                    detection.confidence = confidence;
                    detection.handType = "unknown";
                    detection.landmarks = contour;
                    detection.palmCenter = findPalmCenterFast(contour);
                    detection.fingerTips = findFingerTipsFast(contour);
                    detection.isOpen = isHandOpenFast(contour);
                    detection.isClosed = isHandClosedFast(contour);
                    detection.isRaised = detection.boundingBox.y < m_frameHeight / 2;
                    
                    detections.append(detection);
                    
                    // Limit to 2 detections for speed
                    if (detections.size() >= 2) break;
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: CPU optimized detection error:" << e.what();
    }
    
    return detections;
}

cv::UMat HandDetector::createOpenCLSkinMask(const cv::UMat& gpuImage)
{
    cv::UMat gpuSkinMask;
    
    try {
        // Convert to HSV on GPU
        cv::UMat gpuHSV;
        cv::cvtColor(gpuImage, gpuHSV, cv::COLOR_BGR2HSV);
        
        // Create skin color range mask
        cv::UMat gpuMask1, gpuMask2;
        cv::inRange(gpuHSV, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), gpuMask1);
        cv::inRange(gpuHSV, cv::Scalar(170, 20, 70), cv::Scalar(180, 255, 255), gpuMask2);
        
        // Combine masks
        cv::add(gpuMask1, gpuMask2, gpuSkinMask);
        
        // Apply morphological operations
        applyOpenCLMorphology(gpuSkinMask, cv::MORPH_CLOSE);
        applyOpenCLMorphology(gpuSkinMask, cv::MORPH_OPEN);
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error creating skin mask:" << e.what();
    }
    
    return gpuSkinMask;
}

std::vector<std::vector<cv::Point>> HandDetector::findOpenCLContours(const cv::UMat& gpuMask)
{
    std::vector<std::vector<cv::Point>> contours;
    
    try {
        // Download mask to CPU for contour detection (OpenCV contour detection is CPU-only)
        cv::Mat cpuMask = convertFromOpenCL(gpuMask);
        
        // Find contours
        cv::findContours(cpuMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Filter contours by area
        std::vector<std::vector<cv::Point>> filteredContours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > m_minMotionArea && area < m_frameWidth * m_frameHeight * 0.5) {
                filteredContours.push_back(contour);
            }
        }
        
        contours = filteredContours;
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error finding contours:" << e.what();
    }
    
    return contours;
}

bool HandDetector::isOpenCLHandShape(const std::vector<cv::Point>& contour, const cv::UMat& /*gpuImage*/)
{
    if (contour.size() < 50) return false;
    
    // Calculate contour properties
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter == 0) return false;
    
    // Calculate circularity
    double circularity = 4 * M_PI * cv::contourArea(contour) / (perimeter * perimeter);
    
    // Calculate aspect ratio
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    
    // Hand shape criteria
    double area = cv::contourArea(contour);
    bool isHandShape = (circularity > 0.1 && circularity < 0.8) && // Not too circular, not too irregular
                      (aspectRatio > 0.3 && aspectRatio < 2.0) &&   // Reasonable aspect ratio
                      (area > m_minMotionArea);                     // Minimum size
    
    return isHandShape;
}

double HandDetector::calculateOpenCLHandConfidence(const std::vector<cv::Point>& contour, const cv::UMat& /*gpuImage*/)
{
    double confidence = 0.0;
    
    try {
        // Calculate contour properties
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        
        if (perimeter == 0) return 0.0;
        
        // Calculate circularity
        double circularity = 4 * M_PI * area / (perimeter * perimeter);
        
        // Calculate convexity
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hullArea = cv::contourArea(hull);
        double solidity = hullArea > 0 ? area / hullArea : 0;
        
        // Calculate confidence based on multiple factors
        double areaConfidence = std::min(1.0, area / (m_frameWidth * m_frameHeight * 0.1));
        double circularityConfidence = 1.0 - std::abs(circularity - 0.3) / 0.3; // Optimal around 0.3
        double solidityConfidence = solidity;
        
        confidence = (areaConfidence + circularityConfidence + solidityConfidence) / 3.0;
        confidence = std::max(0.0, std::min(1.0, confidence));
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error calculating confidence:" << e.what();
    }
    
    return confidence;
}

std::vector<cv::Point> HandDetector::findOpenCLFingerTips(const std::vector<cv::Point>& contour)
{
    std::vector<cv::Point> fingerTips;
    
    try {
        if (contour.size() < 3) {
            return fingerTips;
        }
        
        // Find convex hull and defects
        std::vector<cv::Point> hull;
        std::vector<int> hullIndices;
        cv::convexHull(contour, hullIndices, false);
        
        // Get hull points from indices
        for (int idx : hullIndices) {
            if (idx >= 0 && idx < static_cast<int>(contour.size())) {
                hull.push_back(contour[idx]);
            }
        }
        
        std::vector<cv::Vec4i> defects;
        if (hullIndices.size() > 3) {
            cv::convexityDefects(contour, hullIndices, defects);
        }
        
        // Find finger tips from convex hull points
        for (const auto& point : hull) {
            // Check if this point is likely a finger tip
            bool isFingerTip = true;
            for (const auto& defect : defects) {
                cv::Point start = contour[defect[0]];
                cv::Point end = contour[defect[1]];
                cv::Point far = contour[defect[2]];
                
                // If this point is near a defect, it's not a finger tip
                if (cv::norm(point - far) < 20) {
                    isFingerTip = false;
                    break;
                }
            }
            
            if (isFingerTip) {
                fingerTips.push_back(point);
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error finding finger tips:" << e.what();
    }
    
    return fingerTips;
}

cv::Point HandDetector::findOpenCLPalmCenter(const std::vector<cv::Point>& contour)
{
    cv::Point palmCenter(0, 0);
    
    try {
        // Calculate centroid
        cv::Moments moments = cv::moments(contour);
        if (moments.m00 != 0) {
            palmCenter.x = static_cast<int>(moments.m10 / moments.m00);
            palmCenter.y = static_cast<int>(moments.m01 / moments.m00);
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error finding palm center:" << e.what();
    }
    
    return palmCenter;
}

// OpenCL utility methods
cv::UMat HandDetector::convertToOpenCL(const cv::Mat& cpuImage)
{
    cv::UMat gpuImage;
    cpuImage.copyTo(gpuImage);
    return gpuImage;
}

cv::Mat HandDetector::convertFromOpenCL(const cv::UMat& gpuImage)
{
    cv::Mat cpuImage;
    gpuImage.copyTo(cpuImage);
    return cpuImage;
}

void HandDetector::applyOpenCLGaussianBlur(cv::UMat& gpuImage, int kernelSize)
{
    cv::GaussianBlur(gpuImage, gpuImage, cv::Size(kernelSize, kernelSize), 1.0);
}

void HandDetector::applyOpenCLMorphology(cv::UMat& gpuImage, int operation)
{
    cv::morphologyEx(gpuImage, gpuImage, operation, m_morphKernel);
}

void HandDetector::preallocateOpenCLMemory(int width, int height)
{
    try {
        // Preallocate OpenCL memory for better performance
        m_openclGray.create(height, width, CV_8UC1);
        m_openclPrevGray.create(height, width, CV_8UC1);
        m_openclMotionMask.create(height, width, CV_8UC1);
        m_openclSkinMask.create(height, width, CV_8UC1);
        m_openclTemp1.create(height, width, CV_8UC1);
        m_openclTemp2.create(height, width, CV_8UC1);
        
        qDebug() << "HandDetector: Preallocated GPU memory for" << width << "x" << height;
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error preallocating GPU memory:" << e.what();
    }
}

void HandDetector::releaseOpenCLMemory()
{
    try {
        m_openclGray.release();
        m_openclPrevGray.release();
        m_openclMotionMask.release();
        m_openclSkinMask.release();
        m_openclTemp1.release();
        m_openclTemp2.release();
        
        qDebug() << "HandDetector: Released GPU memory";
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error releasing GPU memory:" << e.what();
    }
}

void HandDetector::updatePerformanceStats(double processingTime)
{
    m_processingTimes.append(processingTime);
    m_totalFramesProcessed++;
    
    // Keep only last 100 processing times for average calculation
    if (m_processingTimes.size() > 100) {
        m_processingTimes.removeFirst();
    }
    
    // Calculate average processing time
    double sum = 0.0;
    for (double time : m_processingTimes) {
        sum += time;
    }
    m_averageProcessingTime = sum / m_processingTimes.size();
    
    // Calculate current FPS
    if (m_averageProcessingTime > 0) {
        m_currentFPS = 1000.0 / m_averageProcessingTime;
    }
}

// Stub implementations for OpenCL methods that aren't fully implemented yet
QList<HandDetection> HandDetector::detectHandsByOpenCLShape(const cv::UMat& /*gpuImage*/)
{
    return QList<HandDetection>();
}

QList<HandDetection> HandDetector::detectHandsByOpenCLMotion(const cv::UMat& /*gpuImage*/)
{
    return QList<HandDetection>();
}

QList<HandDetection> HandDetector::detectHandsByOpenCLKeypoints(const cv::UMat& /*gpuImage*/)
{
    return QList<HandDetection>();
}

cv::UMat HandDetector::createOpenCLMotionMask(const cv::UMat& /*gpuImage*/)
{
    return cv::UMat();
}

bool HandDetector::detectOpenCLMotion(const cv::UMat& /*gpuImage*/)
{
    return false;
}

bool HandDetector::acquireOpenCLRoiFromMotion(const cv::UMat& /*gpuGray*/, const cv::UMat& /*gpuMotionMask*/)
{
    return false;
}

void HandDetector::trackOpenCLRoiLK(const cv::UMat& /*gpuGrayPrev*/, const cv::UMat& /*gpuGrayCurr*/)
{
}

void HandDetector::updateOpenCLMotionHistory(const cv::UMat& /*gpuMotionMask*/)
{
}

bool HandDetector::isOpenCLMotionStable()
{
    return false;
}

bool HandDetector::analyzeOpenCLGestureClosed(const cv::UMat& /*gpuGray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

bool HandDetector::analyzeOpenCLGestureOpen(const cv::UMat& /*gpuGray*/, const cv::Rect& /*roi*/) const
{
    return false;
}

// Optimized helper methods for faster processing
cv::UMat HandDetector::createOpenCLSkinMaskOptimized(const cv::UMat& gpuImage)
{
    cv::UMat gpuSkinMask;
    
    try {
        // Simplified skin detection for speed
        cv::UMat gpuHSV;
        cv::cvtColor(gpuImage, gpuHSV, cv::COLOR_BGR2HSV);
        
        // Single skin range for speed
        cv::inRange(gpuHSV, cv::Scalar(0, 30, 60), cv::Scalar(20, 255, 255), gpuSkinMask);
        
        // Minimal morphological operations
        applyOpenCLMorphology(gpuSkinMask, cv::MORPH_CLOSE);
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error creating optimized skin mask:" << e.what();
    }
    
    return gpuSkinMask;
}

std::vector<std::vector<cv::Point>> HandDetector::findOpenCLContoursOptimized(const cv::UMat& gpuMask)
{
    std::vector<std::vector<cv::Point>> contours;
    
    try {
        cv::Mat cpuMask = convertFromOpenCL(gpuMask);
        cv::findContours(cpuMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Simplified filtering
        std::vector<std::vector<cv::Point>> filteredContours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > m_minMotionArea * 0.3 && area < m_frameWidth * m_frameHeight * 0.3) {
                filteredContours.push_back(contour);
            }
        }
        
        contours = filteredContours;
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error finding optimized contours:" << e.what();
    }
    
    return contours;
}

bool HandDetector::isOpenCLHandShapeFast(const std::vector<cv::Point>& contour, const cv::UMat& /*gpuImage*/)
{
    if (contour.size() < 30) return false;
    
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter == 0) return false;
    
    double circularity = 4 * M_PI * area / (perimeter * perimeter);
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    
    // Simplified criteria for speed
    return (circularity > 0.05 && circularity < 0.9) && 
           (aspectRatio > 0.2 && aspectRatio < 3.0) && 
           (area > m_minMotionArea * 0.3);
}

double HandDetector::calculateOpenCLHandConfidenceFast(const std::vector<cv::Point>& contour, const cv::UMat& /*gpuImage*/)
{
    double confidence = 0.0;
    
    try {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        
        if (perimeter == 0) return 0.0;
        
        double circularity = 4 * M_PI * area / (perimeter * perimeter);
        double areaConfidence = std::min(1.0, area / (m_frameWidth * m_frameHeight * 0.05));
        double circularityConfidence = 1.0 - std::abs(circularity - 0.25) / 0.25;
        
        confidence = (areaConfidence + circularityConfidence) / 2.0;
        confidence = std::max(0.0, std::min(1.0, confidence));
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error calculating fast confidence:" << e.what();
    }
    
    return confidence;
}



std::vector<cv::Point> HandDetector::findOpenCLFingerTipsFast(const std::vector<cv::Point>& contour)
{
    std::vector<cv::Point> fingerTips;
    
    try {
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull, false);
        
        // Simplified finger tip detection
        for (const auto& point : hull) {
            fingerTips.push_back(point);
            if (fingerTips.size() >= 5) break; // Limit for speed
        }
        
    } catch (const cv::Exception& /*e*/) {
        qWarning() << "HandDetector: Error finding fast finger tips";
    }
    
    return fingerTips;
}

bool HandDetector::isHandOpenFast(const std::vector<cv::Point>& contour)
{
    try {
        // More lenient detection for poor camera quality
        std::vector<cv::Point> fingerTips = findFingerTipsFast(contour);
        
        // For poor camera quality, be more lenient with open hand detection
        if (fingerTips.size() >= 3) {
            return true; // Consider it open if 3 or more finger tips
        }
        
        // Additional check: if contour is elongated, likely open
        cv::Rect boundingRect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
        if (aspectRatio > 0.8 && aspectRatio < 1.5) { // Roughly square-ish suggests open hand
            return true;
        }
        
        return false;
    } catch (const cv::Exception& /*e*/) {
        return false;
    }
}

bool HandDetector::isHandClosedFast(const std::vector<cv::Point>& contour)
{
    try {
        // Much more lenient detection for poor camera quality
        std::vector<cv::Point> fingerTips = findFingerTipsFast(contour);
        
        // For poor camera quality, be very lenient with closed hand detection
        if (fingerTips.size() <= 3) {
            return true; // Consider it closed if 3 or fewer finger tips
        }
        
        // Additional checks for poor camera quality
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        
        if (perimeter > 0) {
            double circularity = 4 * M_PI * area / (perimeter * perimeter);
            if (circularity > 0.5) { // Lower threshold for poor camera quality
                return true;
            }
        }
        
        // Check if contour is very small (likely closed hand)
        if (area < m_minMotionArea * 0.5) {
            return true;
        }
        
        // Check aspect ratio - very compact shapes are likely closed hands
        cv::Rect boundingRect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
        if (aspectRatio > 0.7 && aspectRatio < 1.3) { // Roughly square-ish
            return true;
        }
        
        return false;
    } catch (const cv::Exception& /*e*/) {
        return false;
    }
}

cv::Mat HandDetector::createSkinMaskOptimized(const cv::Mat& image)
{
    cv::Mat skinMask;
    
    try {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        
        // More lenient skin detection for poor camera quality
        cv::inRange(hsv, cv::Scalar(0, 20, 40), cv::Scalar(25, 255, 255), skinMask);
        
        // Additional skin range for poor lighting conditions
        cv::Mat skinMask2;
        cv::inRange(hsv, cv::Scalar(0, 10, 30), cv::Scalar(30, 255, 255), skinMask2);
        
        // Combine both masks
        cv::bitwise_or(skinMask, skinMask2, skinMask);
        
        // Minimal morphological operations
        cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)));
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error creating optimized CPU skin mask:" << e.what();
    }
    
    return skinMask;
}

bool HandDetector::isHandShapeFast(const std::vector<cv::Point>& contour)
{
    if (contour.size() < 15) return false; // Reduced minimum size for poor camera quality
    
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    if (perimeter == 0) return false;
    
    // Simplified calculations for better performance
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;
    
    // More lenient criteria for poor camera quality
    return (aspectRatio > 0.15 && aspectRatio < 4.0) && 
           (area > m_minMotionArea * 0.2); // Reduced area threshold
}

double HandDetector::calculateHandConfidenceFast(const std::vector<cv::Point>& contour)
{
    double confidence = 0.0;
    
    try {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        
        if (perimeter == 0) return 0.0;
        
        double circularity = 4 * M_PI * area / (perimeter * perimeter);
        double areaConfidence = std::min(1.0, area / (m_frameWidth * m_frameHeight * 0.05));
        double circularityConfidence = 1.0 - std::abs(circularity - 0.25) / 0.25;
        
        confidence = (areaConfidence + circularityConfidence) / 2.0;
        confidence = std::max(0.0, std::min(1.0, confidence));
        
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error calculating fast CPU confidence:" << e.what();
    }
    
    return confidence;
}

cv::Point HandDetector::findPalmCenterFast(const std::vector<cv::Point>& contour)
{
    cv::Point palmCenter(0, 0);
    
    try {
        cv::Moments moments = cv::moments(contour);
        if (moments.m00 != 0) {
            palmCenter.x = static_cast<int>(moments.m10 / moments.m00);
            palmCenter.y = static_cast<int>(moments.m01 / moments.m00);
        }
    } catch (const cv::Exception& e) {
        qWarning() << "HandDetector: Error finding fast palm center:" << e.what();
    }
    
    return palmCenter;
}

std::vector<cv::Point> HandDetector::findFingerTipsFast(const std::vector<cv::Point>& contour)
{
    std::vector<cv::Point> fingerTips;
    
    try {
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull, false);
        
        for (const auto& point : hull) {
            fingerTips.push_back(point);
            if (fingerTips.size() >= 5) break; // Limit for speed
        }
        
    } catch (const cv::Exception& /*e*/) {
        qWarning() << "HandDetector: Error finding fast CPU finger tips";
    }
    
    return fingerTips;
}


