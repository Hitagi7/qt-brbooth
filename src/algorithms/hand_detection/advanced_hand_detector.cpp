#include "algorithms/advanced_hand_detector.h"
#include <QDebug>
#include <QTimer>
#include <QThread>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

AdvancedHandDetector::AdvancedHandDetector(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_confidenceThreshold(0.8) // STRICT threshold to prevent false positives
    , m_showBoundingBox(true)
    , m_performanceMode(1) // Balanced mode
    , m_backgroundInitialized(false)
    , m_frameCount(0)
    , m_motionStabilityCount(0)
    , m_motionDetected(false)
    , m_lastProcessingTime(0.0)
    , m_detectionFPS(0)
    , m_trackingFrames(0)
    , m_hasStableTracking(false)
    , m_wasHandClosed(false)
    , m_wasHandOpen(false)
    , m_gestureStableFrames(0)
{
}

AdvancedHandDetector::~AdvancedHandDetector() {
}

bool AdvancedHandDetector::initialize() {
    QMutexLocker locker(&m_mutex);
    
    if (m_initialized) {
        return true;
    }
    
    try {
        m_initialized = true;
        m_backgroundInitialized = false;
        m_frameCount = 0;
        m_trackingFrames = 0;
        m_hasStableTracking = false;
        qDebug() << "Advanced hand detector initialized successfully with palm-based detection";
        return true;
    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV exception during advanced hand detector initialization:" << e.what();
        m_initialized = false;
        return false;
    } catch (const std::exception& e) {
        qWarning() << "Exception during advanced hand detector initialization:" << e.what();
        m_initialized = false;
        return false;
    }
}

bool AdvancedHandDetector::isInitialized() const {
    QMutexLocker locker(&m_mutex);
    return m_initialized;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detect(const cv::Mat& image) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_initialized || image.empty()) {
        return QList<AdvancedHandDetection>();
    }
    
    // Performance optimization: only process every few frames
    m_frameCount++;
    if (m_frameCount % PROCESSING_INTERVAL != 0) {
        return QList<AdvancedHandDetection>();
    }
    
    // Debug output only every 120 frames to reduce console spam
    if (m_frameCount % 120 == 0) {
        qDebug() << "🔄 Processing hand detection frame" << m_frameCount;
    }
    
    m_processingTimer.start();
    
    try {
        // Optimize image for processing
        cv::Mat processedImage = image.clone();
        optimizeForPerformance(processedImage);
        
        // Update background model for motion detection
        updateBackgroundModel(processedImage);
        
        // MOTION-BASED DETECTION: Only detect hands if there's motion
        QList<AdvancedHandDetection> detections;
        
        if (detectMotion(processedImage)) {
            qDebug() << "🎯 MOTION DETECTED! Checking for hand shapes...";
            detections = detectHandsPalmBased(processedImage);
        } else {
            qDebug() << "⏸️ No motion detected - skipping hand detection";
        }
        
        // Update performance metrics
        m_lastProcessingTime = m_processingTimer.elapsed();
        m_detectionFPS = static_cast<int>(1000.0 / m_lastProcessingTime);
        
        // Debug output
        if (!detections.isEmpty()) {
            qDebug() << "Palm-based hand detection found" << detections.size() << "hands, FPS:" << m_detectionFPS;
            for (int i = 0; i < detections.size(); ++i) {
                const auto& det = detections[i];
                qDebug() << "Hand" << i << "confidence:" << det.confidence 
                         << "raised:" << det.isRaised
                         << "palm center:" << det.palmCenter.x << det.palmCenter.y
                         << "bbox:" << det.boundingBox.x << det.boundingBox.y 
                         << det.boundingBox.width << det.boundingBox.height;
            }
        }
        
        return detections;
    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV exception during advanced hand detection:" << e.what();
        return QList<AdvancedHandDetection>();
    } catch (const std::exception& e) {
        qWarning() << "Exception during advanced hand detection:" << e.what();
        return QList<AdvancedHandDetection>();
    }
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHandsPalmBased(const cv::Mat& image) {
    QList<AdvancedHandDetection> detections;
    
    // Method 1: Palm keypoint detection (most accurate)
    QList<AdvancedHandDetection> palmDetections = detectHandsByKeypoints(image);
    detections.append(palmDetections);
    
    // Method 2: Motion-based detection (backup for moving hands)
    if (detections.isEmpty()) {
        QList<AdvancedHandDetection> motionDetections = detectHandsByMotion(image);
        detections.append(motionDetections);
    }
    
    // Apply hand tracking stability and gesture detection
    if (!detections.isEmpty()) {
        // Track gesture states for the best detection
        AdvancedHandDetection bestDetection = detections.first();
        
        // Check if hand is open or closed
        bool isCurrentlyOpen = isHandOpen(bestDetection.landmarks);
        bool isCurrentlyClosed = isHandClosed(bestDetection.landmarks);
        
        // Update gesture state tracking - IMMEDIATE RESPONSE
        if (isCurrentlyOpen) {
            m_wasHandOpen = true;
            m_gestureStableFrames = 0;
            qDebug() << "🖐️ Hand OPEN detected - waiting for close gesture";
        } else if (isCurrentlyClosed) {
            if (m_wasHandOpen) { // Only trigger if we were previously open
                m_wasHandClosed = true;
                qDebug() << "🎯 Hand CLOSED detected! Ready for IMMEDIATE capture trigger!";
            }
            m_gestureStableFrames++;
        } else {
            // Hand is in intermediate state, maintain stability
            m_gestureStableFrames++;
        }
        
        // If we have previous detections, try to maintain continuity
        if (!m_lastHandDetections.isEmpty()) {
            QList<AdvancedHandDetection> stableDetections;
            
            for (const auto& currentDet : detections) {
                bool isStable = false;
                
                // Check if this detection is close to any previous detection
                for (const auto& lastDet : m_lastHandDetections) {
                    double distance = cv::norm(currentDet.palmCenter - lastDet.palmCenter);
                    double areaDiff = abs(cv::contourArea(currentDet.landmarks) - cv::contourArea(lastDet.landmarks));
                    
                    // If close enough, consider it stable
                    if (distance < 50 && areaDiff < 1000) {
                        isStable = true;
                        break;
                    }
                }
                
                if (isStable) {
                    stableDetections.append(currentDet);
                }
            }
            
            // If we have stable detections, use them; otherwise use all detections
            if (!stableDetections.isEmpty()) {
                detections = stableDetections;
            }
        }
        
        // Update last detections for next frame
        m_lastHandDetections = detections;
        
                    // Debug output for gesture detection
            if (isCurrentlyOpen || isCurrentlyClosed) {
                double closureRatio = calculateHandClosureRatio(bestDetection.landmarks);
                qDebug() << "Hand gesture - Open:" << isCurrentlyOpen << "Closed:" << isCurrentlyClosed 
                         << "Closure ratio:" << closureRatio << "Stable frames:" << m_gestureStableFrames
                         << "State: Open=" << m_wasHandOpen << " Closed=" << m_wasHandClosed
                         << "Confidence:" << static_cast<int>(bestDetection.confidence * 100) << "%";
            }
    } else {
        // If no detections, gradually clear last detections (hysteresis)
        if (!m_lastHandDetections.isEmpty()) {
            m_trackingFrames++;
            if (m_trackingFrames > 5) { // Keep last detections for 5 frames
                m_lastHandDetections.clear();
                m_trackingFrames = 0;
            }
        }
        
        // Reset gesture states if no hand detected for too long
        if (m_trackingFrames > 10) {
            m_wasHandOpen = false;
            m_wasHandClosed = false;
            m_gestureStableFrames = 0;
        }
    }
    
    // Sort by confidence and limit detections
    std::sort(detections.begin(), detections.end(), 
              [](const AdvancedHandDetection& a, const AdvancedHandDetection& b) {
                  return a.confidence > b.confidence;
              });
    
    if (detections.size() > MAX_DETECTIONS) {
        detections = detections.mid(0, MAX_DETECTIONS);
    }
    
    return detections;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHandsByKeypoints(const cv::Mat& image) {
    QList<AdvancedHandDetection> detections;
    
    // METHOD 1: HAND SHAPE DETECTION (NOT SKIN COLOR)
    QList<AdvancedHandDetection> shapeDetections = detectHandsByShape(image);
    detections.append(shapeDetections);
    
    // METHOD 2: Edge-based hand detection (backup)
    if (detections.isEmpty()) {
        QList<AdvancedHandDetection> edgeDetections = detectHandsByEdges(image);
        detections.append(edgeDetections);
    }
    
    return detections;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHandsByShape(const cv::Mat& image) {
    QList<AdvancedHandDetection> detections;
    
    // ULTRA-LIGHT: Convert to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // ULTRA-LIGHT: Simple threshold
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 120, 255, cv::THRESH_BINARY_INV);
    
    // ULTRA-LIGHT: Find contours with minimal processing
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
    
    // BALANCED: Process contours to find hand
    for (const auto& contour : contours) {
        if (isHandShapeAdvanced(contour, image)) {
            AdvancedHandDetection detection;
            detection.boundingBox = cv::boundingRect(contour);
            detection.confidence = 0.9; // High confidence for strict detection
            detection.handType = "shape";
            detection.landmarks = contour;
            detection.isRaised = true;
            detection.palmCenter = cv::Point(detection.boundingBox.x + detection.boundingBox.width/2, 
                                           detection.boundingBox.y + detection.boundingBox.height/2);
            
            detections.append(detection);
            qDebug() << "🎯 Hand detected! Area:" << cv::contourArea(contour) << "Closed:" << isHandClosed(contour);
        }
    }
    
    return detections;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHandsByEdges(const cv::Mat& image) {
    QList<AdvancedHandDetection> detections;
    
    // Convert to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Apply Canny edge detection
    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);
    
    // Dilate edges to connect finger edges
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::dilate(edges, edges, kernel);
    
    // Find contours in edge image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Analyze edge contours for hand-like patterns
    for (const auto& contour : contours) {
        if (isHandEdgePattern(contour, image)) {
            double confidence = calculateEdgeHandConfidence(contour, image);
            
            if (confidence >= m_confidenceThreshold * 0.8) { // Lower threshold for edge detection
                AdvancedHandDetection detection;
                detection.boundingBox = cv::boundingRect(contour);
                detection.confidence = confidence;
                detection.handType = "edge";
                detection.landmarks = contour;
                detection.isRaised = isRaisedHand(detection.boundingBox.tl(), image);
                detection.palmCenter = findPalmCenter(contour);
                detection.fingerTips = findFingerTipsFromContour(contour);
                detection.palmKeypoints = extractPalmKeypoints(contour);
                
                detections.append(detection);
                
                qDebug() << "Hand edge detected - Confidence:" << confidence;
            }
        }
    }
    
    return detections;
}

std::vector<cv::Point> AdvancedHandDetector::detectPalmKeypoints(const cv::Mat& skinMask) {
    std::vector<cv::Point> palmKeypoints;
    
    // Find contours in skin mask
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return palmKeypoints;
    }
    
    // Find the largest contour (most likely to be a hand)
    auto largestContour = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    
    if (largestContour == contours.end()) {
        return palmKeypoints;
    }
    
    const std::vector<cv::Point>& contour = *largestContour;
    double area = cv::contourArea(contour);
    
    // Check area constraints
    if (area < MIN_HAND_AREA || area > MAX_HAND_AREA) {
        return palmKeypoints;
    }
    
    // Find convex hull and defects for palm keypoints
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Extract palm keypoints from convex hull
    for (int i = 0; i < std::min(static_cast<int>(hullIndices.size()), PALM_KEYPOINTS_COUNT); i++) {
        palmKeypoints.push_back(contour[hullIndices[i]]);
    }
    
    // If we don't have enough keypoints, add some from the contour
    while (palmKeypoints.size() < PALM_KEYPOINTS_COUNT) {
        int index = static_cast<int>(palmKeypoints.size()) * static_cast<int>(contour.size()) / PALM_KEYPOINTS_COUNT;
        if (index < static_cast<int>(contour.size())) {
            palmKeypoints.push_back(contour[index]);
        } else {
            break;
        }
    }
    
    return palmKeypoints;
}

// Hand gesture detection methods
bool AdvancedHandDetector::isHandClosed(const std::vector<cv::Point>& contour) {
    // STRICT: Multiple detection methods for closed hand
    double area = cv::contourArea(contour);
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
    
    // Method 1: Area density (closed hand is more compact)
    double areaDensity = area / (boundingRect.width * boundingRect.height);
    bool densityCheck = areaDensity > 0.3; // Strict threshold
    
    // Method 2: Aspect ratio (closed hand is more square-like)
    bool aspectCheck = aspectRatio < 2.0; // Strict threshold
    
    // Method 3: Size check (closed hand should be reasonably sized)
    bool sizeCheck = area > 500 && area < 15000;
    
    // Method 4: Position check (should be in upper part of image)
    bool positionCheck = boundingRect.y < boundingRect.height * 2;
    
    // Accept if ALL checks pass (very strict)
    int passedChecks = (densityCheck ? 1 : 0) + (aspectCheck ? 1 : 0) + (sizeCheck ? 1 : 0) + (positionCheck ? 1 : 0);
    
    return passedChecks == 4; // ALL 4 checks must pass (very strict)
}

bool AdvancedHandDetector::isHandOpen(const std::vector<cv::Point>& contour) {
    // STRICT: Multiple detection methods for open hand
    double area = cv::contourArea(contour);
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
    
    // Method 1: Area density (open hand is less compact)
    double areaDensity = area / (boundingRect.width * boundingRect.height);
    bool densityCheck = areaDensity < 0.4; // Strict threshold
    
    // Method 2: Aspect ratio (open hand is more elongated)
    bool aspectCheck = aspectRatio > 1.5; // Strict threshold
    
    // Method 3: Size check (open hand should be reasonably sized)
    bool sizeCheck = area > 500 && area < 15000;
    
    // Method 4: Position check (should be in upper part of image)
    bool positionCheck = boundingRect.y < boundingRect.height * 2;
    
    // Accept if ALL checks pass (very strict)
    int passedChecks = (densityCheck ? 1 : 0) + (aspectCheck ? 1 : 0) + (sizeCheck ? 1 : 0) + (positionCheck ? 1 : 0);
    
    return passedChecks == 4; // ALL 4 checks must pass (very strict)
}

double AdvancedHandDetector::calculateHandClosureRatio(const std::vector<cv::Point>& contour) {
    // ULTRA-FAST: Simple area density calculation
    double area = cv::contourArea(contour);
    cv::Rect boundingRect = cv::boundingRect(contour);
    double areaDensity = area / (boundingRect.width * boundingRect.height);
    
    // ULTRA-FAST: Return area density as closure ratio
    return std::max(0.0, std::min(1.0, areaDensity));
}

bool AdvancedHandDetector::shouldTriggerCapture() {
    // Trigger capture when hand transitions from open to closed (IMMEDIATE response)
    bool shouldTrigger = m_wasHandOpen && m_wasHandClosed;
    
    // Reset state after triggering
    if (shouldTrigger) {
        m_wasHandOpen = false;
        m_wasHandClosed = false;
        m_gestureStableFrames = 0;
        qDebug() << "🎬 CAPTURE TRIGGERED! Hand closed gesture detected IMMEDIATELY!";
    }
    
    return shouldTrigger;
}

void AdvancedHandDetector::resetGestureState() {
    m_wasHandOpen = false;
    m_wasHandClosed = false;
    m_gestureStableFrames = 0;
    qDebug() << "🔄 Hand detection gesture state RESET";
}

cv::Point AdvancedHandDetector::findPalmCenter(const std::vector<cv::Point>& palmKeypoints) {
    if (palmKeypoints.empty()) {
        return cv::Point(0, 0);
    }
    
    // Calculate centroid of palm keypoints
    cv::Point center(0, 0);
    for (const auto& point : palmKeypoints) {
        center += point;
    }
    center.x /= static_cast<int>(palmKeypoints.size());
    center.y /= static_cast<int>(palmKeypoints.size());
    
    return center;
}

std::vector<cv::Point> AdvancedHandDetector::findFingerTips(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter) {
    std::vector<cv::Point> fingerTips;
    
    if (palmKeypoints.empty()) {
        return fingerTips;
    }
    
    // Find the points furthest from palm center (likely finger tips)
    std::vector<std::pair<double, cv::Point>> distances;
    for (const auto& point : palmKeypoints) {
        double distance = cv::norm(point - palmCenter);
        distances.push_back({distance, point});
    }
    
    // Sort by distance (furthest first)
    std::sort(distances.begin(), distances.end(),
        [](const std::pair<double, cv::Point>& a, const std::pair<double, cv::Point>& b) {
            return a.first > b.first;
        });
    
    // Take the top 3-5 points as finger tips
    for (int i = 0; i < std::min(static_cast<int>(distances.size()), 5); i++) {
        if (distances[i].first > 20) { // Minimum distance from palm center
            fingerTips.push_back(distances[i].second);
        }
    }
    
    return fingerTips;
}

bool AdvancedHandDetector::validateHandStructure(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter) {
    if (palmKeypoints.size() < 5) {
        return false;
    }
    
    // Check if keypoints form a reasonable hand structure
    double totalDistance = 0;
    for (const auto& point : palmKeypoints) {
        totalDistance += cv::norm(point - palmCenter);
    }
    double avgDistance = totalDistance / palmKeypoints.size();
    
    // Hand should have reasonable spread of keypoints
    return avgDistance > 15 && avgDistance < 100;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHandsByMotion(const cv::Mat& image) {
    QList<AdvancedHandDetection> detections;
    
    if (!m_backgroundInitialized) {
        return detections;
    }
    
    // Create motion mask
    cv::Mat motionMask = createMotionMask(image);
    
    // Find contours in motion mask
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(motionMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter contours for hand-like motion
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        
        // Check area constraints
        if (area < MIN_HAND_AREA || area > MAX_HAND_AREA) {
            continue;
        }
        
        cv::Rect boundingRect = cv::boundingRect(contour);
        
        // Check if motion is in upper portion (raised hand)
        double imageHeight = image.rows;
        double motionTop = boundingRect.y;
        
        if (motionTop > imageHeight * 0.4) { // Upper 40% of image
            continue;
        }
        
        // Check aspect ratio for hand-like shape
        double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
        if (aspectRatio < 0.8 || aspectRatio > 3.0) {
            continue;
        }
        
        // Calculate confidence based on motion intensity and position
        double motionIntensity = cv::countNonZero(motionMask(boundingRect)) / (double)(boundingRect.width * boundingRect.height);
        double positionConfidence = 1.0 - (motionTop / (imageHeight * 0.4));
        double confidence = motionIntensity * 0.6 + positionConfidence * 0.4;
        
        if (confidence >= m_confidenceThreshold) {
            AdvancedHandDetection detection;
            detection.boundingBox = boundingRect;
            detection.confidence = confidence;
            detection.handType = "motion";
            detection.landmarks = contour;
            detection.isRaised = true;
            detection.palmCenter = findPalmCenter(contour);
            detection.fingerTips = findFingerTips(contour, detection.palmCenter);
            
            detections.append(detection);
            
            qDebug() << "Motion hand detected - Area:" << area << "Confidence:" << confidence;
        }
    }
    
    return detections;
}

cv::Mat AdvancedHandDetector::createEnhancedSkinMask(const cv::Mat& image) {
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
    
    // MUCH MORE COMPREHENSIVE skin color ranges for better accuracy
    std::vector<std::pair<cv::Scalar, cv::Scalar>> skinRanges = {
        // Very light skin tones
        {cv::Scalar(0, 5, 80), cv::Scalar(15, 255, 255)},
        {cv::Scalar(0, 10, 70), cv::Scalar(20, 255, 255)},
        {cv::Scalar(0, 15, 60), cv::Scalar(25, 255, 255)},
        
        // Light skin tones
        {cv::Scalar(0, 10, 60), cv::Scalar(20, 255, 255)},
        {cv::Scalar(0, 20, 50), cv::Scalar(25, 255, 255)},
        {cv::Scalar(0, 30, 40), cv::Scalar(30, 255, 255)},
        
        // Medium skin tones
        {cv::Scalar(0, 15, 50), cv::Scalar(25, 255, 255)},
        {cv::Scalar(0, 25, 40), cv::Scalar(35, 255, 255)},
        {cv::Scalar(0, 35, 30), cv::Scalar(40, 255, 255)},
        
        // Darker skin tones
        {cv::Scalar(0, 20, 30), cv::Scalar(35, 255, 255)},
        {cv::Scalar(0, 30, 20), cv::Scalar(40, 255, 255)},
        {cv::Scalar(0, 40, 15), cv::Scalar(45, 255, 255)},
        
        // Reddish skin tones (wider range)
        {cv::Scalar(150, 10, 50), cv::Scalar(180, 255, 255)},
        {cv::Scalar(160, 15, 40), cv::Scalar(180, 255, 255)},
        {cv::Scalar(170, 20, 30), cv::Scalar(180, 255, 255)},
        
        // Additional ranges for better coverage
        {cv::Scalar(0, 5, 70), cv::Scalar(15, 255, 255)},
        {cv::Scalar(0, 10, 80), cv::Scalar(20, 255, 255)},
        {cv::Scalar(0, 8, 90), cv::Scalar(18, 255, 255)}
    };
    
    cv::Mat combinedMask = cv::Mat::zeros(image.size(), CV_8UC1);
    
    // Combine masks from different skin color ranges
    for (const auto& range : skinRanges) {
        cv::Mat skinMask;
        cv::inRange(hsvImage, range.first, range.second, skinMask);
        cv::bitwise_or(combinedMask, skinMask, combinedMask);
    }
    
    // Apply enhanced morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, kernel);
    
    // Apply Gaussian blur for smoother edges
    cv::GaussianBlur(combinedMask, combinedMask, cv::Size(7, 7), 0);
    
    return combinedMask;
}

cv::Mat AdvancedHandDetector::createMotionMask(const cv::Mat& image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    cv::Mat motionMask;
    cv::absdiff(grayImage, m_backgroundModel, motionMask);
    
    // Apply threshold to get significant motion
    cv::threshold(motionMask, motionMask, 20, 255, cv::THRESH_BINARY);
    
    // Apply morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, kernel);
    
    return motionMask;
}

void AdvancedHandDetector::updateBackgroundModel(const cv::Mat& frame) {
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    
    if (!m_backgroundInitialized) {
        grayFrame.copyTo(m_backgroundModel);
        m_backgroundInitialized = true;
    } else {
        // Update background model using running average
        cv::addWeighted(m_backgroundModel, 0.95, grayFrame, 0.05, 0, m_backgroundModel);
    }
}

cv::Mat AdvancedHandDetector::getBackgroundModel() {
    return m_backgroundModel.clone();
}

void AdvancedHandDetector::optimizeForPerformance(cv::Mat& image) {
    // ULTRA-LIGHT: Resize to tiny size for maximum speed
    if (image.cols > 60 || image.rows > 45) {
        double scale = std::min(60.0 / image.cols, 45.0 / image.rows);
        cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_NEAREST);
    }
}

bool AdvancedHandDetector::isHandShape(const std::vector<cv::Point>& contour, const cv::Mat& /*image*/) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    
    // Calculate shape properties
    double circularity = 4 * M_PI * area / (perimeter * perimeter);
    double aspectRatio = static_cast<double>(cv::boundingRect(contour).height) / cv::boundingRect(contour).width;
    
    // Check convexity defects (fingers)
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Hand shape criteria
    bool hasReasonableArea = (area > MIN_HAND_AREA && area < MAX_HAND_AREA);
    bool hasReasonableAspectRatio = (aspectRatio > 0.8 && aspectRatio < 3.0);
    bool hasFingerDefects = (defects.size() >= 2 && defects.size() <= 8);
    bool isNotTooCircular = (circularity < 0.7);
    
    return hasReasonableArea && hasReasonableAspectRatio && hasFingerDefects && isNotTooCircular;
}

double AdvancedHandDetector::calculateHandConfidence(const std::vector<cv::Point>& palmKeypoints, const cv::Point& palmCenter, const cv::Mat& image) {
    if (palmKeypoints.empty()) {
        return 0.0;
    }
    
    // Calculate confidence based on palm keypoint distribution
    double totalDistance = 0;
    double maxDistance = 0;
    for (const auto& point : palmKeypoints) {
        double distance = cv::norm(point - palmCenter);
        totalDistance += distance;
        maxDistance = std::max(maxDistance, distance);
    }
    double avgDistance = totalDistance / palmKeypoints.size();
    
    // Position confidence (higher for upper portion)
    double positionConfidence = 1.0 - (palmCenter.y / (image.rows * 0.6));
    
    // Distribution confidence (good spread of keypoints)
    double distributionConfidence = std::min(1.0, avgDistance / 30.0);
    
    // Keypoint count confidence
    double keypointConfidence = std::min(1.0, palmKeypoints.size() / static_cast<double>(PALM_KEYPOINTS_COUNT));
    
    // Weighted confidence calculation
    double confidence = positionConfidence * 0.4 + distributionConfidence * 0.3 + keypointConfidence * 0.3;
    
    return std::min(1.0, confidence);
}

bool AdvancedHandDetector::isRaisedHand(const cv::Point& palmCenter, const cv::Mat& image) {
    // Check if palm center is in upper portion of image
    double imageHeight = image.rows;
    return palmCenter.y < imageHeight * 0.4;
}

void AdvancedHandDetector::setConfidenceThreshold(double threshold) {
    QMutexLocker locker(&m_mutex);
    m_confidenceThreshold = std::max(0.0, std::min(1.0, threshold));
}

double AdvancedHandDetector::getConfidenceThreshold() const {
    QMutexLocker locker(&m_mutex);
    return m_confidenceThreshold;
}

void AdvancedHandDetector::setShowBoundingBox(bool show) {
    QMutexLocker locker(&m_mutex);
    m_showBoundingBox = show;
}

bool AdvancedHandDetector::getShowBoundingBox() const {
    QMutexLocker locker(&m_mutex);
    return m_showBoundingBox;
}

void AdvancedHandDetector::setPerformanceMode(int mode) {
    QMutexLocker locker(&m_mutex);
    m_performanceMode = std::max(0, std::min(2, mode));
}

int AdvancedHandDetector::getPerformanceMode() const {
    QMutexLocker locker(&m_mutex);
    return m_performanceMode;
}

bool AdvancedHandDetector::isHandShapeAdvanced(const std::vector<cv::Point>& contour, const cv::Mat& image) {
    // STRICT: Only detect actual raised hands
    if (contour.size() < 15) {
        return false;
    }
    
    double area = cv::contourArea(contour);
    
    // STRICT: Hand-sized area range
    if (area < 1000 || area > 8000) {
        return false;
    }
    
    // STRICT: Simple bounding box
    cv::Rect boundingRect = cv::boundingRect(contour);
    
    // STRICT: Hand-like aspect ratio (taller than wide)
    double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
    if (aspectRatio < 1.5 || aspectRatio > 2.5) {
        return false;
    }
    
    // STRICT: Must be in upper 40% of image (definitely raised hand)
    double imageHeight = image.rows;
    double imageWidth = image.cols;
    
    if (boundingRect.y > imageHeight * 0.4) {
        return false;
    }
    
    // STRICT: Must be well-centered
    double centerX = boundingRect.x + boundingRect.width / 2.0;
    if (centerX < imageWidth * 0.3 || centerX > imageWidth * 0.7) {
        return false;
    }
    
    // STRICT: Must be reasonable size relative to image
    double handAreaRatio = area / (imageHeight * imageWidth);
    if (handAreaRatio < 0.01 || handAreaRatio > 0.1) {
        return false;
    }
    
    // STRICT: Must have reasonable solidity (hand-like shape)
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    std::vector<cv::Point> hull;
    for (int idx : hullIndices) {
        hull.push_back(contour[idx]);
    }
    double hullArea = cv::contourArea(hull);
    double solidity = area / hullArea;
    
    if (solidity < 0.7 || solidity > 0.95) {
        return false;
    }
    
    // STRICT: Only accept if all checks pass
    qDebug() << "🎯 STRICT hand check passed - Area:" << area << "Aspect:" << aspectRatio << "Position:" << boundingRect.y << "Solidity:" << solidity;
    return true;
}

bool AdvancedHandDetector::validateAdvancedHandShape(const std::vector<cv::Point>& contour, int /*fingerCount*/) {
    // Calculate convex hull area ratio
    std::vector<int> validateHullIndices;
    cv::convexHull(contour, validateHullIndices);
    std::vector<cv::Point> validateHull;
    for (int idx : validateHullIndices) {
        validateHull.push_back(contour[idx]);
    }
    
    double validateHullArea = cv::contourArea(validateHull);
    double contourArea = cv::contourArea(contour);
    double solidity = contourArea / validateHullArea;
    
    // Hand should have reasonable solidity (not too convex, not too concave)
    if (solidity < 0.6 || solidity > 0.95) {
        return false;
    }
    
    // Check if the shape has the characteristic hand structure
    // (wider at the bottom, narrower at the top)
    cv::Rect boundingRect = cv::boundingRect(contour);
    cv::Point topCenter(boundingRect.x + boundingRect.width / 2, boundingRect.y);
    cv::Point bottomCenter(boundingRect.x + boundingRect.width / 2, boundingRect.y + boundingRect.height);
    
    // Count contour points near top vs bottom
    int topPoints = 0, bottomPoints = 0;
    for (const auto& point : contour) {
        if (abs(point.y - topCenter.y) < boundingRect.height * 0.2) {
            topPoints++;
        }
        if (abs(point.y - bottomCenter.y) < boundingRect.height * 0.2) {
            bottomPoints++;
        }
    }
    
    // Hand should have more points at the bottom (palm) than at the top (fingers) - EXTREMELY STRICT
    return bottomPoints > topPoints * 2.5; // Extremely strict ratio to exclude faces
}

bool AdvancedHandDetector::isHandEdgePattern(const std::vector<cv::Point>& contour, const cv::Mat& /*image*/) {
    if (contour.size() < 20) {
        return false;
    }
    
    double area = cv::contourArea(contour);
    if (area < MIN_HAND_AREA || area > MAX_HAND_AREA) {
        return false;
    }
    
    // Check for finger-like protrusions in edge pattern
    std::vector<int> edgeHullIndices;
    cv::convexHull(contour, edgeHullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, edgeHullIndices, defects);
    
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 8) {
            significantDefects++;
        }
    }
    
    return significantDefects >= 2 && significantDefects <= 6;
}

double AdvancedHandDetector::calculateHandShapeConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    double circularity = 4 * M_PI * area / (perimeter * perimeter);
    
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
    
    // Analyze convexity defects
    std::vector<int> confidenceHullIndices;
    cv::convexHull(contour, confidenceHullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, confidenceHullIndices, defects);
    
    int fingerCount = 0;
    double totalDefectDepth = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 10) {
            fingerCount++;
            totalDefectDepth += depth;
        }
    }
    
    // Calculate confidence components
    double areaConfidence = std::min(1.0, area / 3000.0);
    double aspectRatioConfidence = (aspectRatio >= 1.0) ? 1.0 : aspectRatio;
    double circularityConfidence = 1.0 - circularity; // Lower circularity is better for hands
    double fingerConfidence = std::min(1.0, fingerCount / 4.0);
    double positionConfidence = 1.0 - (boundingRect.y / (image.rows * 0.6));
    
    // Weighted confidence calculation
    double confidence = areaConfidence * 0.2 + 
                       aspectRatioConfidence * 0.25 + 
                       circularityConfidence * 0.2 + 
                       fingerConfidence * 0.25 + 
                       positionConfidence * 0.1;
    
    return std::min(1.0, confidence);
}

double AdvancedHandDetector::calculateEdgeHandConfidence(const std::vector<cv::Point>& contour, const cv::Mat& image) {
    double area = cv::contourArea(contour);
    cv::Rect boundingRect = cv::boundingRect(contour);
    double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
    
    // Analyze edge pattern complexity
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    int significantDefects = 0;
    for (const auto& defect : defects) {
        double depth = defect[3] / 256.0;
        if (depth > 8) {
            significantDefects++;
        }
    }
    
    double areaConfidence = std::min(1.0, area / 2000.0);
    double aspectRatioConfidence = (aspectRatio >= 1.0) ? 1.0 : aspectRatio;
    double defectConfidence = std::min(1.0, significantDefects / 4.0);
    double positionConfidence = 1.0 - (boundingRect.y / (image.rows * 0.6));
    
    return (areaConfidence * 0.3 + aspectRatioConfidence * 0.3 + defectConfidence * 0.3 + positionConfidence * 0.1);
}

std::vector<cv::Point> AdvancedHandDetector::findFingerTipsFromContour(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> fingerTips;
    
    // Find convex hull
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    // Find convexity defects
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Find finger tips from convex hull points
    for (int i = 0; i < hullIndices.size(); i++) {
        cv::Point tip = contour[hullIndices[i]];
        
        // Check if this point is a finger tip (not too close to palm center)
        cv::Point palmCenter = findPalmCenter(contour);
        double distance = cv::norm(tip - palmCenter);
        
        if (distance > 25) { // Minimum distance from palm center
            fingerTips.push_back(tip);
        }
    }
    
    // Sort by distance from palm center (furthest first)
    cv::Point palmCenter = findPalmCenter(contour);
    std::sort(fingerTips.begin(), fingerTips.end(),
        [palmCenter](const cv::Point& a, const cv::Point& b) {
            return cv::norm(a - palmCenter) > cv::norm(b - palmCenter);
        });
    
    // Limit to top 5 finger tips
    if (fingerTips.size() > 5) {
        fingerTips.resize(5);
    }
    
    return fingerTips;
}

std::vector<cv::Point> AdvancedHandDetector::extractPalmKeypoints(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> palmKeypoints;
    
    // Find convex hull
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices);
    
    // Extract keypoints from convex hull
    for (int i = 0; i < std::min(static_cast<int>(hullIndices.size()), PALM_KEYPOINTS_COUNT); i++) {
        palmKeypoints.push_back(contour[hullIndices[i]]);
    }
    
    // If we don't have enough keypoints, add some from the contour
    while (palmKeypoints.size() < PALM_KEYPOINTS_COUNT) {
        int index = static_cast<int>(palmKeypoints.size()) * static_cast<int>(contour.size()) / PALM_KEYPOINTS_COUNT;
        if (index < static_cast<int>(contour.size())) {
            palmKeypoints.push_back(contour[index]);
        } else {
            break;
        }
    }
    
    return palmKeypoints;
}

// MOTION DETECTION IMPLEMENTATION
bool AdvancedHandDetector::detectMotion(const cv::Mat& image) {
    if (image.empty()) {
        return false;
    }
    
    // Convert to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Initialize background model if needed
    if (!m_backgroundInitialized) {
        grayImage.copyTo(m_backgroundModel);
        m_backgroundInitialized = true;
        return false; // No motion on first frame
    }
    
    // Calculate difference from background
    cv::Mat diffImage;
    cv::absdiff(grayImage, m_backgroundModel, diffImage);
    
    // Apply threshold to get motion mask
    cv::Mat motionMask;
    cv::threshold(diffImage, motionMask, MOTION_THRESHOLD, 255, cv::THRESH_BINARY);
    
    // Apply morphological operations to clean up noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, kernel);
    
    // Find contours in motion mask
    std::vector<std::vector<cv::Point>> motionContours;
    cv::findContours(motionMask, motionContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Check if any motion contour is large enough
    bool significantMotion = false;
    std::vector<cv::Point> motionCenters;
    
    for (const auto& contour : motionContours) {
        double area = cv::contourArea(contour);
        if (area > MIN_MOTION_AREA) {
            // Calculate center of motion
            cv::Moments moments = cv::moments(contour);
            if (moments.m00 > 0) {
                cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
                motionCenters.push_back(center);
                significantMotion = true;
            }
        }
    }
    
    // Update motion history
    updateMotionHistory(motionMask);
    
    // Check motion stability
    if (significantMotion && isMotionStable()) {
        m_motionDetected = true;
        m_lastMotionCenters = motionCenters;
        qDebug() << "🎯 SIGNIFICANT MOTION DETECTED! Centers:" << motionCenters.size();
        return true;
    } else {
        m_motionDetected = false;
        return false;
    }
}

void AdvancedHandDetector::updateMotionHistory(const cv::Mat& motionMask) {
    // Add current motion mask to history
    m_motionHistory.push_back(motionMask.clone());
    
    // Keep only the last N frames
    if (m_motionHistory.size() > MOTION_HISTORY_FRAMES) {
        m_motionHistory.erase(m_motionHistory.begin());
    }
    
    // Check if motion is stable across frames
    if (m_motionHistory.size() >= MOTION_STABILITY_FRAMES) {
        int stableFrames = 0;
        for (const auto& mask : m_motionHistory) {
            if (cv::countNonZero(mask) > MIN_MOTION_AREA) {
                stableFrames++;
            }
        }
        
        if (stableFrames >= MOTION_STABILITY_FRAMES) {
            m_motionStabilityCount++;
        } else {
            m_motionStabilityCount = 0;
        }
    }
}

bool AdvancedHandDetector::isMotionStable() {
    return m_motionStabilityCount >= MOTION_STABILITY_FRAMES;
}


