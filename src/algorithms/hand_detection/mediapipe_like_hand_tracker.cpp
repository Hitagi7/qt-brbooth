#include "algorithms/mediapipe_like_hand_tracker.h"
#include <QDebug>
#include <cmath>

HandTrackerMP::HandTrackerMP(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_width(640)
    , m_height(480)
    , m_triggerThreshold(30)
    , m_stableFrameCount(0)
    , m_triggerReady(false)
{
}

HandTrackerMP::~HandTrackerMP()
{
}

bool HandTrackerMP::initialize(int width, int height)
{
    m_width = width;
    m_height = height;
    m_initialized = true;
    reset();
    
    qDebug() << "HandTrackerMP initialized with size:" << width << "x" << height;
    return true;
}

void HandTrackerMP::update(const cv::Mat &frame)
{
    if (!m_initialized || frame.empty()) {
        return;
    }

    // Detect hand position
    cv::Point currentPos = detectHandPosition(frame);
    
    // Check if hand is stable
    if (isHandStable(currentPos)) {
        m_stableFrameCount++;
        if (m_stableFrameCount >= m_triggerThreshold) {
            m_triggerReady = true;
        }
    } else {
        m_stableFrameCount = 0;
        m_triggerReady = false;
    }
    
    // Store position history
    m_previousHandPositions.push_back(currentPos);
    if (m_previousHandPositions.size() > 10) {
        m_previousHandPositions.erase(m_previousHandPositions.begin());
    }
    
    // Classify gesture if we have enough history
    if (m_previousHandPositions.size() >= 5) {
        QString gesture = classifyGesture(m_previousHandPositions);
        if (!gesture.isEmpty()) {
            emit handGestureDetected(gesture);
        }
    }
}

bool HandTrackerMP::shouldTriggerCapture() const
{
    return m_triggerReady;
}

void HandTrackerMP::setTriggerThreshold(int threshold)
{
    m_triggerThreshold = qBound(10, threshold, 100);
}

int HandTrackerMP::getTriggerThreshold() const
{
    return m_triggerThreshold;
}

void HandTrackerMP::reset()
{
    m_previousHandPositions.clear();
    m_stableFrameCount = 0;
    m_triggerReady = false;
}

cv::Point HandTrackerMP::detectHandPosition(const cv::Mat &frame)
{
    // Simple hand detection using skin color
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // Define skin color range
    cv::Scalar lowerSkin(0, 20, 70);
    cv::Scalar upperSkin(20, 255, 255);
    
    cv::Mat skinMask;
    cv::inRange(hsv, lowerSkin, upperSkin, skinMask);
    
    // Apply morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return cv::Point(-1, -1); // No hand detected
    }
    
    // Find the largest contour (likely the hand)
    int maxArea = 0;
    int maxIdx = -1;
    
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea && area > 1000) { // Minimum area threshold
            maxArea = area;
            maxIdx = i;
        }
    }
    
    if (maxIdx == -1) {
        return cv::Point(-1, -1);
    }
    
    // Calculate centroid of the hand
    cv::Moments moments = cv::moments(contours[maxIdx]);
    if (moments.m00 != 0) {
        int centerX = static_cast<int>(moments.m10 / moments.m00);
        int centerY = static_cast<int>(moments.m01 / moments.m00);
        return cv::Point(centerX, centerY);
    }
    
    return cv::Point(-1, -1);
}

bool HandTrackerMP::isHandStable(const cv::Point &currentPos)
{
    if (currentPos.x == -1 || currentPos.y == -1) {
        return false; // No hand detected
    }
    
    if (m_previousHandPositions.empty()) {
		return true;
	}
    
    // Check if current position is close to the last position
    cv::Point lastPos = m_previousHandPositions.back();
    double distance = std::sqrt(std::pow(currentPos.x - lastPos.x, 2) + 
                               std::pow(currentPos.y - lastPos.y, 2));
    
    // Consider stable if movement is less than 20 pixels
    return distance < 20;
}

QString HandTrackerMP::classifyGesture(const std::vector<cv::Point> &positions)
{
    if (positions.size() < 5) {
        return QString();
    }
    
    // Simple gesture classification based on movement pattern
    cv::Point first = positions.front();
    cv::Point last = positions.back();
    
    double distance = std::sqrt(std::pow(last.x - first.x, 2) + 
                               std::pow(last.y - first.y, 2));
    
    // Calculate movement direction
    double dx = last.x - first.x;
    double dy = last.y - first.y;
    
    if (distance < 30) {
        return "stable"; // Hand is stable
    } else if (std::abs(dx) > std::abs(dy)) {
        if (dx > 0) {
            return "right"; // Moving right
        } else {
            return "left"; // Moving left
        }
    } else {
        if (dy > 0) {
            return "down"; // Moving down
        } else {
            return "up"; // Moving up
        }
    }
}
