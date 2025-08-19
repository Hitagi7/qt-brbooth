#ifndef MEDIAPIPE_LIKE_HAND_TRACKER_H
#define MEDIAPIPE_LIKE_HAND_TRACKER_H

#include <QObject>
#include <QPoint>
#include <QList>
#include <opencv2/opencv.hpp>

class HandTrackerMP : public QObject
{
    Q_OBJECT

public:
    explicit HandTrackerMP(QObject *parent = nullptr);
    ~HandTrackerMP();

    bool initialize(int width, int height);
    void update(const cv::Mat &frame);
    bool shouldTriggerCapture() const;
    
    void setTriggerThreshold(int threshold);
    int getTriggerThreshold() const;
    
    void reset();

signals:
    void handGestureDetected(const QString &gesture);

private:
    bool m_initialized;
    int m_width;
    int m_height;
    int m_triggerThreshold;
    
    // Tracking state
    std::vector<cv::Point> m_previousHandPositions;
    int m_stableFrameCount;
    bool m_triggerReady;
    
    cv::Point detectHandPosition(const cv::Mat &frame);
    bool isHandStable(const cv::Point &currentPos);
    QString classifyGesture(const std::vector<cv::Point> &positions);
};

#endif // MEDIAPIPE_LIKE_HAND_TRACKER_H
