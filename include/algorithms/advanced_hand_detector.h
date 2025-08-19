#ifndef ADVANCED_HAND_DETECTOR_H
#define ADVANCED_HAND_DETECTOR_H

#include <QObject>
#include <QImage>
#include <QList>
#include <opencv2/opencv.hpp>
#include "core/common_types.h"

class AdvancedHandDetector : public QObject
{
    Q_OBJECT

public:
    explicit AdvancedHandDetector(QObject *parent = nullptr);
    ~AdvancedHandDetector();

    bool initialize();
    bool isInitialized() const;
    
    QList<AdvancedHandDetection> detectHands(const QImage &image);
    QList<AdvancedHandDetection> detectHands(const cv::Mat &frame);
    
    void setConfidenceThreshold(double threshold);
    double getConfidenceThreshold() const;
    
    void setMaxHands(int maxHands);
    int getMaxHands() const;

signals:
    void handsDetected(const QList<AdvancedHandDetection> &detections);
    void detectionError(const QString &error);

private:
    bool m_initialized;
    double m_confidenceThreshold;
    int m_maxHands;
    
    // OpenCV hand detection components
    cv::CascadeClassifier m_handCascade;
    cv::HOGDescriptor m_hogDetector;
    
    bool loadCascadeClassifier();
    QList<AdvancedHandDetection> processDetections(const std::vector<cv::Rect> &detections, const cv::Mat &frame);
};

#endif // ADVANCED_HAND_DETECTOR_H
