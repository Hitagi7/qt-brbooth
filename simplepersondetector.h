#ifndef SIMPLEPERSONDETECTOR_H
#define SIMPLEPERSONDETECTOR_H

#include <QList>
#include <QString>
#include <opencv2/opencv.hpp>

struct SimpleDetection {
    cv::Rect boundingBox;
    double confidence;
    QString className;
    
    // Add equality operator for comparison
    bool operator==(const SimpleDetection& other) const {
        return boundingBox == other.boundingBox && 
               confidence == other.confidence && 
               className == other.className;
    }
    
    bool operator!=(const SimpleDetection& other) const {
        return !(*this == other);
    }
};

class SimplePersonDetector
{
public:
    SimplePersonDetector();
    ~SimplePersonDetector();
    
    bool initialize();
    bool isInitialized() const;
    QList<SimpleDetection> detect(const cv::Mat& image);
    void drawDetections(cv::Mat& image, const QList<SimpleDetection>& detections);

private:
    cv::HOGDescriptor m_hog;
    bool m_initialized;
    
    SimpleDetection createFallbackDetection(const cv::Mat& image, double horizontalPosition);
};

#endif // SIMPLEPERSONDETECTOR_H 