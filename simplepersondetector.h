#ifndef SIMPLEPERSONDETECTOR_H
#define SIMPLEPERSONDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <QString>
#include <QList>
#include <QMutex>

struct SimpleDetection {
    cv::Rect boundingBox;
    double confidence;
    QString className;
    
    // Equality operator for QList operations - simplified for practical use
    bool operator==(const SimpleDetection& other) const {
        return boundingBox == other.boundingBox && 
               className == other.className;
    }
};

class SimplePersonDetector {
public:
    SimplePersonDetector();
    ~SimplePersonDetector();
    
    bool initialize();
    QList<SimpleDetection> detect(const cv::Mat& image);
    bool isInitialized() const;
    
private:
    cv::HOGDescriptor m_hog;
    bool m_initialized;
    
    void drawDetections(cv::Mat& image, const QList<SimpleDetection>& detections);
    SimpleDetection createFallbackDetection(const cv::Mat& image, double horizontalPosition = 0.5);
};

#endif // SIMPLEPERSONDETECTOR_H 