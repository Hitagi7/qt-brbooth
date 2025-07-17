#ifndef PERSONDETECTOR_H
#define PERSONDETECTOR_H

#include <QObject>
#include <QImage>
#include <QRect>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

struct PersonDetection {
    QRectF bbox;        // Bounding box
    float confidence;   // Detection confidence (0.0 - 1.0)
    QPointF center;     // Center point of the person
};

class PersonDetector : public QObject
{
    Q_OBJECT

public:
    explicit PersonDetector(QObject *parent = nullptr);
    ~PersonDetector();

    // Load the YOLOv5n model
    bool loadModel(const QString& modelPath);
    bool isModelLoaded() const { return modelLoaded; }

    // Detect people in image (returns only person detections)
    std::vector<PersonDetection> detectPeople(const QImage& image, float confThreshold = 0.5f);

    // Get count of detected people
    int getPersonCount(const QImage& image, float confThreshold = 0.5f);

    // Draw bounding boxes on image
    QImage drawDetections(const QImage& image, const std::vector<PersonDetection>& detections);

signals:
    void peopleDetected(int count);
    void personEntered(QRectF bbox);
    void personLeft();

private:
    cv::dnn::Net net;
    bool modelLoaded;

    // Helper functions
    cv::Mat qImageToCvMat(const QImage& qimg);
    QImage cvMatToQImage(const cv::Mat& mat);
    std::vector<PersonDetection> postProcess(const cv::Mat& output,
                                             int originalWidth, int originalHeight,
                                             float confThreshold);
    float calculateIoU(const cv::Rect& a, const cv::Rect& b);
    void applyNMS(std::vector<PersonDetection>& detections, float nmsThreshold = 0.4f);
};

#endif // PERSONDETECTOR_H
