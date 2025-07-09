#include "persondetector.h"
#include <QPainter>
#include <algorithm>

PersonDetector::PersonDetector(QObject *parent)
    : QObject(parent), modelLoaded(false)
{
}

PersonDetector::~PersonDetector()
{
}

bool PersonDetector::loadModel(const QString& modelPath)
{
    try {
        net = cv::dnn::readNetFromONNX(modelPath.toStdString());

        if (net.empty()) {
            qDebug() << "Failed to load ONNX model from:" << modelPath;
            return false;
        }

        // Try different backend/target combinations
        std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = {
                                                                              {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
                                                                              {cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_TARGET_CPU},
                                                                              };

        bool backendSet = false;
        for (const auto& backend : backends) {
            try {
                net.setPreferableBackend(backend.first);
                net.setPreferableTarget(backend.second);

                // Create test blob using blobFromImage (simpler and more reliable)
                cv::Mat dummyImage = cv::Mat::zeros(640, 640, CV_8UC3);
                cv::Mat testBlob;
                cv::dnn::blobFromImage(dummyImage, testBlob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
                net.setInput(testBlob);

                std::vector<cv::Mat> testOutputs;
                net.forward(testOutputs, net.getUnconnectedOutLayersNames());

                qDebug() << "Successfully set backend:" << backend.first << "target:" << backend.second;
                backendSet = true;
                break;

            } catch (const cv::Exception& e) {
                qDebug() << "Backend" << backend.first << "failed:" << e.what();
                continue;
            }
        }

        if (!backendSet) {
            qDebug() << "All backends failed!";
            return false;
        }

        modelLoaded = true;
        qDebug() << "YOLOv5n model loaded successfully from:" << modelPath;
        return true;

    } catch (const cv::Exception& e) {
        qDebug() << "OpenCV exception loading model:" << e.what();
        return false;
    }
}

std::vector<PersonDetection> PersonDetector::detectPeople(const QImage& image, float confThreshold)
{
    if (!modelLoaded) {
        qDebug() << "Model not loaded!";
        return {};
    }

    // Convert QImage to OpenCV Mat
    cv::Mat frame = qImageToCvMat(image);
    if (frame.empty()) {
        qDebug() << "Empty image!";
        return {};
    }

    int originalWidth = frame.cols;
    int originalHeight = frame.rows;

    // Prepare input blob
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);

    // Set input to the network
    net.setInput(blob);

    // Run forward pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Post-process results (filter for people only)
    std::vector<PersonDetection> detections = postProcess(outputs[0], originalWidth, originalHeight, confThreshold);

    // Apply Non-Maximum Suppression
    applyNMS(detections);

    // Emit signal with person count
    emit peopleDetected(detections.size());

    return detections;
}

int PersonDetector::getPersonCount(const QImage& image, float confThreshold)
{
    auto detections = detectPeople(image, confThreshold);
    return detections.size();
}

QImage PersonDetector::drawDetections(const QImage& image, const std::vector<PersonDetection>& detections)
{
    QImage result = image.copy();
    QPainter painter(&result);

    // Set drawing style
    painter.setPen(QPen(Qt::red, 3));
    painter.setFont(QFont("Arial", 12, QFont::Bold));

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];

        // Draw bounding box
        painter.drawRect(detection.bbox);

        // Draw confidence and person label
        QString label = QString("Person %1: %2%")
                            .arg(i + 1)
                            .arg(detection.confidence * 100, 0, 'f', 1);

        QRectF textRect = detection.bbox;
        textRect.setHeight(25);
        textRect.moveTop(detection.bbox.top() - 25);

        painter.fillRect(textRect, QBrush(Qt::red));
        painter.setPen(Qt::white);
        painter.drawText(textRect, Qt::AlignCenter, label);
        painter.setPen(QPen(Qt::red, 3));

        // Draw center point
        painter.fillRect(detection.center.x() - 3, detection.center.y() - 3, 6, 6, Qt::yellow);
    }

    // Draw total count
    painter.setPen(Qt::blue);
    painter.setFont(QFont("Arial", 16, QFont::Bold));
    painter.drawText(10, 30, QString("People Count: %1").arg(detections.size()));

    return result;
}

std::vector<PersonDetection> PersonDetector::postProcess(const cv::Mat& output,
                                                         int originalWidth, int originalHeight,
                                                         float confThreshold)
{
    std::vector<PersonDetection> detections;

    // YOLOv5 output format: [1, 25200, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    // We only want class 0 (person)

    const float* data = (float*)output.data;
    const int numDetections = output.size[1]; // 25200
    const int numElements = output.size[2];   // 85

    for (int i = 0; i < numDetections; ++i) {
        const float* detection = &data[i * numElements];

        float confidence = detection[4]; // Object confidence
        if (confidence < confThreshold) continue;

        // Get class scores (classes start at index 5)
        float personScore = detection[5] * confidence; // Class 0 = person

        if (personScore < confThreshold) continue; // Only keep person detections

        // Extract bounding box (center format)
        float cx = detection[0] * originalWidth / 640.0f;
        float cy = detection[1] * originalHeight / 640.0f;
        float w = detection[2] * originalWidth / 640.0f;
        float h = detection[3] * originalHeight / 640.0f;

        // Convert to corner format
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        PersonDetection person;
        person.bbox = QRectF(x, y, w, h);
        person.confidence = personScore;
        person.center = QPointF(cx, cy);

        detections.push_back(person);
    }

    return detections;
}

void PersonDetector::applyNMS(std::vector<PersonDetection>& detections, float nmsThreshold)
{
    // Sort by confidence (highest first)
    std::sort(detections.begin(), detections.end(),
              [](const PersonDetection& a, const PersonDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> keep(detections.size(), true);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;

        cv::Rect rectA(detections[i].bbox.x(), detections[i].bbox.y(),
                       detections[i].bbox.width(), detections[i].bbox.height());

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j]) continue;

            cv::Rect rectB(detections[j].bbox.x(), detections[j].bbox.y(),
                           detections[j].bbox.width(), detections[j].bbox.height());

            float iou = calculateIoU(rectA, rectB);
            if (iou > nmsThreshold) {
                keep[j] = false;
            }
        }
    }

    // Remove suppressed detections
    std::vector<PersonDetection> filtered;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(detections[i]);
        }
    }
    detections = filtered;
}

float PersonDetector::calculateIoU(const cv::Rect& a, const cv::Rect& b)
{
    cv::Rect intersection = a & b;
    float intersectionArea = intersection.area();
    float unionArea = a.area() + b.area() - intersectionArea;

    return (unionArea > 0) ? intersectionArea / unionArea : 0.0f;
}

cv::Mat PersonDetector::qImageToCvMat(const QImage& qimg)
{
    QImage swapped = qimg.rgbSwapped();
    return cv::Mat(swapped.height(), swapped.width(), CV_8UC3,
                   (void*)swapped.constBits(), swapped.bytesPerLine()).clone();
}

QImage PersonDetector::cvMatToQImage(const cv::Mat& mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return qimg.rgbSwapped();
    }
    case CV_8UC3: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return qimg.rgbSwapped();
    }
    case CV_8UC1: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return qimg;
    }
    }
    return QImage();
}
