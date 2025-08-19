#include "algorithms/advanced_hand_detector.h"
#include <QDebug>
#include <QDir>

AdvancedHandDetector::AdvancedHandDetector(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_confidenceThreshold(0.5)
    , m_maxHands(2)
{
}

AdvancedHandDetector::~AdvancedHandDetector()
{
}

bool AdvancedHandDetector::initialize()
{
    if (m_initialized) {
        return true;
    }
    
    // Try to load hand cascade classifier
    if (!loadCascadeClassifier()) {
        qWarning() << "Failed to load hand cascade classifier";
        // Continue with HOG detector only
    }

    // Initialize HOG detector for hand detection
    m_hogDetector.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
        m_initialized = true;
    qDebug() << "AdvancedHandDetector initialized successfully";
        return true;
}

bool AdvancedHandDetector::isInitialized() const
{
    return m_initialized;
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHands(const QImage &image)
{
    if (!m_initialized) {
        qWarning() << "Hand detector not initialized";
        return QList<AdvancedHandDetection>();
    }
    
    // Convert QImage to cv::Mat
    cv::Mat frame;
    if (image.format() == QImage::Format_RGB888) {
        frame = cv::Mat(image.height(), image.width(), CV_8UC3, 
                       const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        } else {
        QImage converted = image.convertToFormat(QImage::Format_RGB888);
        frame = cv::Mat(converted.height(), converted.width(), CV_8UC3, 
                       const_cast<uchar*>(converted.bits()), converted.bytesPerLine());
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
    }

    return detectHands(frame);
}

QList<AdvancedHandDetection> AdvancedHandDetector::detectHands(const cv::Mat &frame)
{
    if (!m_initialized || frame.empty()) {
        return QList<AdvancedHandDetection>();
}

    QList<AdvancedHandDetection> detections;
    
    try {
        // Convert to grayscale for detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Use HOG detector for hand-like objects
        std::vector<cv::Rect> found;
        m_hogDetector.detectMultiScale(gray, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

        // Process detections
        detections = processDetections(found, frame);

        // Limit to max hands
        if (detections.size() > m_maxHands) {
            detections = detections.mid(0, m_maxHands);
        }

        emit handsDetected(detections);

    } catch (const cv::Exception &e) {
        qWarning() << "OpenCV error in hand detection:" << e.what();
        emit detectionError(QString("OpenCV error: %1").arg(e.what()));
    } catch (const std::exception &e) {
        qWarning() << "Error in hand detection:" << e.what();
        emit detectionError(QString("Detection error: %1").arg(e.what()));
    }
    
    return detections;
}

void AdvancedHandDetector::setConfidenceThreshold(double threshold)
{
    m_confidenceThreshold = qBound(0.0, threshold, 1.0);
}

double AdvancedHandDetector::getConfidenceThreshold() const
{
    return m_confidenceThreshold;
}

void AdvancedHandDetector::setMaxHands(int maxHands)
{
    m_maxHands = qBound(1, maxHands, 10);
}

int AdvancedHandDetector::getMaxHands() const
{
    return m_maxHands;
}

bool AdvancedHandDetector::loadCascadeClassifier()
{
    // Try to load hand cascade classifier
    QString cascadePath = "haarcascades/haarcascade_hand.xml";
    
    if (!QFile::exists(cascadePath)) {
        // Try alternative paths
        QStringList possiblePaths = {
            "cascades/haarcascade_hand.xml",
            "models/haarcascade_hand.xml",
            "data/haarcascade_hand.xml"
        };
        
        for (const QString &path : possiblePaths) {
            if (QFile::exists(path)) {
                cascadePath = path;
                break;
            }
        }
    }

    if (QFile::exists(cascadePath)) {
        return m_handCascade.load(cascadePath.toStdString());
    }

        return false;
    }
    
QList<AdvancedHandDetection> AdvancedHandDetector::processDetections(const std::vector<cv::Rect> &detections, const cv::Mat &frame)
{
    QList<AdvancedHandDetection> result;

    for (const cv::Rect &rect : detections) {
            AdvancedHandDetection detection;
        detection.boundingBox = QRect(rect.x, rect.y, rect.width, rect.height);
        detection.confidence = 0.8; // Default confidence for HOG detections
        detection.className = "hand";
        
        // Create a simple mask (rectangle)
        detection.mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::rectangle(detection.mask, rect, cv::Scalar(255), -1);

        result.append(detection);
    }

    return result;
}
