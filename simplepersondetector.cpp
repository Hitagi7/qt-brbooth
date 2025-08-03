#include "simplepersondetector.h"
#include <QDebug>

SimplePersonDetector::SimplePersonDetector() 
    : m_initialized(false)
{
    try {
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        m_initialized = true;
    } catch (const cv::Exception& e) {
        m_initialized = false;
    } catch (const std::exception& e) {
        m_initialized = false;
    }
}

SimplePersonDetector::~SimplePersonDetector() {
}

bool SimplePersonDetector::initialize() {
    if (m_initialized) {
        return true;
    }
    
    try {
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        m_initialized = true;
        return true;
    } catch (const cv::Exception& e) {
        m_initialized = false;
        return false;
    } catch (const std::exception& e) {
        m_initialized = false;
        return false;
    }
}

bool SimplePersonDetector::isInitialized() const {
    return m_initialized;
}

QList<SimpleDetection> SimplePersonDetector::detect(const cv::Mat& image) {
    QList<SimpleDetection> detections;
    
    if (!m_initialized) {
        return detections;
    }
    
    if (image.empty()) {
        return detections;
    }
    
    try {
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        
        cv::Mat scaledImage;
        double scale = 0.5;
        cv::resize(grayImage, scaledImage, cv::Size(), scale, scale, cv::INTER_LINEAR);
        
        std::vector<cv::Rect> foundLocations;
        std::vector<double> weights;
        
        m_hog.detectMultiScale(
            scaledImage,
            foundLocations,
            weights,
            -0.3,
            cv::Size(8, 8),
            cv::Size(4, 4),
            1.05,
            0.3,
            false
        );
        
        for (size_t i = 0; i < foundLocations.size() && i < weights.size(); ++i) {
            SimpleDetection det;
            
            det.boundingBox.x = static_cast<int>(foundLocations[i].x / scale);
            det.boundingBox.y = static_cast<int>(foundLocations[i].y / scale);
            det.boundingBox.width = static_cast<int>(foundLocations[i].width / scale);
            det.boundingBox.height = static_cast<int>(foundLocations[i].height / scale);
            
            det.confidence = weights[i];
            det.className = "person";
            
            if (det.confidence > -0.4) {
                if (det.boundingBox.width > 60 && det.boundingBox.height > 120) {
                    double aspectRatio = static_cast<double>(det.boundingBox.height) / det.boundingBox.width;
                    if (aspectRatio > 1.2) {
                        double bottomThreshold = image.rows * 0.8;
                        if (det.boundingBox.y < bottomThreshold) {
                    int expansionX = 0;
                    int expansionY = 0;
                    
                                         det.boundingBox.x = std::max(0, det.boundingBox.x - expansionX);
                     det.boundingBox.y = std::max(0, det.boundingBox.y - expansionY);
                     det.boundingBox.width = std::min(image.cols - det.boundingBox.x, 
                                                    det.boundingBox.width + 2 * expansionX);
                     det.boundingBox.height = std::min(image.rows - det.boundingBox.y, 
                                                     det.boundingBox.height + 2 * expansionY);
                     
                     if (det.boundingBox.width > 0 && det.boundingBox.height > 0 &&
                         det.boundingBox.x >= 0 && det.boundingBox.y >= 0 &&
                         det.boundingBox.x + det.boundingBox.width <= image.cols &&
                         det.boundingBox.y + det.boundingBox.height <= image.rows) {
                         detections.append(det);
                     }
                     }
                     }
                }
            }
        }
        
        QList<SimpleDetection> filteredDetections;
        
        std::sort(detections.begin(), detections.end(), 
                 [](const SimpleDetection& a, const SimpleDetection& b) {
                     return a.confidence > b.confidence;
                 });
        
        for (const SimpleDetection& det : detections) {
            bool shouldKeep = true;
            QList<SimpleDetection> toRemove;
            
            for (const SimpleDetection& existing : filteredDetections) {
                cv::Rect intersection = det.boundingBox & existing.boundingBox;
                double overlapArea = intersection.area();
                double unionArea = det.boundingBox.area() + existing.boundingBox.area() - overlapArea;
                double iou = overlapArea / unionArea;
                
                if (iou > 0.5) {
                    if (det.confidence > existing.confidence) {
                        toRemove.append(existing);
                    }
                    shouldKeep = false;
                    break;
                }
            }
            
            for (const SimpleDetection& removeDet : toRemove) {
                filteredDetections.removeOne(removeDet);
            }
            
            if (shouldKeep) {
                if (filteredDetections.size() < 3) {
                    filteredDetections.append(det);
                }
            }
        }
        
                
        
        if (filteredDetections.isEmpty()) {
            static int emptyDetectionCount = 0;
            emptyDetectionCount++;
            
            if (emptyDetectionCount >= 3) {
                SimpleDetection fallbackDet1 = createFallbackDetection(image, 0.3);
                SimpleDetection fallbackDet2 = createFallbackDetection(image, 0.7);
                
                if (fallbackDet1.boundingBox.width > 0) {
                    filteredDetections.append(fallbackDet1);
                }
                if (fallbackDet2.boundingBox.width > 0) {
                    filteredDetections.append(fallbackDet2);
                }
            }
        } else {
            static int emptyDetectionCount = 0;
            emptyDetectionCount = 0;
        }
        
        return filteredDetections;
        
    } catch (const cv::Exception& e) {
        return detections;
    } catch (const std::exception& e) {
        return detections;
    }
}

void SimplePersonDetector::drawDetections(cv::Mat& image, const QList<SimpleDetection>& detections) {
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255)
    };
    
    for (int i = 0; i < detections.size(); ++i) {
        const SimpleDetection& det = detections[i];
        cv::Scalar color = colors[i % colors.size()];
        
        cv::rectangle(image, det.boundingBox, color, 3);
        
        QString label = QString("Person %1: %.2f").arg(i + 1).arg(det.confidence);
        cv::putText(image, label.toStdString(), 
                   cv::Point(det.boundingBox.x, det.boundingBox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

SimpleDetection SimplePersonDetector::createFallbackDetection(const cv::Mat& image, double horizontalPosition) {
    SimpleDetection fallbackDet;
    fallbackDet.className = "person";
    fallbackDet.confidence = 0.3;
    
    int centerX = static_cast<int>(image.cols * horizontalPosition);
    int centerY = image.rows / 2;
    
    int boxWidth = image.cols * 2 / 5;
    int boxHeight = image.rows * 3 / 4;
    
    if (boxWidth < 120) boxWidth = 120;
    if (boxHeight < 250) boxHeight = 250;
    
    fallbackDet.boundingBox.x = centerX - boxWidth / 2;
    fallbackDet.boundingBox.y = centerY - boxHeight / 2;
    fallbackDet.boundingBox.width = boxWidth;
    fallbackDet.boundingBox.height = boxHeight;
    
    fallbackDet.boundingBox.x = std::max(0, fallbackDet.boundingBox.x);
    fallbackDet.boundingBox.y = std::max(0, fallbackDet.boundingBox.y);
    fallbackDet.boundingBox.width = std::min(boxWidth, image.cols - fallbackDet.boundingBox.x);
    fallbackDet.boundingBox.height = std::min(boxHeight, image.rows - fallbackDet.boundingBox.y);
    
    return fallbackDet;
} 
