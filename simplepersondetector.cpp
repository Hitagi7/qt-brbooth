#include "simplepersondetector.h"
#include <QDebug>

SimplePersonDetector::SimplePersonDetector() 
    : m_initialized(false)
{
    try {
        // Initialize HOG descriptor with default people detector
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        m_initialized = true;
        qDebug() << "✅ SimplePersonDetector constructor: HOG initialized successfully!";
    } catch (const cv::Exception& e) {
        qWarning() << "❌ SimplePersonDetector constructor failed:" << e.what();
        m_initialized = false;
    } catch (const std::exception& e) {
        qWarning() << "❌ SimplePersonDetector constructor failed:" << e.what();
        m_initialized = false;
    }
}

SimplePersonDetector::~SimplePersonDetector() {
}

bool SimplePersonDetector::initialize() {
    if (m_initialized) {
        qDebug() << "✅ Person detector already initialized!";
        return true;
    }
    
    try {
        // Initialize HOG descriptor for person detection
        m_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        m_initialized = true;
        qDebug() << "✅ Optimized person detector initialized successfully!";
        return true;
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Failed to initialize optimized person detector:" << e.what();
        m_initialized = false;
        return false;
    } catch (const std::exception& e) {
        qWarning() << "❌ Failed to initialize optimized person detector:" << e.what();
        m_initialized = false;
        return false;
    }
}

bool SimplePersonDetector::isInitialized() const {
    return m_initialized;
}

QList<SimpleDetection> SimplePersonDetector::detect(const cv::Mat& image) {
    QList<SimpleDetection> detections;
    
    qDebug() << "🎯 SimplePersonDetector::detect() called with image size:" << image.cols << "x" << image.rows;
    
    if (!m_initialized) {
        qWarning() << "❌ Optimized person detector not initialized";
        return detections;
    }
    
    if (image.empty()) {
        qWarning() << "❌ Empty image provided to person detector";
        return detections;
    }
    
    try {
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        
        // Scale down the image for better detection quality and coverage
        cv::Mat scaledImage;
        double scale = 0.5; // Scale down to 50% for better detection quality and coverage
        cv::resize(grayImage, scaledImage, cv::Size(), scale, scale, cv::INTER_LINEAR);
        
        std::vector<cv::Rect> foundLocations;
        std::vector<double> weights;
        
        // Balanced HOG parameters to detect real people without false positives
        m_hog.detectMultiScale(
            scaledImage,
            foundLocations,
            weights,
            -0.3,  // hitThreshold - balanced sensitivity
            cv::Size(8, 8),    // winStride - moderate stride
            cv::Size(4, 4),    // padding - moderate padding
            1.05,  // scale - moderate scale factor
            0.3,   // finalThreshold - moderate threshold
            false  // useMeanshiftGrouping - disabled for speed
        );
        
        // Convert to our detection format and scale back to original image size
        for (size_t i = 0; i < foundLocations.size() && i < weights.size(); ++i) {
            SimpleDetection det;
            
            // Scale the bounding box back to original image size
            det.boundingBox.x = static_cast<int>(foundLocations[i].x / scale);
            det.boundingBox.y = static_cast<int>(foundLocations[i].y / scale);
            det.boundingBox.width = static_cast<int>(foundLocations[i].width / scale);
            det.boundingBox.height = static_cast<int>(foundLocations[i].height / scale);
            
            det.confidence = weights[i];
            det.className = "person";
            
            // Debug output for all detections
            qDebug() << "🔍 Processing detection" << i << ":" << det.boundingBox.width << "x" << det.boundingBox.height 
                     << "confidence:" << det.confidence;
            
            // Balanced threshold to detect real people
            if (det.confidence > -0.4) { // Moderate threshold to avoid false positives
                qDebug() << "✅ Confidence passed:" << det.confidence << "> -0.4";
                // Ensure bounding box is valid and reasonably sized for actual people
                // Use moderate size requirements to avoid detecting small objects
                if (det.boundingBox.width > 60 && det.boundingBox.height > 120) {
                    qDebug() << "✅ Size passed:" << det.boundingBox.width << "x" << det.boundingBox.height;
                    // Check aspect ratio to ensure it's a person (height should be > width)
                    double aspectRatio = static_cast<double>(det.boundingBox.height) / det.boundingBox.width;
                    if (aspectRatio > 1.2) { // Person should be taller than wide (less strict)
                        qDebug() << "✅ Aspect ratio passed:" << aspectRatio << "> 1.2";
                        
                        // Avoid detecting UI elements in the bottom area
                        double bottomThreshold = image.rows * 0.8; // Bottom 20% of image
                        if (det.boundingBox.y < bottomThreshold) {
                            qDebug() << "✅ Not in UI area (y:" << det.boundingBox.y << "<" << bottomThreshold << ")";
                    // No bounding box expansion for maximum accuracy
                    // Keep boxes exactly as detected for tight fit around people
                    int expansionX = 0; // No expansion
                    int expansionY = 0; // No expansion
                    
                    det.boundingBox.x = std::max(0, det.boundingBox.x - expansionX);
                    det.boundingBox.y = std::max(0, det.boundingBox.y - expansionY);
                    det.boundingBox.width = std::min(image.cols - det.boundingBox.x, 
                                                   det.boundingBox.width + 2 * expansionX);
                    det.boundingBox.height = std::min(image.rows - det.boundingBox.y, 
                                                    det.boundingBox.height + 2 * expansionY);
                    
                                         // Final validation before adding
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
        
        // Simple non-maximum suppression for overlapping detections
        QList<SimpleDetection> filteredDetections;
        
        // Sort detections by confidence (highest first) to keep the best ones
        std::sort(detections.begin(), detections.end(), 
                 [](const SimpleDetection& a, const SimpleDetection& b) {
                     return a.confidence > b.confidence;
                 });
        
        for (const SimpleDetection& det : detections) {
            bool shouldKeep = true;
            QList<SimpleDetection> toRemove;
            
            for (const SimpleDetection& existing : filteredDetections) {
                // Calculate overlap
                cv::Rect intersection = det.boundingBox & existing.boundingBox;
                double overlapArea = intersection.area();
                double unionArea = det.boundingBox.area() + existing.boundingBox.area() - overlapArea;
                double iou = overlapArea / unionArea;
                
                if (iou > 0.5) { // If overlap is more than 50% (more aggressive filtering to remove duplicates)
                    if (det.confidence > existing.confidence) {
                        // Mark for removal instead of removing during iteration
                        toRemove.append(existing);
                    }
                    shouldKeep = false;
                    break;
                }
            }
            
            // Remove marked detections after iteration
            for (const SimpleDetection& removeDet : toRemove) {
                filteredDetections.removeOne(removeDet);
            }
            
            if (shouldKeep) {
                // Limit to maximum 3 detections to avoid too many false positives
                if (filteredDetections.size() < 3) {
                    filteredDetections.append(det);
                }
            }
        }
        
                qDebug() << "🔍 Raw HOG detections:" << foundLocations.size() << "locations found with weights:";
        for (size_t i = 0; i < foundLocations.size() && i < weights.size(); ++i) {
            qDebug() << "  Raw detection" << i << ":" << foundLocations[i].x << foundLocations[i].y
                     << foundLocations[i].width << "x" << foundLocations[i].height
                     << "confidence:" << weights[i] << "position:" 
                     << (foundLocations[i].x < scaledImage.cols/3 ? "LEFT" : 
                         foundLocations[i].x > 2*scaledImage.cols/3 ? "RIGHT" : "CENTER")
                     << "aspect_ratio:" << (static_cast<double>(foundLocations[i].height) / foundLocations[i].width);
        }
        qDebug() << "✅ Multi-person detector found" << filteredDetections.size() << "persons (after filtering)";
        
        // Enhanced debugging for multi-person detection
        if (filteredDetections.size() > 1) {
            qDebug() << "👥 MULTIPLE PEOPLE DETECTED! Found" << filteredDetections.size() << "persons";
        }
        
        // Show final detection details
        for (int i = 0; i < filteredDetections.size(); ++i) {
            const SimpleDetection& det = filteredDetections[i];
            qDebug() << "🎯 Final Person" << (i + 1) << ":" 
                     << "box:" << det.boundingBox.x << det.boundingBox.y 
                     << det.boundingBox.width << "x" << det.boundingBox.height
                     << "confidence:" << det.confidence;
        }
        
        // If no HOG detections found, try a simple fallback detection
        if (filteredDetections.isEmpty()) {
            qDebug() << "🔄 No HOG detections, trying fallback detection...";
            // Only use fallback if we've had multiple consecutive empty detections
            static int emptyDetectionCount = 0;
            emptyDetectionCount++;
            
            if (emptyDetectionCount >= 3) { // Faster fallback for multiple people
                // Create multiple fallback detections for different areas
                SimpleDetection fallbackDet1 = createFallbackDetection(image, 0.3); // Left side
                SimpleDetection fallbackDet2 = createFallbackDetection(image, 0.7); // Right side
                
                if (fallbackDet1.boundingBox.width > 0) {
                    filteredDetections.append(fallbackDet1);
                    qDebug() << "🆘 Fallback detection 1 (left) added after" << emptyDetectionCount << "empty detections:"
                             << fallbackDet1.boundingBox.x << fallbackDet1.boundingBox.y 
                             << fallbackDet1.boundingBox.width << "x" << fallbackDet1.boundingBox.height;
                }
                if (fallbackDet2.boundingBox.width > 0) {
                    filteredDetections.append(fallbackDet2);
                    qDebug() << "🆘 Fallback detection 2 (right) added after" << emptyDetectionCount << "empty detections:"
                             << fallbackDet2.boundingBox.x << fallbackDet2.boundingBox.y 
                             << fallbackDet2.boundingBox.width << "x" << fallbackDet2.boundingBox.height;
                }
            }
        } else {
            // Reset empty detection counter when we find detections
            static int emptyDetectionCount = 0;
            emptyDetectionCount = 0;
        }
        
        return filteredDetections;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Error during optimized person detection:" << e.what();
        return detections;
    } catch (const std::exception& e) {
        qWarning() << "Unexpected error during optimized person detection:" << e.what();
        return detections;
    }
}

void SimplePersonDetector::drawDetections(cv::Mat& image, const QList<SimpleDetection>& detections) {
    // Different colors for different people
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(255, 255, 0),  // Cyan
        cv::Scalar(255, 0, 255),  // Magenta
        cv::Scalar(0, 255, 255)   // Yellow
    };
    
    for (int i = 0; i < detections.size(); ++i) {
        const SimpleDetection& det = detections[i];
        cv::Scalar color = colors[i % colors.size()]; // Cycle through colors
        
        cv::rectangle(image, det.boundingBox, color, 3); // Thicker lines for better visibility
        
        // Add label with person number
        QString label = QString("Person %1: %.2f").arg(i + 1).arg(det.confidence);
        cv::putText(image, label.toStdString(), 
                   cv::Point(det.boundingBox.x, det.boundingBox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

SimpleDetection SimplePersonDetector::createFallbackDetection(const cv::Mat& image, double horizontalPosition) {
    SimpleDetection fallbackDet;
    fallbackDet.className = "person";
    fallbackDet.confidence = 0.3; // Lower confidence for fallback
    
    // Create a dynamic bounding box at the specified horizontal position
    int centerX = static_cast<int>(image.cols * horizontalPosition);
    int centerY = image.rows / 2;
    
    // Calculate box size based on image dimensions - balanced proportions
    // Use more realistic person proportions that cover the whole person
    int boxWidth = image.cols * 2 / 5;   // 2/5 of image width (realistic person width)
    int boxHeight = image.rows * 3 / 4;  // 3/4 of image height (realistic person height)
    
    // Ensure minimum reasonable size
    if (boxWidth < 120) boxWidth = 120;
    if (boxHeight < 250) boxHeight = 250;
    
    fallbackDet.boundingBox.x = centerX - boxWidth / 2;
    fallbackDet.boundingBox.y = centerY - boxHeight / 2;
    fallbackDet.boundingBox.width = boxWidth;
    fallbackDet.boundingBox.height = boxHeight;
    
    // Ensure the box is within image bounds
    fallbackDet.boundingBox.x = std::max(0, fallbackDet.boundingBox.x);
    fallbackDet.boundingBox.y = std::max(0, fallbackDet.boundingBox.y);
    fallbackDet.boundingBox.width = std::min(boxWidth, image.cols - fallbackDet.boundingBox.x);
    fallbackDet.boundingBox.height = std::min(boxHeight, image.rows - fallbackDet.boundingBox.y);
    
    qDebug() << "🆘 Created fallback detection box:" << fallbackDet.boundingBox.x 
             << fallbackDet.boundingBox.y << fallbackDet.boundingBox.width 
             << "x" << fallbackDet.boundingBox.height;
    
    return fallbackDet;
} 