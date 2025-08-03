#include "fast_segmentation.h"
#include "capture.h" // For BoundingBox struct
#include "optimized_detector.h" // For OptimizedDetection struct
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

FastSegmentationProcessor::FastSegmentationProcessor()
    : m_segmentationMethod(WATERSHED_FAST)
    , m_maxProcessingTime(5.0)  // 5ms target for real-time
    , m_avgProcessingTime(0.0)
    , m_currentFPS(30)
    , m_frameCount(0)
{
    qDebug() << "🚀 FastSegmentationProcessor: Real-time segmentation initialized";
    qDebug() << "⚡ Target processing time:" << m_maxProcessingTime << "ms per frame";
    
    // Initialize SLIC superpixel processor (reusable)
    try {
        // Initialize watershed kernel instead of SLIC processor
    m_watershedKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    } catch (const cv::Exception& e) {
        qWarning() << "⚠️ SLIC not available, falling back to watershed:" << e.what();
        m_segmentationMethod = WATERSHED_FAST;
    }
}

FastSegmentationProcessor::~FastSegmentationProcessor() {
    qDebug() << "✅ FastSegmentationProcessor destroyed";
}

QList<FastSegmentationResult> FastSegmentationProcessor::segmentPersonsFast(const cv::Mat& image, 
                                                                          const QList<BoundingBox>& detections,
                                                                          double minConfidence) {
    startTiming();
    
    QList<FastSegmentationResult> results;
    
    if (image.empty() || detections.isEmpty()) {
        endTiming();
        return results;
    }
    
    qDebug() << "⚡ Fast segmentation: Processing" << detections.size() << "detections";
    
    try {
        for (const BoundingBox& bbox : detections) {
            if (bbox.confidence < minConfidence) {
                continue;
            }
            
            // Convert BoundingBox to cv::Rect
            cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
            
            // Validate bounding box
            if (rect.width <= 0 || rect.height <= 0 || 
                rect.x < 0 || rect.y < 0 ||
                rect.x + rect.width > image.cols || 
                rect.y + rect.height > image.rows) {
                continue;
            }
            
            FastSegmentationResult result;
            
            // Choose segmentation method based on performance mode
            switch (m_segmentationMethod) {
                case WATERSHED_FAST:
                    result = performWatershedSegmentation(image, rect, bbox.confidence);
                    break;
                    
                case EDGE_BASED:
                    result = performEdgeBasedSegmentation(image, rect, bbox.confidence);
                    break;
                    
                case SUPERPIXEL_SLIC:
                    result = performSuperpixelSegmentation(image, rect, bbox.confidence);
                    break;
                    
                default:
                    result = performWatershedSegmentation(image, rect, bbox.confidence);
                    break;
            }
            
            if (result.isValid) {
                results.append(result);
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Fast segmentation error:" << e.what();
    }
    
    endTiming();
    updatePerformanceStats();
    
    qDebug() << "✅ Fast segmentation complete:" << results.size() << "valid masks in" 
             << m_processingTimes.last() << "ms";
    
    return results;
}

QList<FastSegmentationResult> FastSegmentationProcessor::segmentFromOptimizedDetections(const cv::Mat& image,
                                                                                       const QList<OptimizedDetection>& detections) {
    startTiming();
    
    QList<FastSegmentationResult> results;
    
    if (image.empty() || detections.isEmpty()) {
        endTiming();
        return results;
    }
    
    qDebug() << "🎯 Processing" << detections.size() << "optimized detections with masks";
    
    try {
        for (const OptimizedDetection& detection : detections) {
            FastSegmentationResult result;
            
            if (!detection.mask.empty()) {
                // Use YOLO segmentation mask directly (fastest method)
                result = processYOLOMask(image, detection.mask, detection.boundingBox, detection.confidence);
            } else {
                // Fallback to fast segmentation methods
                result = performWatershedSegmentation(image, detection.boundingBox, detection.confidence);
            }
            
            if (result.isValid) {
                results.append(result);
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Optimized segmentation error:" << e.what();
    }
    
    endTiming();
    updatePerformanceStats();
    
    qDebug() << "✅ Optimized segmentation complete:" << results.size() << "valid masks";
    
    return results;
}

FastSegmentationResult FastSegmentationProcessor::performWatershedSegmentation(const cv::Mat& image, 
                                                                             const cv::Rect& bbox, 
                                                                             double confidence) {
    FastSegmentationResult result;
    result.boundingBox = bbox;
    result.confidence = confidence;
    result.isValid = false;
    
    try {
        // Extract ROI
        cv::Mat roi = image(bbox);
        cv::Mat roiGray;
        cv::cvtColor(roi, roiGray, cv::COLOR_BGR2GRAY);
        
        // Apply Gaussian blur to reduce noise
        cv::Mat blurred;
        cv::GaussianBlur(roiGray, blurred, cv::Size(3, 3), 0);
        
        // Apply threshold to get binary image
        cv::Mat binary;
        cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Noise removal using morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat cleaned;
        cv::morphologyEx(binary, cleaned, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
        
        // Find sure background area
        cv::Mat sureBg;
        cv::dilate(cleaned, sureBg, kernel, cv::Point(-1, -1), 3);
        
        // Find sure foreground area
        cv::Mat distTransform;
        cv::distanceTransform(cleaned, distTransform, cv::DIST_L2, 5);
        cv::Mat sureFg;
        cv::threshold(distTransform, sureFg, 0.5 * 255, 255, cv::THRESH_BINARY);
        sureFg.convertTo(sureFg, CV_8UC1);
        
        // Find unknown region
        cv::Mat unknown;
        cv::subtract(sureBg, sureFg, unknown);
        
        // Marker labelling
        cv::Mat markers;
        cv::connectedComponents(sureFg, markers);
        markers = markers + 1;  // Background becomes 1
        markers.setTo(0, unknown == 255);  // Unknown regions become 0
        
        // Apply watershed
        cv::Mat watershedInput;
        roi.copyTo(watershedInput);
        cv::watershed(watershedInput, markers);
        
        // Extract person mask (excluding background label 1)
        cv::Mat personMask = cv::Mat::zeros(roi.size(), CV_8UC1);
        personMask.setTo(255, markers > 1);
        
        // Clean up the mask
        personMask = cleanupMask(personMask);
        
        // Create full image mask
        cv::Mat fullMask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat maskROI = fullMask(bbox);
        personMask.copyTo(maskROI);
        
        // Validate mask
        if (isValidMask(fullMask)) {
            result.mask = fullMask;
            result.segmentedImage = createTransparentBackground(image, fullMask);
            result.isValid = true;
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Watershed segmentation error:" << e.what();
    }
    
    return result;
}

FastSegmentationResult FastSegmentationProcessor::performEdgeBasedSegmentation(const cv::Mat& image, 
                                                                              const cv::Rect& bbox, 
                                                                              double confidence) {
    FastSegmentationResult result;
    result.boundingBox = bbox;
    result.confidence = confidence;
    result.isValid = false;
    
    try {
        // Extract ROI
        cv::Mat roi = image(bbox);
        cv::Mat roiGray;
        cv::cvtColor(roi, roiGray, cv::COLOR_BGR2GRAY);
        
        // Edge detection
        cv::Mat edges;
        cv::Canny(roiGray, edges, 50, 150);
        
        // Dilate edges to create closed contours
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat dilatedEdges;
        cv::dilate(edges, dilatedEdges, kernel, cv::Point(-1, -1), 2);
        
        // Fill the largest contour (should be the person)
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilatedEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Mat personMask = cv::Mat::zeros(roi.size(), CV_8UC1);
        
        if (!contours.empty()) {
            // Find the largest contour
            size_t largestContourIdx = 0;
            double largestArea = 0;
            
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestContourIdx = i;
                }
            }
            
            // Fill the largest contour
            cv::fillPoly(personMask, std::vector<std::vector<cv::Point>>{contours[largestContourIdx]}, cv::Scalar(255));
        } else {
            // Fallback: simple threshold
            cv::threshold(roiGray, personMask, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        }
        
        // Clean up the mask
        personMask = cleanupMask(personMask);
        
        // Create full image mask
        cv::Mat fullMask = cv::Mat::zeros(image.size(), CV_8UC1);
        cv::Mat maskROI = fullMask(bbox);
        personMask.copyTo(maskROI);
        
        // Validate mask
        if (isValidMask(fullMask)) {
            result.mask = fullMask;
            result.segmentedImage = createTransparentBackground(image, fullMask);
            result.isValid = true;
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Edge-based segmentation error:" << e.what();
    }
    
    return result;
}

FastSegmentationResult FastSegmentationProcessor::performSuperpixelSegmentation(const cv::Mat& image, 
                                                                               const cv::Rect& bbox, 
                                                                               double confidence) {
    FastSegmentationResult result;
    result.boundingBox = bbox;
    result.confidence = confidence;
    result.isValid = false;
    
    // Always use watershed segmentation (SLIC/ximgproc not available)
    return performWatershedSegmentation(image, bbox, confidence);
}

FastSegmentationResult FastSegmentationProcessor::processYOLOMask(const cv::Mat& image,
                                                                 const cv::Mat& mask,
                                                                 const cv::Rect& bbox,
                                                                 double confidence) {
    FastSegmentationResult result;
    result.boundingBox = bbox;
    result.confidence = confidence;
    result.isValid = false;
    
    try {
        if (mask.empty()) {
            return result;
        }
        
        // Clone the mask and ensure it's binary
        cv::Mat processedMask = mask.clone();
        if (processedMask.type() != CV_8UC1) {
            processedMask.convertTo(processedMask, CV_8UC1);
        }
        
        // Apply light cleanup for YOLO masks
        processedMask = cleanupMask(processedMask);
        
        // Validate mask
        if (isValidMask(processedMask)) {
            result.mask = processedMask;
            result.segmentedImage = createTransparentBackground(image, processedMask);
            result.isValid = true;
            
            qDebug() << "✅ YOLO mask processed successfully";
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ YOLO mask processing error:" << e.what();
    }
    
    return result;
}

cv::Mat FastSegmentationProcessor::createCombinedSegmentation(const cv::Mat& originalImage, 
                                                            const QList<FastSegmentationResult>& results) {
    if (results.isEmpty() || originalImage.empty()) {
        return cv::Mat();
    }
    
    try {
        // Create combined mask
        cv::Mat combinedMask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
        
        for (const FastSegmentationResult& result : results) {
            if (result.isValid && !result.mask.empty()) {
                cv::bitwise_or(combinedMask, result.mask, combinedMask);
            }
        }
        
        // Create transparent background image
        return createTransparentBackground(originalImage, combinedMask);
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Error combining segmentations:" << e.what();
        return cv::Mat();
    }
}

cv::Mat FastSegmentationProcessor::cleanupMask(const cv::Mat& mask) {
    if (mask.empty()) {
        return cv::Mat();
    }
    
    cv::Mat cleaned = mask.clone();
    
    try {
        // Fill holes
        cleaned = fillHoles(cleaned);
        
        // Remove small noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE));
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);
        
        // Smooth edges
        cleaned = smoothEdges(cleaned);
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Mask cleanup error:" << e.what();
        return mask; // Return original if cleanup fails
    }
    
    return cleaned;
}

cv::Mat FastSegmentationProcessor::fillHoles(const cv::Mat& mask) {
    cv::Mat filled = mask.clone();
    
    try {
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(filled, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Fill all external contours
        cv::fillPoly(filled, contours, cv::Scalar(255));
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Fill holes error:" << e.what();
    }
    
    return filled;
}

cv::Mat FastSegmentationProcessor::smoothEdges(const cv::Mat& mask, int kernelSize) {
    cv::Mat smoothed;
    
    try {
        // Light Gaussian blur followed by threshold to maintain binary nature
        cv::GaussianBlur(mask, smoothed, cv::Size(kernelSize, kernelSize), 0);
        cv::threshold(smoothed, smoothed, 127, 255, cv::THRESH_BINARY);
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Edge smoothing error:" << e.what();
        return mask;
    }
    
    return smoothed;
}

cv::Mat FastSegmentationProcessor::createTransparentBackground(const cv::Mat& originalImage, const cv::Mat& mask) {
    if (originalImage.empty() || mask.empty()) {
        return cv::Mat();
    }
    
    try {
        // Create 4-channel image (BGRA)
        cv::Mat transparentImage;
        cv::cvtColor(originalImage, transparentImage, cv::COLOR_BGR2BGRA);
        
        // Apply mask to alpha channel
        std::vector<cv::Mat> channels;
        cv::split(transparentImage, channels);
        
        // Set alpha channel based on mask
        channels[3] = mask.clone();
        
        // Merge channels back
        cv::merge(channels, transparentImage);
        
        return transparentImage;
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Error creating transparent background:" << e.what();
        return cv::Mat();
    }
}

bool FastSegmentationProcessor::isValidMask(const cv::Mat& mask, double minAreaRatio) {
    if (mask.empty()) {
        return false;
    }
    
    try {
        int nonZeroPixels = cv::countNonZero(mask);
        double totalPixels = mask.rows * mask.cols;
        double ratio = nonZeroPixels / totalPixels;
        
        return ratio >= minAreaRatio && ratio <= 0.9; // Not too small, not too large
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Mask validation error:" << e.what();
        return false;
    }
}

void FastSegmentationProcessor::startTiming() {
    m_timingStart = std::chrono::steady_clock::now();
}

void FastSegmentationProcessor::endTiming() {
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_timingStart);
    double processingTime = duration.count() / 1000.0; // Convert to milliseconds
    
    m_processingTimes.append(processingTime);
}

void FastSegmentationProcessor::updatePerformanceStats() {
    // Keep only recent samples
    while (m_processingTimes.size() > MAX_TIMING_SAMPLES) {
        m_processingTimes.removeFirst();
    }
    
    // Calculate average
    if (!m_processingTimes.isEmpty()) {
        double sum = 0.0;
        for (double time : m_processingTimes) {
            sum += time;
        }
        m_avgProcessingTime = sum / m_processingTimes.size();
        m_currentFPS = static_cast<int>(1000.0 / m_avgProcessingTime);
    }
    
    m_frameCount++;
}