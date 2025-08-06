#include "tflite_deeplabv3.h"
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp> // For general DNN operations if needed, though not directly used for TFLite in fallback
#include <algorithm> // For std::max and std::min

TFLiteDeepLabv3::TFLiteDeepLabv3(QObject *parent)
    : QObject(parent)
    , m_modelLoaded(false)
    , m_inputWidth(513)
    , m_inputHeight(513)
    , m_confidenceThreshold(0.5f)
         , m_processingInterval(16) // ~60 FPS for better performance
    , m_performanceMode(Balanced)
    , m_processingActive(false)
    , hasValidTracking(false)
    , trackerInitialized(false)
    , trackingFrames(0)
{
    m_processingTimer = new QTimer(this);
    m_processingTimer->setSingleShot(false);
    connect(m_processingTimer, &QTimer::timeout, this, &TFLiteDeepLabv3::processQueuedFrames);
    
    m_processingThread = new QThread(this);
    m_processingThread->setObjectName("SegmentationThread");
    
    initializeColorPalette();
    
    qDebug() << "TFLiteDeepLabv3 initialized (OpenCV fallback mode)";
}

TFLiteDeepLabv3::~TFLiteDeepLabv3()
{
    stopRealtimeProcessing();
    
    if (m_processingThread) {
        m_processingThread->quit();
        m_processingThread->wait();
    }
}

bool TFLiteDeepLabv3::initializeModel(const QString &modelPath)
{
    // Handle OpenCV fallback case
    if (modelPath == "opencv_fallback") {
        qDebug() << "Using OpenCV-based segmentation fallback";
        m_modelLoaded = true;
        emit modelLoaded(true);
        return true;
    }
    
    QFileInfo fileInfo(modelPath);
    if (!fileInfo.exists()) {
        qWarning() << "Model file not found:" << modelPath;
        emit processingError(QString("Model file not found: %1").arg(modelPath));
        emit modelLoaded(false);
        return false;
    }

    // For now, we'll use OpenCV-based segmentation as a fallback
    // since TensorFlow Lite compilation is problematic
    qDebug() << "Using OpenCV-based segmentation fallback";
    m_modelLoaded = true;
    emit modelLoaded(true);
    return true;
}

bool TFLiteDeepLabv3::loadModel(const QString &modelPath)
{
    return initializeModel(modelPath);
}

cv::Mat TFLiteDeepLabv3::segmentFrame(const cv::Mat &inputFrame)
{
    if (!m_modelLoaded) {
        qWarning() << "Model not loaded";
        return inputFrame.clone();
    }

    try {
        // Use OpenCV-based person segmentation as fallback
        cv::Mat segmentedFrame = performOpenCVSegmentation(inputFrame);
        return segmentedFrame;
        
    } catch (const std::exception& e) {
        qWarning() << "Exception during segmentation:" << e.what();
        return inputFrame.clone();
    }
}

cv::Mat TFLiteDeepLabv3::performOpenCVSegmentation(const cv::Mat &inputFrame)
{
    // Simplified and more effective person detection and segmentation
    cv::Mat frame = inputFrame.clone();
    
    // Initialize HOG person detector
    static cv::HOGDescriptor hog;
    static bool hogInitialized = false;
    if (!hogInitialized) {
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        hogInitialized = true;
        qDebug() << "HOG person detector initialized";
    }
    
    // Initialize tracking variables
    static cv::Rect lastDetection;
    static bool hasValidTracking = false;
    static int trackingFrames = 0;
    static bool trackerInitialized = false;
    
         // Resize frame for faster detection - use smaller scale for better performance
     cv::Mat resizedFrame;
     double scale = 0.25; // Process at quarter resolution for much better speed
     cv::resize(frame, resizedFrame, cv::Size(), scale, scale);
    
                   // Detect people with HOG - simplified for better performance
      std::vector<cv::Rect> foundLocations;
      std::vector<double> weights;
      hog.detectMultiScale(resizedFrame, foundLocations, weights, 0, cv::Size(8, 8), cv::Size(8, 8), 1.1, 1, false);
    
    // Debug output
    if (!foundLocations.empty()) {
        qDebug() << "HOG detected" << foundLocations.size() << "person(s) with confidences:" << weights;
    }
    
    // Find the best detection
    cv::Rect currentDetection;
    bool detectionSuccess = false;
    
    if (!foundLocations.empty()) {
        // Find the best detection (highest confidence and reasonable size)
        double bestConfidence = 0;
        size_t bestIndex = 0;
        
        for (size_t i = 0; i < foundLocations.size(); i++) {
            double confidence = weights[i];
            cv::Rect detection = foundLocations[i];
            
                                                                                                                                                                                                               // More permissive person detection - back to working version
                if (confidence > 0.1 && detection.width > 15 && detection.height > 30) {
                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestIndex = i;
                }
            }
        }
        
                                                                                                                                               if (bestConfidence > 0.1) {
            currentDetection = foundLocations[bestIndex];
            detectionSuccess = true;
            
            // Initialize tracking
            if (!trackerInitialized) {
                trackerInitialized = true;
                hasValidTracking = true;
                trackingFrames = 0;
                qDebug() << "New detection with confidence:" << bestConfidence;
            }
        }
    }
    
                             // If no HOG detection, try to use last known position - more permissive
      if (!detectionSuccess && hasValidTracking && trackingFrames < 20 && !lastDetection.empty()) {
          currentDetection = lastDetection;
          detectionSuccess = true;
          trackingFrames++;
          qDebug() << "Using last known position, frames:" << trackingFrames;
      }
      
                           // Fallback: if no detection at all, create a default detection in the center
        if (!detectionSuccess) {
            int centerX = resizedFrame.cols / 2;
            int centerY = resizedFrame.rows / 2;
            int defaultWidth = resizedFrame.cols / 3;
            int defaultHeight = resizedFrame.rows / 2;
            
            currentDetection = cv::Rect(centerX - defaultWidth/2, centerY - defaultHeight/2, 
                                       defaultWidth, defaultHeight);
            detectionSuccess = true;
            qDebug() << "Using fallback center detection";
        }
    
    // Create person mask
    cv::Mat personMask = cv::Mat::zeros(resizedFrame.size(), CV_8UC1);
    
    if (detectionSuccess) {
        // Store for next frame
        lastDetection = currentDetection;
        trackingFrames = 0;
        
                                   // Balanced detection expansion for complete person
          cv::Rect expandedDetection = currentDetection;
          int expandX = static_cast<int>(currentDetection.width * 0.15);  // Moderate horizontal expansion
          int expandY = static_cast<int>(currentDetection.height * 0.3);  // Moderate vertical expansion
          
          expandedDetection.x = std::max(0, expandedDetection.x - expandX);
          expandedDetection.y = std::max(0, expandedDetection.y - expandY);
          expandedDetection.width = std::min(resizedFrame.cols - expandedDetection.x, 
                                            expandedDetection.width + 2 * expandX);
          expandedDetection.height = std::min(resizedFrame.rows - expandedDetection.y, 
                                             expandedDetection.height + 2 * expandY);
        
                          // Extract the region of interest with some padding
         cv::Mat roi = resizedFrame(expandedDetection);
         
         // Check if ROI is valid
         if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) {
             qDebug() << "Invalid ROI, skipping processing";
             return frame.clone();
         }
         
         // Convert to grayscale for edge detection
         cv::Mat gray;
         cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
         
         // Apply Gaussian blur to reduce noise
         cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
         
                            // Enhanced edge detection for complete person capture
          cv::Mat edges1, edges2, edges3, combinedEdges;
          
                                                                                       // Simplified single-scale edge detection for better performance
             cv::Canny(gray, combinedEdges, 30, 90);   // Single scale for speed
          
                                           // Simplified morphological operations for better performance
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(combinedEdges, combinedEdges, cv::MORPH_CLOSE, kernel);
         
         // Find contours in the edge image
         std::vector<std::vector<cv::Point>> edgeContours;
         cv::findContours(combinedEdges, edgeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
         
         // Create a mask for the person silhouette
         cv::Mat silhouetteMask = cv::Mat::zeros(roi.size(), CV_8UC1);
    
                           // Permissive contour filtering for complete person capture
          std::vector<std::vector<cv::Point>> validContours;
          
          for (const auto& contour : edgeContours) {
              double area = cv::contourArea(contour);
              cv::Rect boundingRect = cv::boundingRect(contour);
              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               // More permissive contour filtering - back to working version
                     if (area > 10 && area < roi.rows * roi.cols * 0.95) { // Very wide area range
                         double aspectRatio = static_cast<double>(boundingRect.height) / boundingRect.width;
                         
                         // Accept very wide range of proportions
                         if (aspectRatio > 0.1 && aspectRatio < 10.0) { // Very flexible proportions
                          // Check if contour is within the detection area
                          cv::Point2f roiCenter(roi.cols / 2.0, roi.rows / 2.0);
                          cv::Point2f contourCenter;
                          cv::Moments moments = cv::moments(contour);
                          if (moments.m00 != 0) {
                              contourCenter.x = moments.m10 / moments.m00;
                              contourCenter.y = moments.m01 / moments.m00;
                              
                                                             double distance = cv::norm(roiCenter - contourCenter);
                               double maxDistance = std::min(roi.cols, roi.rows) * 0.9; // Very relaxed distance constraint
                              
                              if (distance < maxDistance) {
                                  validContours.push_back(contour);
                              }
                          }
                      }
                  }
          }
        
        if (!validContours.empty()) {
            // Sort contours by area (largest first)
            std::sort(validContours.begin(), validContours.end(), 
                     [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                         return cv::contourArea(a) > cv::contourArea(b);
                     });
            
                                      // Aggressive silhouette formation for complete person capture
              cv::Mat tempSilhouette = cv::Mat::zeros(roi.size(), CV_8UC1);
              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               // More aggressive silhouette formation - back to working version
                   for (size_t i = 0; i < std::min(validContours.size(), size_t(8)); i++) {
                       cv::drawContours(tempSilhouette, validContours, static_cast<int>(i), cv::Scalar(255), -1);
                   }
                  
                  // Simplified morphological operations for speed
                  cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                  cv::morphologyEx(tempSilhouette, silhouetteMask, cv::MORPH_CLOSE, closeKernel);
            
            // Fill holes gently
            cv::Mat floodFillMask = cv::Mat::zeros(silhouetteMask.rows + 2, silhouetteMask.cols + 2, CV_8UC1);
            silhouetteMask.copyTo(floodFillMask(cv::Rect(1, 1, silhouetteMask.cols, silhouetteMask.rows)));
            cv::floodFill(floodFillMask, cv::Point(0, 0), cv::Scalar(255));
            cv::Mat floodFillMaskInverted;
            cv::bitwise_not(floodFillMask, floodFillMaskInverted);
            silhouetteMask = silhouetteMask | floodFillMaskInverted(cv::Rect(1, 1, silhouetteMask.cols, silhouetteMask.rows));
            
                                                   // Simplified smoothing for better performance
              cv::Mat smoothKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
              cv::morphologyEx(silhouetteMask, silhouetteMask, cv::MORPH_OPEN, smoothKernel);
             
             // Final refinement: keep only the largest connected region
             std::vector<std::vector<cv::Point>> finalContours;
             cv::findContours(silhouetteMask, finalContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
             
             if (!finalContours.empty()) {
                 // Sort by area and keep the largest
                 std::sort(finalContours.begin(), finalContours.end(), 
                          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                              return cv::contourArea(a) > cv::contourArea(b);
                          });
                 
                 cv::Mat refinedMask = cv::Mat::zeros(roi.size(), CV_8UC1);
                 cv::drawContours(refinedMask, finalContours, 0, cv::Scalar(255), -1);
                 silhouetteMask = refinedMask;
             }
             
                                                                                                                                                                                                                                                                                                                                                                                                                                                               // Simplified background removal for better performance
                 // Skip complex background removal for speed - just use the silhouette as is
            
                                   } else {
              // Enhanced fallback for complete person capture
              qDebug() << "No valid contours found, using fallback silhouette";
              if (hasValidTracking && trackerInitialized && !lastSilhouette.empty()) {
                 // Use previous silhouette as a guide for consistency
                 cv::Mat previousSilhouette;
                 cv::resize(lastSilhouette, previousSilhouette, roi.size(), 0, 0, cv::INTER_LINEAR);
                 
                 // Apply dilation to account for movement and ensure complete coverage
                 cv::Mat dilatedPrevious;
                 cv::dilate(previousSilhouette, dilatedPrevious, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
                 
                 // Use the dilated previous silhouette as a starting point
                 silhouetteMask = dilatedPrevious.clone();
                 
                           } else {
                                                                       // Create a better person-shaped fallback using edge detection
                   // Use the original detection rectangle as a starting point
                   cv::rectangle(silhouetteMask, cv::Rect(0, 0, roi.cols, roi.rows), cv::Scalar(255), -1);
                   
                   // Apply edge-based refinement to make it more person-shaped
                   cv::Mat edgeRefined = cv::Mat::zeros(roi.size(), CV_8UC1);
                   
                   // Use the existing edge detection to refine the shape
                   if (!combinedEdges.empty()) {
                       // Find contours in the edge image for refinement
                       std::vector<std::vector<cv::Point>> refinementContours;
                       cv::findContours(combinedEdges, refinementContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                       
                       // Draw edge contours to refine the shape
                       for (size_t i = 0; i < std::min(refinementContours.size(), size_t(3)); i++) {
                           double area = cv::contourArea(refinementContours[i]);
                           if (area > 50) { // Reasonable threshold for meaningful contours
                               cv::drawContours(edgeRefined, refinementContours, static_cast<int>(i), cv::Scalar(255), -1);
                           }
                       }
                       
                       // Combine with the rectangle mask
                       cv::Mat combinedMask;
                       cv::bitwise_and(silhouetteMask, edgeRefined, combinedMask);
                       silhouetteMask = combinedMask;
                   }
                   
                                                           // Simplified morphological operations for speed
                     cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                     cv::morphologyEx(silhouetteMask, silhouetteMask, cv::MORPH_CLOSE, closeKernel);
                     
                                                                                       // Simple fallback rectangle
                       if (cv::countNonZero(silhouetteMask) < 50) {
                           cv::rectangle(silhouetteMask, cv::Rect(roi.cols/4, roi.rows/8, roi.cols/2, roi.rows*3/4), cv::Scalar(255), -1);
                       }
             }
        }
        
            // Copy the silhouette to the main mask
    cv::Mat roiMask = personMask(expandedDetection);
    silhouetteMask.copyTo(roiMask);
    
                                       // Simplified final smoothing for better performance
       cv::Mat finalKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
       cv::morphologyEx(personMask, personMask, cv::MORPH_CLOSE, finalKernel);
     
           // Simple final refinement to keep only the largest connected region
      std::vector<std::vector<cv::Point>> finalContours;
      cv::findContours(personMask, finalContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      
      if (!finalContours.empty()) {
          // Sort by area and keep the largest
          std::sort(finalContours.begin(), finalContours.end(), 
                   [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                       return cv::contourArea(a) > cv::contourArea(b);
                   });
          
          cv::Mat refinedMask = cv::Mat::zeros(personMask.size(), CV_8UC1);
          cv::drawContours(refinedMask, finalContours, 0, cv::Scalar(255), -1);
          personMask = refinedMask;
      }
    
         // Update tracking information for consistency
     if (!silhouetteMask.empty()) {
         lastSilhouette = silhouetteMask.clone();
     }
     lastDetection = expandedDetection;
     lastCenter = cv::Point2f(expandedDetection.x + expandedDetection.width/2, 
                             expandedDetection.y + expandedDetection.height/2);
     hasValidTracking = true;
     trackerInitialized = true;
     trackingFrames = 0;
        
    } else {
                 // Reset tracking if no detection for too long - more permissive
         if (trackingFrames > 20) {
            hasValidTracking = false;
            trackerInitialized = false;
            qDebug() << "Tracking lost, resetting";
        }
    }
    
    // Resize back to original size
    cv::Mat fullSizeMask;
    cv::resize(personMask, fullSizeMask, frame.size(), 0, 0, cv::INTER_LINEAR);
    
    // Create segmentation visualization
    cv::Mat blendedFrame = frame.clone();
    
    // Create masks for person and background
    cv::Mat personMaskFinal = fullSizeMask.clone();
    cv::Mat backgroundMask;
    cv::bitwise_not(personMaskFinal, backgroundMask);
    
    // Apply background darkening
    cv::Mat darkenedBackground;
    cv::addWeighted(frame, 0.1, cv::Mat::zeros(frame.size(), CV_8UC3), 0.9, 0, darkenedBackground);
    darkenedBackground.copyTo(blendedFrame, backgroundMask);
    
                   // Simplified green outline for better performance
      cv::Mat outline;
      cv::Canny(personMaskFinal, outline, 30, 90);
      cv::cvtColor(outline, outline, cv::COLOR_GRAY2BGR);
      outline.setTo(cv::Scalar(0, 255, 0), outline);
      cv::addWeighted(blendedFrame, 1.0, outline, 0.8, 0, blendedFrame);
    
    return blendedFrame;
}

cv::Mat TFLiteDeepLabv3::preprocessFrame(const cv::Mat &inputFrame)
{
    cv::Mat resizedFrame;
    cv::resize(inputFrame, resizedFrame, cv::Size(m_inputWidth, m_inputHeight));
    
    // Convert BGR to RGB
    cv::Mat rgbFrame;
    cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1] and convert to float
    cv::Mat floatFrame;
    rgbFrame.convertTo(floatFrame, CV_32F, 1.0/255.0);
    
    // Reshape to match model input format (1, height, width, 3)
    cv::Mat reshapedFrame = floatFrame.reshape(1, 1);
    
    return reshapedFrame;
}

cv::Mat TFLiteDeepLabv3::postprocessSegmentation(const cv::Mat &inputFrame, const std::vector<float> &output)
{
    // This method is not used in the OpenCV fallback mode
    Q_UNUSED(output)
    return inputFrame.clone();
}

void TFLiteDeepLabv3::startRealtimeProcessing()
{
    if (!m_modelLoaded) {
        qWarning() << "Cannot start processing: model not loaded";
        return;
    }
    
    m_processingActive = true;
    m_processingTimer->start(m_processingInterval);
    qDebug() << "Started real-time segmentation processing (OpenCV mode)";
}

void TFLiteDeepLabv3::stopRealtimeProcessing()
{
    m_processingActive = false;
    m_processingTimer->stop();
    
    // Clear frame queue
    QMutexLocker locker(&m_frameMutex);
    m_frameQueue.clear();
    
    qDebug() << "Stopped real-time segmentation processing";
}

void TFLiteDeepLabv3::setInputSize(int width, int height)
{
    m_inputWidth = width;
    m_inputHeight = height;
}

void TFLiteDeepLabv3::setConfidenceThreshold(float threshold)
{
    m_confidenceThreshold = threshold;
}

void TFLiteDeepLabv3::setProcessingInterval(int msec)
{
    m_processingInterval = msec;
    if (m_processingTimer->isActive()) {
        m_processingTimer->setInterval(m_processingInterval);
    }
}

void TFLiteDeepLabv3::setPerformanceMode(PerformanceMode mode)
{
    m_performanceMode = mode;
    
    // Adjust processing parameters based on performance mode
    switch (mode) {
        case HighQuality:
            m_processingInterval = 50; // 20 FPS
            m_confidenceThreshold = 0.7f;
            break;
                 case Balanced:
             m_processingInterval = 16; // 60 FPS
             m_confidenceThreshold = 0.5f;
             break;
        case HighSpeed:
            m_processingInterval = 16; // 60 FPS
            m_confidenceThreshold = 0.3f;
            break;
                 case Adaptive:
             // Will be adjusted dynamically based on performance
             m_processingInterval = 16;
             m_confidenceThreshold = 0.5f;
             break;
    }
    
    // Update timer interval if active
    if (m_processingTimer->isActive()) {
        m_processingTimer->setInterval(m_processingInterval);
    }
}

void TFLiteDeepLabv3::processFrame(const QImage &frame)
{
    cv::Mat cvFrame = qImageToCvMat(frame);
    processFrame(cvFrame);
}

void TFLiteDeepLabv3::processFrame(const cv::Mat &frame)
{
    if (!m_processingActive) {
        return;
    }
    
    QMutexLocker locker(&m_frameMutex);
    
    // Keep only the latest frame to avoid queue buildup
    m_frameQueue.clear();
    m_frameQueue.enqueue(frame.clone());
    
    m_frameCondition.wakeOne();
}

void TFLiteDeepLabv3::processQueuedFrames()
{
    QMutexLocker locker(&m_frameMutex);
    
    if (m_frameQueue.isEmpty()) {
        return;
    }
    
    cv::Mat frame = m_frameQueue.dequeue();
    locker.unlock();
    
    // Process the frame
    cv::Mat segmentedFrame = segmentFrame(frame);
    
    // Convert to QImage and emit signal
    QImage resultImage = cvMatToQImage(segmentedFrame);
    emit segmentationResultReady(resultImage);
}

QImage TFLiteDeepLabv3::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC3: {
        QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    case CV_8UC4: {
        QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return img;
    }
    default:
        qWarning() << "Unsupported image format for conversion";
        return QImage();
    }
}

cv::Mat TFLiteDeepLabv3::qImageToCvMat(const QImage &image)
{
    switch (image.format()) {
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied: {
        cv::Mat mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine());
        cv::Mat mat2;
        cv::cvtColor(mat, mat2, cv::COLOR_BGRA2BGR);
        return mat2;
    }
    case QImage::Format_RGB888: {
        cv::Mat mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
        cv::Mat mat2;
        cv::cvtColor(mat, mat2, cv::COLOR_RGB2BGR);
        return mat2;
    }
    default:
        qWarning() << "Unsupported QImage format for conversion";
        return cv::Mat();
    }
}

void TFLiteDeepLabv3::initializeColorPalette()
{
    // Initialize color palette for segmentation visualization
    // These colors correspond to different segmentation classes
    m_colorPalette = {
        cv::Vec3b(0, 0, 0),        // Background - Black
        cv::Vec3b(128, 0, 0),      // Person - Dark Red
        cv::Vec3b(0, 128, 0),      // Animal - Green
        cv::Vec3b(128, 128, 0),    // Vehicle - Olive
        cv::Vec3b(0, 0, 128),      // Object - Navy
        cv::Vec3b(128, 0, 128),    // Building - Purple
        cv::Vec3b(0, 128, 128),    // Nature - Teal
        cv::Vec3b(128, 128, 128),  // Other - Gray
        cv::Vec3b(64, 0, 0),       // Additional classes...
        cv::Vec3b(192, 0, 0),
        cv::Vec3b(64, 128, 0),
        cv::Vec3b(192, 128, 0),
        cv::Vec3b(64, 0, 128),
        cv::Vec3b(192, 0, 128),
        cv::Vec3b(64, 128, 128),
        cv::Vec3b(192, 128, 128),
        cv::Vec3b(0, 64, 0),
        cv::Vec3b(128, 64, 0),
        cv::Vec3b(0, 192, 0),
        cv::Vec3b(128, 192, 0)
    };
}

// SegmentationThread implementation
SegmentationThread::SegmentationThread(TFLiteDeepLabv3 *processor, QObject *parent)
    : QThread(parent)
    , m_processor(processor)
    , m_running(false)
{
}

void SegmentationThread::run()
{
    m_running = true;
    
    while (m_running) {
        // Process frames in the background
        QThread::msleep(10); // Small delay to prevent busy waiting
    }
}

void SegmentationThread::stop()
{
    m_running = false;
} 