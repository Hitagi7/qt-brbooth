// Segmentation Processing Implementation
// Extracted from capture.cpp for better code organization

#include "core/capture.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QCoreApplication>
#include <QMutexLocker>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <chrono>

// Fixed segmentation rectangle constants
static const int kFixedRectWidth = 1440;   // constant width in pixels
static const int kFixedRectHeight = 720;  // constant height in pixels
static const int kFixedRectX = 0;       // left offset in pixels (adjustable)
static const int kFixedRectY = 100;        // top offset in pixels (adjustable)

// Compute a fixed rectangle and clamp it to the frame bounds to ensure it stays inside
static cv::Rect getFixedSegmentationRect(const cv::Size &frameSize)
{
    int w = std::min(kFixedRectWidth, frameSize.width);
    int h = std::min(kFixedRectHeight, frameSize.height);
    int x = std::max(0, std::min(kFixedRectX, frameSize.width - w));
    int y = std::max(0, std::min(kFixedRectY, frameSize.height - h));
    return cv::Rect(x, y, w, h);
}

cv::Mat Capture::processFrameWithGPUOnlyPipeline(const cv::Mat &frame)
{
    if (frame.empty()) {
        return cv::Mat();
    }

    updateGreenBackgroundModel(frame);
    m_personDetectionTimer.start();

    try {
        qDebug() << "Phase 2A: Using GPU-only processing pipeline";

        // Upload frame to GPU (single transfer)
        m_gpuVideoFrame.upload(frame);

        // GREEN SCREEN MODE: Use GPU-accelerated green screen masking
        if (m_greenScreenEnabled && m_segmentationEnabledInCapture) {
            qDebug() << "Processing green screen with GPU acceleration";
            
            // VALIDATION: Ensure GPU frame is valid
            if (m_gpuVideoFrame.empty() || m_gpuVideoFrame.cols == 0 || m_gpuVideoFrame.rows == 0) {
                qWarning() << "GPU video frame is invalid for green screen, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // Create GPU-accelerated green screen mask with crash protection
            cv::cuda::GpuMat gpuPersonMask;
            try {
                gpuPersonMask = createGreenScreenPersonMaskGPU(m_gpuVideoFrame);
            } catch (const cv::Exception &e) {
                qWarning() << "GPU green screen mask creation failed:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            } catch (const std::exception &e) {
                qWarning() << "Exception in GPU green screen:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // VALIDATION: Ensure mask is valid
            if (gpuPersonMask.empty()) {
                qWarning() << "GPU green screen mask is empty, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // GPU SYNCHRONIZATION: Wait for all GPU operations to complete before downloading
            cv::cuda::Stream::Null().waitForCompletion();
            
            // REMOVE GREEN SPILL: Desaturate green tint from person pixels
            cv::cuda::GpuMat gpuCleanedFrame;
            cv::Mat cleanedFrame;
            try {
                gpuCleanedFrame = removeGreenSpillGPU(m_gpuVideoFrame, gpuPersonMask);
                if (!gpuCleanedFrame.empty()) {
                    gpuCleanedFrame.download(cleanedFrame);
                    qDebug() << "Green spill removal applied to person pixels";
                } else {
                    cleanedFrame = frame.clone();
                }
            } catch (const cv::Exception &e) {
                qWarning() << "Green spill removal failed:" << e.what() << "- using original frame";
                cleanedFrame = frame.clone();
            }
            
            // Download mask to derive detections on CPU (for bounding boxes)
            cv::Mat personMask;
            try {
                gpuPersonMask.download(personMask);
            } catch (const cv::Exception &e) {
                qWarning() << "Failed to download GPU mask:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            if (personMask.empty()) {
                qWarning() << "Downloaded mask is empty, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            std::vector<cv::Rect> detections = deriveDetectionsFromMask(personMask);
            m_lastDetections = detections;
            
            qDebug() << "Derived" << detections.size() << "detections from green screen mask";
            
            // Use cleaned frame (with green spill removed) for GPU-only segmentation
            cv::Mat segmentedFrame;
            try {
                // Upload cleaned frame to GPU for segmentation
                m_gpuVideoFrame.upload(cleanedFrame);
                segmentedFrame = createSegmentedFrameGPUOnly(cleanedFrame, detections);
            } catch (const cv::Exception &e) {
                qWarning() << "GPU segmentation failed:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            } catch (const std::exception &e) {
                qWarning() << "Exception in GPU segmentation:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // VALIDATION: Ensure segmented frame is valid
            if (segmentedFrame.empty()) {
                qWarning() << "GPU segmented frame is empty, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
            m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;
            
            qDebug() << "GPU green screen processing completed successfully";
            return segmentedFrame;
        }

        // Optimized processing for 30 FPS with GPU
        cv::cuda::GpuMat processFrame = m_gpuVideoFrame;
        if (frame.cols > 640) {
            double scale = 640.0 / frame.cols;
            cv::cuda::resize(m_gpuVideoFrame, processFrame, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }

        // Use a fixed, bounded segmentation rectangle instead of person detection
        std::vector<cv::Rect> fixedDetections;
        fixedDetections.push_back(getFixedSegmentationRect(frame.size()));

        // Store detections for UI display
        m_lastDetections = fixedDetections;

        // Create segmented frame with GPU-only processing
        cv::Mat segmentedFrame = createSegmentedFrameGPUOnly(frame, fixedDetections);

        // Update timing info
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;

        qDebug() << "Phase 2A: GPU-only processing completed successfully";

        return segmentedFrame;

    } catch (const cv::Exception& e) {
        qWarning() << "GPU-only processing failed, falling back to CPU:" << e.what();
        // Fallback to CPU processing
        return processFrameWithUnifiedDetection(frame);
    } catch (const std::exception& e) {
        qWarning() << "Exception in GPU-only processing, falling back to CPU:" << e.what();
        return processFrameWithUnifiedDetection(frame);
    } catch (...) {
        qWarning() << "Unknown error in GPU-only processing, falling back to CPU";
        return processFrameWithUnifiedDetection(frame);
    }
}

cv::Mat Capture::processFrameWithUnifiedDetection(const cv::Mat &frame)
{
    // Validate input frame
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
        qWarning() << "Invalid frame received, returning empty result";
        return cv::Mat::zeros(480, 640, CV_8UC3);
    }

    //  PERFORMANCE OPTIMIZATION: NEVER apply lighting during real-time processing
    // Lighting is ONLY applied in post-processing after recording, just like static mode

    // If green-screen is enabled, bypass HOG and derive mask directly
    if (m_greenScreenEnabled && m_segmentationEnabledInCapture) {
        cv::Mat personMask = createGreenScreenPersonMask(frame);
        std::vector<cv::Rect> detections = deriveDetectionsFromMask(personMask);
        m_lastDetections = detections;
        cv::Mat segmentedFrame = createSegmentedFrame(frame, detections);
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;
        return segmentedFrame;
    }

    // Phase 2A: Use GPU-only processing if available
    if (isGPUOnlyProcessingAvailable()) {
        return processFrameWithGPUOnlyPipeline(frame);
    }

    m_personDetectionTimer.start();

    try {
        // Optimized processing for 30 FPS with GPU (matching peopledetect_v1.cpp)
        cv::Mat processFrame = frame;
        if (frame.cols > 640) {
            double scale = 640.0 / frame.cols;
            cv::resize(frame, processFrame, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }

        // Use a fixed, bounded segmentation rectangle instead of person detection
        std::vector<cv::Rect> fixedDetections;
        fixedDetections.push_back(getFixedSegmentationRect(frame.size()));

        // Store detections for UI display
        m_lastDetections = fixedDetections;

        // Create segmented frame with fixed rectangle
        // NO LIGHTING APPLIED HERE - only segmentation for display
        cv::Mat segmentedFrame = createSegmentedFrame(frame, fixedDetections);

        // Update timing info
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;

        // Log detections for visibility (reduced frequency for performance)
        if (fixedDetections.size() > 0) {
            // qDebug() << "FIXED RECTANGLE ACTIVE:" << fixedDetections[0].x << fixedDetections[0].y << fixedDetections[0].width << "x" << fixedDetections[0].height;
        } else {
            qDebug() << "NO FIXED RECTANGLE (unexpected)";

            // For dynamic video backgrounds, always create a segmented frame even without people detection
            // This ensures the video background is always visible
            if (m_segmentationEnabledInCapture && m_useDynamicVideoBackground) {
                qDebug() << "Dynamic video mode: Creating segmented frame without people detection to show video background";
                // Don't add fake detection, just let createSegmentedFrame handle the background
            }
        }

        return segmentedFrame;

    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV exception in unified detection:" << e.what();
        return frame.clone();
    } catch (const std::exception& e) {
        qWarning() << "Exception in unified detection:" << e.what();
        return frame.clone();
    } catch (...) {
        qWarning() << "Unknown error in unified detection, returning original frame";
        return frame.clone();
    }
}

cv::Mat Capture::createSegmentedFrame(const cv::Mat &frame, const std::vector<cv::Rect> &detections)
{
    // Process only first 3 detections for better performance (matching peopledetect_v1.cpp)
    int maxDetections = std::min(3, (int)detections.size());

    if (m_segmentationEnabledInCapture) {
        qDebug() << "SEGMENTATION MODE (CPU): Creating background + edge-based silhouettes";
        qDebug() << "- m_useDynamicVideoBackground:" << m_useDynamicVideoBackground;
        qDebug() << "- m_videoPlaybackActive:" << m_videoPlaybackActive;
        qDebug() << "- detections count:" << detections.size();

        // Create background for edge-based segmentation
        cv::Mat segmentedFrame;

        // Use cached background template for performance
        static cv::Mat cachedBackgroundTemplate;
        static QString lastBackgroundPath;

        //  PERFORMANCE OPTIMIZATION: Always use lightweight processing during recording
        if (m_isRecording) {
            // Use lightweight background during recording
            // CRASH FIX: Add mutex lock and validation when accessing dynamic video frame
            if (m_useDynamicVideoBackground) {
                QMutexLocker locker(&m_dynamicVideoMutex);
                if (!m_dynamicVideoFrame.empty() && m_dynamicVideoFrame.cols > 0 && m_dynamicVideoFrame.rows > 0) {
                    try {
                        cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                        qDebug() << " RECORDING: Using dynamic video frame as background";
                    } catch (const cv::Exception &e) {
                        qWarning() << " RECORDING: Failed to resize dynamic video frame:" << e.what();
                        segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                    }
                } else {
                    qWarning() << " RECORDING: Dynamic video frame invalid, using black background";
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                }
            } else {
                // Use black background for performance
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else if (m_useDynamicVideoBackground) {
            // CRITICAL FIX: ALWAYS read a NEW frame from video to ensure it's MOVING
            // Never use cached frames - always advance the video position
            try {
                cv::Mat nextBg;
                bool frameRead = false;
                
                // ALWAYS read directly from video to advance position
                if (m_dynamicVideoCap.isOpened()) {
                    frameRead = m_dynamicVideoCap.read(nextBg);
                    if (!frameRead || nextBg.empty()) {
                        // Video reached end, loop it
                        m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        frameRead = m_dynamicVideoCap.read(nextBg);
                    }
                } else if (!m_dynamicGpuReader.empty()) {
                    // Try GPU reader
                    try {
                        cv::cuda::GpuMat gpu;
                        if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
                            if (gpu.type() == CV_8UC4) {
                                cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                            }
                            gpu.download(nextBg);
                            frameRead = !nextBg.empty();
                            // Update cached GPU frame
                            {
                                QMutexLocker locker(&m_dynamicVideoMutex);
                                m_dynamicGpuFrame = gpu;
                            }
                        } else {
                            // GPU reader reached end, restart it
                            m_dynamicGpuReader.release();
                            m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
                            cv::cuda::GpuMat gpuRetry;
                            if (!m_dynamicGpuReader.empty() && m_dynamicGpuReader->nextFrame(gpuRetry) && !gpuRetry.empty()) {
                                if (gpuRetry.type() == CV_8UC4) {
                                    cv::cuda::cvtColor(gpuRetry, gpuRetry, cv::COLOR_BGRA2BGR);
                                }
                                gpuRetry.download(nextBg);
                                frameRead = !nextBg.empty();
                                {
                                    QMutexLocker locker(&m_dynamicVideoMutex);
                                    m_dynamicGpuFrame = gpuRetry;
                                }
                            }
                        }
                    } catch (...) {
                        // GPU reader failed, try to open CPU reader as fallback
                        if (!m_dynamicVideoPath.isEmpty()) {
                            m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                            if (!m_dynamicVideoCap.isOpened()) {
                                m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                            }
                            if (m_dynamicVideoCap.isOpened()) {
                                frameRead = m_dynamicVideoCap.read(nextBg);
                                if (!frameRead || nextBg.empty()) {
                                    m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                                    frameRead = m_dynamicVideoCap.read(nextBg);
                                }
                            }
                        }
                    }
                } else {
                    // No reader open, try to open CPU reader
                    if (!m_dynamicVideoPath.isEmpty()) {
                        m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                        if (!m_dynamicVideoCap.isOpened()) {
                            m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                        }
                        if (m_dynamicVideoCap.isOpened()) {
                            frameRead = m_dynamicVideoCap.read(nextBg);
                            if (!frameRead || nextBg.empty()) {
                                m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                                frameRead = m_dynamicVideoCap.read(nextBg);
                            }
                        }
                    }
                }
                
                if (frameRead && !nextBg.empty()) {
                    cv::resize(nextBg, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    // Update the cached frame
                    {
                        QMutexLocker locker(&m_dynamicVideoMutex);
                        m_dynamicVideoFrame = nextBg.clone();
                    }
                } else {
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                    qWarning() << "Failed to read video frame from:" << m_dynamicVideoPath;
                }
            } catch (const cv::Exception &e) {
                qWarning() << "CPU segmentation crashed:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            } catch (const std::exception &e) {
                qWarning() << "Exception in CPU segmentation:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else {
            // Debug why dynamic video background is not being used
            if (m_useDynamicVideoBackground) {
                if (!m_videoPlaybackActive) {
                    qWarning() << "Dynamic video background enabled but playback not active!";
                } else if (m_dynamicVideoFrame.empty()) {
                    qWarning() << "Dynamic video background enabled and playback active but no video frame available!";
                }
            } else {
                qDebug() << "Dynamic video background not enabled - using template or black background";
            }
        }
        
        // Only process background templates if we're not using dynamic video background
        if (!m_useDynamicVideoBackground && m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
            // Check if we need to reload the background template
            bool needReload = cachedBackgroundTemplate.empty() ||
                             lastBackgroundPath != m_selectedBackgroundTemplate;

            if (needReload) {
                qDebug() << "Loading background template:" << m_selectedBackgroundTemplate;

                // Check if this is image6 (white background special case)
                if (m_selectedBackgroundTemplate.contains("bg6.png")) {
                    // Create white background instead of loading a file
                    // OpenCV uses BGR format, so we need to set all channels to 255 for white
                    cachedBackgroundTemplate = cv::Mat(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
                    lastBackgroundPath = m_selectedBackgroundTemplate;
                    qDebug() << "White background created for image6, size:" << frame.cols << "x" << frame.rows;
                } else {
                    // Resolve to an existing filesystem path similar to dynamic asset loading
                    QString requestedPath = m_selectedBackgroundTemplate; // e.g., templates/background/bg1.png

                    QStringList candidates;
                    candidates << requestedPath
                               << QDir::currentPath() + "/" + requestedPath
                               << QCoreApplication::applicationDirPath() + "/" + requestedPath
                               << QCoreApplication::applicationDirPath() + "/../" + requestedPath
                               << QCoreApplication::applicationDirPath() + "/../../" + requestedPath
                               << "../" + requestedPath
                               << "../../" + requestedPath
                               << "../../../" + requestedPath;

                    QString resolvedPath;
                    for (const QString &p : candidates) {
                        if (QFile::exists(p)) { resolvedPath = p; break; }
                    }

                    if (resolvedPath.isEmpty()) {
                        qWarning() << "Background template not found in expected locations for request:" << requestedPath
                                   << "- falling back to black background";
                        cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                    } else {
                        // Load background image directly using OpenCV for performance
                        cv::Mat backgroundImage = cv::imread(resolvedPath.toStdString());
                        if (!backgroundImage.empty()) {
                            // Resize background to match frame size
                            cv::resize(backgroundImage, cachedBackgroundTemplate, frame.size(), 0, 0, cv::INTER_LINEAR);
                            lastBackgroundPath = m_selectedBackgroundTemplate;
                            qDebug() << "Background template loaded from" << resolvedPath
                                     << "and cached at" << frame.cols << "x" << frame.rows;
                        } else {
                            qWarning() << "Failed to decode background template from:" << resolvedPath
                                       << "- using black background";
                            cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                        }
                    }
                }
            }

            // Use cached background template
            segmentedFrame = cachedBackgroundTemplate.clone();
        } else if (!m_useDynamicVideoBackground) {
            // Only use black background if we're not using dynamic video background
            segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            qDebug() << "Using black background (no template selected)";
        }
        // If m_useDynamicVideoBackground is true, segmentedFrame should already be set with video frame

        {
            // ALWAYS use green-screen removal (pure green-only detection)
            cv::Mat personMask = createGreenScreenPersonMask(frame);

            int nonZeroPixels = cv::countNonZero(personMask);
            qDebug() << "Green-screen person mask non-zero:" << nonZeroPixels;

            // Apply mask to extract person from camera frame
            cv::Mat personRegion;
            frame.copyTo(personRegion, personMask);

            // CRITICAL FIX: Use mutex to protect shared person data from race conditions
            // Store raw person data for post-processing (lighting will be applied after capture)
            {
                QMutexLocker locker(&m_personDetectionMutex);
                m_lastRawPersonRegion = personRegion.clone();
                m_lastRawPersonMask = personMask.clone();
            }

            // Store template background if using background template
            if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
                // Use cached template background if available, otherwise load it
                if (m_lastTemplateBackground.empty() || lastBackgroundPath != m_selectedBackgroundTemplate) {
                    // Check if this is bg6.png (white background special case)
                    if (m_selectedBackgroundTemplate.contains("bg6.png")) {
                        // Create white background instead of loading a file
                        m_lastTemplateBackground = cv::Mat(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
                        qDebug() << "White template background cached for post-processing (bg6.png)";
                    } else {
                        QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
                        if (!resolvedPath.isEmpty()) {
                            cv::Mat templateBg = cv::imread(resolvedPath.toStdString());
                            if (!templateBg.empty()) {
                                cv::resize(templateBg, m_lastTemplateBackground, frame.size());
                                qDebug() << "Template background cached for post-processing from:" << resolvedPath;
                            } else {
                                qWarning() << "Failed to load template background from resolved path:" << resolvedPath;
                                m_lastTemplateBackground = cv::Mat();
                            }
                        } else {
                            qWarning() << "Could not resolve template background path:" << m_selectedBackgroundTemplate;
                            m_lastTemplateBackground = cv::Mat();
                        }
                    }
                }
            }

            // Scale the person region with person-only scaling for background template mode and dynamic video mode
            cv::Mat scaledPersonRegion, scaledPersonMask;

            if ((m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground) {
                cv::Size backgroundSize = segmentedFrame.size();
                cv::Size scaledPersonSize;

                if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
                    int scaledWidth = static_cast<int>(backgroundSize.width * m_personScaleFactor + 0.5);
                    int scaledHeight = static_cast<int>(backgroundSize.height * m_personScaleFactor + 0.5);
                    
                    //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
                    scaledWidth = qMax(1, scaledWidth);
                    scaledHeight = qMax(1, scaledHeight);
                    
                    scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
                    qDebug() << "Person scaled to" << scaledWidth << "x" << scaledHeight << "with factor" << m_personScaleFactor;
                } else {
                    scaledPersonSize = backgroundSize;
                }

                //  CRASH PREVENTION: Validate size before resize
                if (scaledPersonSize.width > 0 && scaledPersonSize.height > 0 &&
                    personRegion.cols > 0 && personRegion.rows > 0) {
                    cv::resize(personRegion, scaledPersonRegion, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
                    cv::resize(personMask, scaledPersonMask, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
                } else {
                    qWarning() << " CRASH PREVENTION: Invalid size for scaling - using original size";
                    scaledPersonRegion = personRegion.clone();
                    scaledPersonMask = personMask.clone();
                }

                //  CRASH PREVENTION: Validate scaled mats before compositing
                if (!scaledPersonRegion.empty() && !scaledPersonMask.empty() &&
                    scaledPersonRegion.cols > 0 && scaledPersonRegion.rows > 0 &&
                    scaledPersonMask.cols > 0 && scaledPersonMask.rows > 0) {
                    
                    // Use actual scaled dimensions instead of calculated size
                    cv::Size actualScaledSize(scaledPersonRegion.cols, scaledPersonRegion.rows);
                    int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
                    int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;

                    if (xOffset >= 0 && yOffset >= 0 &&
                        xOffset + actualScaledSize.width <= backgroundSize.width &&
                        xOffset + actualScaledSize.height <= backgroundSize.height &&
                        scaledPersonRegion.cols == scaledPersonMask.cols &&
                        scaledPersonRegion.rows == scaledPersonMask.rows) {
                        
                        try {
                            cv::Rect backgroundRect(cv::Point(xOffset, yOffset), actualScaledSize);
                            cv::Mat backgroundROI = segmentedFrame(backgroundRect);
                            scaledPersonRegion.copyTo(backgroundROI, scaledPersonMask);
                            qDebug() << " COMPOSITING: Successfully composited scaled person at offset" << xOffset << "," << yOffset;
                        } catch (const cv::Exception& e) {
                            qWarning() << " CRASH PREVENTION: Compositing failed:" << e.what() << "- using fallback";
                            scaledPersonRegion.copyTo(segmentedFrame, scaledPersonMask);
                        }
                    } else {
                        scaledPersonRegion.copyTo(segmentedFrame, scaledPersonMask);
                        qDebug() << " COMPOSITING: Using fallback compositing due to bounds check";
                    }
                } else {
                    qWarning() << " CRASH PREVENTION: Scaled mats are empty or invalid - skipping compositing";
                }
            } else {
                //  CRASH PREVENTION: Validate before resize and composite
                if (!personRegion.empty() && !personMask.empty() && 
                    segmentedFrame.cols > 0 && segmentedFrame.rows > 0) {
                    cv::resize(personRegion, scaledPersonRegion, segmentedFrame.size(), 0, 0, cv::INTER_LINEAR);
                    cv::resize(personMask, scaledPersonMask, segmentedFrame.size(), 0, 0, cv::INTER_LINEAR);
                    
                    if (!scaledPersonRegion.empty() && !scaledPersonMask.empty()) {
                        scaledPersonRegion.copyTo(segmentedFrame, scaledPersonMask);
                    }
                }
            }
        }

        // Ensure we always return the video background in segmentation mode
        if (segmentedFrame.empty() && m_useDynamicVideoBackground && !m_dynamicVideoFrame.empty()) {
            qDebug() << "Segmented frame is empty, using video frame directly";
            cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        qDebug() << "Segmentation complete, returning segmented frame - size:" << segmentedFrame.cols << "x" << segmentedFrame.rows << "empty:" << segmentedFrame.empty();
        return segmentedFrame;
    } else {
        // Show original frame with detection rectangles
        cv::Mat displayFrame = frame.clone();

        qDebug() << "Drawing" << maxDetections << "detection rectangles";

        for (int i = 0; i < maxDetections; i++) {
            const auto& detection = detections[i];

            // Draw detection rectangles with thick green lines
            cv::rectangle(displayFrame, detection.tl(), detection.br(), cv::Scalar(0, 255, 0), 3);

            qDebug() << "Rectangle" << i << "at" << detection.x << detection.y << detection.width << "x" << detection.height;
        }

        return displayFrame;
    }
}

// Phase 2A: GPU-Only Segmentation Frame Creation
cv::Mat Capture::createSegmentedFrameGPUOnly(const cv::Mat &frame, const std::vector<cv::Rect> &detections)
{
    // Process only first 3 detections for better performance
    int maxDetections = std::min(3, (int)detections.size());

    if (m_segmentationEnabledInCapture) {
        qDebug() << "SEGMENTATION MODE (GPU): GPU-only segmentation frame creation";
        qDebug() << "- m_useDynamicVideoBackground:" << m_useDynamicVideoBackground;
        qDebug() << "- m_videoPlaybackActive:" << m_videoPlaybackActive;
        qDebug() << "- detections count:" << detections.size();
        qDebug() << "- m_isRecording:" << m_isRecording;

        // Create background for edge-based segmentation
        cv::Mat segmentedFrame;

        // Use cached background template for performance
        static cv::Mat cachedBackgroundTemplate;
        static QString lastBackgroundPath;

        //  PERFORMANCE OPTIMIZATION: Lightweight GPU processing during recording
        if (m_isRecording && m_useDynamicVideoBackground) {
            qDebug() << "RECORDING MODE: Using lightweight GPU processing";
            try {
                // THREAD SAFETY: Lock mutex for safe GPU frame access
                QMutexLocker locker(&m_dynamicVideoMutex);
                
                // CRASH FIX: Validate frames before GPU operations
                if (!m_dynamicGpuFrame.empty() && m_dynamicGpuFrame.cols > 0 && m_dynamicGpuFrame.rows > 0) {
                    cv::cuda::resize(m_dynamicGpuFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.download(segmentedFrame);
                    qDebug() << "RECORDING: Using GPU frame for background";
                } else if (!m_dynamicVideoFrame.empty() && m_dynamicVideoFrame.cols > 0 && m_dynamicVideoFrame.rows > 0) {
                    m_gpuBackgroundFrame.upload(m_dynamicVideoFrame);
                    cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.download(segmentedFrame);
                    qDebug() << "RECORDING: Using CPU frame for background (uploaded to GPU)";
                } else {
                    qWarning() << "RECORDING: No valid video frame, using black background";
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                }
            } catch (const cv::Exception &e) {
                qWarning() << "RECORDING: GPU processing failed:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else if (m_useDynamicVideoBackground) {
            // CRITICAL FIX: ALWAYS read a NEW frame from video to ensure it's MOVING
            // Never use cached frames - always advance the video position
            try {
                cv::Mat nextBg;
                bool frameRead = false;
                
                // ALWAYS read directly from video to advance position
                if (!m_dynamicGpuReader.empty()) {
                    // Try GPU reader first
                    try {
                        cv::cuda::GpuMat gpu;
                        if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
                            if (gpu.type() == CV_8UC4) {
                                cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                            }
                            cv::cuda::resize(gpu, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                            m_gpuSegmentedFrame.download(segmentedFrame);
                            // Update cached GPU frame
                            {
                                QMutexLocker locker(&m_dynamicVideoMutex);
                                m_dynamicGpuFrame = gpu;
                            }
                            frameRead = true;
                        } else {
                            // GPU reader reached end, restart it
                            m_dynamicGpuReader.release();
                            m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
                            cv::cuda::GpuMat gpuRetry;
                            if (!m_dynamicGpuReader.empty() && m_dynamicGpuReader->nextFrame(gpuRetry) && !gpuRetry.empty()) {
                                if (gpuRetry.type() == CV_8UC4) {
                                    cv::cuda::cvtColor(gpuRetry, gpuRetry, cv::COLOR_BGRA2BGR);
                                }
                                cv::cuda::resize(gpuRetry, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                                m_gpuSegmentedFrame.download(segmentedFrame);
                                {
                                    QMutexLocker locker(&m_dynamicVideoMutex);
                                    m_dynamicGpuFrame = gpuRetry;
                                }
                                frameRead = true;
                            }
                        }
                    } catch (...) {
                        // GPU reader failed, try CPU below
                    }
                }
                
                // CPU fallback
                if (!frameRead) {
                    if (m_dynamicVideoCap.isOpened()) {
                        frameRead = m_dynamicVideoCap.read(nextBg);
                        if (!frameRead || nextBg.empty()) {
                            // Video reached end, loop it
                            m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            frameRead = m_dynamicVideoCap.read(nextBg);
                        }
                        if (frameRead && !nextBg.empty()) {
                            // Upload to GPU and resize
                            m_gpuBackgroundFrame.upload(nextBg);
                            cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                            m_gpuSegmentedFrame.download(segmentedFrame);
                            // Update cached frames
                            {
                                QMutexLocker locker(&m_dynamicVideoMutex);
                                m_dynamicVideoFrame = nextBg.clone();
                            }
                        }
                    } else {
                        // No reader open, try to open CPU reader
                        if (!m_dynamicVideoPath.isEmpty()) {
                            m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                            if (!m_dynamicVideoCap.isOpened()) {
                                m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                            }
                            if (m_dynamicVideoCap.isOpened()) {
                                frameRead = m_dynamicVideoCap.read(nextBg);
                                if (!frameRead || nextBg.empty()) {
                                    m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                                    frameRead = m_dynamicVideoCap.read(nextBg);
                                }
                                if (frameRead && !nextBg.empty()) {
                                    m_gpuBackgroundFrame.upload(nextBg);
                                    cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                                    m_gpuSegmentedFrame.download(segmentedFrame);
                                    {
                                        QMutexLocker locker(&m_dynamicVideoMutex);
                                        m_dynamicVideoFrame = nextBg.clone();
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (!frameRead || segmentedFrame.empty()) {
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                    qWarning() << "Failed to read video frame for GPU segmentation from:" << m_dynamicVideoPath;
                }
            } catch (const cv::Exception &e) {
                qWarning() << "GPU segmentation crashed:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            } catch (const std::exception &e) {
                qWarning() << "Exception in GPU segmentation:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
            // GPU-only background template processing
            if (lastBackgroundPath != m_selectedBackgroundTemplate) {
                QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
                if (!resolvedPath.isEmpty()) {
                    cachedBackgroundTemplate = cv::imread(resolvedPath.toStdString());
                    if (cachedBackgroundTemplate.empty()) {
                        qWarning() << "Failed to load background template from resolved path:" << resolvedPath;
                        cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                    } else {
                        // Only show success message once per template change
                        static QString lastLoggedTemplate;
                        if (lastLoggedTemplate != m_selectedBackgroundTemplate) {
                            qDebug() << "GPU: Background template loaded from resolved path:" << resolvedPath;
                            lastLoggedTemplate = m_selectedBackgroundTemplate;
                        }
                    }
                } else {
                    qWarning() << "GPU: Could not resolve background template path:" << m_selectedBackgroundTemplate;
                    cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                }
                lastBackgroundPath = m_selectedBackgroundTemplate;
            }

            if (!cachedBackgroundTemplate.empty()) {
                // Upload template to GPU
                m_gpuBackgroundFrame.upload(cachedBackgroundTemplate);

                // Resize on GPU
                cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);

                // Download result
                m_gpuSegmentedFrame.download(segmentedFrame);
            } else {
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else {
            // Black background
            segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
        }

        // Process detections with GPU-only silhouette segmentation
        for (int i = 0; i < maxDetections; i++) {
            try {
                // GPU MEMORY PROTECTION: Validate GPU buffer before processing
                if (m_gpuVideoFrame.empty()) {
                    qWarning() << "GPU video frame is empty, skipping detection" << i;
                    continue;
                }
                
                cv::Mat personSegment = enhancedSilhouetteSegmentGPUOnly(m_gpuVideoFrame, detections[i]);
                if (!personSegment.empty()) {
                    // Composite person onto background
                    cv::addWeighted(segmentedFrame, 1.0, personSegment, 1.0, 0.0, segmentedFrame);
                }
            } catch (const cv::Exception &e) {
                qWarning() << "GPU segmentation failed for detection" << i << ":" << e.what();
                // Continue with next detection
            } catch (const std::exception &e) {
                qWarning() << "Exception processing detection" << i << ":" << e.what();
                // Continue with next detection
            }
        }

        // Ensure we always return the video background in segmentation mode
        if (segmentedFrame.empty() && m_useDynamicVideoBackground && !m_dynamicVideoFrame.empty()) {
            qDebug() << "GPU segmented frame is empty, using video frame directly";
            cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        qDebug() << "GPU segmentation complete, returning segmented frame - size:" << segmentedFrame.cols << "x" << segmentedFrame.rows << "empty:" << segmentedFrame.empty();
        return segmentedFrame;

    } else {
        // Rectangle mode - draw rectangles on original frame
        cv::Mat result = frame.clone();
        for (int i = 0; i < maxDetections; i++) {
            cv::rectangle(result, detections[i], cv::Scalar(0, 255, 0), 2);
        }
        return result;
    }
}

cv::Mat Capture::enhancedSilhouetteSegment(const cv::Mat &frame, const cv::Rect &detection)
{
    // Optimized frame skipping for GPU-accelerated segmentation - process every 4th frame
    static int frameCounter = 0;
    static double lastProcessingTime = 0.0;
    frameCounter++;

    // RECORDING: Disable frame skipping during recording for smooth capture
    bool shouldProcess = m_isRecording; // Always process during recording
    
    if (!m_isRecording) {
        // OPTIMIZATION: More aggressive skipping during live preview to maintain video template speed
        shouldProcess = (frameCounter % 5 == 0); // Process every 5th frame by default during preview

        // If processing is taking too long, skip even more frames
        if (lastProcessingTime > 20.0) {
            shouldProcess = (frameCounter % 8 == 0); // Process every 8th frame
        } else if (lastProcessingTime < 10.0) {
            shouldProcess = (frameCounter % 3 == 0); // Process every 3rd frame
        }
    }

    if (!shouldProcess) {
        // Return cached result for skipped frames
        static cv::Mat lastMask;
        if (!lastMask.empty()) {
            return lastMask.clone();
        }
    }

    // Start timing for adaptive processing
    auto startTime = std::chrono::high_resolution_clock::now();

    // qDebug() << "Starting enhanced silhouette segmentation for detection at" << detection.x << detection.y << detection.width << "x" << detection.height;

    // Person-focused silhouette segmentation with enhanced edge detection
    // Validate and clip detection rectangle to frame bounds
    qDebug() << "Frame size:" << frame.cols << "x" << frame.rows;
    qDebug() << "Original detection rectangle:" << detection.x << detection.y << detection.width << "x" << detection.height;

    // Create a clipped version of the detection rectangle
    cv::Rect clippedDetection = detection;

    // Clip to frame bounds
    clippedDetection.x = std::max(0, clippedDetection.x);
    clippedDetection.y = std::max(0, clippedDetection.y);
    clippedDetection.width = std::min(clippedDetection.width, frame.cols - clippedDetection.x);
    clippedDetection.height = std::min(clippedDetection.height, frame.rows - clippedDetection.y);

    qDebug() << "Clipped detection rectangle:" << clippedDetection.x << clippedDetection.y << clippedDetection.width << "x" << clippedDetection.height;

    // Check if the clipped rectangle is still valid
    if (clippedDetection.width <= 0 || clippedDetection.height <= 0) {
        qDebug() << "Clipped detection rectangle is invalid, returning empty mask";
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }

    // Create expanded rectangle for full body coverage
    cv::Rect expandedRect = clippedDetection;
    expandedRect.x = std::max(0, expandedRect.x - 25); // Larger expansion for full body
    expandedRect.y = std::max(0, expandedRect.y - 25);
    expandedRect.width = std::min(frame.cols - expandedRect.x, expandedRect.width + 50); // Larger expansion
    expandedRect.height = std::min(frame.rows - expandedRect.y, expandedRect.height + 50);

    qDebug() << "Expanded rectangle:" << expandedRect.x << expandedRect.y << expandedRect.width << "x" << expandedRect.height;

    // Validate expanded rectangle
    if (expandedRect.width <= 0 || expandedRect.height <= 0) {
        qDebug() << "Invalid expanded rectangle, returning empty mask";
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }

    // Create ROI for silhouette extraction
    cv::Mat roi = frame(expandedRect);
    cv::Mat roiMask = cv::Mat::zeros(roi.size(), CV_8UC1);

    qDebug() << "ROI created, size:" << roi.cols << "x" << roi.rows;

    // GPU-accelerated edge detection for full body segmentation
    cv::Mat edges;

    if (m_useCUDA) {
        try {
            // Upload ROI to GPU
            cv::cuda::GpuMat gpu_roi;
            gpu_roi.upload(roi);

            // CRASH PREVENTION: Validate ROI has 3 channels before BGR2GRAY conversion
            if (roi.empty() || roi.channels() != 3) {
                qWarning() << "Invalid ROI for GPU processing: empty or not 3 channels";
                return roiMask; // Return empty mask
            }

            // Convert to grayscale on GPU
            cv::cuda::GpuMat gpu_gray;
            cv::cuda::cvtColor(gpu_roi, gpu_gray, cv::COLOR_BGR2GRAY);

            // Apply Gaussian blur on GPU using CUDA filters
            cv::cuda::GpuMat gpu_blurred;
            cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(gpu_gray.type(), gpu_blurred.type(), cv::Size(5, 5), 0);
            gaussian_filter->apply(gpu_gray, gpu_blurred);

            // CUDA-accelerated Canny edge detection
            cv::cuda::GpuMat gpu_edges;
            cv::Ptr<cv::cuda::CannyEdgeDetector> canny_detector = cv::cuda::createCannyEdgeDetector(15, 45);
            canny_detector->detect(gpu_blurred, gpu_edges);

            // CUDA-accelerated morphological dilation
            cv::cuda::GpuMat gpu_dilated;
            cv::Ptr<cv::cuda::Filter> dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpu_edges.type(), cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            dilate_filter->apply(gpu_edges, gpu_dilated);

            // Download result back to CPU
            gpu_dilated.download(edges);

            qDebug() << "GPU-accelerated edge detection applied";

        } catch (const cv::Exception& e) {
            qWarning() << "CUDA edge detection failed, falling back to CPU:" << e.what();
            // Fallback to CPU processing
            // CRASH PREVENTION: Validate ROI has 3 channels before BGR2GRAY conversion
            if (roi.empty() || roi.channels() != 3) {
                qWarning() << "Invalid ROI for CPU fallback: empty or not 3 channels";
                return roiMask; // Return empty mask
            }
            cv::Mat gray;
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
            cv::Mat blurred;
            cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
            cv::Canny(blurred, edges, 15, 45);
            cv::Mat kernel_edge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::dilate(edges, edges, kernel_edge);
        }
    } else {
        // CPU fallback
        // CRASH PREVENTION: Validate ROI has 3 channels before BGR2GRAY conversion
        if (roi.empty() || roi.channels() != 3) {
            qWarning() << "Invalid ROI for CPU processing: empty or not 3 channels";
            return edges; // Return empty edges
        }
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edges, 15, 45);
        cv::Mat kernel_edge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(edges, edges, kernel_edge);
    }

    // Find contours from edges
    std::vector<std::vector<cv::Point>> edgeContours;
    cv::findContours(edges, edgeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    qDebug() << "Found" << edgeContours.size() << "edge contours";

    // Filter contours based on person-like characteristics
    std::vector<std::vector<cv::Point>> validContours;
    cv::Point detectionCenter(expandedRect.width/2, expandedRect.height/2);

    // Only process edge contours if they exist
    if (!edgeContours.empty()) {
        qDebug() << "Filtering" << edgeContours.size() << "contours for person-like characteristics";

    for (const auto& contour : edgeContours) {
        double area = cv::contourArea(contour);

            // Optimized size constraints for full body detection
            if (area > 10 && area < expandedRect.width * expandedRect.height * 0.98) {
            // Get bounding rectangle
            cv::Rect contourRect = cv::boundingRect(contour);

                // Check if contour is centered in the detection area (very lenient)
            cv::Point contourCenter(contourRect.x + contourRect.width/2, contourRect.y + contourRect.height/2);
            double distance = cv::norm(contourCenter - detectionCenter);
                double maxDistance = std::min(expandedRect.width, expandedRect.height) * 0.9; // Very lenient distance

                            // Optimized aspect ratio check for full body
            double aspectRatio = (double)contourRect.height / contourRect.width;

            if (distance < maxDistance && aspectRatio > 0.2) { // Allow very wide aspect ratios for full body
                validContours.push_back(contour);
            }
        }
    }

        qDebug() << "After filtering:" << validContours.size() << "valid contours";
    } else {
        qDebug() << "No edge contours found, skipping to background subtraction";
    }

    // If no valid edge contours found, use background subtraction approach
    if (validContours.empty()) {
        qDebug() << "No valid edge contours, trying background subtraction";
        
        cv::Mat fgMask;
        
        // Check if static reference image(s) are available - use them for subtraction
        if (!m_subtractionReferenceImage.empty() || !m_subtractionReferenceImage2.empty()) {
            cv::Mat refResized;
            
            // If both reference images are available, blend them
            if (!m_subtractionReferenceImage.empty() && !m_subtractionReferenceImage2.empty()) {
                cv::Mat ref1, ref2;
                if (m_subtractionReferenceImage.size() != roi.size()) {
                    cv::resize(m_subtractionReferenceImage, ref1, roi.size(), 0, 0, cv::INTER_LINEAR);
                } else {
                    ref1 = m_subtractionReferenceImage;
                }
                if (m_subtractionReferenceImage2.size() != roi.size()) {
                    cv::resize(m_subtractionReferenceImage2, ref2, roi.size(), 0, 0, cv::INTER_LINEAR);
                } else {
                    ref2 = m_subtractionReferenceImage2;
                }
                
                // Blend the two reference images: weight * ref2 + (1-weight) * ref1
                double alpha = m_subtractionBlendWeight;
                double beta = 1.0 - alpha;
                cv::addWeighted(ref1, beta, ref2, alpha, 0.0, refResized);
            } else if (!m_subtractionReferenceImage.empty()) {
                // Use only first reference image
                if (m_subtractionReferenceImage.size() != roi.size()) {
                    cv::resize(m_subtractionReferenceImage, refResized, roi.size(), 0, 0, cv::INTER_LINEAR);
                } else {
                    refResized = m_subtractionReferenceImage;
                }
            } else {
                // Use only second reference image
                if (m_subtractionReferenceImage2.size() != roi.size()) {
                    cv::resize(m_subtractionReferenceImage2, refResized, roi.size(), 0, 0, cv::INTER_LINEAR);
                } else {
                    refResized = m_subtractionReferenceImage2;
                }
            }
            
            if (m_useCUDA) {
                try {
                    // GPU-accelerated absolute difference
                    cv::cuda::GpuMat gpu_roi, gpu_ref, gpu_diff;
                    gpu_roi.upload(roi);
                    gpu_ref.upload(refResized);
                    
                    cv::cuda::absdiff(gpu_roi, gpu_ref, gpu_diff);
                    
                    // Convert to grayscale and threshold
                    cv::cuda::GpuMat gpu_gray;
                    cv::cuda::cvtColor(gpu_diff, gpu_gray, cv::COLOR_BGR2GRAY);
                    
                    cv::cuda::GpuMat gpu_mask;
                    cv::cuda::threshold(gpu_gray, gpu_mask, 30, 255, cv::THRESH_BINARY);
                    
                    gpu_mask.download(fgMask);
                } catch (...) {
                    // Fallback to CPU
                    cv::Mat diff;
                    cv::absdiff(roi, refResized, diff);
                    // CRASH PREVENTION: Validate diff has 3 channels before BGR2GRAY conversion
                    if (diff.empty() || diff.channels() != 3) {
                        qWarning() << "Invalid diff for CPU processing: empty or not 3 channels";
                        fgMask = cv::Mat::zeros(roi.size(), CV_8UC1);
                        return fgMask; // Return empty mask
                    }
                    cv::Mat gray;
                    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
                    cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
                }
            } else {
                // CPU static reference subtraction
                cv::Mat diff;
                cv::absdiff(roi, refResized, diff);
                cv::Mat gray;
                cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
                cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
            }
            qDebug() << "Using static reference image(s) for background subtraction";
        } else {
            // CRASH FIX: Check if background subtractor is initialized
            if (!m_bgSubtractor) {
                qWarning() << "Background subtractor not initialized, cannot perform segmentation";
                // Return empty mask - let caller handle this gracefully
                return cv::Mat::zeros(roi.size(), CV_8UC1);
            }
            // Use MOG2 background subtraction for motion-based segmentation
            m_bgSubtractor->apply(roi, fgMask);
        }

        // GPU-accelerated morphological operations for full body
        if (m_useCUDA) {
            try {
                // Upload mask to GPU
                cv::cuda::GpuMat gpu_fgMask;
                gpu_fgMask.upload(fgMask);

                // Create morphological kernels
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

                // GPU-accelerated morphological operations
                cv::Ptr<cv::cuda::Filter> open_filter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, gpu_fgMask.type(), kernel);
                cv::Ptr<cv::cuda::Filter> close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpu_fgMask.type(), kernel);
                cv::Ptr<cv::cuda::Filter> dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpu_fgMask.type(), kernel_dilate);

                open_filter->apply(gpu_fgMask, gpu_fgMask);
                close_filter->apply(gpu_fgMask, gpu_fgMask);
                dilate_filter->apply(gpu_fgMask, gpu_fgMask);

                // Download result back to CPU
                gpu_fgMask.download(fgMask);

                qDebug() << "GPU-accelerated morphological operations applied";

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA morphological operations failed, falling back to CPU:" << e.what();
                // Fallback to CPU processing
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                cv::dilate(fgMask, fgMask, kernel_dilate);
            }
        } else {
            // CPU fallback
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
            cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::dilate(fgMask, fgMask, kernel_dilate);
        }

        // Find contours from background subtraction
        cv::findContours(fgMask, validContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        qDebug() << "Background subtraction found" << validContours.size() << "contours";
    }

    // If still no valid contours, try color-based segmentation
    if (validContours.empty()) {
        qDebug() << "No contours from background subtraction, trying color-based segmentation";

        // GPU-accelerated color space conversion and thresholding
        cv::Mat combinedMask;

        if (m_useCUDA) {
            try {
                // Upload ROI to GPU
                cv::cuda::GpuMat gpu_roi;
                gpu_roi.upload(roi);

                // Convert to HSV on GPU
                cv::cuda::GpuMat gpu_hsv;
                cv::cuda::cvtColor(gpu_roi, gpu_hsv, cv::COLOR_BGR2HSV);

                // Create masks for skin-like colors and non-background colors on GPU
                cv::cuda::GpuMat gpu_skinMask, gpu_colorMask;
                // Widened skin range and relaxed saturation/value to better capture varied tones/lighting
                cv::cuda::inRange(gpu_hsv, cv::Scalar(0, 10, 40), cv::Scalar(25, 255, 255), gpu_skinMask);
                // Broader general color mask with relaxed S/V to include darker/low-saturation clothing
                cv::cuda::inRange(gpu_hsv, cv::Scalar(0, 15, 35), cv::Scalar(180, 255, 255), gpu_colorMask);

                // Combine masks on GPU using bitwise_or
                cv::cuda::GpuMat gpu_combinedMask;
                cv::cuda::bitwise_or(gpu_skinMask, gpu_colorMask, gpu_combinedMask);

                // Download result back to CPU
                gpu_combinedMask.download(combinedMask);

                qDebug() << "GPU-accelerated color segmentation applied";

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA color segmentation failed, falling back to CPU:" << e.what();
                // Fallback to CPU processing
                cv::Mat hsv;
                cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
                cv::Mat skinMask;
                cv::inRange(hsv, cv::Scalar(0, 10, 40), cv::Scalar(25, 255, 255), skinMask);
                cv::Mat colorMask;
                cv::inRange(hsv, cv::Scalar(0, 15, 35), cv::Scalar(180, 255, 255), colorMask);
                cv::bitwise_or(skinMask, colorMask, combinedMask);
            }
        } else {
            // CPU fallback
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
            cv::Mat skinMask;
            cv::inRange(hsv, cv::Scalar(0, 10, 40), cv::Scalar(25, 255, 255), skinMask);
            cv::Mat colorMask;
            cv::inRange(hsv, cv::Scalar(0, 15, 35), cv::Scalar(180, 255, 255), colorMask);
            cv::bitwise_or(skinMask, colorMask, combinedMask);
        }

        // GPU-accelerated morphological operations for color segmentation
        if (m_useCUDA) {
            try {
                // Upload mask to GPU
                cv::cuda::GpuMat gpu_combinedMask;
                gpu_combinedMask.upload(combinedMask);

                // Create morphological kernel
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

                // GPU-accelerated morphological operations
                cv::Ptr<cv::cuda::Filter> open_filter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, gpu_combinedMask.type(), kernel);
                cv::Ptr<cv::cuda::Filter> close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpu_combinedMask.type(), kernel);

                open_filter->apply(gpu_combinedMask, gpu_combinedMask);
                close_filter->apply(gpu_combinedMask, gpu_combinedMask);

                // Download result back to CPU
                gpu_combinedMask.download(combinedMask);

                qDebug() << "GPU-accelerated color morphological operations applied";

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA color morphological operations failed, falling back to CPU:" << e.what();
                // Fallback to CPU processing
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, kernel);
            }
        } else {
            // CPU fallback
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, kernel);
        }

        // Find contours from color segmentation
        cv::findContours(combinedMask, validContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        qDebug() << "Color-based segmentation found" << validContours.size() << "contours";
    }

    // Create mask from valid contours
    if (!validContours.empty()) {
        qDebug() << "Creating mask from" << validContours.size() << "valid contours";
        // Sort contours by area
        std::sort(validContours.begin(), validContours.end(),
                 [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                     return cv::contourArea(a) > cv::contourArea(b);
                 });

        // Enhanced contour usage for full body coverage
        int maxContours = std::min(4, (int)validContours.size()); // Use up to 4 largest contours for full body
        for (int i = 0; i < maxContours; i++) {
            cv::drawContours(roiMask, validContours, i, cv::Scalar(255), -1);
        }

        // Fill holes in the silhouette
        cv::Mat filledMask = roiMask.clone();
        cv::floodFill(filledMask, cv::Point(0, 0), cv::Scalar(128));
        cv::floodFill(filledMask, cv::Point(filledMask.cols-1, 0), cv::Scalar(128));
        cv::floodFill(filledMask, cv::Point(0, filledMask.rows-1), cv::Scalar(128));
        cv::floodFill(filledMask, cv::Point(filledMask.cols-1, filledMask.rows-1), cv::Scalar(128));

        // Create final mask
        for (int y = 0; y < filledMask.rows; y++) {
            for (int x = 0; x < filledMask.cols; x++) {
                if (filledMask.at<uchar>(y, x) != 128) {
                    roiMask.at<uchar>(y, x) = 255;
                } else {
                    roiMask.at<uchar>(y, x) = 0;
                }
            }
        }

        // GPU-accelerated final morphological cleanup for full body
        if (m_useCUDA) {
            try {
                // Upload mask to GPU
                cv::cuda::GpuMat gpu_roiMask;
                gpu_roiMask.upload(roiMask);

                // Create morphological kernels
                cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

                // GPU-accelerated morphological operations
                cv::Ptr<cv::cuda::Filter> close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpu_roiMask.type(), kernel_clean);
                cv::Ptr<cv::cuda::Filter> dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpu_roiMask.type(), kernel_dilate);

                close_filter->apply(gpu_roiMask, gpu_roiMask);
                dilate_filter->apply(gpu_roiMask, gpu_roiMask);

                // Download result back to CPU
                gpu_roiMask.download(roiMask);

                qDebug() << "GPU-accelerated final morphological cleanup applied";

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA final morphological cleanup failed, falling back to CPU:" << e.what();
                // Fallback to CPU processing
                cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(roiMask, roiMask, cv::MORPH_CLOSE, kernel_clean);
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                cv::dilate(roiMask, roiMask, kernel_dilate);
            }
        } else {
            // CPU fallback
            cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            cv::morphologyEx(roiMask, roiMask, cv::MORPH_CLOSE, kernel_clean);
            cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::dilate(roiMask, roiMask, kernel_dilate);
        }
    } else {
        qDebug() << "No valid contours found, creating empty mask";
    }

    // Create final mask for the entire frame
    cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    roiMask.copyTo(finalMask(expandedRect));

    int finalNonZeroPixels = cv::countNonZero(finalMask);
    qDebug() << "Enhanced silhouette segmentation complete, final mask has" << finalNonZeroPixels << "non-zero pixels";

    // Cache the result for frame skipping
    static cv::Mat lastMask;
    lastMask = finalMask.clone();

    // End timing and update adaptive processing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    lastProcessingTime = duration.count() / 1000.0; // Convert to milliseconds

    return finalMask;
}

// Phase 2A: GPU-Only Silhouette Segmentation
cv::Mat Capture::enhancedSilhouetteSegmentGPUOnly(const cv::cuda::GpuMat &gpuFrame, const cv::Rect &detection)
{
    if (gpuFrame.empty()) {
        return cv::Mat();
    }

    qDebug() << "Phase 2A: GPU-only silhouette segmentation";

    // Validate and clip detection rectangle to frame bounds
    cv::Rect clippedDetection = detection;
    clippedDetection.x = std::max(0, clippedDetection.x);
    clippedDetection.y = std::max(0, clippedDetection.y);
    clippedDetection.width = std::min(clippedDetection.width, gpuFrame.cols - clippedDetection.x);
    clippedDetection.height = std::min(clippedDetection.height, gpuFrame.rows - clippedDetection.y);

    if (clippedDetection.width <= 0 || clippedDetection.height <= 0) {
        return cv::Mat::zeros(gpuFrame.size(), CV_8UC1);
    }

    // Create expanded rectangle for full body coverage
    cv::Rect expandedRect = clippedDetection;
    expandedRect.x = std::max(0, expandedRect.x - 25);
    expandedRect.y = std::max(0, expandedRect.y - 25);
    expandedRect.width = std::min(gpuFrame.cols - expandedRect.x, expandedRect.width + 50);
    expandedRect.height = std::min(gpuFrame.rows - expandedRect.y, expandedRect.height + 50);

    if (expandedRect.width <= 0 || expandedRect.height <= 0) {
        return cv::Mat::zeros(gpuFrame.size(), CV_8UC1);
    }

    //  GPU MEMORY POOL OPTIMIZED PIPELINE - REUSABLE BUFFERS + ASYNC STREAMS

    // Check if GPU Memory Pool is available
    if (!m_gpuMemoryPoolInitialized || !m_gpuMemoryPool.isInitialized()) {
        qWarning() << " GPU Memory Pool not available, falling back to standard GPU processing";
        // Fallback to standard GPU processing (existing code)
    cv::cuda::GpuMat gpuRoi = gpuFrame(expandedRect);
        cv::cuda::GpuMat gpuRoiMask(gpuRoi.size(), CV_8UC1, cv::Scalar(0));

        // CRASH PREVENTION: Validate gpuRoi has 3 channels before BGR2GRAY conversion
        if (gpuRoi.empty() || gpuRoi.channels() != 3) {
            qWarning() << "Invalid gpuRoi for GPU processing: empty or not 3 channels";
            return cv::Mat::zeros(gpuFrame.size(), CV_8UC1); // Return empty mask
        }
        
        // Use standard GPU processing without memory pool
    cv::cuda::GpuMat gpuGray, gpuEdges;
    cv::cuda::cvtColor(gpuRoi, gpuGray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::cuda::CannyEdgeDetector> canny_detector = cv::cuda::createCannyEdgeDetector(50, 150);
        canny_detector->detect(gpuGray, gpuEdges);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Ptr<cv::cuda::Filter> close_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpuEdges.type(), kernel);
        cv::Ptr<cv::cuda::Filter> open_filter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, gpuEdges.type(), kernel);
        cv::Ptr<cv::cuda::Filter> dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpuEdges.type(), kernel);

        close_filter->apply(gpuEdges, gpuRoiMask);
        open_filter->apply(gpuRoiMask, gpuRoiMask);
        dilate_filter->apply(gpuRoiMask, gpuRoiMask);

        cv::cuda::GpuMat gpuConnectedMask;
        cv::cuda::threshold(gpuRoiMask, gpuConnectedMask, 127, 255, cv::THRESH_BINARY);
        close_filter->apply(gpuConnectedMask, gpuConnectedMask);

        cv::Mat kernel_final = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Ptr<cv::cuda::Filter> final_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpuConnectedMask.type(), kernel_final);
        final_filter->apply(gpuConnectedMask, gpuConnectedMask);

        cv::Mat finalMask;
        gpuConnectedMask.download(finalMask);

        cv::cuda::GpuMat gpuFullMask(gpuFrame.size(), CV_8UC1, cv::Scalar(0));
    cv::cuda::GpuMat gpuFinalMask;
    gpuFinalMask.upload(finalMask);
        gpuFinalMask.copyTo(gpuFullMask(expandedRect));

        cv::Mat fullMask;
        gpuFullMask.download(fullMask);

        qDebug() << " Phase 2A: Standard GPU processing completed (memory pool not available)";
        return fullMask;
    }

    // Extract ROI on GPU using memory pool
    cv::cuda::GpuMat gpuRoi = gpuFrame(expandedRect);
    cv::cuda::GpuMat& gpuRoiMask = m_gpuMemoryPool.getNextSegmentationBuffer();
    gpuRoiMask.create(gpuRoi.size(), CV_8UC1);
    gpuRoiMask.setTo(cv::Scalar(0));

    // Get CUDA streams for parallel processing
    cv::cuda::Stream& detectionStream = m_gpuMemoryPool.getDetectionStream();
    cv::cuda::Stream& segmentationStream = m_gpuMemoryPool.getSegmentationStream();

    // CRASH PREVENTION: Validate gpuRoi has 3 channels before BGR2GRAY conversion
    if (gpuRoi.empty() || gpuRoi.channels() != 3) {
        qWarning() << "Invalid gpuRoi for GPU memory pool processing: empty or not 3 channels";
        return cv::Mat::zeros(gpuFrame.size(), CV_8UC1); // Return empty mask
    }
    
    // Step 1: GPU Color Conversion (async)
    cv::cuda::GpuMat& gpuGray = m_gpuMemoryPool.getNextTempBuffer();
    cv::cuda::GpuMat& gpuEdges = m_gpuMemoryPool.getNextDetectionBuffer();
    cv::cuda::cvtColor(gpuRoi, gpuGray, cv::COLOR_BGR2GRAY, 0, detectionStream);

    // Step 2: GPU Canny Edge Detection (async) - using pre-created detector
    cv::Ptr<cv::cuda::CannyEdgeDetector>& canny_detector = m_gpuMemoryPool.getCannyDetector();
    canny_detector->detect(gpuGray, gpuEdges, detectionStream);

    // Step 3: GPU Morphological Operations (async) - using pre-created filters
    cv::Ptr<cv::cuda::Filter>& close_filter = m_gpuMemoryPool.getMorphCloseFilter();
    cv::Ptr<cv::cuda::Filter>& open_filter = m_gpuMemoryPool.getMorphOpenFilter();
    cv::Ptr<cv::cuda::Filter>& dilate_filter = m_gpuMemoryPool.getMorphDilateFilter();

    // Apply GPU morphological pipeline (async)
    close_filter->apply(gpuEdges, gpuRoiMask, detectionStream);      // Close gaps
    open_filter->apply(gpuRoiMask, gpuRoiMask, detectionStream);     // Remove noise
    dilate_filter->apply(gpuRoiMask, gpuRoiMask, detectionStream);   // Expand regions

    // Step 4: GPU-accelerated area-based filtering (async)
    cv::cuda::GpuMat& gpuConnectedMask = m_gpuMemoryPool.getNextSegmentationBuffer();

    // Create a mask for large connected regions (person-like areas) - async
    cv::cuda::threshold(gpuRoiMask, gpuConnectedMask, 127, 255, cv::THRESH_BINARY, segmentationStream);

    // Apply additional GPU morphological cleanup (async)
    close_filter->apply(gpuConnectedMask, gpuConnectedMask, segmentationStream);

    // Step 5: Final GPU morphological cleanup (async)
    cv::Mat kernel_final = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Ptr<cv::cuda::Filter> final_filter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpuConnectedMask.type(), kernel_final);
    final_filter->apply(gpuConnectedMask, gpuConnectedMask, segmentationStream);

    //  SYNCHRONIZE STREAMS BEFORE DOWNLOAD
    detectionStream.waitForCompletion();
    segmentationStream.waitForCompletion();

    // Step 6: Single download at the end (minimize GPU-CPU transfers)
    cv::Mat finalMask;
    gpuConnectedMask.download(finalMask);

    //  GPU-OPTIMIZED: Create full-size mask directly on GPU
    cv::cuda::GpuMat& gpuFullMask = m_gpuMemoryPool.getNextFrameBuffer();
    gpuFullMask.create(gpuFrame.size(), CV_8UC1);
    gpuFullMask.setTo(cv::Scalar(0));

    // Copy the processed ROI back to the full-size mask on GPU (async)
    cv::cuda::GpuMat gpuFinalMask;
    gpuFinalMask.upload(finalMask, m_gpuMemoryPool.getCompositionStream());
    gpuFinalMask.copyTo(gpuFullMask(expandedRect), m_gpuMemoryPool.getCompositionStream());

    // Synchronize composition stream and download
    m_gpuMemoryPool.getCompositionStream().waitForCompletion();

    // Single download at the very end
    cv::Mat fullMask;
    gpuFullMask.download(fullMask);

    qDebug() << " Phase 2A: GPU MEMORY POOL + ASYNC STREAMS silhouette segmentation completed";

    return fullMask;
}

