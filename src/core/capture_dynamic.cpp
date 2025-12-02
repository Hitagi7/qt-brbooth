//  DYNAMIC VIDEO PROCESSING
//  This file contains all dynamic video capture and post-processing functions
//  Separated from static image processing for easier code review and maintenance

#include "core/capture.h"
#include "core/camera.h"  // For cvMatToQImage helper function
#include "ui/foreground.h"
#include "ui_capture.h"
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include <QEasingCurve>
#include <vector>
#include <QFont>
#include <QResizeEvent>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QPainter>
#include <QKeyEvent>
#include <QApplication>
#include <QPushButton>
#include <QCoreApplication>
#include <QDir>
#include <QMessageBox>
#include <QDateTime>
#include <QStackedLayout>
#include <QThread>
#include <QFileInfo>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <QtConcurrent/QtConcurrent>
#include <QThreadPool>
#include <QMutexLocker>
#include <chrono>
#include <QFutureWatcher>
#include "algorithms/lighting_correction/lighting_corrector.h"
#include "core/system_monitor.h"

// Forward declarations for helper functions defined in capture.cpp
// These are used by both static and dynamic processing
extern cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                                 GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

// ============================================================================
//  DYNAMIC VIDEO POST-PROCESSING
// ============================================================================

QList<QPixmap> Capture::processRecordedVideoWithLighting(const QList<QPixmap> &inputFrames, double fps)
{
    Q_UNUSED(fps); // FPS parameter kept for future use but not currently needed
    
    //  DYNAMIC VIDEO POST-PROCESSING: Fast path for video frames
    //  NOTE: Static mode uses applyPostProcessingLighting() which has full edge blending
    //  This function uses simplified processing for speed (dynamic videos have many frames)
    const int total = inputFrames.size();
    qDebug() << "Starting fast post-processing for dynamic video:" << total << "frames";
    
    //  CRASH PREVENTION: Check if lighting corrector is properly initialized
    bool lightingAvailable = (m_lightingCorrector != nullptr);
    qDebug() << "Lighting corrector available:" << lightingAvailable;
    
    //  CRASH PREVENTION: If no lighting available, return frames as-is
    if (!lightingAvailable) {
        qDebug() << "No lighting correction available, returning original frames";
        return inputFrames;
    }
    
    //  OPTIMIZATION: Use parallel processing for faster frame processing with GPU acceleration
    //  Process frames in parallel using QtConcurrent::mapped for better performance
    qDebug() << "OPTIMIZATION: Using parallel frame processing with GPU acceleration for faster post-processing";
    
    // Create a list of frame indices for parallel processing
    QList<int> frameIndices;
    for (int i = 0; i < total; ++i) {
        frameIndices.append(i);
    }
    
    // OPTIMIZATION: Use more threads if CUDA is available (GPU can handle parallel operations)
    // With CUDA, we can process more frames in parallel since GPU operations are async
    int optimalThreads = QThread::idealThreadCount();
    if (m_useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        // With GPU, we can use more threads (GPU handles parallel operations efficiently)
        optimalThreads = qMin(optimalThreads, 8); // Allow up to 8 threads with GPU
        qDebug() << "GPU-ACCELERATED: Using" << optimalThreads << "threads for parallel GPU processing";
        // Set thread pool for QtConcurrent to use more threads with GPU
        QThreadPool::globalInstance()->setMaxThreadCount(optimalThreads);
    } else {
        // Without GPU, cap at 4 threads to avoid CPU overload
        if (optimalThreads > 4) optimalThreads = 4;
        qDebug() << "CPU-ONLY: Using" << optimalThreads << "threads for parallel processing";
        QThreadPool::globalInstance()->setMaxThreadCount(optimalThreads);
    }
    
    // THREAD SAFETY: Make local copies of data needed for processing to avoid accessing member variables from multiple threads
    QList<cv::Mat> localPersonRegions = m_recordedRawPersonRegions;
    QList<cv::Mat> localPersonMasks = m_recordedRawPersonMasks;
    QList<cv::Mat> localBackgroundFrames = m_recordedBackgroundFrames;
    
    // CRITICAL: Capture all member variables needed for processing as local copies to ensure thread safety
    // These are accessed from multiple threads during parallel processing
    LightingCorrector* localLightingCorrector = m_lightingCorrector; // Pointer is safe to copy
    double localPersonScaleFactor = m_recordedPersonScaleFactor;
    cv::Mat localLastTemplateBackground = m_lastTemplateBackground.clone(); // Clone to avoid shared access
    bool localUseCUDA = m_useCUDA;
    GPUMemoryPool* localGpuMemoryPool = &m_gpuMemoryPool; // Pointer is safe to copy
    
    QAtomicInt processedCount(0);
    
    // Progress tracking: Emit initial progress
    QMetaObject::invokeMethod(this, "videoProcessingProgress", Qt::QueuedConnection, Q_ARG(int, 0));
    
    // OPTIMIZED: Process frames in parallel with thread-safe data access
    // Helper function to process a single frame - uses only local copies, captures this only for member function calls
    auto processFrame = [this, inputFrames, localPersonRegions, localPersonMasks, localBackgroundFrames, 
                         localLightingCorrector, localPersonScaleFactor, localLastTemplateBackground,
                         localUseCUDA, localGpuMemoryPool,
                         &processedCount, total](int i) -> QPixmap {
            try {
                //  CRASH PREVENTION: Validate frame before processing
                if (i >= inputFrames.size() || inputFrames.at(i).isNull()) {
                    return QPixmap(640, 480);
                }

                // Get current frame
                QPixmap currentFrame = inputFrames.at(i);
                
                // Convert to cv::Mat for processing
                QImage frameImage = currentFrame.toImage().convertToFormat(QImage::Format_BGR888);
                if (frameImage.isNull()) {
                    return currentFrame;
                }

                cv::Mat composedFrame(frameImage.height(), frameImage.width(), CV_8UC3,
                                      const_cast<uchar*>(frameImage.bits()), frameImage.bytesPerLine());
                
                if (composedFrame.empty()) {
                    return currentFrame;
                }

                cv::Mat composedCopy = composedFrame.clone();
                cv::Mat finalFrame;

                // DYNAMIC VIDEO PROCESSING: Apply full lighting correction and edge blending
                // Uses the same quality algorithms as static mode but optimized for video processing
                bool hasRawPersonData = (i < localPersonRegions.size() && 
                                         i < localPersonMasks.size() &&
                                         !localPersonRegions[i].empty() &&
                                         !localPersonMasks[i].empty());

                if (hasRawPersonData) {
                    // FULL EDGE BLENDING AND LIGHTING: Apply complete post-processing for dynamic videos
                    // Uses applyDynamicFrameEdgeBlending which includes:
                    // - Full lighting correction via applyLightingToRawPersonRegion
                    // - Guided filter edge blending for smooth transitions
                    // - Edge blurring for seamless compositing
                    try {
                        cv::Mat bgFrame = (i < localBackgroundFrames.size() && !localBackgroundFrames[i].empty()) 
                                         ? localBackgroundFrames[i] : cv::Mat();
                        
                        if (!bgFrame.empty() && bgFrame.size() != composedFrame.size()) {
                            cv::resize(bgFrame, bgFrame, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
                        }
                        
                        if (bgFrame.empty()) {
                            bgFrame = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
                        }
                        
                        // Apply full edge blending and lighting correction
                        // CRASH PREVENTION: Pass all necessary data as parameters to avoid member variable access
                        finalFrame = applyDynamicFrameEdgeBlendingSafe(composedCopy,
                                                                      localPersonRegions[i],
                                                                      localPersonMasks[i],
                                                                      bgFrame,
                                                                      localLightingCorrector,
                                                                      localPersonScaleFactor,
                                                                      localLastTemplateBackground,
                                                                      localUseCUDA,
                                                                      localGpuMemoryPool);
                        
                        if (finalFrame.empty()) {
                            qWarning() << "Edge blending returned empty frame for frame" << i << "- using simple compositing fallback";
                            // Fallback to simple compositing if edge blending fails
                            finalFrame = applySimpleDynamicCompositingSafe(composedCopy,
                                                                           localPersonRegions[i],
                                                                           localPersonMasks[i],
                                                                           bgFrame,
                                                                           localLightingCorrector,
                                                                           localPersonScaleFactor,
                                                                           localUseCUDA);
                            if (finalFrame.empty()) {
                                finalFrame = composedCopy;
                            }
                        }
                    } catch (const std::exception& e) {
                        qWarning() << "Edge blending failed for frame" << i << ":" << e.what() << "- using simple compositing fallback";
                        // Fallback to simple compositing on error
                        try {
                            cv::Mat bgFrame = (i < localBackgroundFrames.size() && !localBackgroundFrames[i].empty()) 
                                             ? localBackgroundFrames[i] : cv::Mat();
                            if (!bgFrame.empty() && bgFrame.size() != composedFrame.size()) {
                                cv::resize(bgFrame, bgFrame, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
                            }
                            if (bgFrame.empty()) {
                                bgFrame = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
                            }
                            finalFrame = applySimpleDynamicCompositingSafe(composedCopy,
                                                                           localPersonRegions[i],
                                                                           localPersonMasks[i],
                                                                           bgFrame,
                                                                           localLightingCorrector,
                                                                           localPersonScaleFactor,
                                                                           localUseCUDA);
                            if (finalFrame.empty()) {
                                finalFrame = composedCopy;
                            }
                        } catch (const std::exception& e2) {
                            qWarning() << "Simple compositing fallback also failed:" << e2.what();
                            finalFrame = composedCopy;
                        }
                    }
                } else {
                    // No raw person data - return as-is
                    finalFrame = composedCopy;
                }

                // CRITICAL: Ensure output frame size matches input frame size (no scaling down)
                if (finalFrame.size() != composedFrame.size()) {
                    cv::resize(finalFrame, finalFrame, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
                }
                
                // Convert back to QPixmap
                QImage outImage = cvMatToQImage(finalFrame);
                if (outImage.isNull()) {
                    return currentFrame;
                }
                
                // Ensure QPixmap size matches original (exact size match, no aspect ratio preservation)
                QPixmap outputPixmap = QPixmap::fromImage(outImage);
                if (outputPixmap.size() != currentFrame.size()) {
                    outputPixmap = outputPixmap.scaled(currentFrame.size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
                }
                
                // Update progress counter (thread-safe atomic operation)
                int current = processedCount.fetchAndAddAcquire(1) + 1;
                
                // Emit progress update every 5 frames or at milestones (thread-safe via queued connection)
                if (current % 5 == 0 || current == total) {
                    int progress = total > 0 ? (current * 100) / total : 0;
                    progress = qMin(progress, 99); // Cap at 99% until all frames are done
                    QMetaObject::invokeMethod(this, "videoProcessingProgress", Qt::QueuedConnection, Q_ARG(int, progress));
                }
                
                return outputPixmap;

            } catch (const std::exception& e) {
                qWarning() << "Exception processing frame" << i << ":" << e.what();
                if (i < inputFrames.size()) {
                    return inputFrames.at(i);
                } else {
                    return QPixmap(640, 480);
                }
            }
    };
    
    // Use blockingMapped for synchronous parallel processing with proper type inference
    qDebug() << "DYNAMIC VIDEO: Starting parallel frame processing with" << total << "frames";
    QList<QPixmap> outputFrames = QtConcurrent::blockingMapped(frameIndices, processFrame);
    
    // Update progress to 100% after processing completes
    QMetaObject::invokeMethod(this, "videoProcessingProgress", Qt::QueuedConnection, Q_ARG(int, 100));
    qDebug() << "DYNAMIC VIDEO: Processing complete - processed" << outputFrames.size() << "frames";
    
    // Validate output
    if (outputFrames.size() != total) {
        qWarning() << "Output frame count mismatch:" << outputFrames.size() << "vs" << total;
        // Fill missing frames with originals
        while (outputFrames.size() < total) {
            int idx = outputFrames.size();
            if (idx < inputFrames.size()) {
                outputFrames.append(inputFrames[idx]);
            } else {
                outputFrames.append(QPixmap(640, 480));
            }
        }
    }

    // Ensure 100% at end
    emit videoProcessingProgress(100);

    // Clear per-frame buffers for next recording (safely)
    m_recordedRawPersonRegions.clear();
    m_recordedRawPersonMasks.clear();
    m_recordedBackgroundFrames.clear();

    qDebug() << "Enhanced post-processing with edge blending completed for" << total << "frames - output:" << outputFrames.size() << "frames";
    return outputFrames;
}

// ============================================================================
//  THREAD-SAFE HELPER FUNCTIONS
// ============================================================================

// Thread-safe wrapper for applyLightingToRawPersonRegion
// This function accesses member variables but is called from main thread context via lambda
// The lambda captures 'this' pointer, so we need to ensure thread safety
static cv::Mat applyLightingToRawPersonRegionSafe(const cv::Mat &personRegion, 
                                                   const cv::Mat &personMask,
                                                   LightingCorrector* lightingCorrector)
{
    // CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs for lighting correction - returning empty mat";
        return cv::Mat();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch between person region and mask - returning original";
        return personRegion.clone();
    }
    
    if (!lightingCorrector) {
        qWarning() << "No lighting corrector provided - returning original";
        return personRegion.clone();
    }
    
    try {
        // Apply global lighting correction as a simplified approach
        // This avoids accessing member variables that might not be thread-safe
        return lightingCorrector->applyGlobalLightingCorrection(personRegion);
    } catch (const std::exception& e) {
        qWarning() << "Lighting correction failed:" << e.what() << "- returning original";
        return personRegion.clone();
    }
}

// ============================================================================
//  DYNAMIC VIDEO EDGE BLENDING FUNCTIONS
// ============================================================================

// THREAD-SAFE WRAPPER: Takes all parameters instead of accessing member variables
cv::Mat Capture::applyDynamicFrameEdgeBlendingSafe(const cv::Mat &composedFrame,
                                                   const cv::Mat &rawPersonRegion,
                                                   const cv::Mat &rawPersonMask,
                                                   const cv::Mat &backgroundFrame,
                                                   LightingCorrector* lightingCorrector,
                                                   double personScaleFactor,
                                                   const cv::Mat &lastTemplateBackground,
                                                   bool useCUDA,
                                                   GPUMemoryPool* gpuMemoryPool)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        qWarning() << "Invalid input data for edge blending, using global correction";
        if (lightingCorrector) {
            return lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    }
    
    try {
        // Start with clean background or use provided background frame
        cv::Mat result;
        cv::Mat cleanBackground;
        
        if (!backgroundFrame.empty()) {
            cv::resize(backgroundFrame, cleanBackground, composedFrame.size());
        } else {
            // Extract background from dynamic template or use clean template
            if (!lastTemplateBackground.empty()) {
                cv::resize(lastTemplateBackground, cleanBackground, composedFrame.size());
            } else {
                // Fallback to zero background
                cleanBackground = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
            }
        }
        result = cleanBackground.clone();
        
        // VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        // Follows the same algorithm as static mode but with video-optimized parameters
        cv::Mat lightingCorrectedPerson = applyVideoOptimizedLighting(rawPersonRegion, rawPersonMask, lightingCorrector);
        
        // SCALING PRESERVATION: Scale the lighting-corrected person using the recorded scaling factor
        cv::Mat scaledPerson, scaledMask;
        
        // Calculate the scaled size using the recorded scaling factor
        cv::Size backgroundSize = result.size();
        cv::Size scaledPersonSize;
        
        if (qAbs(personScaleFactor - 1.0) > 0.01) {
            int scaledWidth = static_cast<int>(backgroundSize.width * personScaleFactor + 0.5);
            int scaledHeight = static_cast<int>(backgroundSize.height * personScaleFactor + 0.5);
            
            //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
            scaledWidth = qMax(1, scaledWidth);
            scaledHeight = qMax(1, scaledHeight);
            
            scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
        } else {
            scaledPersonSize = backgroundSize;
        }
        
        // Scale person and mask to the calculated size
        cv::resize(lightingCorrectedPerson, scaledPerson, scaledPersonSize);
        cv::resize(rawPersonMask, scaledMask, scaledPersonSize);
        
        // Calculate centered offset for placing the scaled person
        cv::Size actualScaledSize(scaledPerson.cols, scaledPerson.rows);
        int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
        int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;
        
        // If person is scaled down, place it on a full-size canvas at the centered position
        cv::Mat fullSizePerson, fullSizeMask;
        if (actualScaledSize != backgroundSize) {
            // Create full-size images initialized to zeros
            fullSizePerson = cv::Mat::zeros(backgroundSize, scaledPerson.type());
            fullSizeMask = cv::Mat::zeros(backgroundSize, CV_8UC1);
            
            // Ensure offsets are valid
            if (xOffset >= 0 && yOffset >= 0 &&
                xOffset + actualScaledSize.width <= backgroundSize.width &&
                yOffset + actualScaledSize.height <= backgroundSize.height) {
                
                // Place scaled person at centered position
                cv::Rect roi(xOffset, yOffset, actualScaledSize.width, actualScaledSize.height);
                scaledPerson.copyTo(fullSizePerson(roi));
                
                // Convert mask to grayscale if needed, then copy to ROI
                if (scaledMask.type() != CV_8UC1) {
                    cv::Mat grayMask;
                    cv::cvtColor(scaledMask, grayMask, cv::COLOR_BGR2GRAY);
                    grayMask.copyTo(fullSizeMask(roi));
                } else {
                    scaledMask.copyTo(fullSizeMask(roi));
                }
            } else {
                qWarning() << "Invalid offset, using direct copy";
                cv::resize(scaledPerson, fullSizePerson, backgroundSize);
                cv::resize(scaledMask, fullSizeMask, backgroundSize);
            }
        } else {
            // Person is full size, use as is
            fullSizePerson = scaledPerson;
            if (scaledMask.type() != CV_8UC1) {
                cv::cvtColor(scaledMask, fullSizeMask, cv::COLOR_BGR2GRAY);
            } else {
                fullSizeMask = scaledMask;
            }
        }
        
        // Now use fullSizePerson and fullSizeMask for blending
        scaledPerson = fullSizePerson;
        scaledMask = fullSizeMask;
        
        // Apply guided filter edge blending
        cv::Mat binMask;
        cv::threshold(scaledMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // First: shrink mask slightly to avoid fringe, then hard-copy interior
        cv::Mat interiorMask;
        cv::erode(binMask, interiorMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1)));
        scaledPerson.copyTo(result, interiorMask);

        //  ENHANCED: Pre-smooth the mask before guided filtering for better edge quality
        cv::Mat smoothedBinMask;
        cv::GaussianBlur(binMask, smoothedBinMask, cv::Size(9, 9), 2.0); // Smooth mask edges first
        
        //  CUDA-Accelerated Guided image filtering
        const int gfRadius = 12; // Increased window size for smoother edges
        const float gfEps = 5e-3f; // Reduced regularization for better edge preservation
        
        // CRASH PREVENTION: Validate GPU memory pool before use
        cv::Mat alphaFloat;
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& guidedFilterStream = gpuMemoryPool->getCompositionStream();
                alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, smoothedBinMask, gfRadius, gfEps, *gpuMemoryPool, guidedFilterStream);
            } catch (const cv::Exception& e) {
                qWarning() << "GPU guided filter failed:" << e.what() << "- using CPU fallback";
                // CPU fallback would go here if needed
                alphaFloat = cv::Mat::ones(result.size(), CV_32F);
            }
        } else {
            // CPU fallback
            alphaFloat = cv::Mat::ones(result.size(), CV_32F);
        }
        
        //  ENHANCED: Apply edge blurring
        const float edgeBlurRadius = 5.0f; // Increased blur radius for smoother transitions
        cv::Mat edgeBlurredPerson;
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& guidedFilterStream = gpuMemoryPool->getCompositionStream();
                edgeBlurredPerson = applyEdgeBlurringCUDA(scaledPerson, binMask, cleanBackground, edgeBlurRadius, *gpuMemoryPool, guidedFilterStream);
                if (!edgeBlurredPerson.empty()) {
                    scaledPerson = edgeBlurredPerson;
                }
            } catch (const cv::Exception& e) {
                qWarning() << "GPU edge blurring failed:" << e.what();
            }
        }
        
        if (edgeBlurredPerson.empty()) {
            edgeBlurredPerson = applyEdgeBlurringAlternative(scaledPerson, binMask, edgeBlurRadius);
            if (!edgeBlurredPerson.empty()) {
                scaledPerson = edgeBlurredPerson;
            }
        }
        
        // Build thin inner/outer rings around the boundary
        cv::Mat inner, outer, ringInner, ringOuter;
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // Wider transition
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*6+1, 2*6+1))); // Wider feather
        cv::subtract(binMask, inner, ringInner);
        cv::subtract(outer, binMask, ringOuter);
        
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);
        alphaFloat.setTo(0.0f, outer == 0);
        alphaFloat = alphaFloat * 0.6f; // Less aggressive reduction for smoother blending

        // Composite
        cv::Mat personF, bgF; 
        scaledPerson.convertTo(personF, CV_32F); 
        cleanBackground.convertTo(bgF, CV_32F);
        std::vector<cv::Mat> a3 = {alphaFloat, alphaFloat, alphaFloat};
        cv::Mat alpha3; 
        cv::merge(a3, alpha3);
        
        cv::Mat alphaSafe;
        cv::max(alpha3, 0.05f, alphaSafe);
        cv::Mat Fclean = (personF - bgF.mul(1.0f - alpha3)).mul(1.0f / alphaSafe);
        cv::Mat compF = Fclean.mul(alpha3) + bgF.mul(1.0f - alpha3);
        cv::Mat out8u; 
        compF.convertTo(out8u, CV_8U);
        out8u.copyTo(result, ringInner);

        cleanBackground.copyTo(result, ringOuter);
        
        //  FINAL EDGE BLURRING
        const float finalEdgeBlurRadius = 6.0f; // Increased for smoother final edges
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& finalStream = gpuMemoryPool->getCompositionStream();
                cv::Mat finalEdgeBlurred = applyEdgeBlurringCUDA(result, binMask, cleanBackground, finalEdgeBlurRadius, *gpuMemoryPool, finalStream);
                if (!finalEdgeBlurred.empty()) {
                    result = finalEdgeBlurred;
                }
            } catch (const cv::Exception& e) {
                qWarning() << "Final GPU edge blurring failed:" << e.what();
            }
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "DYNAMIC EDGE BLENDING: Edge blending failed:" << e.what() << "- using global correction";
        if (lightingCorrector) {
            return lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    } catch (const std::exception& e) {
        qWarning() << "DYNAMIC EDGE BLENDING: Exception:" << e.what() << "- using original";
        return composedFrame;
    }
}

cv::Mat Capture::applyDynamicFrameEdgeBlending(const cv::Mat &composedFrame, 
                                               const cv::Mat &rawPersonRegion, 
                                               const cv::Mat &rawPersonMask, 
                                               const cv::Mat &backgroundFrame)
{
    qDebug() << "DYNAMIC EDGE BLENDING: Applying edge blending to dynamic frame";
    
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        qWarning() << "Invalid input data for edge blending, using global correction";
        return m_lightingCorrector->applyGlobalLightingCorrection(composedFrame);
    }
    
    try {
        // Start with clean background or use provided background frame
        cv::Mat result;
        cv::Mat cleanBackground;
        
        if (!backgroundFrame.empty()) {
            cv::resize(backgroundFrame, cleanBackground, composedFrame.size());
        } else {
            // Extract background from dynamic template or use clean template
            if (!m_lastTemplateBackground.empty()) {
                cv::resize(m_lastTemplateBackground, cleanBackground, composedFrame.size());
            } else {
                // Fallback to zero background
                cleanBackground = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
            }
        }
        result = cleanBackground.clone();
        
        // VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        // Follows the same algorithm as static mode but with video-optimized parameters
        cv::Mat lightingCorrectedPerson = applyVideoOptimizedLighting(rawPersonRegion, rawPersonMask, m_lightingCorrector);
        qDebug() << "DYNAMIC: Applied video-optimized lighting correction (based on static mode)";
        
        // SCALING PRESERVATION: Scale the lighting-corrected person using the recorded scaling factor
        cv::Mat scaledPerson, scaledMask;
        
        // Calculate the scaled size using the recorded scaling factor
        cv::Size backgroundSize = result.size();
        cv::Size scaledPersonSize;
        
        if (qAbs(m_recordedPersonScaleFactor - 1.0) > 0.01) {
            int scaledWidth = static_cast<int>(backgroundSize.width * m_recordedPersonScaleFactor + 0.5);
            int scaledHeight = static_cast<int>(backgroundSize.height * m_recordedPersonScaleFactor + 0.5);
            
            //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
            scaledWidth = qMax(1, scaledWidth);
            scaledHeight = qMax(1, scaledHeight);
            
            scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
            qDebug() << "SCALING PRESERVATION: Scaling person to" << scaledWidth << "x" << scaledHeight 
                     << "with recorded factor" << m_recordedPersonScaleFactor;
        } else {
            scaledPersonSize = backgroundSize;
            qDebug() << "SCALING PRESERVATION: No scaling needed, using full size";
        }
        
        // Scale person and mask to the calculated size
        cv::resize(lightingCorrectedPerson, scaledPerson, scaledPersonSize);
        cv::resize(rawPersonMask, scaledMask, scaledPersonSize);
        
        // Calculate centered offset for placing the scaled person (same as static mode)
        cv::Size actualScaledSize(scaledPerson.cols, scaledPerson.rows);
        int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
        int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;
        
        // If person is scaled down, place it on a full-size canvas at the centered position (same as static mode)
        cv::Mat fullSizePerson, fullSizeMask;
        if (actualScaledSize != backgroundSize) {
            // Create full-size images initialized to zeros
            fullSizePerson = cv::Mat::zeros(backgroundSize, scaledPerson.type());
            fullSizeMask = cv::Mat::zeros(backgroundSize, CV_8UC1);
            
            // Ensure offsets are valid
            if (xOffset >= 0 && yOffset >= 0 &&
                xOffset + actualScaledSize.width <= backgroundSize.width &&
                yOffset + actualScaledSize.height <= backgroundSize.height) {
                
                // Place scaled person at centered position
                cv::Rect roi(xOffset, yOffset, actualScaledSize.width, actualScaledSize.height);
                scaledPerson.copyTo(fullSizePerson(roi));
                
                // Convert mask to grayscale if needed, then copy to ROI
                if (scaledMask.type() != CV_8UC1) {
                    cv::Mat grayMask;
                    cv::cvtColor(scaledMask, grayMask, cv::COLOR_BGR2GRAY);
                    grayMask.copyTo(fullSizeMask(roi));
                } else {
                    scaledMask.copyTo(fullSizeMask(roi));
                }
                
                qDebug() << "DYNAMIC: Placed scaled person at offset" << xOffset << "," << yOffset;
            } else {
                qWarning() << "DYNAMIC: Invalid offset, using direct copy";
                cv::resize(scaledPerson, fullSizePerson, backgroundSize);
                cv::resize(scaledMask, fullSizeMask, backgroundSize);
            }
        } else {
            // Person is full size, use as is
            fullSizePerson = scaledPerson;
            if (scaledMask.type() != CV_8UC1) {
                cv::cvtColor(scaledMask, fullSizeMask, cv::COLOR_BGR2GRAY);
            } else {
                fullSizeMask = scaledMask;
            }
        }
        
        // Now use fullSizePerson and fullSizeMask for blending (same as static mode)
        scaledPerson = fullSizePerson;
        scaledMask = fullSizeMask;
        
        // Apply guided filter edge blending (same algorithm as static mode)
        cv::Mat binMask;
        cv::threshold(scaledMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // First: shrink mask slightly to avoid fringe, then hard-copy interior
        cv::Mat interiorMask;
        cv::erode(binMask, interiorMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // ~2px shrink
        scaledPerson.copyTo(result, interiorMask);

        //  ENHANCED: Pre-smooth the mask before guided filtering for better edge quality
        cv::Mat smoothedBinMask;
        cv::GaussianBlur(binMask, smoothedBinMask, cv::Size(9, 9), 2.0); // Smooth mask edges first
        
        //  CUDA-Accelerated Guided image filtering to refine a soft alpha only on a thin edge ring
        const int gfRadius = 12; // Increased window size for smoother edges
        const float gfEps = 5e-3f; // Reduced regularization for better edge preservation
        
        // Use GPU memory pool stream and buffers for optimized guided filtering
        cv::cuda::Stream& guidedFilterStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, smoothedBinMask, gfRadius, gfEps, m_gpuMemoryPool, guidedFilterStream);
        
        //  ENHANCED: Apply edge blurring to create smooth transitions between background and segmented object
        const float edgeBlurRadius = 5.0f; // Increased blur radius for smoother background-object transition
        cv::Mat edgeBlurredPerson = applyEdgeBlurringCUDA(scaledPerson, binMask, cleanBackground, edgeBlurRadius, m_gpuMemoryPool, guidedFilterStream);
        if (!edgeBlurredPerson.empty()) {
            scaledPerson = edgeBlurredPerson;
            qDebug() << "DYNAMIC MODE: Applied CUDA edge blurring with radius" << edgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
            edgeBlurredPerson = applyEdgeBlurringAlternative(scaledPerson, binMask, edgeBlurRadius);
            if (!edgeBlurredPerson.empty()) {
                scaledPerson = edgeBlurredPerson;
                qDebug() << "DYNAMIC MODE: Applied alternative edge blurring with radius" << edgeBlurRadius;
            }
        }
        
        // Build thin inner/outer rings around the boundary for localized updates only
        cv::Mat inner, outer, ringInner, ringOuter;
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // shrink by ~2px for inner ring (wider transition)
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*6+1, 2*6+1))); // expand by ~6px for outer ring (wider feather)
        cv::subtract(binMask, inner, ringInner);   // just inside the boundary
        cv::subtract(outer, binMask, ringOuter);   // just outside the boundary
        
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);  // full person interior remains 1
        alphaFloat.setTo(0.0f, outer == 0); // outside remains 0
        // Less aggressive alpha reduction for smoother blending (was 0.3f, now 0.6f)
        alphaFloat = alphaFloat * 0.6f;

        // Composite only where outer>0 to avoid touching background
        cv::Mat personF, bgF; 
        scaledPerson.convertTo(personF, CV_32F); 
        cleanBackground.convertTo(bgF, CV_32F);
        std::vector<cv::Mat> a3 = {alphaFloat, alphaFloat, alphaFloat};
        cv::Mat alpha3; 
        cv::merge(a3, alpha3);
        
        // Inner ring: solve for decontaminated foreground using matting equation, then composite
        cv::Mat alphaSafe;
        cv::max(alpha3, 0.05f, alphaSafe); // avoid division by very small alpha
        cv::Mat Fclean = (personF - bgF.mul(1.0f - alpha3)).mul(1.0f / alphaSafe);
        cv::Mat compF = Fclean.mul(alpha3) + bgF.mul(1.0f - alpha3);
        cv::Mat out8u; 
        compF.convertTo(out8u, CV_8U);
        out8u.copyTo(result, ringInner);

        // Outer ring: copy template directly to eliminate any colored outline
        cleanBackground.copyTo(result, ringOuter);
        
        //  FINAL EDGE BLURRING: Apply edge blurring to the final composite result
        const float finalEdgeBlurRadius = 6.0f; // Increased blur radius for smoother final edges
        cv::cuda::Stream& finalStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat finalEdgeBlurred = applyEdgeBlurringCUDA(result, binMask, cleanBackground, finalEdgeBlurRadius, m_gpuMemoryPool, finalStream);
        if (!finalEdgeBlurred.empty()) {
            result = finalEdgeBlurred;
            qDebug() << "DYNAMIC MODE: Applied final CUDA edge blurring to composite result with radius" << finalEdgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
            finalEdgeBlurred = applyEdgeBlurringAlternative(result, binMask, finalEdgeBlurRadius);
            if (!finalEdgeBlurred.empty()) {
                result = finalEdgeBlurred;
                qDebug() << "DYNAMIC MODE: Applied final alternative edge blurring to composite result with radius" << finalEdgeBlurRadius;
            }
        }
        
        qDebug() << "DYNAMIC EDGE BLENDING: Successfully applied edge blending";
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "DYNAMIC EDGE BLENDING: Edge blending failed:" << e.what() << "- using global correction";
        return m_lightingCorrector->applyGlobalLightingCorrection(composedFrame);
    }
}

cv::Mat Capture::applyFastEdgeBlendingForVideo(const cv::Mat &composedFrame, 
                                               const cv::Mat &rawPersonRegion, 
                                               const cv::Mat &rawPersonMask, 
                                               const cv::Mat &backgroundFrame)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        if (m_lightingCorrector) {
            return m_lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    }
    
    try {
        // Get background - CRITICAL: Use exact frame from recording to ensure perfect synchronization
        cv::Mat cleanBackground;
        if (!backgroundFrame.empty()) {
            // Use the exact background frame that was recorded with this person frame
            // This ensures perfect synchronization between person and background
            if (backgroundFrame.size() != composedFrame.size()) {
                cv::resize(backgroundFrame, cleanBackground, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
            } else {
                cleanBackground = backgroundFrame.clone();
            }
            qDebug() << "DYNAMIC VIDEO: Using synchronized background frame:" << cleanBackground.cols << "x" << cleanBackground.rows;
        } else if (!m_lastTemplateBackground.empty()) {
            cv::resize(m_lastTemplateBackground, cleanBackground, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        } else {
            cleanBackground = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
        }
        
        // VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        cv::Mat lightingCorrectedPerson = applyVideoOptimizedLighting(rawPersonRegion, rawPersonMask, m_lightingCorrector);
        
        // Scale person if needed (preserve scaling factor)
        cv::Mat scaledPerson, scaledMask;
        cv::Size bgSize = cleanBackground.size();
        if (qAbs(m_recordedPersonScaleFactor - 1.0) > 0.01) {
            cv::Size scaledSize(static_cast<int>(bgSize.width * m_recordedPersonScaleFactor),
                               static_cast<int>(bgSize.height * m_recordedPersonScaleFactor));
            cv::resize(lightingCorrectedPerson, scaledPerson, scaledSize, 0, 0, cv::INTER_LINEAR);
            cv::resize(rawPersonMask, scaledMask, scaledSize, 0, 0, cv::INTER_LINEAR);
        } else {
            scaledPerson = lightingCorrectedPerson;
            scaledMask = rawPersonMask;
        }
        
        // Place person in center
        cv::Mat result = cleanBackground.clone();
        int xOffset = (bgSize.width - scaledPerson.cols) / 2;
        int yOffset = (bgSize.height - scaledPerson.rows) / 2;
        cv::Rect roi(xOffset, yOffset, scaledPerson.cols, scaledPerson.rows);
        
        // FAST EDGE BLENDING: Simple Gaussian blur + alpha blending (much faster than guided filters)
        // Step 1: Create binary mask
        cv::Mat binMask;
        if (scaledMask.channels() == 3) {
            cv::cvtColor(scaledMask, binMask, cv::COLOR_BGR2GRAY);
        } else {
            binMask = scaledMask.clone();
        }
        cv::threshold(binMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // Step 2: Apply Gaussian blur to mask for smooth edges
        cv::Mat blurredMask;
        cv::GaussianBlur(binMask, blurredMask, cv::Size(13, 13), 3.0); // Increased blur radius for smoother edges
        
        // Step 3: Normalize mask to [0, 1] range for alpha blending
        cv::Mat alphaMask;
        blurredMask.convertTo(alphaMask, CV_32F, 1.0/255.0);
        
        // Step 4: Resize alpha mask to match ROI if needed
        if (alphaMask.size() != roi.size()) {
            cv::resize(alphaMask, alphaMask, roi.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Step 5: Extract ROI from background and person
        cv::Mat bgROI = result(roi);
        cv::Mat personROI = scaledPerson;
        if (personROI.size() != roi.size()) {
            cv::resize(scaledPerson, personROI, roi.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Step 6: Fast alpha blending (GPU-accelerated if available)
        // FORCE GPU USAGE: Always use GPU if available for dynamic video processing
        bool useGPU = m_useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0;
        if (useGPU) {
            try {
                qDebug() << "DYNAMIC VIDEO: Using GPU acceleration for edge blending";
                // Upload to GPU
                cv::cuda::GpuMat gpuPerson, gpuBg, gpuAlpha, gpuResult;
                gpuPerson.upload(personROI);
                gpuBg.upload(bgROI);
                gpuAlpha.upload(alphaMask);
                
                // Convert to float for blending
                cv::cuda::GpuMat gpuPersonF, gpuBgF;
                gpuPerson.convertTo(gpuPersonF, CV_32F);
                gpuBg.convertTo(gpuBgF, CV_32F);
                
                // Create 3-channel alpha
                std::vector<cv::cuda::GpuMat> alphaChannels = {gpuAlpha, gpuAlpha, gpuAlpha};
                cv::cuda::GpuMat gpuAlpha3;
                cv::cuda::merge(alphaChannels, gpuAlpha3);
                
                // Alpha blend: result = person * alpha + bg * (1 - alpha)
                cv::cuda::GpuMat gpuResultF, gpuPersonBlended, gpuBgBlended;
                
                // person * alpha
                cv::cuda::multiply(gpuPersonF, gpuAlpha3, gpuPersonBlended);
                
                // Create (1 - alpha) by creating ones matrix and subtracting alpha
                cv::cuda::GpuMat gpuOnes;
                gpuOnes.create(gpuAlpha3.size(), gpuAlpha3.type());
                gpuOnes.setTo(cv::Scalar(1.0, 1.0, 1.0));
                cv::cuda::GpuMat gpuOneMinusAlpha;
                cv::cuda::subtract(gpuOnes, gpuAlpha3, gpuOneMinusAlpha);
                
                // bg * (1 - alpha)
                cv::cuda::multiply(gpuBgF, gpuOneMinusAlpha, gpuBgBlended);
                
                // Add them together
                cv::cuda::add(gpuPersonBlended, gpuBgBlended, gpuResultF);
                
                // Convert back to uint8
                gpuResultF.convertTo(gpuResult, CV_8U);
                gpuResult.download(bgROI);
                qDebug() << "DYNAMIC VIDEO: GPU blending successful for frame";
            } catch (const cv::Exception& e) {
                qWarning() << "DYNAMIC VIDEO: GPU blending failed:" << e.what() << "- falling back to CPU";
                // Fallback to CPU blending
                cv::Mat personF, bgF;
                personROI.convertTo(personF, CV_32F);
                bgROI.convertTo(bgF, CV_32F);
                
                std::vector<cv::Mat> alphaChannels = {alphaMask, alphaMask, alphaMask};
                cv::Mat alpha3;
                cv::merge(alphaChannels, alpha3);
                
                cv::Mat resultF = personF.mul(alpha3) + bgF.mul(cv::Scalar(1.0, 1.0, 1.0) - alpha3);
                resultF.convertTo(bgROI, CV_8U);
            }
        } else {
            qDebug() << "DYNAMIC VIDEO: Using CPU blending (GPU not available)";
            // CPU blending
            cv::Mat personF, bgF;
            personROI.convertTo(personF, CV_32F);
            bgROI.convertTo(bgF, CV_32F);
            
            std::vector<cv::Mat> alphaChannels = {alphaMask, alphaMask, alphaMask};
            cv::Mat alpha3;
            cv::merge(alphaChannels, alpha3);
            
            cv::Mat resultF = personF.mul(alpha3) + bgF.mul(cv::Scalar(1.0, 1.0, 1.0) - alpha3);
            resultF.convertTo(bgROI, CV_8U);
        }
        
        // CRITICAL: Ensure output size matches input composedFrame size (preserve original resolution)
        if (result.size() != composedFrame.size()) {
            cv::resize(result, result, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Fast edge blending failed:" << e.what() << "- using global correction";
        if (m_lightingCorrector) {
            return m_lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    }
}

// THREAD-SAFE WRAPPER: Takes all parameters instead of accessing member variables
cv::Mat Capture::applySimpleDynamicCompositingSafe(const cv::Mat &composedFrame,
                                                    const cv::Mat &rawPersonRegion,
                                                    const cv::Mat &rawPersonMask,
                                                    const cv::Mat &backgroundFrame,
                                                    LightingCorrector* lightingCorrector,
                                                    double /*personScaleFactor*/,
                                                    bool useCUDA)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        return composedFrame;
    }
    
    try {
        // Get background - use provided frame or fallback
        cv::Mat bg = backgroundFrame.empty() ? cv::Mat::zeros(composedFrame.size(), composedFrame.type()) : backgroundFrame.clone();
        if (bg.size() != composedFrame.size()) {
            cv::resize(bg, bg, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Scale person region to match frame size if needed
        cv::Mat personRegion = rawPersonRegion;
        cv::Mat personMask = rawPersonMask;
        if (personRegion.size() != composedFrame.size()) {
            cv::resize(personRegion, personRegion, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(personMask, personMask, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // STEP 1: VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        cv::Mat lightingCorrectedPerson = applyVideoOptimizedLighting(personRegion, personMask, lightingCorrector);
        
        // STEP 2: Create smoothed mask for edge blending
        cv::Mat binMask;
        if (personMask.channels() == 3) {
            cv::cvtColor(personMask, binMask, cv::COLOR_BGR2GRAY);
        } else {
            binMask = personMask.clone();
        }
        cv::threshold(binMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // Enhanced edge smoothing for better quality
        cv::Mat smoothedMask;
        cv::GaussianBlur(binMask, smoothedMask, cv::Size(11, 11), 2.5); // Larger blur for smoother edges
        
        // Normalize to [0, 1] for alpha blending
        cv::Mat alphaMask;
        smoothedMask.convertTo(alphaMask, CV_32F, 1.0/255.0);
        
        // STEP 3: Fast GPU-accelerated alpha blending
        if (useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::GpuMat gpuBg, gpuPerson, gpuAlpha, gpuResult;
                gpuBg.upload(bg);
                gpuPerson.upload(lightingCorrectedPerson);
                gpuAlpha.upload(alphaMask);
                
                cv::cuda::GpuMat gpuBgF, gpuPersonF;
                gpuBg.convertTo(gpuBgF, CV_32F);
                gpuPerson.convertTo(gpuPersonF, CV_32F);
                
                std::vector<cv::cuda::GpuMat> alphaChannels = {gpuAlpha, gpuAlpha, gpuAlpha};
                cv::cuda::GpuMat gpuAlpha3;
                cv::cuda::merge(alphaChannels, gpuAlpha3);
                
                cv::cuda::GpuMat gpuResultF, gpuPersonBlended, gpuBgBlended;
                cv::cuda::multiply(gpuPersonF, gpuAlpha3, gpuPersonBlended);
                
                cv::cuda::GpuMat gpuOnes;
                gpuOnes.create(gpuAlpha3.size(), gpuAlpha3.type());
                gpuOnes.setTo(cv::Scalar(1.0, 1.0, 1.0));
                cv::cuda::GpuMat gpuOneMinusAlpha;
                cv::cuda::subtract(gpuOnes, gpuAlpha3, gpuOneMinusAlpha);
                cv::cuda::multiply(gpuBgF, gpuOneMinusAlpha, gpuBgBlended);
                cv::cuda::add(gpuPersonBlended, gpuBgBlended, gpuResultF);
                gpuResultF.convertTo(gpuResult, CV_8U);
                
                cv::Mat result;
                gpuResult.download(result);
                return result;
            } catch (const cv::Exception& e) {
                qWarning() << "GPU compositing failed:" << e.what() << "- using CPU";
            }
        }
        
        // CPU fallback
        cv::Mat personF, bgF;
        lightingCorrectedPerson.convertTo(personF, CV_32F);
        bg.convertTo(bgF, CV_32F);
        
        std::vector<cv::Mat> alphaChannels = {alphaMask, alphaMask, alphaMask};
        cv::Mat alpha3;
        cv::merge(alphaChannels, alpha3);
        
        cv::Mat resultF = personF.mul(alpha3) + bgF.mul(cv::Scalar(1.0, 1.0, 1.0) - alpha3);
        cv::Mat result;
        resultF.convertTo(result, CV_8U);
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Dynamic compositing failed:" << e.what();
        return composedFrame;
    } catch (const std::exception& e) {
        qWarning() << "Dynamic compositing exception:" << e.what();
        return composedFrame;
    }
}

cv::Mat Capture::applySimpleDynamicCompositing(const cv::Mat &composedFrame,
                                               const cv::Mat &rawPersonRegion,
                                               const cv::Mat &rawPersonMask,
                                               const cv::Mat &backgroundFrame)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        return composedFrame;
    }
    
    try {
        // Get background - use provided frame or fallback
        cv::Mat bg = backgroundFrame.empty() ? cv::Mat::zeros(composedFrame.size(), composedFrame.type()) : backgroundFrame.clone();
        if (bg.size() != composedFrame.size()) {
            cv::resize(bg, bg, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Scale person region to match frame size if needed
        cv::Mat personRegion = rawPersonRegion;
        cv::Mat personMask = rawPersonMask;
        if (personRegion.size() != composedFrame.size()) {
            cv::resize(personRegion, personRegion, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(personMask, personMask, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // STEP 1: VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        cv::Mat lightingCorrectedPerson = applyVideoOptimizedLighting(personRegion, personMask, m_lightingCorrector);
        
        // STEP 2: Create smoothed mask for edge blending (fast Gaussian blur)
        cv::Mat binMask;
        if (personMask.channels() == 3) {
            cv::cvtColor(personMask, binMask, cv::COLOR_BGR2GRAY);
        } else {
            binMask = personMask.clone();
        }
        cv::threshold(binMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // Enhanced edge smoothing for better quality
        cv::Mat smoothedMask;
        cv::GaussianBlur(binMask, smoothedMask, cv::Size(11, 11), 2.5); // Larger blur for smoother edges
        
        // Normalize to [0, 1] for alpha blending
        cv::Mat alphaMask;
        smoothedMask.convertTo(alphaMask, CV_32F, 1.0/255.0);
        
        // STEP 3: Fast GPU-accelerated alpha blending with lighting-corrected person
        if (m_useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::GpuMat gpuBg, gpuPerson, gpuAlpha, gpuResult;
                gpuBg.upload(bg);
                gpuPerson.upload(lightingCorrectedPerson);
                gpuAlpha.upload(alphaMask);
                
                // Convert to float for blending
                cv::cuda::GpuMat gpuBgF, gpuPersonF;
                gpuBg.convertTo(gpuBgF, CV_32F);
                gpuPerson.convertTo(gpuPersonF, CV_32F);
                
                // Create 3-channel alpha
                std::vector<cv::cuda::GpuMat> alphaChannels = {gpuAlpha, gpuAlpha, gpuAlpha};
                cv::cuda::GpuMat gpuAlpha3;
                cv::cuda::merge(alphaChannels, gpuAlpha3);
                
                // Alpha blend: result = person * alpha + bg * (1 - alpha)
                cv::cuda::GpuMat gpuResultF, gpuPersonBlended, gpuBgBlended;
                
                // person * alpha
                cv::cuda::multiply(gpuPersonF, gpuAlpha3, gpuPersonBlended);
                
                // Create (1 - alpha)
                cv::cuda::GpuMat gpuOnes;
                gpuOnes.create(gpuAlpha3.size(), gpuAlpha3.type());
                gpuOnes.setTo(cv::Scalar(1.0, 1.0, 1.0));
                cv::cuda::GpuMat gpuOneMinusAlpha;
                cv::cuda::subtract(gpuOnes, gpuAlpha3, gpuOneMinusAlpha);
                
                // bg * (1 - alpha)
                cv::cuda::multiply(gpuBgF, gpuOneMinusAlpha, gpuBgBlended);
                
                // Add them together
                cv::cuda::add(gpuPersonBlended, gpuBgBlended, gpuResultF);
                
                // Convert back to uint8
                gpuResultF.convertTo(gpuResult, CV_8U);
                
                cv::Mat result;
                gpuResult.download(result);
                return result;
            } catch (const cv::Exception& e) {
                qWarning() << "GPU compositing failed:" << e.what() << "- using CPU";
            }
        }
        
        // CPU fallback: alpha blending with lighting-corrected person
        cv::Mat personF, bgF;
        lightingCorrectedPerson.convertTo(personF, CV_32F);
        bg.convertTo(bgF, CV_32F);
        
        std::vector<cv::Mat> alphaChannels = {alphaMask, alphaMask, alphaMask};
        cv::Mat alpha3;
        cv::merge(alphaChannels, alpha3);
        
        // Alpha blend: result = person * alpha + bg * (1 - alpha)
        cv::Mat resultF = personF.mul(alpha3) + bgF.mul(cv::Scalar(1.0, 1.0, 1.0) - alpha3);
        cv::Mat result;
        resultF.convertTo(result, CV_8U);
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Dynamic compositing failed:" << e.what();
        return composedFrame;
    }
}

// ============================================================================
//  VIDEO-OPTIMIZED LIGHTING CORRECTION (Based on Static Mode Algorithm)
// ============================================================================

cv::Mat Capture::applyVideoOptimizedLighting(const cv::Mat &personRegion, 
                                              const cv::Mat &personMask,
                                              LightingCorrector* lightingCorrector)
{
    // VIDEO-SPECIFIC: Follow static mode algorithm but optimized for video processing
    // This uses the same LAB color space matching as static mode but with video-optimized parameters
    
    // CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.type() != CV_8UC3) {
        qWarning() << "Invalid person region format for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personMask.type() != CV_8UC1) {
        qWarning() << "Invalid mask format for video lighting - returning original";
        return personRegion.clone();
    }
    
    // Start with exact copy of person region
    cv::Mat result = personRegion.clone();
    
    // If no lighting corrector, return original
    if (!lightingCorrector) {
        return result;
    }
    
    try {
        // Get template reference for color matching (same as static mode)
        cv::Mat templateRef = lightingCorrector->getReferenceTemplate();
        
        if (templateRef.empty()) {
            // No template: Apply very subtle brightness adjustment for video (even more subtle than static)
            // VIDEO-OPTIMIZED: More conservative adjustments for video frames
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    if (y < personMask.rows && x < personMask.cols && 
                        personMask.at<uchar>(y, x) > 0) {  // Person pixel
                        cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                        // VIDEO-SPECIFIC: Very subtle changes (less aggressive than static mode)
                        pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 1.05);  // Very slight blue boost
                        pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 1.02);  // Very slight green boost
                        pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 1.04);  // Very slight red boost
                    }
                }
            }
        } else {
            // VIDEO-OPTIMIZED: Use same LAB color space matching as static mode but with video-specific parameters
            cv::resize(templateRef, templateRef, personRegion.size());
            
            // Convert to LAB for color matching (same as static mode)
            cv::Mat personLab, templateLab;
            cv::cvtColor(personRegion, personLab, cv::COLOR_BGR2Lab);
            cv::cvtColor(templateRef, templateLab, cv::COLOR_BGR2Lab);
            
            // Calculate template statistics (same as static mode)
            cv::Scalar templateMean, templateStd;
            cv::meanStdDev(templateLab, templateMean, templateStd);
            
            // Apply color matching to person region (same algorithm as static mode)
            cv::Mat resultLab = personLab.clone();
            std::vector<cv::Mat> channels;
            cv::split(resultLab, channels);
            
            // Calculate person statistics for comparison
            cv::Scalar personMean, personStd;
            cv::meanStdDev(personLab, personMean, personStd);
            
            // VIDEO-SPECIFIC: More conservative adjustments than static mode
            // Static mode uses 15% adjustment, video uses 10% for smoother frame-to-frame transitions
            for (int c = 0; c < 3; c++) {
                // Calculate the difference between template and person
                double lightingDiff = templateMean[c] - personMean[c];
                
                // VIDEO-OPTIMIZED: Apply more conservative adjustment (10% vs 15% in static)
                // This prevents jarring changes between video frames
                channels[c] = channels[c] + lightingDiff * 0.10;
            }
            
            // VIDEO-SPECIFIC: More conservative brightness adjustment
            // Static mode uses 10%, video uses 5% for smoother transitions
            double brightnessDiff = templateMean[0] - personMean[0]; // L channel
            if (brightnessDiff > 0) {
                channels[0] = channels[0] + brightnessDiff * 0.05; // Very slight brightness boost
            }
            
            cv::merge(channels, resultLab);
            cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
            
            // Apply mask to ensure only person pixels are affected
            cv::Mat maskedResult;
            result.copyTo(maskedResult, personMask);
            personRegion.copyTo(maskedResult, ~personMask);
            result = maskedResult;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        qWarning() << "Video lighting correction exception:" << e.what() << "- returning original";
        return personRegion.clone();
    }
}

