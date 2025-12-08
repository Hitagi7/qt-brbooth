// Video Processing Orchestrator
// Main function for processing recorded video frames with lighting correction

#include "core/capture.h"
#include "core/camera.h"  // For cvMatToQImage helper function
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QThread>
#include <QtConcurrent/QtConcurrent>
#include <QThreadPool>
#include <QAtomicInt>
#include <opencv2/opencv.hpp>
#include "core/lighting_corrector.h"

// Forward declarations for helper functions defined in capture.cpp
extern cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                                 GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

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
                        finalFrame = this->applyDynamicFrameEdgeBlendingSafe(composedCopy,
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
                            finalFrame = this->applySimpleDynamicCompositingSafe(composedCopy,
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
                            finalFrame = this->applySimpleDynamicCompositingSafe(composedCopy,
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
