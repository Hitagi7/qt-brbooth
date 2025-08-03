#include "segmentation_manager.h"
#include "optimized_detector.h"
#include <QDebug>
#include <QMutexLocker>
#include <QDir>
#include <QDateTime>
#include <chrono>

SegmentationManager::SegmentationManager(QObject* parent)
    : QObject(parent)
    , m_segmentationProcessor(new PersonSegmentationProcessor())
    , m_fastSegmentationProcessor(new FastSegmentationProcessor())
    , m_showSegmentation(false)
    , m_confidenceThreshold(0.7)
    , m_averageProcessingTime(0.0)
    , m_frameCount(0)
{
    qDebug() << "🔧 SegmentationManager initialized";
    
    // Connect fast segmentation processor signals if it has any
    // connect(m_fastSegmentationProcessor, &FastSegmentationProcessor::processingFinished,
    //         this, &SegmentationManager::onSegmentationProcessingFinished);
}

SegmentationManager::~SegmentationManager() {
    delete m_segmentationProcessor;
    delete m_fastSegmentationProcessor;
    qDebug() << "🔧 SegmentationManager destroyed";
}

void SegmentationManager::setShowSegmentation(bool show) {
    m_showSegmentation = show;
    qDebug() << "🎭 Segmentation display:" << (show ? "enabled" : "disabled");
}

void SegmentationManager::setConfidenceThreshold(double threshold) {
    m_confidenceThreshold = qBound(0.1, threshold, 1.0);
    qDebug() << "🎯 Segmentation confidence threshold set to:" << m_confidenceThreshold;
}

void SegmentationManager::setSegmentationMethod(PersonSegmentationProcessor::PerformanceMode mode) {
    if (m_segmentationProcessor) {
        m_segmentationProcessor->setPerformanceMode(mode);
        qDebug() << "⚙️ Segmentation performance mode changed to:" << static_cast<int>(mode);
    }
}

void SegmentationManager::processPersonSegmentation(const cv::Mat& frame, const QList<BoundingBox>& detections) {
    if (!m_showSegmentation || !isValidForSegmentation(frame) || detections.isEmpty()) {
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        qDebug() << "🎭 Processing person segmentation for" << detections.size() << "detections";
        
        // Process segmentation
        QList<SegmentationResult> results = m_segmentationProcessor->segmentPersons(frame, detections, m_confidenceThreshold);
        
        // Update results
        {
            QMutexLocker locker(&m_segmentationMutex);
            m_currentSegmentations = results;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        updatePerformanceStats(processingTime);
        
        qDebug() << "✅ Person segmentation complete:" << results.size() << "masks in" << processingTime << "ms";
        
        emit segmentationCompleted(results);
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Error in person segmentation:" << e.what();
        emit segmentationError(QString("Person segmentation failed: %1").arg(e.what()));
    }
}

void SegmentationManager::processOptimizedSegmentation(const cv::Mat& frame, const QList<OptimizedDetection>& detections) {
    if (!m_showSegmentation || !m_fastSegmentationProcessor || !isValidForSegmentation(frame) || detections.isEmpty()) {
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        qDebug() << "⚡ Processing optimized segmentation for" << detections.size() << "detections";
        
        // Process fast segmentation
        QList<FastSegmentationResult> results = 
            m_fastSegmentationProcessor->segmentFromOptimizedDetections(frame, detections);
        
        // Update results
        {
            QMutexLocker locker(&m_fastSegmentationMutex);
            m_currentFastSegmentations = results;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        updatePerformanceStats(processingTime);
        
        qDebug() << "✅ Optimized segmentation complete:" << results.size() << "masks in" << processingTime << "ms";
        
        emit fastSegmentationCompleted(results);
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Error in optimized segmentation:" << e.what();
        emit segmentationError(QString("Optimized segmentation failed: %1").arg(e.what()));
    }
}

QList<SegmentationResult> SegmentationManager::getCurrentSegmentationResults() const {
    QMutexLocker locker(&m_segmentationMutex);
    return m_currentSegmentations;
}

QList<FastSegmentationResult> SegmentationManager::getCurrentFastSegmentationResults() const {
    QMutexLocker locker(&m_fastSegmentationMutex);
    return m_currentFastSegmentations;
}

double SegmentationManager::getAverageProcessingTime() const {
    return m_averageProcessingTime;
}

int SegmentationManager::getCurrentFPS() const {
    if (m_averageProcessingTime > 0) {
        return static_cast<int>(1000.0 / m_averageProcessingTime);
    }
    return 0;
}

void SegmentationManager::applySegmentationToFrame(cv::Mat& frame) const {
    try {
        // Apply regular segmentation results
        {
            QMutexLocker locker(&m_segmentationMutex);
            if (!m_currentSegmentations.isEmpty()) {
                for (const SegmentationResult& result : m_currentSegmentations) {
                    if (result.isValid && !result.segmentedImage.empty()) {
                        // Apply segmented image with transparent background
                        cv::Rect roi(result.boundingBox.x, result.boundingBox.y, 
                                   result.boundingBox.width, result.boundingBox.height);
                        
                        // Ensure ROI is within frame bounds
                        roi.x = std::max(0, roi.x);
                        roi.y = std::max(0, roi.y);
                        roi.width = std::min(roi.width, frame.cols - roi.x);
                        roi.height = std::min(roi.height, frame.rows - roi.y);
                        
                        if (roi.width > 0 && roi.height > 0) {
                            cv::Mat resizedSegmented;
                            cv::resize(result.segmentedImage, resizedSegmented, roi.size());
                            resizedSegmented.copyTo(frame(roi));
                        }
                    }
                }
                return; // Regular segmentation takes priority
            }
        }
        
        // Apply fast segmentation results if no regular results
        {
            QMutexLocker locker(&m_fastSegmentationMutex);
            if (!m_currentFastSegmentations.isEmpty()) {
                for (const FastSegmentationResult& result : m_currentFastSegmentations) {
                    if (result.isValid && !result.segmentedImage.empty()) {
                        cv::Rect roi(result.boundingBox.x, result.boundingBox.y, 
                                   result.boundingBox.width, result.boundingBox.height);
                        
                        // Ensure ROI is within frame bounds
                        roi.x = std::max(0, roi.x);
                        roi.y = std::max(0, roi.y);
                        roi.width = std::min(roi.width, frame.cols - roi.x);
                        roi.height = std::min(roi.height, frame.rows - roi.y);
                        
                        if (roi.width > 0 && roi.height > 0) {
                            cv::Mat resizedSegmented;
                            cv::resize(result.segmentedImage, resizedSegmented, roi.size());
                            resizedSegmented.copyTo(frame(roi));
                        }
                    }
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ OpenCV error applying segmentation:" << e.what();
    } catch (const std::exception& e) {
        qWarning() << "❌ Standard error applying segmentation:" << e.what();
    }
}

void SegmentationManager::saveSegmentedFrame(const QString& filename) {
    try {
        QList<SegmentationResult> results = getCurrentSegmentationResults();
        QList<FastSegmentationResult> fastResults = getCurrentFastSegmentationResults();
        
        if (results.isEmpty() && fastResults.isEmpty()) {
            qWarning() << "⚠️ No segmentation results to save";
            return;
        }
        
        // Create output directory if it doesn't exist
        QDir outputDir("segmented_output");
        if (!outputDir.exists()) {
            outputDir.mkpath(".");
        }
        
        QString fullPath = outputDir.absoluteFilePath(filename);
        
        // Save regular segmentation results
        if (!results.isEmpty()) {
            for (int i = 0; i < results.size(); ++i) {
                const SegmentationResult& result = results[i];
                if (result.isValid && !result.segmentedImage.empty()) {
                    QString indexedFilename = QString("segmented_%1_%2").arg(i).arg(filename);
                    QString indexedPath = outputDir.absoluteFilePath(indexedFilename);
                    
                    if (cv::imwrite(indexedPath.toStdString(), result.segmentedImage)) {
                        qDebug() << "💾 Saved segmented frame:" << indexedPath;
                    } else {
                        qWarning() << "❌ Failed to save segmented frame:" << indexedPath;
                    }
                }
            }
        }
        
        // Save fast segmentation results
        else if (!fastResults.isEmpty()) {
            for (int i = 0; i < fastResults.size(); ++i) {
                const FastSegmentationResult& result = fastResults[i];
                if (result.isValid && !result.segmentedImage.empty()) {
                    QString indexedFilename = QString("fast_segmented_%1_%2").arg(i).arg(filename);
                    QString indexedPath = outputDir.absoluteFilePath(indexedFilename);
                    
                    if (cv::imwrite(indexedPath.toStdString(), result.segmentedImage)) {
                        qDebug() << "💾 Saved fast segmented frame:" << indexedPath;
                    } else {
                        qWarning() << "❌ Failed to save fast segmented frame:" << indexedPath;
                    }
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ OpenCV error saving segmented frame:" << e.what();
        emit segmentationError(QString("Failed to save segmented frame: %1").arg(e.what()));
    } catch (const std::exception& e) {
        qWarning() << "❌ Standard error saving segmented frame:" << e.what();
        emit segmentationError(QString("Failed to save segmented frame: %1").arg(e.what()));
    }
}

void SegmentationManager::onSegmentationProcessingFinished() {
    qDebug() << "🏁 Segmentation processing finished";
}

void SegmentationManager::updatePerformanceStats(double processingTime) {
    m_processingTimes.append(processingTime);
    
    // Keep only recent samples
    while (m_processingTimes.size() > 30) {
        m_processingTimes.removeFirst();
    }
    
    // Calculate average
    double sum = 0.0;
    for (double time : m_processingTimes) {
        sum += time;
    }
    m_averageProcessingTime = sum / m_processingTimes.size();
    
    m_frameCount++;
}

void SegmentationManager::clearResults() {
    {
        QMutexLocker locker(&m_segmentationMutex);
        m_currentSegmentations.clear();
    }
    {
        QMutexLocker locker(&m_fastSegmentationMutex);
        m_currentFastSegmentations.clear();
    }
}

bool SegmentationManager::isValidForSegmentation(const cv::Mat& frame) const {
    if (frame.empty()) {
        return false;
    }
    
    // Minimum size check
    if (frame.cols < 100 || frame.rows < 100) {
        return false;
    }
    
    return true;
}