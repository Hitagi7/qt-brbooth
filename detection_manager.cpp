#include "detection_manager.h"
#include "simplepersondetector.h"
#include "optimized_detector.h"
#include <QFile>
#include <QDebug>
#include <QMutexLocker>
#include <QCoreApplication>
#include <chrono>

DetectionManager::DetectionManager(QObject* parent)
    : QObject(parent)
    , m_yoloProcess(new QProcess(this))
    , m_simpleDetector(new SimplePersonDetector())
    , m_optimizedDetector(new OptimizedPersonDetector(this))
    , m_currentMode(OPTIMIZED_ONNX)
    , m_useCppDetector(false)
    , m_useOptimizedDetector(true)
    , m_showBoundingBoxes(true)
    , m_isProcessingFrame(false)
    , m_averageDetectionTime(0.0)
    , m_frameCount(0)
    , m_performanceTimer(this)
{
    qDebug() << "🔧 DetectionManager initialized";
    
    initializeDetectors();
    
    // Setup performance monitoring
    m_performanceTimer.setInterval(5000); // Log every 5 seconds
    connect(&m_performanceTimer, &QTimer::timeout, this, &DetectionManager::printPerformanceStats);
    m_performanceTimer.start();
}

DetectionManager::~DetectionManager() {
    if (m_yoloProcess && m_yoloProcess->state() != QProcess::NotRunning) {
        m_yoloProcess->kill();
        m_yoloProcess->waitForFinished(3000);
    }
    
    delete m_simpleDetector;
    // m_optimizedDetector is deleted by Qt parent-child system
    
    qDebug() << "🔧 DetectionManager destroyed";
}

void DetectionManager::setDetectionMode(DetectionMode mode) {
    m_currentMode = mode;
    
    switch (mode) {
        case YOLO_PYTHON:
            m_useCppDetector = false;
            m_useOptimizedDetector = false;
            qDebug() << "🐍 Detection mode: YOLO Python subprocess";
            break;
        case SIMPLE_CPP:
            m_useCppDetector = true;
            m_useOptimizedDetector = false;
            qDebug() << "🔧 Detection mode: Simple C++ HOG detector";
            break;
        case OPTIMIZED_ONNX:
            m_useCppDetector = false;
            m_useOptimizedDetector = true;
            qDebug() << "⚡ Detection mode: Optimized C++ ONNX detector";
            break;
    }
}

void DetectionManager::setUseCppDetector(bool use) {
    m_useCppDetector = use;
    if (use) {
        m_useOptimizedDetector = false;
        m_currentMode = SIMPLE_CPP;
    }
}

void DetectionManager::setUseOptimizedDetector(bool use) {
    m_useOptimizedDetector = use;
    if (use) {
        m_useCppDetector = false;
        m_currentMode = OPTIMIZED_ONNX;
    }
}

void DetectionManager::setShowBoundingBoxes(bool show) {
    m_showBoundingBoxes = show;
    qDebug() << "📦 Bounding boxes:" << (show ? "enabled" : "disabled");
}

void DetectionManager::detectPersonInImage(const QString& imagePath) {
    if (m_isProcessingFrame) {
        qDebug() << "⚠️ Detection already in progress, skipping frame";
        return;
    }
    
    m_isProcessingFrame = true;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        cv::Mat image = cv::imread(imagePath.toStdString());
        if (image.empty()) {
            qWarning() << "❌ Failed to load image:" << imagePath;
            m_isProcessingFrame = false;
            emit detectionError("Failed to load image");
            return;
        }
        
        detectPersonsInFrame(image);
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Error in image detection:" << e.what();
        m_isProcessingFrame = false;
        emit detectionError(QString("Image detection failed: %1").arg(e.what()));
    }
}

void DetectionManager::detectPersonsInFrame(const cv::Mat& frame) {
    if (m_isProcessingFrame || !isValidForDetection(frame)) {
        return;
    }
    
    m_isProcessingFrame = true;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        if (m_useOptimizedDetector && m_optimizedDetector) {
            // Use optimized ONNX detector
            qDebug() << "⚡ Using optimized detector for frame detection";
            m_optimizedDetector->detectPersonsAsync(frame);
            // Results will come via signal
            
        } else if (m_useCppDetector && m_simpleDetector) {
            // Use simple C++ HOG detector
            qDebug() << "🔧 Using simple C++ detector for frame detection";
            QList<SimpleDetection> simpleDetections = m_simpleDetector->detect(frame);
            
            // Convert to BoundingBox format
            QList<BoundingBox> detections;
            for (const SimpleDetection& det : simpleDetections) {
                BoundingBox box(
                    det.boundingBox.x,
                    det.boundingBox.y,
                    det.boundingBox.x + det.boundingBox.width,
                    det.boundingBox.y + det.boundingBox.height,
                    det.confidence
                );
                detections.append(box);
            }
            
            updateDetectionResults(detections);
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double detectionTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            updatePerformanceStats(detectionTime);
            
            m_isProcessingFrame = false;
            emit detectionsReady(detections);
            emit detectionProcessingFinished();
            
        } else {
            // Use YOLO Python subprocess (fallback)
            qDebug() << "🐍 Using YOLO Python subprocess for frame detection";
            
            // Save frame temporarily
            QString tempPath = QCoreApplication::applicationDirPath() + "/temp_detection.jpg";
            cv::imwrite(tempPath.toStdString(), frame);
            
            // Start YOLO process
            QString program = "python";
            QStringList arguments;
            arguments << "yolov5/detect.py" 
                     << "--weights" << "yolov5/yolov5n.pt"
                     << "--source" << tempPath
                     << "--img" << "640"
                     << "--conf" << "0.5"
                     << "--save-txt"
                     << "--project" << "runs/detect"
                     << "--name" << "exp"
                     << "--exist-ok";
            
            m_yoloProcess->start(program, arguments);
            // Results will come via handleYoloOutput()
        }
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Error in frame detection:" << e.what();
        m_isProcessingFrame = false;
        emit detectionError(QString("Frame detection failed: %1").arg(e.what()));
    }
}

void DetectionManager::testDetection() {
    qDebug() << "🧪 Testing detection systems...";
    
    // Test with a sample image
    QString testImagePath = "pics/1.png";
    if (QFile::exists(testImagePath)) {
        detectPersonInImage(testImagePath);
    } else {
        qWarning() << "⚠️ Test image not found:" << testImagePath;
        emit detectionError("Test image not found");
    }
}

QList<BoundingBox> DetectionManager::getCurrentDetections() const {
    QMutexLocker locker(&m_detectionMutex);
    return m_currentDetections;
}

QList<OptimizedDetection> DetectionManager::getCurrentOptimizedDetections() const {
    QMutexLocker locker(&m_optimizedDetectionMutex);
    return m_currentOptimizedDetections;
}

double DetectionManager::getAverageDetectionTime() const {
    return m_averageDetectionTime;
}

int DetectionManager::getCurrentFPS() const {
    if (m_averageDetectionTime > 0) {
        return static_cast<int>(1000.0 / m_averageDetectionTime);
    }
    return 0;
}

void DetectionManager::printPerformanceStats() {
    if (m_frameCount > 0) {
        qDebug() << "📊 Detection Performance Stats:";
        qDebug() << "   Mode:" << static_cast<int>(m_currentMode);
        qDebug() << "   Avg time:" << m_averageDetectionTime << "ms";
        qDebug() << "   Current FPS:" << getCurrentFPS();
        qDebug() << "   Frames processed:" << m_frameCount;
    }
}

// SLOTS

void DetectionManager::handleYoloOutput() {
    QString output = QString::fromUtf8(m_yoloProcess->readAllStandardOutput());
    processYoloOutput(output);
}

void DetectionManager::handleYoloError() {
    QString error = QString::fromUtf8(m_yoloProcess->readAllStandardError());
    qWarning() << "🐍 YOLO Error:" << error;
}

void DetectionManager::handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    m_isProcessingFrame = false;
    
    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        qDebug() << "🐍 YOLO detection completed successfully";
        // Process results from output files
        // Implementation depends on YOLO output format
        emit detectionProcessingFinished();
    } else {
        qWarning() << "🐍 YOLO detection failed with exit code:" << exitCode;
        emit detectionError("YOLO detection process failed");
    }
}

void DetectionManager::handleYoloErrorOccurred(QProcess::ProcessError error) {
    m_isProcessingFrame = false;
    qWarning() << "🐍 YOLO process error:" << error;
    emit detectionError("YOLO process error occurred");
}

void DetectionManager::onOptimizedDetectionsReady(const QList<OptimizedDetection>& detections) {
    updateOptimizedDetectionResults(detections);
    emit optimizedDetectionsReady(detections);
}

void DetectionManager::onOptimizedProcessingFinished() {
    m_isProcessingFrame = false;
    emit detectionProcessingFinished();
}

// PRIVATE METHODS

void DetectionManager::initializeDetectors() {
    // Initialize simple detector
    if (m_simpleDetector && !m_simpleDetector->initialize()) {
        qWarning() << "⚠️ Failed to initialize simple person detector";
    }
    
    // Initialize optimized detector
    if (m_optimizedDetector) {
        connect(m_optimizedDetector, &OptimizedPersonDetector::detectionsReady,
                this, &DetectionManager::onOptimizedDetectionsReady);
        connect(m_optimizedDetector, &OptimizedPersonDetector::processingFinished,
                this, &DetectionManager::onOptimizedProcessingFinished);
        
        // Initialize with segmentation model for best results
        m_optimizedDetector->initialize(OptimizedPersonDetector::YOLO_SEGMENTATION,
                                      OptimizedPersonDetector::REAL_TIME);
    }
    
    // Setup YOLO process signals
    connect(m_yoloProcess, &QProcess::readyReadStandardOutput,
            this, &DetectionManager::handleYoloOutput);
    connect(m_yoloProcess, &QProcess::readyReadStandardError,
            this, &DetectionManager::handleYoloError);
    connect(m_yoloProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &DetectionManager::handleYoloFinished);
    connect(m_yoloProcess, &QProcess::errorOccurred,
            this, &DetectionManager::handleYoloErrorOccurred);
}

void DetectionManager::processYoloOutput(const QString& output) {
    // Implementation depends on YOLO output format
    // This is a placeholder for processing YOLO detection results
    qDebug() << "🐍 Processing YOLO output:" << output.left(100) << "...";
}

void DetectionManager::updatePerformanceStats(double detectionTime) {
    m_detectionTimes.append(detectionTime);
    
    // Keep only recent samples
    while (m_detectionTimes.size() > 30) {
        m_detectionTimes.removeFirst();
    }
    
    // Calculate average
    double sum = 0.0;
    for (double time : m_detectionTimes) {
        sum += time;
    }
    m_averageDetectionTime = sum / m_detectionTimes.size();
    
    m_frameCount++;
}

void DetectionManager::updateDetectionResults(const QList<BoundingBox>& detections) {
    {
        QMutexLocker locker(&m_detectionMutex);
        m_currentDetections = detections;
    }
    
    qDebug() << "🎯 Detection results updated:" << detections.size() << "persons detected";
}

void DetectionManager::updateOptimizedDetectionResults(const QList<OptimizedDetection>& detections) {
    {
        QMutexLocker locker(&m_optimizedDetectionMutex);
        m_currentOptimizedDetections = detections;
    }
    
    qDebug() << "⚡ Optimized detection results updated:" << detections.size() << "persons detected";
}

bool DetectionManager::isValidForDetection(const cv::Mat& frame) const {
    if (frame.empty()) {
        return false;
    }
    
    // Minimum size check
    if (frame.cols < 100 || frame.rows < 100) {
        return false;
    }
    
    return true;
}

void DetectionManager::clearResults() {
    {
        QMutexLocker locker(&m_detectionMutex);
        m_currentDetections.clear();
    }
    {
        QMutexLocker locker(&m_optimizedDetectionMutex);
        m_currentOptimizedDetections.clear();
    }
}