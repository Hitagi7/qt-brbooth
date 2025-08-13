#include "capture.h"
#include "ui_capture.h"
#include "foreground.h"
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QPainter>
#include <QKeyEvent>
#include <QCheckBox>
#include <QApplication>
#include <QPushButton>
#include <QMessageBox>
#include <QDateTime>
#include <QStackedLayout>
#include <opencv2/opencv.hpp>
#include <QtConcurrent/QtConcurrent>
#include <QMutexLocker>

Capture::Capture(QWidget *parent, Foreground *fg)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)
    , stackedLayout(nullptr)
    , videoLabelFPS(nullptr)
    , totalTime(0)
    , frameCount(0)
    , m_tfliteSegmentation(new TFLiteDeepLabv3(this))
    , m_segmentationWidget(new TFLiteSegmentationWidget(this))
    , m_showSegmentation(true)
    , m_segmentationConfidenceThreshold(0.5)
    , m_currentFrame()
    , m_lastSegmentedFrame()
    , m_segmentationMutex()
    , m_segmentationTimer()
    , m_lastSegmentationTime(0.0)
    , m_segmentationFPS(0)
    , m_tfliteModelLoaded(false)
    , foreground(fg)
    , overlayImageLabel(nullptr)
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , fpsLabel(nullptr)
    , segmentationLabel(nullptr)
    , segmentationButton(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_segmentationWatcher(nullptr)
    , m_processingAsync(false)
    , m_asyncMutex()
    , m_lastProcessedFrame()
    , m_handDetector(new AdvancedHandDetector())
    , m_showHandDetection(true)
    , m_handDetectionMutex()
    , m_handDetectionTimer()
    , m_lastHandDetectionTime(0.0)
    , m_handDetectionFPS(0)
{
    ui->setupUi(this);

    // Ensure no margins or spacing for the Capture widget itself
    setContentsMargins(0, 0, 0, 0);
    
    // Enable keyboard focus for this widget
    setFocusPolicy(Qt::StrongFocus);
    setFocus();

    // Ensure no margins or spacing for the Capture widget itself
    // Note: Don't create a new layout here as the UI file already provides one

    // Setup Debug Display
    setupDebugDisplay();
    
    // Update segmentation button to reflect initial state
    updateSegmentationButton();
    
    // Update hand detection button to reflect initial state
    updateHandDetectionButton();
    
    // Ensure video label fills the entire window
    if (ui->videoLabel) {
        ui->videoLabel->setMinimumSize(this->size());
        ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }

    // Foreground Overlay Setup
    overlayImageLabel = new QLabel(ui->overlayWidget);
    QString selectedOverlay;
    if (foreground) {
        selectedOverlay = foreground->getSelectedForeground();
    } else {
        qWarning() << "Error: foreground is nullptr!";
    }
    qDebug() << "Selected overlay path:" << selectedOverlay;
    overlayImageLabel->setAttribute(Qt::WA_TranslucentBackground);
    overlayImageLabel->setStyleSheet("background: transparent;");
    overlayImageLabel->setScaledContents(true);
    overlayImageLabel->resize(this->size());
    overlayImageLabel->hide();
    connect(foreground, &Foreground::foregroundChanged, this, &Capture::updateForegroundOverlay);
    qDebug() << "Selected overlay path:" << selectedOverlay;
    QPixmap overlayPixmap(selectedOverlay);
    overlayImageLabel->setPixmap(overlayPixmap);

    // Setup stacked layout
    setupStackedLayoutHybrid();

    // Initialize TFLite Segmentation
    initializeTFLiteSegmentation();
    
    // Initialize Hand Detection (disabled by default)
    initializeHandDetection();
    m_handDetectionEnabled = false; // Start with hand detection disabled
    // Initialize MediaPipe-like tracker
    m_handTracker.initialize(640, 480);

    // Run segmentation in a background thread so UI stays responsive (Debug-safe)
    m_segmentationThread = new QThread(this);
    m_tfliteSegmentation->moveToThread(m_segmentationThread);
    connect(m_segmentationThread, &QThread::started, m_tfliteSegmentation, &TFLiteDeepLabv3::startRealtimeProcessing);
    connect(m_segmentationThread, &QThread::finished, m_tfliteSegmentation, &TFLiteDeepLabv3::stopRealtimeProcessing);
    m_segmentationThread->start();

    // Connect TFLite signals
    connect(m_tfliteSegmentation, &TFLiteDeepLabv3::segmentationResultReady,
            this, &Capture::onSegmentationResultReady);
    connect(m_tfliteSegmentation, &TFLiteDeepLabv3::processingError,
            this, &Capture::onSegmentationError);
    connect(m_tfliteSegmentation, &TFLiteDeepLabv3::modelLoaded,
            this, &Capture::onTFLiteModelLoaded);

    // Connect segmentation widget signals
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::showSegmentationChanged,
            this, &Capture::setShowSegmentation);
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::confidenceThresholdChanged,
            this, &Capture::setSegmentationConfidenceThreshold);
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::performanceModeChanged,
            this, &Capture::setSegmentationPerformanceMode);

    // Initialize camera
    cap.open(0);
    if (!cap.isOpened()) {
        qWarning() << "Failed to open camera";
        QMessageBox::warning(this, "Camera Error", "Failed to open camera. Please check your camera connection.");
        return;
    }

    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // Minimize buffer to reduce latency
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); // Use MJPG for better performance

    // Setup timers
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(16); // ~60 FPS - ULTRA-LIGHT processing

    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    // Performance tracking
    loopTimer.start();
    frameTimer.start();

    // Debug update timer
    debugUpdateTimer = new QTimer(this);
    connect(debugUpdateTimer, &QTimer::timeout, this, &Capture::updateDebugDisplay);
    debugUpdateTimer->start(1000); // Lighter UI updates in Debug

    // Initialize async processing
    m_segmentationWatcher = new QFutureWatcher<cv::Mat>(this);
    connect(m_segmentationWatcher, &QFutureWatcher<cv::Mat>::finished, 
            this, &Capture::onSegmentationFinished);
            
    // Pre-allocate pixmap for better performance
    m_cachedPixmap = QPixmap(640, 480);

    qDebug() << "Capture widget initialized successfully with TRUE REAL-TIME frame-by-frame segmentation";
}

Capture::~Capture()
{
    if (cameraTimer && cameraTimer->isActive()) {
        cameraTimer->stop();
    }
    if (m_segmentationThread) {
        m_segmentationThread->quit();
        m_segmentationThread->wait();
        m_segmentationThread = nullptr;
    }
    if (cap.isOpened()) {
        cap.release();
    }
    
    // Cancel any running async operations
    if (m_segmentationWatcher && m_segmentationWatcher->isRunning()) {
        m_segmentationWatcher->cancel();
        m_segmentationWatcher->waitForFinished();
    }
    
    delete ui;
}

void Capture::initializeTFLiteSegmentation()
{
    // Try to load the TFLite model
    QString modelPath = "deeplabv3.tflite";
    if (QFileInfo::exists(modelPath)) {
        qDebug() << "Loading TFLite model from:" << modelPath;
        m_tfliteSegmentation->initializeModel(modelPath);
            } else {
        qDebug() << "TFLite model not found, using OpenCV fallback segmentation";
        // Initialize with OpenCV fallback instead of showing error
        m_tfliteSegmentation->initializeModel("opencv_fallback");
    }

    // Set initial parameters
    m_tfliteSegmentation->setConfidenceThreshold(m_segmentationConfidenceThreshold);
    m_tfliteSegmentation->setPerformanceMode(TFLiteDeepLabv3::Balanced);

    // Initialize async processing
    m_segmentationWatcher = new QFutureWatcher<cv::Mat>(this);
    connect(m_segmentationWatcher, &QFutureWatcher<cv::Mat>::finished, 
            this, &Capture::onSegmentationFinished);
}

void Capture::initializeHandDetection()
{
    if (m_handDetector) {
        if (m_handDetector->initialize()) {
            qDebug() << "Advanced hand detector initialized successfully";
            m_handDetector->setConfidenceThreshold(0.8); // STRICT threshold to prevent false positives
            m_handDetector->setShowBoundingBox(true);
            m_handDetector->setPerformanceMode(1); // Set to Balanced mode
        } else {
            qWarning() << "Failed to initialize hand detector";
        }
    }
}

void Capture::enableHandDetection(bool enable)
{
    m_handDetectionEnabled = enable;
    if (enable && m_handDetector) {
        // Reset hand detection state when enabling to prevent false triggers
        m_handDetector->resetGestureState();
        // Initially disable detection to prevent false triggers
        m_handDetectionEnabled = false;
        qDebug() << "🖐️ Hand detection DISABLED during initialization";
        
        // Add a longer delay before enabling detection to prevent false triggers
        QTimer::singleShot(5000, [this]() {
            // Re-arm trackers cleanly
            m_handDetector->resetGestureState();
            m_handTracker.reset();
            m_handDetectionEnabled = true;
            qDebug() << "🖐️ Hand detection now ACTIVE - ready for gestures";
        });
    }
    if (enable) {
        qDebug() << "🖐️ Hand detection ENABLED - State reset (5 second delay with proper disable)";
    } else {
        qDebug() << "🖐️ Hand detection DISABLED";
    }
}

void Capture::setHandDetectionEnabled(bool enabled)
{
    enableHandDetection(enabled);
}

void Capture::setCaptureMode(CaptureMode mode)
{
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to:" << (mode == ImageCaptureMode ? "Image" : "Video");
}

void Capture::setVideoTemplate(const VideoTemplate &templateData)
{
    m_currentVideoTemplate = templateData;
    qDebug() << "Video template set:" << templateData.name << "Duration:" << templateData.durationSeconds;
}

void Capture::setShowSegmentation(bool show)
{
    m_showSegmentation = show;
    updateSegmentationButton();
    qDebug() << "Segmentation display set to:" << show;
}

bool Capture::getShowSegmentation() const
{
    return m_showSegmentation;
}

void Capture::setSegmentationConfidenceThreshold(double threshold)
{
    m_segmentationConfidenceThreshold = threshold;
    m_tfliteSegmentation->setConfidenceThreshold(threshold);
    qDebug() << "Segmentation confidence threshold set to:" << threshold;
}

double Capture::getSegmentationConfidenceThreshold() const
{
    return m_segmentationConfidenceThreshold;
}

cv::Mat Capture::getLastSegmentedFrame() const
{
    QMutexLocker locker(&m_segmentationMutex);
    return m_lastSegmentedFrame.clone();
}

void Capture::saveSegmentedFrame(const QString& filename)
{
    QMutexLocker locker(&m_segmentationMutex);
    if (!m_lastSegmentedFrame.empty()) {
        QString savePath = filename.isEmpty() ? 
            QString("segmented_frame_%1.png").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss")) :
            filename;
        
        if (cv::imwrite(savePath.toStdString(), m_lastSegmentedFrame)) {
            qDebug() << "Segmented frame saved to:" << savePath;
            QMessageBox::information(this, "Success", "Segmented frame saved successfully.");
        } else {
            qWarning() << "Failed to save segmented frame to:" << savePath;
            QMessageBox::warning(this, "Error", "Failed to save segmented frame.");
        }
    } else {
        QMessageBox::information(this, "Info", "No segmented frame available to save.");
    }
}

double Capture::getSegmentationProcessingTime() const
{
    return m_lastSegmentationTime;
}

void Capture::setSegmentationPerformanceMode(TFLiteDeepLabv3::PerformanceMode mode)
{
    m_tfliteSegmentation->setPerformanceMode(mode);
    qDebug() << "Segmentation performance mode set to:" << mode;
}

void Capture::toggleSegmentation()
{
    m_showSegmentation = !m_showSegmentation;
    updateSegmentationButton();
    qDebug() << "Segmentation toggled to:" << (m_showSegmentation ? "ON" : "OFF");
}

void Capture::updateSegmentationButton()
{
    if (segmentationButton) {
        if (m_showSegmentation) {
            segmentationButton->setText("Disable Segmentation");
            segmentationButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; }");
        } else {
            segmentationButton->setText("Enable Segmentation");
            segmentationButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #388e3c; border: 1px solid white; padding: 5px; }");
        }
    }
}

bool Capture::isTFLiteModelLoaded() const
{
    return m_tfliteModelLoaded;
}

// Hand Detection Control Methods
void Capture::setShowHandDetection(bool show)
{
    m_showHandDetection = show;
    updateHandDetectionButton();
    qDebug() << "Hand detection display set to:" << show;
}

bool Capture::getShowHandDetection() const
{
    return m_showHandDetection;
}

void Capture::setHandDetectionConfidenceThreshold(double threshold)
{
    if (m_handDetector) {
        m_handDetector->setConfidenceThreshold(threshold);
        qDebug() << "Hand detection confidence threshold set to:" << threshold;
    }
}

double Capture::getHandDetectionConfidenceThreshold() const
{
    if (m_handDetector) {
        return m_handDetector->getConfidenceThreshold();
    }
    return 0.5;
}

QList<AdvancedHandDetection> Capture::getLastHandDetections() const
{
    QMutexLocker locker(&m_handDetectionMutex);
    return m_lastHandDetections;
}

void Capture::toggleHandDetection()
{
    m_showHandDetection = !m_showHandDetection;
    updateHandDetectionButton();
    qDebug() << "Hand detection toggled to:" << (m_showHandDetection ? "ON" : "OFF");
}

void Capture::updateHandDetectionButton()
{
    if (handDetectionButton) {
        if (m_showHandDetection) {
            handDetectionButton->setText("Disable Hand Detection");
            handDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; }");
        } else {
            handDetectionButton->setText("Enable Hand Detection");
            handDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #388e3c; border: 1px solid white; padding: 5px; }");
        }
    }
}

double Capture::getHandDetectionProcessingTime() const
{
    return m_lastHandDetectionTime;
}

void Capture::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    
    // Ensure video label fills the entire window
    if (ui->videoLabel) {
        ui->videoLabel->setMinimumSize(this->size());
    }
    
    if (overlayImageLabel) {
        overlayImageLabel->resize(this->size());
    }
    updateOverlayStyles();
}

void Capture::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_Space:
            on_capture_clicked();
            break;
        case Qt::Key_Escape:
            on_back_clicked();
            break;
        case Qt::Key_D:
            // Toggle debug widget visibility
            if (debugWidget) {
                debugWidget->setVisible(!debugWidget->isVisible());
            }
            break;
        case Qt::Key_S:
            // Toggle segmentation
            toggleSegmentation();
            break;
        case Qt::Key_H:
            // Toggle hand detection
            toggleHandDetection();
            break;
        case Qt::Key_F12:
            // Save debug frame
            if (!m_currentFrame.empty()) {
                cv::imwrite("debug_current_frame.png", m_currentFrame);
                qDebug() << "Saved debug frame to debug_current_frame.png";
                
                // Also trigger hand detection on this frame for debugging
                if (m_handDetector && m_handDetector->isInitialized()) {
                    QList<AdvancedHandDetection> debugDetections = m_handDetector->detect(m_currentFrame);
                    qDebug() << "Manual hand detection test found" << debugDetections.size() << "hands";
                }
            }
            break;
        default:
            QWidget::keyPressEvent(event);
    }
}

void Capture::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    if (cameraTimer && !cameraTimer->isActive()) {
        cameraTimer->start(16); // ~60 FPS - ULTRA-LIGHT processing
    }
    qDebug() << "Capture widget shown - ENABLING hand detection with 5-second delay";
    enableHandDetection(true); // Enable hand detection when page is shown
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    if (cameraTimer && cameraTimer->isActive()) {
        cameraTimer->stop();
    }
    qDebug() << "Capture widget hidden - DISABLING hand detection";
    enableHandDetection(false); // Disable hand detection when page is hidden
}

void Capture::updateCameraFeed()
{
    if (!cap.isOpened()) {
        return;
    }

    cv::Mat frame;
    if (cap.read(frame)) {
        // Mirror the camera frame horizontally
        cv::flip(frame, frame, 1);
        
        // Store current frame for TFLite processing
        m_currentFrame = frame.clone();

        cv::Mat displayFrame = frame.clone();
        bool frameProcessed = false;

        // Process frame with hand detection if enabled (optimized performance)
        if (m_handDetectionEnabled) {
            // Update MediaPipe-like tracker first
            m_handTracker.update(frame);
            if (m_handTracker.shouldTriggerCapture()) {
                startCountdown();
            }
            // Keep existing detector as secondary signal (can be disabled later)
            if (m_handDetector && m_handDetector->isInitialized()) {
                qDebug() << "🖐️ Processing hand detection - ENABLED";
                processFrameWithHandDetection(displayFrame);
                frameProcessed = true;
            }
        } else {
            qDebug() << "🖐️ Hand detection DISABLED - skipping processing";
        }

        // Process frame with segmentation if enabled - ASYNC PROCESSING
        if (m_showSegmentation) {
            // Check if we should start a new async processing
            QMutexLocker locker(&m_asyncMutex);
            if (!m_processingAsync && m_segmentationWatcher && !m_segmentationWatcher->isRunning()) {
                m_processingAsync = true;
                m_lastProcessedFrame = frame.clone();
                
                // Start async processing
                QFuture<cv::Mat> future = QtConcurrent::run(processFrameAsync, frame, m_tfliteSegmentation);
                m_segmentationWatcher->setFuture(future);
            }
            
            // Display the last available segmented frame or original frame
            QPixmap pixmap;
            cv::Mat segmentedFrame;
            QList<AdvancedHandDetection> handDetections;
            
            {
                QMutexLocker segLocker(&m_segmentationMutex);
                if (!m_lastSegmentedFrame.empty()) {
                    segmentedFrame = m_lastSegmentedFrame.clone();
                }
            }
            
            if (!segmentedFrame.empty()) {
                // Apply hand detection to segmented frame if enabled
                if (m_handDetectionEnabled && m_handDetector && m_handDetector->isInitialized()) {
                    QMutexLocker handLocker(&m_handDetectionMutex);
                    handDetections = m_lastHandDetections;
                } else {
                    // Clear hand detections when disabled
                    handDetections.clear();
                }
                drawHandBoundingBoxes(segmentedFrame, handDetections);
                QImage qImage = cvMatToQImage(segmentedFrame);
                pixmap = QPixmap::fromImage(qImage);
            }
            
            if (pixmap.isNull()) {
                QImage qImage = cvMatToQImage(displayFrame);
                pixmap = QPixmap::fromImage(qImage);
            }
            
            if (ui->videoLabel) {
                ui->videoLabel->setPixmap(pixmap.scaled(ui->videoLabel->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
            }
        } else {
            // Display original frame when segmentation is disabled
            QImage qImage = cvMatToQImage(displayFrame);
            QPixmap pixmap = QPixmap::fromImage(qImage);
            
            if (ui->videoLabel) {
                ui->videoLabel->setPixmap(pixmap.scaled(ui->videoLabel->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
            }
        }

        // Update performance metrics more frequently for real-time feedback
        frameCount++;
        if (frameCount % 10 == 0) { // Update FPS every 10 frames instead of 30
            qint64 elapsed = loopTimer.elapsed();
            m_currentFPS = (elapsed > 0) ? static_cast<int>((10 * 1000) / elapsed) : 0;
            loopTimer.restart();
        }
    }
}

// Static function for async processing
cv::Mat Capture::processFrameAsync(const cv::Mat &frame, TFLiteDeepLabv3 *segmentation)
{
    if (!segmentation) {
        return frame.clone();
    }
    
    try {
        return segmentation->segmentFrame(frame);
    } catch (const std::exception& e) {
        qWarning() << "Exception in async segmentation:" << e.what();
        return frame.clone();
    }
}

// Slot to handle finished async segmentation
void Capture::onSegmentationFinished()
{
    if (m_segmentationWatcher && m_segmentationWatcher->isFinished()) {
        try {
            cv::Mat result = m_segmentationWatcher->result();
            
            // Store the result
            {
                QMutexLocker locker(&m_segmentationMutex);
                m_lastSegmentedFrame = result.clone();
            }
            
            // Update timing info
            m_lastSegmentationTime = m_segmentationTimer.elapsed() / 1000.0;
            m_segmentationFPS = (m_lastSegmentationTime > 0) ? 1.0 / m_lastSegmentationTime : 0;
            
        } catch (const std::exception& e) {
            qWarning() << "Exception getting async result:" << e.what();
        }
    }
    
    // Mark as not processing
    QMutexLocker locker(&m_asyncMutex);
    m_processingAsync = false;
}

void Capture::onHandDetectionFinished()
{
    // This slot can be used for async hand detection if needed in the future
    qDebug() << "Hand detection processing completed";
}

void Capture::processFrameWithHandDetection(const cv::Mat &frame)
{
    if (!m_handDetector || !m_handDetector->isInitialized()) {
        return;
    }
    
    m_handDetectionTimer.start();
    
    try {
        // Detect hands in the frame
        QList<AdvancedHandDetection> detections = m_handDetector->detect(frame);
        
        // Store the detections
        {
            QMutexLocker locker(&m_handDetectionMutex);
            m_lastHandDetections = detections;
        }
        
        // Update timing info
        m_lastHandDetectionTime = m_handDetectionTimer.elapsed() / 1000.0;
        m_handDetectionFPS = (m_lastHandDetectionTime > 0) ? 1.0 / m_lastHandDetectionTime : 0;
        
        // Apply hand detection visualization to the frame
        cv::Mat displayFrame = frame.clone();
        drawHandBoundingBoxes(displayFrame, detections);
        
    } catch (const std::exception& e) {
        qWarning() << "Exception in hand detection:" << e.what();
    }
}

void Capture::drawHandBoundingBoxes(cv::Mat &/*frame*/, const QList<AdvancedHandDetection> &detections)
{
    // REMOVED: No more bounding box drawing to avoid conflicts with segmentation
    // Only show gesture status in console for debugging
    for (const auto& detection : detections) {
        if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
            // Check if capture should be triggered
            if (m_handDetector->shouldTriggerCapture()) {
                qDebug() << "🎯 CAPTURE TRIGGERED! Hand closed gesture detected!";
                // Trigger countdown and capture
                startCountdown();
            }
            
            // Show gesture status in console only
            bool isOpen = m_handDetector->isHandOpen(detection.landmarks);
            bool isClosed = m_handDetector->isHandClosed(detection.landmarks);
            double closureRatio = m_handDetector->calculateHandClosureRatio(detection.landmarks);
            
            if (isOpen || isClosed) {
                QString gestureStatus = isOpen ? "OPEN" : "CLOSED";
                qDebug() << "Hand detected - Gesture:" << gestureStatus 
                         << "Confidence:" << static_cast<int>(detection.confidence * 100) << "%"
                         << "Closure ratio:" << closureRatio;
            }
        }
    }
}

// These methods are no longer needed since we process frames directly in updateCameraFeed
// Keeping them for compatibility but they're not used in the new real-time implementation

void Capture::updateSegmentationDisplay(const QImage &segmentedImage)
{
    // Convert QImage to cv::Mat and store
    cv::Mat segmentedMat = qImageToCvMat(segmentedImage);
    {
        QMutexLocker locker(&m_segmentationMutex);
        m_lastSegmentedFrame = segmentedMat.clone();
    }
}

void Capture::showSegmentationNotification()
{
    qDebug() << "Segmentation processing completed";
}

void Capture::captureRecordingFrame()
{
    if (m_isRecording && cap.isOpened()) {
        cv::Mat frame;
        if (cap.read(frame)) {
            QImage qImage = cvMatToQImage(frame);
            QPixmap pixmap = QPixmap::fromImage(qImage);
            m_recordedFrames.append(pixmap);
        }
    }
}

void Capture::on_back_clicked()
{
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    if (m_currentCaptureMode == VideoRecordMode) {
        if (!m_isRecording) {
            startRecording();
                    } else {
            stopRecording();
                }
            } else {
        performImageCapture();
    }
}

void Capture::startCountdown()
{
    // Only start countdown if not already running
    if (countdownTimer && !countdownTimer->isActive()) {
        countdownValue = 3; // 3 second countdown to prepare
        if (countdownLabel) {
            countdownLabel->setText(QString::number(countdownValue));
            countdownLabel->show();
            countdownLabel->raise(); // Bring to front
        }
        countdownTimer->start(1000); // 1 second intervals
        qDebug() << "🎬 Countdown started! 3 seconds to prepare...";
    }
}

void Capture::updateCountdown()
{
    countdownValue--;
    if (countdownValue <= 0) {
        countdownTimer->stop();
        if (countdownLabel) {
            countdownLabel->hide();
        }
        
        if (m_currentCaptureMode == VideoRecordMode) {
            startRecording();
        } else {
            performImageCapture();
        }
        } else {
        if (countdownLabel) {
            countdownLabel->setText(QString::number(countdownValue));
        }
    }
}

void Capture::updateRecordTimer()
{
    m_recordedSeconds++;
    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        stopRecording();
    }
}

void Capture::updateForegroundOverlay(const QString &path)
{
    if (overlayImageLabel) {
        QPixmap overlayPixmap(path);
        overlayImageLabel->setPixmap(overlayPixmap);
    }
}

void Capture::on_verticalSlider_valueChanged(int value)
{
    // Handle slider value changes if needed
    qDebug() << "Slider value changed to:" << value;
}

void Capture::onSegmentationResultReady(const QImage &segmentedImage)
{
    updateSegmentationDisplay(segmentedImage);
    showSegmentationNotification();
    emit segmentationCompleted();
}

void Capture::onSegmentationError(const QString &error)
{
    qWarning() << "TFLite segmentation error:" << error;
    QMessageBox::warning(this, "Segmentation Error", error);
}

void Capture::onTFLiteModelLoaded(bool success)
{
    m_tfliteModelLoaded = success;
    if (success) {
        qDebug() << "TFLite model loaded successfully";
        QMessageBox::information(this, "Success", "TFLite Deeplabv3 model loaded successfully!");
    } else {
        qWarning() << "Failed to load TFLite model";
        QMessageBox::warning(this, "Error", "Failed to load TFLite Deeplabv3 model.");
    }
}

void Capture::updateDebugDisplay()
{
    if (debugLabel) {
        QString debugInfo = QString("FPS: %1 | Seg: %2 | Hand: %3 | Seg FPS: %4 | Hand FPS: %5")
                           .arg(m_currentFPS)
                           .arg(m_showSegmentation ? "ON" : "OFF")
                           .arg(m_showHandDetection ? "ON" : "OFF")
                           .arg(QString::number(m_segmentationFPS, 'f', 1))
                           .arg(QString::number(m_handDetectionFPS, 'f', 1));
        debugLabel->setText(debugInfo);
    }
    
    if (fpsLabel) {
        fpsLabel->setText(QString("FPS: %1").arg(m_currentFPS));
    }
    
    if (segmentationLabel) {
        QString segStatus = m_showSegmentation ? "ON (Tracking)" : "OFF";
        segmentationLabel->setText(QString("Segmentation: %1").arg(segStatus));
    }
    
    if (handDetectionLabel) {
        QString handStatus = m_showHandDetection ? "ON (Tracking)" : "OFF";
        handDetectionLabel->setText(QString("Hand Detection: %1").arg(handStatus));
    }
}

void Capture::startRecording()
{
    m_isRecording = true;
    m_recordedSeconds = 0;
    m_recordedFrames.clear();
    
    recordingFrameTimer->start(1000 / m_targetRecordingFPS);
    recordTimer->start(1000);
    
    qDebug() << "Recording started";
}

void Capture::stopRecording()
{
    m_isRecording = false;
    recordingFrameTimer->stop();
    recordTimer->stop();

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    
    emit showFinalOutputPage();
    qDebug() << "Recording stopped";
}

void Capture::performImageCapture()
{
    if (cap.isOpened()) {
        cv::Mat frame;
        if (cap.read(frame)) {
            QImage qImage = cvMatToQImage(frame);
            m_capturedImage = QPixmap::fromImage(qImage);
            emit imageCaptured(m_capturedImage);
        }
    }
    emit showFinalOutputPage();
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    if (mat.empty()) {
        return QImage();
    }

    // Optimize for BGR format (most common from camera)
    if (mat.type() == CV_8UC3) {
        // Use faster conversion for BGR
        QImage qImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return qImage.rgbSwapped(); // Convert BGR to RGB
    }
    
    // Fallback for other formats
    switch (mat.type()) {
        case CV_8UC1: {
            QImage qImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
            return qImage.copy(); // Need to copy for grayscale
        }
        case CV_8UC4: {
            QImage qImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGBA8888);
            return qImage.copy(); // Need to copy for RGBA
        }
        default: {
            cv::Mat converted;
            cv::cvtColor(mat, converted, cv::COLOR_BGR2RGB);
            QImage qImage(converted.data, converted.cols, converted.rows, converted.step, QImage::Format_RGB888);
            return qImage.copy();
        }
    }
}

cv::Mat Capture::qImageToCvMat(const QImage &image)
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

void Capture::setupStackedLayoutHybrid()
{
    // Create stacked layout
    stackedLayout = new QStackedLayout();
    
    // Add video label to stacked layout
    if (ui->videoLabel) {
        stackedLayout->addWidget(ui->videoLabel);
    }
    
    // Add countdown label
    countdownLabel = new QLabel(this);
    countdownLabel->setAlignment(Qt::AlignCenter);
    countdownLabel->setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 0.7); color: white; font-size: 48px; font-weight: bold; }");
    countdownLabel->hide();
    stackedLayout->addWidget(countdownLabel);
    
    // Don't set layout on videoLabel if it already has one from the UI file
    // The stacked layout will be managed by the parent widget's layout
}

void Capture::updateOverlayStyles()
{
    if (overlayImageLabel) {
        overlayImageLabel->setStyleSheet("background: transparent;");
    }
}

void Capture::setupDebugDisplay()
{
    // Create debug widget
    debugWidget = new QWidget(this);
    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; }");
    
    QVBoxLayout *debugLayout = new QVBoxLayout(debugWidget);
    
    // Debug info label
    debugLabel = new QLabel("Initializing...", debugWidget);
    debugLabel->setStyleSheet("QLabel { color: white; font-size: 12px; }");
    debugLayout->addWidget(debugLabel);

    // FPS label
    fpsLabel = new QLabel("FPS: 0", debugWidget);
    fpsLabel->setStyleSheet("QLabel { color: white; font-size: 12px; }");
    debugLayout->addWidget(fpsLabel);
    
    // Segmentation label
    segmentationLabel = new QLabel("Segmentation: OFF", debugWidget);
    segmentationLabel->setStyleSheet("QLabel { color: white; font-size: 12px; }");
    debugLayout->addWidget(segmentationLabel);
    
    // Segmentation button
    segmentationButton = new QPushButton("Disable Segmentation", debugWidget);
    segmentationButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; }");
    connect(segmentationButton, &QPushButton::clicked, this, &Capture::toggleSegmentation);
    debugLayout->addWidget(segmentationButton);
    
    // Hand detection label
    handDetectionLabel = new QLabel("Hand Detection: OFF", debugWidget);
    handDetectionLabel->setStyleSheet("QLabel { color: white; font-size: 12px; }");
    debugLayout->addWidget(handDetectionLabel);
    
    // Hand detection button
    handDetectionButton = new QPushButton("Disable Hand Detection", debugWidget);
    handDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; }");
    connect(handDetectionButton, &QPushButton::clicked, this, &Capture::toggleHandDetection);
    debugLayout->addWidget(handDetectionButton);
    
    // Add debug widget to the main widget instead of videoLabel's layout
    debugWidget->setParent(this);
    debugWidget->move(10, 10); // Position in top-left corner
    debugWidget->resize(200, 150); // Increased size for hand detection controls
    debugWidget->raise(); // Ensure it's on top
    
    debugWidget->show(); // Show debug widget so user can enable segmentation and hand detection
}
