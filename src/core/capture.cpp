#include "core/capture.h"
#include "core/camera.h"
#include "ui/foreground.h"
#include "ui_capture.h"
// Hand detection removed per user request
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
#include <opencv2/core/ocl.hpp>  // OpenCL support
#include <QtConcurrent/QtConcurrent>
#include <QThreadPool>
#include <QMutexLocker>
#include <chrono>
#include <QFutureWatcher>
#include "algorithms/lighting_correction/lighting_corrector.h"
#include "core/system_monitor.h"

//  Forward declarations for guided filtering functions
static cv::Mat guidedFilterGrayAlphaCPU(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps);
// Made non-static so it can be used by capture_dynamic.cpp
cv::Mat guidedFilterGrayAlphaOpenCLOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, GPUMemoryPool &memoryPool);

//  Forward declarations for edge blurring functions
// Made non-static so they can be used by capture_dynamic.cpp
cv::Mat applyEdgeBlurringOpenCL(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, GPUMemoryPool &memoryPool);
static cv::Mat applyEdgeBlurringCPU(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius);
cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

// Fixed segmentation rectangle configuration
// Adjust kFixedRectX and kFixedRectY to reposition the rectangle on screen.
// Size of the rectanglez
static const int kFixedRectWidth = 1440;   // constant width in pixels
static const int kFixedRectHeight = 720;  // constant height in pixels
// Move left to right
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

static double intersectionOverUnion(const cv::Rect &a, const cv::Rect &b)
{
    const int interArea = (a & b).area();
    const int unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0) return 0.0;
    return static_cast<double>(interArea) / static_cast<double>(unionArea);
}

// Consolidate near-identical boxes (very high IoU) to ensure one box per person
static std::vector<cv::Rect> enforceOneBoxPerPerson(const std::vector<cv::Rect> &detections)
{
    if (detections.size() <= 1) {
        return detections;
    }

    const double highIoU = 0.75; // only merge near-duplicates; preserves nearby people

    std::vector<cv::Rect> boxes = detections;
    std::vector<bool> removed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            double iou = intersectionOverUnion(boxes[i], boxes[j]);
            if (iou >= highIoU) {
                // Merge duplicates by taking the union to retain full-body coverage
                boxes[i] = boxes[i] | boxes[j];
                removed[j] = true;
            }
        }
    }

    std::vector<cv::Rect> result;
    result.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!removed[i]) {
            result.push_back(boxes[i]);
        }
    }
    return result;
}

std::vector<cv::Rect> Capture::smoothDetections(const std::vector<cv::Rect> &current)
{
    // Parameters: EMA smoothing and IoU matching
    const double iouMatchThreshold = 0.3;
    const double alpha = 0.7; // keep majority of current box to avoid lag

    if (m_prevSmoothedDetections.empty()) {
        m_prevSmoothedDetections = current;
        m_smoothingHoldCounter = m_smoothingHoldFrames;
        return current;
    }

    std::vector<cv::Rect> result;
    std::vector<bool> matchedPrev(m_prevSmoothedDetections.size(), false);

    // Greedy match current to previous by IoU
    for (const auto &cur : current) {
        int bestIdx = -1;
        double bestIou = 0.0;
        for (size_t j = 0; j < m_prevSmoothedDetections.size(); ++j) {
            if (matchedPrev[j]) continue;
            double iou = intersectionOverUnion(cur, m_prevSmoothedDetections[j]);
            if (iou > bestIou) { bestIou = iou; bestIdx = static_cast<int>(j); }
        }

        if (bestIdx >= 0 && bestIou >= iouMatchThreshold) {
            const cv::Rect &prev = m_prevSmoothedDetections[bestIdx];
            matchedPrev[bestIdx] = true;
            // EMA on position and size
            cv::Rect smoothed;
            smoothed.x = cvRound(alpha * cur.x + (1.0 - alpha) * prev.x);
            smoothed.y = cvRound(alpha * cur.y + (1.0 - alpha) * prev.y);
            smoothed.width = cvRound(alpha * cur.width + (1.0 - alpha) * prev.width);
            smoothed.height = cvRound(alpha * cur.height + (1.0 - alpha) * prev.height);
            result.push_back(smoothed);
        } else {
            // New detection, accept as is
            result.push_back(cur);
        }
    }

    // Holdover: keep unmatched previous for a few frames to avoid flicker
    if (result.empty() && m_smoothingHoldCounter > 0) {
        m_smoothingHoldCounter--;
        return m_prevSmoothedDetections;
    }

    m_prevSmoothedDetections = result;
    m_smoothingHoldCounter = m_smoothingHoldFrames;
    return result;
}

Capture::Capture(QWidget *parent, Foreground *fg, Camera *existingCameraWorker, QThread *existingCameraThread)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , foreground(fg)
    , cameraThread(existingCameraThread)
    , cameraWorker(existingCameraWorker)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , flashOverlayLabel(nullptr)
    , flashAnimation(nullptr)
    , recordingTimerLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_actualCameraFPS(30.0)  // Default to 30 FPS
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)
    , m_recordedFrames()
    , m_capturedImage()
    , stackedLayout(nullptr)
    , videoLabelFPS(nullptr)
    , loopTimer()
    , totalTime(0)
    , frameCount(0)
    , frameTimer()
    , overlayImageLabel(nullptr)
    , m_personScaleFactor(1.0) // Initialize to 1.0 (normal size) - matches slider at 0
    // Unified Person Detection and Segmentation
    , m_personDetectionFPS(0)
    , m_lastPersonDetectionTime(0.0)
    , m_currentFrame()
    , m_lastSegmentedFrame()
    , m_personDetectionMutex()
    , m_personDetectionTimer()
    , m_hogDetector()
    , m_hogDetectorDaimler(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    , m_bgSubtractor()
    , m_subtractionReferenceImage()
    , m_subtractionReferenceImage2()
    , m_subtractionBlendWeight(0.5)  // Default: equal blend
    , m_useGPU(false)
    , m_useOpenCL(false)
    , m_gpuUtilized(false)
    , m_openclUtilized(false)
    // Hand detection completely removed
    , m_systemMonitor(nullptr)
    , m_personDetectionWatcher(nullptr)
    , m_lastDetections()
    , m_captureReady(false)
    , m_segmentationEnabledInCapture(false)
    , m_selectedBackgroundTemplate()
    , m_useBackgroundTemplate(false)
    , m_useDynamicVideoBackground(false)
    , m_dynamicVideoPath()
    , m_dynamicVideoCap()
    , m_dynamicVideoFrame()
    , m_dynamicGpuFrame()
    , m_videoPlaybackTimer(nullptr)
    , m_videoFrameRate(30.0)
    , m_videoFrameInterval(33)
    , m_videoPlaybackActive(false)
    , m_videoTotalFrames(0)
    , m_gpuVideoFrame()
    , m_gpuSegmentedFrame()
    , m_gpuPersonMask()
    , m_gpuBackgroundFrame()
    , m_gpuOnlyProcessingEnabled(false)
    , m_gpuProcessingAvailable(false)
    , m_gpuMemoryPool()
    , m_gpuMemoryPoolInitialized(false)
    , m_recordingThread(nullptr)
    , m_recordingFrameTimer(nullptr)
    , m_recordingMutex()
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , m_recordingThreadActive(false)
    // Hand detection UI removed
    , m_recordingFrameQueue()
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0.0)
    , m_processingFpsTimerInitialized(false)
    , m_lastCalculatedFPS(0.0)
    , m_recordingGpuBuffer()
    , m_cachedPixmap(640, 480)
    // CUDA HOG detector removed
    , m_detectionNmsOverlap(0.35)
    , m_detectionMotionOverlap(0.12)
    , m_smoothingHoldFrames(5)
    , m_smoothingHoldCounter(0)
    , m_detectionSkipInterval(2)
    // Lighting Correction Member
    , m_lightingCorrector(nullptr)
    , m_detectionSkipCounter(0)
    //  Simplified Lighting Processing (POST-PROCESSING ONLY)
    , m_lightingProcessingThread(nullptr)
    , m_lightingWatcher(nullptr)
    // Lighting Comparison Storage
    , m_originalCapturedImage()
    , m_lightingCorrectedImage()
    , m_hasLightingComparison(false)
    , m_hasVideoLightingComparison(false)
    , m_recordedPersonScaleFactor(1.0) // Initialize to default scale (100%)

{
    ui->setupUi(this);
    // Dynamic video background defaults
    m_useDynamicVideoBackground = false;
    m_dynamicVideoPath.clear();
    if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.release();
    }
    m_dynamicVideoFrame.release();
    // m_dynamicGpuReader removed - using OpenCL instead
    m_dynamicGpuFrame.release();

    // Initialize video playback timer for Phase 1
    m_videoPlaybackTimer = new QTimer(this);
    m_videoPlaybackTimer->setSingleShot(false); // Continuous timer
    m_videoPlaybackTimer->setTimerType(Qt::PreciseTimer);
    m_videoFrameRate = 30.0; // Default to 30 FPS
    m_videoFrameInterval = 33; // Default interval (1000ms / 30fps ≈ 33ms)
    m_videoPlaybackActive = false;

    // Connect video playback timer to slot
    connect(m_videoPlaybackTimer, &QTimer::timeout, this, &Capture::onVideoPlaybackTimer);

    // Phase 2A: Initialize GPU-only processing
    initializeGPUOnlyProcessing();
    
    // Initialize lighting correction system
    initializeLightingCorrection();
    
    //  Initialize async lighting system
    initializeAsyncLightingSystem();

    setContentsMargins(0, 0, 0, 0);

    // Enable keyboard focus for this widget
    setFocusPolicy(Qt::StrongFocus);
    setFocus();

    // Setup Debug Display
    setupDebugDisplay();


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

    // Watershed segmentation is used for person detection (no configuration needed)

    // Initialize status overlay
    statusOverlay = new QLabel(this);
    statusOverlay->setAlignment(Qt::AlignCenter);
    QFont statusFont = statusOverlay->font();
    statusFont.setPointSize(24);
    statusFont.setBold(true);
    statusOverlay->setFont(statusFont);
    statusOverlay->setStyleSheet(
        "color: #00ff00; "
        "background-color: rgba(0, 0, 0, 0.8); "
        "border-radius: 15px; "
        "padding: 20px 40px; "
        "border: 3px solid #00ff00; "
    );
    statusOverlay->hide();

    // Initialize loading camera label
    loadingCameraLabel = new QLabel(this);
    loadingCameraLabel->setAlignment(Qt::AlignCenter);
    QFont loadingFont = loadingCameraLabel->font();
    loadingFont.setPointSize(28);
    loadingFont.setBold(true);
    loadingCameraLabel->setFont(loadingFont);
    loadingCameraLabel->setStyleSheet(
        "color: white; "
        "background-color: rgba(0, 0, 0, 0.9); "
        "border-radius: 20px; "
        "padding: 30px 50px; "
    );
    loadingCameraLabel->setText("Loading Camera...");
    loadingCameraLabel->hide();

    // Flag to track if camera has been initialized for the first time
    m_cameraFirstInitialized = false;



    ui->videoLabel->show();

    connect(foreground, &Foreground::foregroundChanged, this, &Capture::updateForegroundOverlay);
    QPixmap overlayPixmap(selectedOverlay);
    overlayImageLabel->setPixmap(overlayPixmap);

    setupStackedLayoutHybrid();
    updateOverlayStyles();

    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());
    if (overlayImageLabel) {
        overlayImageLabel->resize(this->size());
    }

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;");
    ui->videoLabel->setScaledContents(false);
    ui->videoLabel->setAlignment(Qt::AlignCenter);

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->overlayWidget->setStyleSheet("background-color: transparent;");

    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setPageStep(tickStep);
    ui->verticalSlider->setValue(0);

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    ui->capture->setEnabled(false);

    if (cameraWorker) {
        connect(cameraWorker, &Camera::frameReady, this, &Capture::updateCameraFeed, Qt::QueuedConnection);
        connect(cameraWorker, &Camera::cameraOpened, this, &Capture::handleCameraOpened, Qt::QueuedConnection);
        connect(cameraWorker, &Camera::error, this, &Capture::handleCameraError, Qt::QueuedConnection);

        // Connect first frame signal to hide loading label
        connect(cameraWorker, &Camera::firstFrameEmitted, this, &Capture::handleFirstFrame, Qt::QueuedConnection);

        // Show loading label only on first initialization
        if (!cameraWorker->isCameraOpen() && !m_cameraFirstInitialized) {
            showLoadingCameraLabel();
            qDebug() << "First time camera initialization - showing loading label";
        }
    } else {
        qCritical() << "Capture: ERROR: cameraWorker is NULL! Camera features will not function.";
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText(
            "Camera worker not provided or is NULL.\nCannot initialize camera.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }

    // Initialize Enhanced Person Detection and Segmentation
    initializePersonDetection();

    // Initialize Hand Detection (disabled by default)
    // Hand detection completely removed
    m_captureReady = true;  // Start with capture ready
    // Initialize MediaPipe-like tracker
    // TODO: Initialize hand tracker when available



    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    recordingFrameTimer->setSingleShot(false);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);

    // Initialize and start performance timers (these are not QTimers, so they're thread-safe)
    loopTimer.start();
    frameTimer.start();

    // Debug update timer
    debugUpdateTimer = new QTimer(this);
    connect(debugUpdateTimer, &QTimer::timeout, this, &Capture::updateDebugDisplay);
    debugUpdateTimer->start(500); // Update twice per second for more responsive display

    // Initialize async processing for person detection
    m_personDetectionWatcher = new QFutureWatcher<cv::Mat>(this);
    connect(m_personDetectionWatcher, &QFutureWatcher<cv::Mat>::finished,
            this, &Capture::onPersonDetectionFinished);

    // Hand detection completely removed

    ui->capture->setEnabled(true);

    // Countdown label overlays on the overlayWidget
    countdownLabel = new QLabel(ui->overlayWidget);
    countdownLabel->setAlignment(Qt::AlignCenter);
    QFont font = countdownLabel->font();
    font.setPointSize(100);
    font.setBold(true);
    countdownLabel->setFont(font);
    countdownLabel->setStyleSheet(
        "color:white; background-color: rgba(0, 0, 0, 150); border-radius: 20px;");
    countdownLabel->setFixedSize(200, 200);
    countdownLabel->hide();

    // Flash overlay for capture animation
    flashOverlayLabel = new QLabel(ui->overlayWidget);
    flashOverlayLabel->setStyleSheet("background-color: white;");
    flashOverlayLabel->resize(ui->overlayWidget->size());
    flashOverlayLabel->move(0, 0);
    flashOverlayLabel->hide();
    flashOverlayLabel->lower(); // Place behind other overlays
    
    // Create flash animation
    QGraphicsOpacityEffect *flashEffect = new QGraphicsOpacityEffect(flashOverlayLabel);
    flashOverlayLabel->setGraphicsEffect(flashEffect);
    flashAnimation = new QPropertyAnimation(flashEffect, "opacity", this);
    flashAnimation->setDuration(150); // 150ms flash animation (faster)
    flashAnimation->setEasingCurve(QEasingCurve::OutCubic);
    
    // Connect animation finished signal to hide overlay (only once)
    connect(flashAnimation, &QPropertyAnimation::finished, this, [this]() {
        if (flashOverlayLabel) {
            flashOverlayLabel->hide();
        }
    });

    // Recording timer label (top right corner)
    recordingTimerLabel = new QLabel(ui->overlayWidget);
    recordingTimerLabel->setAlignment(Qt::AlignCenter);
    QFont timerFont = recordingTimerLabel->font();
    timerFont.setPointSize(24);
    timerFont.setBold(true);
    recordingTimerLabel->setFont(timerFont);
    recordingTimerLabel->setStyleSheet(
        "color: white; "
        "background-color: rgba(255, 0, 0, 200); "
        "border: 3px solid red; "
        "border-radius: 10px; "
        "padding: 10px 20px;");
    recordingTimerLabel->hide();

    qDebug() << "Capture UI initialized. Loading Camera...";
}

Capture::~Capture()
{
    // Disconnect all signals to prevent use-after-free
    if (cameraWorker) {
        disconnect(cameraWorker, nullptr, this, nullptr);
    }
    if (foreground) {
        disconnect(foreground, nullptr, this, nullptr);
    }
    // Hand detection removed

    if (debugUpdateTimer) {
        disconnect(debugUpdateTimer, nullptr, this, nullptr);
    }

    // Stop and delete QTimers owned by Capture
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; countdownTimer = nullptr; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; recordTimer = nullptr; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; recordingFrameTimer = nullptr; }
    if (debugUpdateTimer){ debugUpdateTimer->stop(); delete debugUpdateTimer; debugUpdateTimer = nullptr; }

    // Clean up person detection watcher
    if (m_personDetectionWatcher) {
        if (m_personDetectionWatcher->isRunning()) {
            m_personDetectionWatcher->cancel();
            m_personDetectionWatcher->waitForFinished();
        }
        delete m_personDetectionWatcher;
        m_personDetectionWatcher = nullptr;
    }

    // Delete QLabels created on the heap and parented to Capture or ui->overlayWidget
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; }
    if (statusOverlay){ delete statusOverlay; statusOverlay = nullptr; }


    if (videoLabelFPS){ delete videoLabelFPS; videoLabelFPS = nullptr; }

    // Clean up debug widgets
    if (debugWidget){ delete debugWidget; debugWidget = nullptr; }
    if (debugLabel){ delete debugLabel; debugLabel = nullptr; }
    // Hand detection UI removed

    // Hand detection removed
    
    // Clean up lighting corrector
    if (m_lightingCorrector){ 
        m_lightingCorrector->cleanup();
        delete m_lightingCorrector; 
        m_lightingCorrector = nullptr; 
    }
    
    //  Cleanup async lighting system
    cleanupAsyncLightingSystem();

    // DO NOT DELETE cameraWorker or cameraThread here.
    // They are passed in as existing objects, implying Capture does not own them.
    cameraWorker = nullptr;
    cameraThread = nullptr;

    delete ui; // Deletes the UI object and all its child widgets.
}

void Capture::handleCameraOpened(bool success,
                                 double actual_width,
                                 double actual_height,
                                 double actual_fps)
{
    Q_UNUSED(actual_width);
    Q_UNUSED(actual_height);

    // Store the actual camera FPS for correct video playback speed
    m_actualCameraFPS = actual_fps;
    qDebug() << "Capture: Stored actual camera FPS:" << m_actualCameraFPS;

    if (success) {
        qDebug() << "Capture: Camera worker reported open success. Enabling capture button.";
        ui->capture->setEnabled(true);
        ui->videoLabel->show();


    } else {
        qWarning() << "Capture: Camera worker reported open failure.";
        ui->videoLabel->show();
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText("Camera failed to open.\nCheck connection and drivers.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->capture->setEnabled(false);
    }
}

void Capture::handleCameraError(const QString &msg)
{
    QMessageBox::critical(this, "Camera Error", msg);
    ui->capture->setEnabled(false);
    qWarning() << "Capture: Camera error received:" << msg;

    ui->videoLabel->show();
    ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
    ui->videoLabel->setText(QString("Error: %1").arg(msg));
    ui->videoLabel->setAlignment(Qt::AlignCenter);
}
void Capture::updateCameraFeed(const QImage &image)
{
    // Performance measurement (thread-safe, using QElapsedTimer)
    QElapsedTimer frameTimer;
    frameTimer.start();

    if (image.isNull()) {
        qWarning() << "Capture: Received null QImage from Camera.";
        // Performance stats should still be calculated for every attempt to process a frame
        qint64 currentLoopTime = frameTimer.elapsed();
        totalTime += currentLoopTime;
        frameCount++;
        if (frameCount % 60 == 0) {
            printPerformanceStats();
        }
        return;
    }

    // Show initial performance stats when camera first starts (only once)
    static bool firstFrame = true;
    if (firstFrame) {
        qDebug() << "----------------------------------------";
        qDebug() << "=== CAMERA STARTED - PERFORMANCE MONITORING ACTIVE ===";
        qDebug() << "Performance stats will be displayed automatically every 60 frames";
        qDebug() << "Debug widget can be toggled with 'D' key";
        qDebug() << "----------------------------------------";
        firstFrame = false;

                    // Handle first frame in main thread (thread-safe)
        QMetaObject::invokeMethod(this, &Capture::handleFirstFrame, Qt::QueuedConnection);
    }

    // Store the original image for potential future use
    m_originalCameraImage = image;

    // SIMPLIFIED CAMERA FEED: Display immediately without blocking
    QImage displayImage = image;

    // Check if we have processed segmentation results to display
    if (m_segmentationEnabledInCapture && !m_lastSegmentedFrame.empty()) {
        // Convert the processed OpenCV frame back to QImage for display
        displayImage = cvMatToQImage(m_lastSegmentedFrame);
        qDebug() << "Displaying processed segmentation frame";
    } else {
        // Use original camera image
        displayImage = image;
    }

    QPixmap pixmap = QPixmap::fromImage(displayImage);

    if (ui->videoLabel) {
        QSize labelSize = ui->videoLabel->size();
        QPixmap scaledPixmap = pixmap.scaled(
            labelSize,
            Qt::KeepAspectRatioByExpanding,
            Qt::FastTransformation
        );

        // Apply person-only scaling for background template/dynamic video mode, frame scaling for other modes
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Check if we're in segmentation mode with background template or dynamic video background
            if (m_segmentationEnabledInCapture && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode or dynamic video mode, don't scale the entire frame
                // Person scaling is handled in createSegmentedFrame
                qDebug() << "Person-only scaling applied in segmentation mode (background template or dynamic video)";
            } else {
                // Apply frame scaling for other modes (normal, rectangle, black background)
                QSize originalSize = scaledPixmap.size();
                int newWidth = qRound(originalSize.width() * m_personScaleFactor);
                int newHeight = qRound(originalSize.height() * m_personScaleFactor);

                //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
                newWidth = qMax(1, newWidth);
                newHeight = qMax(1, newHeight);

                scaledPixmap = scaledPixmap.scaled(
                    newWidth, newHeight,
                    Qt::KeepAspectRatio,
                    Qt::FastTransformation
                );

                qDebug() << "Frame scaled to" << newWidth << "x" << newHeight
                         << "with factor" << m_personScaleFactor;
            }
        }

        ui->videoLabel->setPixmap(scaledPixmap);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->videoLabel->update();
    }

    // BACKGROUND PROCESSING: Move heavy work to separate threads (non-blocking)
    // For dynamic video backgrounds, throttle processing to reduce load while keeping video smooth
    int processInterval = (m_useDynamicVideoBackground && m_segmentationEnabledInCapture) ? 6 : 3;
    if (frameCount > 5 && frameCount % processInterval == 0) {
        // Process person detection in background (non-blocking) - only if segmentation is enabled
        if (m_segmentationEnabledInCapture) {
            qDebug() << "Starting person detection processing - frame:" << frameCount << "segmentation enabled:" << m_segmentationEnabledInCapture << "interval:" << processInterval;
            QMutexLocker locker(&m_personDetectionMutex);
            m_currentFrame = qImageToCvMat(image);

            // Process unified detection in background thread
            QFuture<cv::Mat> future = QtConcurrent::run([this]() {
                return processFrameWithUnifiedDetection(m_currentFrame);
            });
            m_personDetectionWatcher->setFuture(future);
        }

        // Hand detection removed
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = frameTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Calculate PROCESSING FPS (system/display rate, not camera input rate)
    // This measures how fast we're actually processing and displaying frames
    // Using instance variables and smoothing for cross-device consistency
    
    if (!m_processingFpsTimerInitialized) {
        m_processingFpsTimer.start();
        m_processingFpsTimerInitialized = true;
        m_processingFrameCount = 0;
        m_fpsHistory.clear();
        m_lastCalculatedFPS = 0.0;
    }
    m_processingFrameCount++;

    // Calculate processing FPS every second (measures actual system performance)
    // Use a more robust calculation that works consistently across different devices
    qint64 elapsedMs = m_processingFpsTimer.elapsed();
    
    // Use a minimum of 800ms to avoid timer precision issues on some devices
    // This ensures we have enough samples for accurate calculation
    if (elapsedMs >= 800) {
        double fpsDuration = elapsedMs / 1000.0;
        if (fpsDuration > 0 && m_processingFrameCount > 0) {
            // Calculate raw FPS
            double rawFPS = m_processingFrameCount / fpsDuration;
            
            // Clamp to reasonable range first (0-120 FPS) to handle timer anomalies
            rawFPS = qBound(0.0, rawFPS, 120.0);
            
            // Add to history for smoothing (helps with device-specific timer variations)
            m_fpsHistory.append(rawFPS);
            if (m_fpsHistory.size() > MAX_FPS_HISTORY) {
                m_fpsHistory.removeFirst();
            }
            
            // Calculate smoothed/averaged FPS (more accurate and consistent across devices)
            double sumFPS = 0.0;
            int validSamples = 0;
            for (double fps : m_fpsHistory) {
                if (fps > 0.0) {  // Only count valid samples
                    sumFPS += fps;
                    validSamples++;
                }
            }
            
            if (validSamples > 0) {
                double smoothedFPS = sumFPS / validSamples;
                
                // Apply additional smoothing to prevent sudden jumps between devices
                // This helps maintain consistency when switching between devices
                if (m_lastCalculatedFPS > 0.0) {
                    // Weighted average: 70% new value, 30% old value for stability
                    m_currentFPS = (smoothedFPS * 0.7) + (m_lastCalculatedFPS * 0.3);
                } else {
                    m_currentFPS = smoothedFPS;
                }
                
                m_lastCalculatedFPS = m_currentFPS;
                
                // Final clamp to ensure reasonable values
                m_currentFPS = qBound(0.0, m_currentFPS, 120.0);
                
                // Update system monitor with processing FPS (system performance metric)
                // updateCameraFeed is already called via Qt::QueuedConnection, so we're in the main thread
                // Direct call is safe here since we're in the same thread as SystemMonitor
                if (m_systemMonitor && m_currentFPS > 0.0) {
                    // Direct call is safe - updateCameraFeed runs in main thread via QueuedConnection
                    m_systemMonitor->updateFPS(m_currentFPS);
                }
            }
        }
        
        m_processingFrameCount = 0;
        m_processingFpsTimer.restart();
    }

    // Print performance stats every 60 frames (approximately every 2 seconds at 30 FPS)
    if (frameCount % 60 == 0) {
        printPerformanceStats();
    }

    // Enable processing modes after camera is stable
    if (frameCount == 50) {
        enableProcessingModes();
    }
}


void Capture::setupStackedLayoutHybrid()
{
    qDebug() << "Setting up hybrid stacked layout...";

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

    if (!stackedLayout) {
        stackedLayout = new QStackedLayout;
        stackedLayout->setStackingMode(QStackedLayout::StackAll);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        stackedLayout->setSpacing(0);

        stackedLayout->addWidget(ui->videoLabel); // Layer 0: Camera feed (background)

        stackedLayout->addWidget(ui->overlayWidget); // Layer 2: UI elements (buttons, slider)
        if (overlayImageLabel) {
            stackedLayout->addWidget(overlayImageLabel); // Layer 3: Foreground image (top)
        }

        if (layout()) {
            delete layout();
        }

        QGridLayout *mainLayout = new QGridLayout(this);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);
        mainLayout->addLayout(stackedLayout, 0, 0);
        mainLayout->setRowStretch(0, 1);
        mainLayout->setColumnStretch(0, 1);

        setLayout(mainLayout);
    }

    if (overlayImageLabel) {
        overlayImageLabel->raise();
    }
    ui->overlayWidget->raise();
    if (ui->back)
        ui->back->raise();
    if (ui->capture)
        ui->capture->raise();
    if (ui->verticalSlider)
        ui->verticalSlider->raise();
    if (countdownLabel)
        countdownLabel->raise();




    qDebug() << "Hybrid stacked layout setup complete.";
}

void Capture::updateOverlayStyles()
{
    qDebug() << "Updating overlay styles with clean professional appearance...";
    ui->back->setStyleSheet("QPushButton {"
                            "   background: transparent;"
                            "   border: none;"
                            "   color: white;"
                            "}");

    ui->capture->setStyleSheet("QPushButton {"
                               "   border-radius: 9px;"
                               "   border-bottom: 3px solid rgba(2, 2, 2, 200);"
                               "   background: rgba(11, 194, 0, 200);"
                               "   color: white;"
                               "   font-size: 16px;"
                               "   font-weight: bold;"
                               "}"
                               "QPushButton:hover {"
                               "   background: rgba(8, 154, 0, 230);"
                               "}"
                               "QPushButton:disabled {"
                               "   background: rgba(100, 100, 100, 150);"
                               "   color: rgba(200, 200, 200, 150);"
                               "   border-bottom: 3px solid rgba(50, 50, 50, 150);"
                               "}");

    ui->verticalSlider->setStyleSheet("QSlider::groove:vertical {"
                                      "   background: rgba(0, 0, 0, 80);"
                                      "   width: 30px;"
                                      "   border-radius: 15px;"
                                      "   border: none;"
                                      "}"
                                      "QSlider::handle:vertical {"
                                      "   background: rgba(13, 77, 38, 220);"
                                      "   border: 1px solid rgba(30, 144, 255, 180);"
                                      "   width: 60px;"
                                      "   height: 13px;"
                                      "   border-radius: 7px;"
                                      "   margin: 0 -15px;"
                                      "}"
                                      "QSlider::sub-page:vertical {"
                                      "   background: rgba(0, 0, 0, 60);"
                                      "   border-top-left-radius: 15px;"
                                      "   border-top-right-radius: 15px;"
                                      "   border-bottom-left-radius: 0px;"
                                      "   border-bottom-right-radius: 0px;"
                                      "}"
                                      "QSlider::add-page:vertical {"
                                      "   background: rgba(11, 194, 0, 180);"
                                      "   border-bottom-left-radius: 15px;"
                                      "   border-bottom-right-radius: 15px;"
                                      "   border-top-left-radius: 0px;"
                                      "   border-top-right-radius: 0px;"
                                      "}");

    ui->overlayWidget->setStyleSheet("background: transparent;");
    qDebug() << "Clean professional overlay styles applied";
}
void Capture::printPerformanceStats() {
    if (frameCount == 0) return; // Avoid division by zero

    double avgLoopTime = (double)totalTime / frameCount; // Average time per updateCameraFeed call

    // Calculate FPS based on the total time elapsed for the batch
    double batchDurationSeconds = (double)frameTimer.elapsed() / 1000.0;
    if (batchDurationSeconds == 0) return; // Avoid division by zero

    double measuredFPS = (double)frameCount / batchDurationSeconds;

    qDebug() << "----------------------------------------";
    qDebug() << "=== AUTOMATIC PERFORMANCE MONITORING ===";
    qDebug() << "Current FPS (real-time):" << QString::number(m_currentFPS, 'f', 1) << "FPS";
    qDebug() << "Avg loop time per frame (measured over " << frameCount << " frames):" << avgLoopTime << "ms";
    qDebug() << "Camera/Display FPS (measured over " << frameCount << " frames):" << measuredFPS << "FPS";
    qDebug() << "Frame processing efficiency:" << (avgLoopTime < 16.67 ? "GOOD" : "NEEDS OPTIMIZATION");
    qDebug() << "Person Detection Enabled:" << (m_segmentationEnabledInCapture ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "Unified Detection Enabled:" << (m_segmentationEnabledInCapture ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "GPU Acceleration:" << (m_useGPU ? "YES (OpenCL)" : "NO (CPU)");
    qDebug() << "GPU Utilized:" << (m_gpuUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "OpenCL Acceleration:" << (m_useOpenCL ? "YES (OpenCL)" : "NO (CPU)");
    qDebug() << "OpenCL Utilized:" << (m_openclUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "Person Detection FPS:" << (m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
    qDebug() << "Unified Detection FPS:" << (m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
    qDebug() << "Hand Detection FPS: N/A (DISABLED)";
    qDebug() << "Person Scale Factor:" << QString::number(m_personScaleFactor * 100, 'f', 0) << "%";
    qDebug() << "----------------------------------------";

    // Reset counters for next batch
    frameCount = 0;
    totalTime = 0;
}



void Capture::captureRecordingFrame()
{
    if (!m_isRecording)
        return;

    //  CRASH PREVENTION: Memory safety check
    const int MAX_FRAMES = 3000; // Prevent memory overflow (100 seconds at 30 FPS)
    if (m_recordedFrames.size() >= MAX_FRAMES) {
        qWarning() << " RECORDING: Maximum frame limit reached (" << MAX_FRAMES << ") - stopping recording";
        stopRecording();
        return;
    }

    //  CAPTURE FULL-RESOLUTION FRAME (not scaled display)
    QPixmap currentDisplayPixmap;

    // CRITICAL: Capture full-resolution segmented frame instead of scaled display
    // This ensures recorded frames are at full resolution, not scaled down
    if (m_segmentationEnabledInCapture) {
        QMutexLocker locker(&m_personDetectionMutex);
        if (!m_lastSegmentedFrame.empty()) {
            // Convert full-resolution segmented frame to QPixmap
            QImage fullResImage = cvMatToQImage(m_lastSegmentedFrame);
            if (!fullResImage.isNull()) {
                currentDisplayPixmap = QPixmap::fromImage(fullResImage);
                qDebug() << " FULL-RES CAPTURE: Captured full-resolution segmented frame:" 
                         << currentDisplayPixmap.width() << "x" << currentDisplayPixmap.height();
            }
        }
        locker.unlock();
    }
    
    // Fallback: Get from video label if full-res capture failed
    if (currentDisplayPixmap.isNull() && ui->videoLabel) {
        QPixmap labelPixmap = ui->videoLabel->pixmap();
        if (!labelPixmap.isNull()) {
            currentDisplayPixmap = labelPixmap;
            qDebug() << " FALLBACK CAPTURE: Using scaled display from video label";
        } else {
            qDebug() << " FALLBACK CAPTURE: Video label pixmap is null, using fallback";
        }
    }
    
    if (currentDisplayPixmap.isNull()) {
        // Fallback: Get the appropriate frame to record
        cv::Mat frameToRecord;

        // CRITICAL FIX: Use mutex when reading segmented frame from background thread
        if (m_segmentationEnabledInCapture) {
            QMutexLocker locker(&m_personDetectionMutex);
            if (!m_lastSegmentedFrame.empty()) {
                frameToRecord = m_lastSegmentedFrame.clone();
                locker.unlock();
                qDebug() << " DIRECT CAPTURE: Fallback - using segmented frame";
            } else {
                locker.unlock();
                if (!m_originalCameraImage.isNull()) {
                    frameToRecord = qImageToCvMat(m_originalCameraImage);
                    qDebug() << " DIRECT CAPTURE: Fallback - using original frame";
                } else {
                    qWarning() << " DIRECT CAPTURE: No frame available for recording";
                    return;
                }
            }
        } else if (!m_originalCameraImage.isNull()) {
            frameToRecord = qImageToCvMat(m_originalCameraImage);
            qDebug() << " DIRECT CAPTURE: Fallback - using original frame";
        } else {
            qWarning() << " DIRECT CAPTURE: No frame available for recording";
            return;
        }

        //  CRASH PREVENTION: Safe conversion to QPixmap for recording
        try {
            QImage qImage = cvMatToQImage(frameToRecord);
            if (qImage.isNull()) {
                qWarning() << " RECORDING: Failed to convert frame to QImage - skipping frame";
                return; // Skip this frame to prevent crash
            }
            currentDisplayPixmap = QPixmap::fromImage(qImage);
            if (currentDisplayPixmap.isNull()) {
                qWarning() << " RECORDING: Failed to convert QImage to QPixmap - skipping frame";
                return; // Skip this frame to prevent crash
            }
        } catch (const std::exception& e) {
            qWarning() << " RECORDING: Exception during frame conversion:" << e.what() << "- skipping frame";
            return; // Skip this frame to prevent crash
        }
    }

    //  CRASH PREVENTION: Safe frame recording
    try {
        if (currentDisplayPixmap.isNull()) {
            qWarning() << " RECORDING: Null pixmap - cannot record frame";
            return;
        }
        
        // Add the current display directly to recorded frames (no additional processing needed)
        m_recordedFrames.append(currentDisplayPixmap);
        qDebug() << " DIRECT CAPTURE: Display frame captured safely, total frames:" << m_recordedFrames.size();
    } catch (const std::exception& e) {
        qWarning() << " RECORDING: Exception during frame recording:" << e.what();
        return;
    }

    //  CRASH PREVENTION: Safe raw person data recording for post-processing
    if (m_segmentationEnabledInCapture) {
        try {
            // CRITICAL FIX: Use mutex to protect shared person data from race conditions
            // These variables are written by background segmentation thread and read by recording thread
            QMutexLocker locker(&m_personDetectionMutex);
            
            if (!m_lastRawPersonRegion.empty() && !m_lastRawPersonMask.empty()) {
                cv::Mat personRegionCopy = m_lastRawPersonRegion.clone();
                cv::Mat personMaskCopy = m_lastRawPersonMask.clone();
                
                // Release lock before doing more work
                locker.unlock();
                
                if (!personRegionCopy.empty() && !personMaskCopy.empty()) {
                    m_recordedRawPersonRegions.append(personRegionCopy);
                    m_recordedRawPersonMasks.append(personMaskCopy);
                } else {
                    qWarning() << " RECORDING: Failed to clone person data - using empty mats";
                    m_recordedRawPersonRegions.append(cv::Mat());
                    m_recordedRawPersonMasks.append(cv::Mat());
                }
            } else {
                locker.unlock();
                m_recordedRawPersonRegions.append(cv::Mat());
                m_recordedRawPersonMasks.append(cv::Mat());
            }
        } catch (const std::exception& e) {
            qWarning() << " RECORDING: Exception during person data recording:" << e.what();
            m_recordedRawPersonRegions.append(cv::Mat());
            m_recordedRawPersonMasks.append(cv::Mat());
        }
        // Background reference: use current dynamic frame if enabled, else selected template if available
        if (m_useDynamicVideoBackground) {
            // THREAD SAFETY: Lock mutex for safe frame access during recording
            QMutexLocker locker(&m_dynamicVideoMutex);
            if (!m_dynamicVideoFrame.empty()) {
                try {
                    m_recordedBackgroundFrames.append(m_dynamicVideoFrame.clone());
                } catch (const cv::Exception &e) {
                    qWarning() << " RECORDING: Failed to clone dynamic video frame:" << e.what();
                    m_recordedBackgroundFrames.append(cv::Mat());
                }
            } else {
                m_recordedBackgroundFrames.append(cv::Mat());
            }
        } else if (!m_selectedTemplate.empty()) {
            m_recordedBackgroundFrames.append(m_selectedTemplate.clone());
        } else {
            m_recordedBackgroundFrames.append(cv::Mat());
        }
    } else {
        // Keep lists aligned
        m_recordedRawPersonRegions.append(cv::Mat());
        m_recordedRawPersonMasks.append(cv::Mat());
        m_recordedBackgroundFrames.append(cv::Mat());
    }
}

void Capture::on_back_clicked()
{
    qDebug() << "DEBUG: Back button clicked in Capture! Emitting backtoPreviousPage.";
    if (countdownTimer->isActive()) {
        countdownTimer->stop();
        countdownLabel->hide();
        countdownValue = 0;
    }
    if (m_isRecording) {
        stopRecording();
    }
    ui->capture->setEnabled(true);
    
    // Reset scaling slider to default position (0 = 100% scale)
    if (ui->verticalSlider) {
        ui->verticalSlider->setValue(0);
        m_personScaleFactor = 1.0; // Reset scale factor to normal size
        qDebug() << "Scaling slider reset to default position (0 = 100% scale)";
    }
    
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    if (!cameraWorker || !cameraWorker->isCameraOpen()) {
        QMessageBox::warning(
            this,
            "Camera Not Ready",
            "Camera is not open. Please ensure it's connected and drivers are installed.");
        return;
    }

    // Hand detection removed

    // If hand detection is already enabled, then start the countdown
    ui->capture->setEnabled(false);

    // Start the countdown timer properly
    if (countdownTimer && !countdownTimer->isActive()) {
        countdownValue = 5; // 5 second countdown
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();
    countdownLabel->raise(); // Bring to front
        countdownTimer->start(1000); // 1 second intervals
        qDebug() << "Manual countdown started! 5 seconds to prepare...";
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
        qDebug() << "Countdown started! 3 seconds to prepare...";
    }
}

void Capture::updateCountdown()
{
    countdownValue--;
    if (countdownValue > 0) {
        countdownLabel->setText(QString::number(countdownValue));
    } else {
        countdownTimer->stop();
        countdownLabel->hide();
        
        if (m_currentCaptureMode == ImageCaptureMode) {
            // Show flash animation immediately
            showCaptureFlash();
            
            // Take the picture after a brief delay (during flash)
            QTimer::singleShot(50, this, [this]() {
                performImageCapture();

                // Reset capture button for next capture
                ui->capture->setEnabled(true);
            });
        } else if (m_currentCaptureMode == VideoRecordMode) {
            startRecording();
        }
    }
}

void Capture::updateRecordTimer()
{
    m_recordedSeconds++;

    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        qDebug() << "RECORDING COMPLETE: Reached video template duration ("
                 << m_currentVideoTemplate.durationSeconds << " seconds)";
        stopRecording();
    } else {
        // Update recording timer label
        if (recordingTimerLabel) {
            int remainingSeconds = m_currentVideoTemplate.durationSeconds - m_recordedSeconds;
            int minutes = remainingSeconds / 60;
            int seconds = remainingSeconds % 60;
            QString timeText = QString("%1:%2").arg(minutes, 2, 10, QChar('0'))
                                                .arg(seconds, 2, 10, QChar('0'));
            recordingTimerLabel->setText(timeText);
            recordingTimerLabel->adjustSize();
            // Reposition in case size changed
            int x = width() - recordingTimerLabel->width() - 20;
            int y = 20;
            recordingTimerLabel->move(x, y);
        }
        
        // Show progress every 2 seconds or when near completion
        if (m_recordedSeconds % 2 == 0 ||
            m_recordedSeconds >= m_currentVideoTemplate.durationSeconds - 2) {
            qDebug() << "RECORDING PROGRESS:" << m_recordedSeconds << "/"
                     << m_currentVideoTemplate.durationSeconds << "seconds";
        }
    }
}

void Capture::updateForegroundOverlay(const QString &path)
{
    qDebug() << "Foreground overlay updated to:" << path;

    if (!overlayImageLabel) {
        qWarning() << "overlayImageLabel is null! Cannot update overlay.";
        return;
    }

    QPixmap overlayPixmap(path);
    if (overlayPixmap.isNull()) {
        qWarning() << "Failed to load overlay image from path:" << path;
        overlayImageLabel->hide();
        return;
    }
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->show();

    // Emit signal to notify final interface about foreground path change
    emit foregroundPathChanged(path);
}

void Capture::on_verticalSlider_valueChanged(int value)
{
    int tickInterval = ui->verticalSlider->tickInterval();
    if (tickInterval == 0)
        return;
    int snappedValue = qRound((double) value / tickInterval) * tickInterval;
    snappedValue = qBound(ui->verticalSlider->minimum(),
                          snappedValue,
                          ui->verticalSlider->maximum());
    if (value != snappedValue) {
        ui->verticalSlider->setValue(snappedValue);
    }

    // Debug: Print actual slider values
    qDebug() << "Slider value:" << value << "Snapped value:" << snappedValue;

    // --- SCALING FUNCTIONALITY (TICK-BASED) ---
    // Convert slider value (0-100) to scale factor (1.0-0.5) in 10-unit steps
    // Since slider default is 0: 0 = 1.0x scale (normal size), 100 = 0.5x scale (50% smaller)
    // Tick intervals: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    double newScaleFactor = 1.0 - (snappedValue / 100.0) * 0.5;

    if (qAbs(newScaleFactor - m_personScaleFactor) > 0.01) { // Only update if change is significant
        m_personScaleFactor = newScaleFactor;
        qDebug() << "=== TICK-BASED SCALING ===";
        qDebug() << "Slider tick position:" << snappedValue << "/100";
        qDebug() << "Person scaling factor:" << m_personScaleFactor;
        qDebug() << "Scale percentage:" << (m_personScaleFactor * 100) << "%";
        qDebug() << "========================";

        // Trigger a refresh of the camera feed to apply the new scaling
        // The scaling will be applied in the next updateCameraFeed call
        if (!m_originalCameraImage.isNull()) {
            updateCameraFeed(m_originalCameraImage);
        }
    }
    // --- END SCALING FUNCTIONALITY ---
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
void Capture::setupDebugDisplay()
{
    qDebug() << "setupDebugDisplay called";
    
    // Create debug widget
    debugWidget = new QWidget(this);
    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 5px; }");

    QVBoxLayout *debugLayout = new QVBoxLayout(debugWidget);

    // Debug info label
    debugLabel = new QLabel("Initializing...", debugWidget);
    debugLabel->setStyleSheet("QLabel { color: white; font-size: 12px; font-weight: bold; }");
    debugLayout->addWidget(debugLabel);



    //  PERFORMANCE OPTIMIZATION: Remove lighting mode toggle button (post-processing only)
    // QPushButton *lightingModeButton = new QPushButton("Toggle Lighting Mode", debugWidget);
    // lightingModeButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #ff6600; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    // connect(lightingModeButton, &QPushButton::clicked, this, &Capture::toggleLightingMode);
    // debugLayout->addWidget(lightingModeButton);



    // Add debug widget to the main widget instead of videoLabel's layout
    debugWidget->setParent(this);
    debugWidget->move(10, 10); // Position in top-left corner
    debugWidget->resize(350, 80); // Compact size without keybind instructions
    debugWidget->raise(); // Ensure it's on top
    debugWidget->setVisible(false); // Hidden by default - press 'D' to toggle


    // Force debug display update to show correct initial state
    updateDebugDisplay();
    
    // Force another update after a short delay to ensure correct display
    QTimer::singleShot(100, [this]() {
        updateDebugDisplay();
    });

    qDebug() << "Debug display setup complete - FPS, GPU, and OpenCL status should be visible";
}

// Hand detection completely removed - enableHandDetectionForCapture removed

void Capture::setCaptureReady(bool ready)
{
    m_captureReady = ready;
    qDebug() << "Capture ready state set to:" << ready;
}

bool Capture::isCaptureReady() const
{
    return m_captureReady;
}

void Capture::resetCapturePage()
{
    qDebug() << "COMPLETE CAPTURE PAGE RESET";

    // Reset all timers and countdown
    if (countdownTimer) {
        countdownTimer->stop();
        qDebug() << "Countdown timer stopped";
    }
    if (recordTimer) {
        recordTimer->stop();
        qDebug() << "Record timer stopped";
    }
    if (recordingFrameTimer) {
        recordingFrameTimer->stop();
        qDebug() << "Recording frame timer stopped";
    }

    // Hide countdown label
    if (countdownLabel) {
        countdownLabel->hide();
        qDebug() << "Countdown label hidden";
    }

    // Reset capture button
    ui->capture->setEnabled(true);
    qDebug() << "Capture button reset to enabled";

    // Hand detection removed
    m_captureReady = true;

    // Reset segmentation state for capture interface
    enableSegmentationInCapture();
    qDebug() << "Segmentation reset for capture interface";

    // Hand detection removed
    // Hand detection removed

    // BUG FIX: Don't reset capture mode - preserve user's mode selection (static/dynamic)
    // The mode should only be changed when user explicitly selects a different template type
    qDebug() << "Preserving capture mode:" << (m_currentCaptureMode == VideoRecordMode ? "VideoRecordMode" : "ImageCaptureMode");

    // Reset video recording state (but keep the mode)
    m_recordedFrames.clear();
    m_originalRecordedFrames.clear();
    m_hasVideoLightingComparison = false;
    m_recordedSeconds = 0;

    // Reset dynamic video background to start from beginning
    // Only reset if video is already fully initialized and active
    if (m_useDynamicVideoBackground && m_videoPlaybackActive && !m_dynamicVideoPath.isEmpty()) {
        resetDynamicVideoToStart();
        qDebug() << "Dynamic video reset to start for re-recording";
    } else if (!m_dynamicVideoPath.isEmpty() && !m_useDynamicVideoBackground) {
        // Video path is set but not active yet - don't reset, let it be enabled
        qDebug() << "Dynamic video path set but not active yet - skipping reset";
    }

    // Reset scaling slider to default position (0 = 100% scale)
    if (ui->verticalSlider) {
        ui->verticalSlider->setValue(0);
        m_personScaleFactor = 1.0; // Reset scale factor to normal size
        m_recordedPersonScaleFactor = 1.0; // Reset recorded scale factor
        qDebug() << "Scaling slider reset to default position (0 = 100% scale)";
    }

    qDebug() << "Capture page completely reset - all state cleared";
}


void Capture::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    ui->videoLabel->resize(size());
    ui->overlayWidget->resize(size());
    if (overlayImageLabel) {
        overlayImageLabel->resize(size());
        overlayImageLabel->move(0, 0);
    }

    // Center the countdown label when window is resized
    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }
    
    // Resize flash overlay to cover entire screen
    if (flashOverlayLabel) {
        flashOverlayLabel->resize(size());
        flashOverlayLabel->move(0, 0);
    }
    
    // Position recording timer label in top right corner
    if (recordingTimerLabel) {
        recordingTimerLabel->adjustSize();
        int x = width() - recordingTimerLabel->width() - 20;
        int y = 20;
        recordingTimerLabel->move(x, y);
    }

    // Center the status overlay when window is resized
    if (statusOverlay && statusOverlay->isVisible()) {
        int x = (width() - statusOverlay->width()) / 2;
        int y = (height() - statusOverlay->height()) / 2;
        statusOverlay->move(x, y);
    }





    // Ensure debug widget is properly positioned (but keep current visibility state)
    if (debugWidget) {
        debugWidget->move(10, 10);
        debugWidget->raise();
        // Don't force visibility - respect user's toggle state
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
                bool isVisible = debugWidget->isVisible();
                debugWidget->setVisible(!isVisible);
                if (!isVisible) {
                    debugWidget->raise();
                    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.9); color: white; border-radius: 8px; border: 2px solid #00ff00; }");
                    qDebug() << "Debug display SHOWN - FPS, GPU, and OpenCL status visible";
                } else {
                    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 5px; }");
                    qDebug() << "Debug display HIDDEN";
                }
            }
            break;
        //  REMOVED: F and T keys since we only use post-processing lighting
        /*
        case Qt::Key_F:
            // Performance mode toggle removed - lighting always in post-processing
            break;
        case Qt::Key_T:
            // Lighting mode toggle removed - always post-processing
            break;
        */
        case Qt::Key_S:
            // Toggle segmentation on/off
            if (m_segmentationEnabledInCapture) {
                m_segmentationEnabledInCapture = false;
                qDebug() << "Segmentation DISABLED";
                
                // Clear any cached segmentation data
                m_lastSegmentedFrame = cv::Mat();
                m_lastDetections.clear();
                
                // Reset GPU utilization flags
                m_gpuUtilized = false;
                // m_cudaUtilized removed - using OpenCL
                } else {
                m_segmentationEnabledInCapture = true;
                qDebug() << "Segmentation ENABLED";
                }

            // Show status overlay
            if (statusOverlay) {
                QString statusText = m_segmentationEnabledInCapture ? 
                    "SEGMENTATION: ENABLED" : "SEGMENTATION: DISABLED";
                statusOverlay->setText(statusText);
                statusOverlay->resize(statusOverlay->sizeHint());
                int x = (width() - statusOverlay->width()) / 2;
                int y = (height() - statusOverlay->height()) / 2;
                statusOverlay->move(x, y);
                statusOverlay->show();
                statusOverlay->raise();
                QTimer::singleShot(2000, [this]() {
                    if (statusOverlay) {
                        statusOverlay->hide();
                    }
                });
            }

            // Update debug display
            updateDebugDisplay();
                            break;
        case Qt::Key_H:
            // Toggle hand detection
            // Hand detection removed - 'H' key disabled
            updateDebugDisplay();
            break;
        case Qt::Key_F12:
            // Temporarily disabled debug frame save
            qDebug() << "Debug frame save disabled";
            break;

        default:
            QWidget::keyPressEvent(event);
    }
}
void Capture::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    qDebug() << "Capture widget shown - camera should already be running continuously";

    // Camera is now managed continuously by brbooth.cpp, no need to start it here
    // Just enable segmentation after a short delay (hand detection is user-controlled)
    QTimer::singleShot(100, [this]() {
        // Hand detection removed
        enableSegmentationInCapture();
        qDebug() << "Segmentation ENABLED for capture interface";
        qDebug() << "Hand detection is DISABLED by default - use debug menu to enable";

        // Restore dynamic video background if a path was previously set
        if (!m_dynamicVideoPath.isEmpty() && !m_useDynamicVideoBackground) {
            qDebug() << "Restoring dynamic video background:" << m_dynamicVideoPath;
            enableDynamicVideoBackground(m_dynamicVideoPath);
        }
    });
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    qDebug() << "Capture widget hidden - OPTIMIZED camera and hand detection shutdown";

    // Keep hand detection state as set by user (don't auto-disable)
    qDebug() << "Hand detection state preserved during page transition";

    // Disable segmentation when leaving capture page
    disableSegmentationOutsideCapture();
    qDebug() << "Segmentation DISABLED outside capture interface";

    // Note: Camera is now controlled by the main page change handler in brbooth.cpp
    // This prevents lag when returning to capture page
}

// Hand detection completely removed - drawHandBoundingBoxes implementation removed



void Capture::updateDebugDisplay()
{
    // Debug output to verify the method is being called
    static int updateCount = 0;
    updateCount++;
    
    // Always log hand detection state for debugging
        qDebug() << "updateDebugDisplay #" << updateCount;
    
    if (updateCount % 10 == 0) { // Log every 5 seconds (10 updates * 500ms)
        qDebug() << "Debug display update #" << updateCount << "FPS:" << m_currentFPS << "GPU:" << m_useGPU << "OpenCL:" << m_useOpenCL;
    }

    if (debugLabel) {
        QString peopleDetected = QString::number(m_lastDetections.size());
        QString segmentationStatus = m_segmentationEnabledInCapture ? "ON" : "OFF";
        QString aiFPS = m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "0.0";
        
        QString debugInfo = QString("FPS: %1 | People: %2 | Seg: %3 | Person FPS: %4")
                           .arg(QString::number(m_currentFPS, 'f', 1))
                           .arg(peopleDetected)
                           .arg(segmentationStatus)
                           .arg(aiFPS);
        debugLabel->setText(debugInfo);
    }


}
void Capture::startRecording()
{
    if (!cameraWorker->isCameraOpen()) {
        qWarning() << "Cannot start recording: Camera not opened by worker.";
        ui->capture->setEnabled(true);
        return;
    }

    // CRASH FIX: Ensure background subtractor is initialized before recording in segmentation mode
    // OPTIMIZATION: Only create if not already initialized (should be initialized in initializePersonDetection)
    if (m_segmentationEnabledInCapture && m_bgSubtractor.empty()) {
        qWarning() << "Background subtractor not initialized, initializing now...";
        // Ensure person detection is initialized first (this will also initialize background subtractor)
        qWarning() << "Person detection may not be initialized, calling initializePersonDetection()...";
        initializePersonDetection();
        // Double-check after initialization
        if (m_bgSubtractor.empty()) {
            qWarning() << "Background subtractor still not initialized after initializePersonDetection, creating directly...";
            m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
            if (m_bgSubtractor.empty()) {
                qWarning() << "Failed to create background subtractor!";
                QMessageBox::warning(this, "Recording Error", "Failed to initialize segmentation system. Please restart the application.");
                ui->capture->setEnabled(true);
                return;
            }
        }
    }

    // CRASH FIX: Validate dynamic video is ready if in dynamic mode
    if (m_useDynamicVideoBackground && m_segmentationEnabledInCapture) {
        if (!m_videoPlaybackActive) {
            qWarning() << "Dynamic video playback not active, attempting to restart...";
            if (m_videoPlaybackTimer && m_videoFrameInterval > 0) {
                m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
                m_videoPlaybackTimer->start();
                m_videoPlaybackActive = true;
                qDebug() << "Video playback timer restarted";
            } else {
                qWarning() << "Cannot start video playback - timer or interval invalid!";
                QMessageBox::warning(this, "Recording Error", "Dynamic video background is not ready. Please return to video selection and try again.");
                ui->capture->setEnabled(true);
                return;
            }
        }
        if (m_dynamicVideoFrame.empty()) {
            qWarning() << "Dynamic video frame is empty, recording may have issues";
        }
    }

    m_recordedFrames.clear();
    m_originalRecordedFrames.clear();
    m_hasVideoLightingComparison = false;
    m_isRecording = true;
    m_recordedSeconds = 0;
    
    // SCALING PRESERVATION: Store the current scaling factor for post-processing
    m_recordedPersonScaleFactor = m_personScaleFactor;
    qDebug() << "SCALING: Stored scaling factor" << m_recordedPersonScaleFactor << "for post-processing";
    
    // Show recording indicators: red border and timer
    if (ui->videoLabel) {
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px; border: 5px solid red;");
    }
    
    // Show and initialize recording timer label
    if (recordingTimerLabel) {
        int totalSeconds = m_currentVideoTemplate.durationSeconds;
        int minutes = totalSeconds / 60;
        int seconds = totalSeconds % 60;
        QString timeText = QString("%1:%2").arg(minutes, 2, 10, QChar('0'))
                                            .arg(seconds, 2, 10, QChar('0'));
        recordingTimerLabel->setText(timeText);
        recordingTimerLabel->adjustSize();
        // Position in top right
        int x = width() - recordingTimerLabel->width() - 20;
        int y = 20;
        recordingTimerLabel->move(x, y);
        recordingTimerLabel->show();
        recordingTimerLabel->raise();
    }

    // Choose recording FPS: use template's native FPS for dynamic video backgrounds, else camera FPS
    if (m_useDynamicVideoBackground && m_videoFrameRate > 0.0) {
        m_adjustedRecordingFPS = m_videoFrameRate;
    } else {
        m_adjustedRecordingFPS = m_actualCameraFPS;
    }

    qDebug() << " DIRECT CAPTURE RECORDING: Starting with FPS:" << m_adjustedRecordingFPS;
    qDebug() << "  - Scale factor:" << m_personScaleFactor;
    qDebug() << "  - Capturing exact display content";
    qDebug() << "  - Recording duration:" << m_currentVideoTemplate.durationSeconds << "seconds";
    qDebug() << "  - Video template:" << m_currentVideoTemplate.name;
    qDebug() << "  - Target frames:" << m_videoTotalFrames;

    // RECORDING OPTIMIZATION: Disable frame skipping during recording for smooth capture
    m_detectionSkipCounter = 0; // Force detection every frame during recording
    qDebug() << "RECORDING: Disabled detection frame skipping for smooth capture";

    int frameIntervalMs = qMax(1, static_cast<int>(1000.0 / m_adjustedRecordingFPS));

    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
    qDebug() << " DIRECT CAPTURE RECORDING: Started at " + QString::number(m_adjustedRecordingFPS)
                    + " frames/sec (interval: " + QString::number(frameIntervalMs) + "ms)";

    // Pre-calculate label size for better performance during recording
    m_cachedLabelSize = ui->videoLabel->size();

    // Reset dynamic video to start when recording begins
    if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
        resetDynamicVideoToStart();
        qDebug() << "Dynamic video reset to start for new recording";
    }
}

void Capture::stopRecording()
{
    if (!m_isRecording)
        return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;
    
    // Hide recording indicators: remove red border and hide timer
    if (ui->videoLabel) {
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;"); // Remove red border, keep other styles
    }
    
    if (recordingTimerLabel) {
        recordingTimerLabel->hide();
    }

    qDebug() << " DIRECT CAPTURE RECORDING: Stopped. Captured " + QString::number(m_recordedFrames.size())
                    + " frames.";

    // SYNCHRONIZATION: Cap recorded frames to match template frame count exactly for perfect timing
    if (m_useDynamicVideoBackground && m_videoTotalFrames > 0 && m_recordedFrames.size() > m_videoTotalFrames) {
        qDebug() << "SYNC: Trimming recorded frames from" << m_recordedFrames.size() << "to" << m_videoTotalFrames << "to match template";
        while (m_recordedFrames.size() > m_videoTotalFrames) {
            m_recordedFrames.removeLast();
        }
    }

    if (!m_recordedFrames.isEmpty()) {
        // Store original frames before lighting correction (just like static mode)
        m_originalRecordedFrames = m_recordedFrames;
        m_hasVideoLightingComparison = (m_lightingCorrector != nullptr);
        
        // NEW FLOW: Send frames to confirm page FIRST for user confirmation
        qDebug() << "Sending recorded frames to confirm page for user review";
        qDebug() << "Recorded frames:" << m_recordedFrames.size() << "at FPS:" << m_adjustedRecordingFPS;
        qDebug() << "Video template FPS:" << m_videoFrameRate;
        emit videoRecordedForConfirm(m_recordedFrames, m_adjustedRecordingFPS);
        
        // Show confirm page (user can review before post-processing)
        qDebug() << "Showing confirm page - waiting for user confirmation";
        emit showConfirmPage();
        
        // POST-PROCESSING NOW HAPPENS IN startPostProcessing() - AFTER USER CONFIRMS
    }

    // Re-enable capture button for re-recording
    ui->capture->setEnabled(true);
}

void Capture::startPostProcessing()
{
    qDebug() << " Starting post-processing after user confirmation";
    
    // CRASH PREVENTION: Validate recorded frames
    if (m_recordedFrames.isEmpty()) {
        qWarning() << " No recorded frames available for post-processing";
        return;
    }
    
    // CRASH PREVENTION: Validate original recorded frames
    if (m_originalRecordedFrames.isEmpty()) {
        qWarning() << " No original recorded frames available - using recorded frames";
        m_originalRecordedFrames = m_recordedFrames;
    }
    
    // CRASH PREVENTION: Validate FPS
    if (m_adjustedRecordingFPS <= 0) {
        qWarning() << " Invalid FPS:" << m_adjustedRecordingFPS << "- using default 30";
        m_adjustedRecordingFPS = 30.0;
    }
    
    try {
        // Send original frames to loading page for background preview
        qDebug() << "Sending original frames to loading page for background preview";
        emit videoRecordedForLoading(m_originalRecordedFrames, m_adjustedRecordingFPS);
        
        //  Show loading UI (now has original frame background)
        qDebug() << "Showing loading UI with original frame background";
        emit showLoadingPage();
        
        if (m_hasVideoLightingComparison) {
            qDebug() << "Starting ASYNC lighting correction for enhanced output";
            
            //  ASYNC POST-PROCESSING: Apply lighting correction in background thread
            qDebug() << "Post-processing recorded video with lighting correction (per-frame) - ASYNC MODE";
            
            // Check if watcher is available
            if (!m_lightingWatcher) {
                qWarning() << " Lighting watcher not initialized! Falling back to synchronous processing";
                try {
                    QList<QPixmap> processedFrames = processRecordedVideoWithLighting(m_recordedFrames, m_adjustedRecordingFPS);
                    if (!processedFrames.isEmpty()) {
                        emit videoRecordedWithComparison(processedFrames, m_originalRecordedFrames, m_adjustedRecordingFPS);
                    } else {
                        qWarning() << " Processed frames empty, using original frames";
                        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
                    }
                    emit showFinalOutputPage();
                } catch (const std::exception& e) {
                    qWarning() << " Synchronous processing failed:" << e.what() << "- using original frames";
                    emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
                    emit showFinalOutputPage();
                }
                return;
            }
            
            // CRASH PREVENTION: Check if previous processing is still running
            if (m_lightingWatcher->isRunning()) {
                qWarning() << " Previous processing still running, cancelling and restarting";
                m_lightingWatcher->cancel();
                m_lightingWatcher->waitForFinished(); // Wait for completion
            }
            
            // CRASH PREVENTION: Make local copies of data needed for processing
            QList<QPixmap> localRecordedFrames = m_recordedFrames;
            double localFPS = m_adjustedRecordingFPS;
            
            // Run processing in background thread using QtConcurrent
            QFuture<QList<QPixmap>> future = QtConcurrent::run(
                [this, localRecordedFrames, localFPS]() {
                    try {
                        return processRecordedVideoWithLighting(localRecordedFrames, localFPS);
                    } catch (const std::exception& e) {
                        qWarning() << " Exception in background processing:" << e.what();
                        return QList<QPixmap>(); // Return empty list on error
                    } catch (...) {
                        qWarning() << " Unknown exception in background processing";
                        return QList<QPixmap>(); // Return empty list on error
                    }
                }
            );
            
            // Set the future on the watcher (will trigger onVideoProcessingFinished when done)
            m_lightingWatcher->setFuture(future);
            
            qDebug() << " Video processing started in background thread - UI will remain responsive";
            
        } else {
            qDebug() << "No lighting correction needed - sending original frames to final output";
            
            // Send original frames to final output page
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            
            // Show final output page immediately
            emit showFinalOutputPage();
            qDebug() << " No processing needed - showing final output page";
        }
    } catch (const std::exception& e) {
        qWarning() << " Exception in startPostProcessing:" << e.what();
        // Fallback: send original frames
        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
        emit showFinalOutputPage();
    } catch (...) {
        qWarning() << " Unknown exception in startPostProcessing";
        // Fallback: send original frames
        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
        emit showFinalOutputPage();
    }
}

void Capture::showCaptureFlash()
{
    if (!flashOverlayLabel || !flashAnimation) {
        return;
    }
    
    // Ensure flash overlay is properly sized
    flashOverlayLabel->resize(ui->overlayWidget->size());
    flashOverlayLabel->move(0, 0);
    
    // Show flash overlay
    flashOverlayLabel->show();
    flashOverlayLabel->raise(); // Bring to front
    
    // Set initial opacity to 0
    QGraphicsOpacityEffect *effect = qobject_cast<QGraphicsOpacityEffect*>(flashOverlayLabel->graphicsEffect());
    if (effect) {
        effect->setOpacity(0.0);
        
        // Stop any ongoing animation
        if (flashAnimation->state() == QPropertyAnimation::Running) {
            flashAnimation->stop();
        }
        
        // Animate flash: fade in quickly, then fade out
        flashAnimation->setStartValue(0.0);
        flashAnimation->setKeyValueAt(0.3, 0.8); // Peak at 30% of duration
        flashAnimation->setEndValue(0.0);
        flashAnimation->start();
    }
}

void Capture::performImageCapture()
{
    // Capture the processed frame that includes background template and segmentation
    if (!m_originalCameraImage.isNull()) {
        QPixmap cameraPixmap;
        QSize labelSize = ui->videoLabel->size();

        // Check if we have a processed segmented frame to capture
        if (m_segmentationEnabledInCapture && !m_lastSegmentedFrame.empty()) {
            // Store original segmented frame for comparison
            cv::Mat originalSegmentedFrame = m_lastSegmentedFrame.clone();
            
            // Apply person-only lighting correction using template reference
            cv::Mat lightingCorrectedFrame;
            qDebug() << "LIGHTING DEBUG - Segmentation mode detected";
            qDebug() << "LIGHTING DEBUG - Background template enabled:" << m_useBackgroundTemplate;
            qDebug() << "LIGHTING DEBUG - Template path:" << m_selectedBackgroundTemplate;
            qDebug() << "LIGHTING DEBUG - Lighting corrector exists:" << (m_lightingCorrector != nullptr);
            
            // POST-PROCESSING: Apply lighting to raw person data and re-composite
            qDebug() << "POST-PROCESSING: Apply lighting to raw person data";
            lightingCorrectedFrame = applyPostProcessingLighting();
            qDebug() << "Post-processing lighting applied";
            
            // Store both versions for saving
            m_originalCapturedImage = originalSegmentedFrame;
            m_lightingCorrectedImage = lightingCorrectedFrame;
            m_hasLightingComparison = true;
            
            qDebug() << "FORCED: Stored both original and lighting-corrected versions for comparison";
            
            // Convert the processed OpenCV frame to QImage for capture
            QImage processedImage = cvMatToQImage(lightingCorrectedFrame);
            cameraPixmap = QPixmap::fromImage(processedImage);
            qDebug() << "Capturing processed segmented frame with background template and person lighting correction";
        } else {
            // For normal mode, apply global lighting correction if enabled
            cv::Mat originalFrame = qImageToCvMat(m_originalCameraImage);
            cv::Mat lightingCorrectedFrame;
            qDebug() << "LIGHTING DEBUG - Normal mode detected";
            qDebug() << "LIGHTING DEBUG - Lighting corrector exists:" << (m_lightingCorrector != nullptr);
            
            if (m_lightingCorrector) {
                lightingCorrectedFrame = m_lightingCorrector->applyGlobalLightingCorrection(originalFrame);
                qDebug() << "Applied global lighting correction (normal mode)";
            } else {
                lightingCorrectedFrame = originalFrame;
                qDebug() << "No lighting correction applied (normal mode)";
            }
            
            // Convert back to QImage
            QImage correctedImage = cvMatToQImage(lightingCorrectedFrame);
            cameraPixmap = QPixmap::fromImage(correctedImage);
            qDebug() << "Capturing original camera frame with lighting correction (normal mode)";
        }

        // Apply the same scaling logic as the live display
        // First scale to fit the label - use FastTransformation for better performance
        QPixmap scaledPixmap = cameraPixmap.scaled(
            labelSize,
            Qt::KeepAspectRatioByExpanding,
            Qt::FastTransformation
        );

        // Apply person-only scaling for background template/dynamic video mode, frame scaling for other modes
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Check if we're in segmentation mode with background template or dynamic video background
            if (m_segmentationEnabledInCapture && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode or dynamic video mode, don't scale the entire frame
                // Person scaling is already applied in createSegmentedFrame
                qDebug() << "Person-only scaling preserved in final output (background template or dynamic video mode)";
            } else {
                // Apply frame scaling for other modes (normal, rectangle, black background)
                QSize originalSize = scaledPixmap.size();
                int newWidth = qRound(originalSize.width() * m_personScaleFactor);
                int newHeight = qRound(originalSize.height() * m_personScaleFactor);

                scaledPixmap = scaledPixmap.scaled(
                    newWidth, newHeight,
                    Qt::KeepAspectRatio,
                    Qt::FastTransformation
                );

                qDebug() << "Frame scaled in final output to" << newWidth << "x" << newHeight
                         << "with factor" << m_personScaleFactor;
            }
        }

        m_capturedImage = scaledPixmap;
        
        // LOADING UI INTEGRATION: Show loading page with original frame background
        if (m_hasLightingComparison && !m_originalCapturedImage.empty()) {
            // Convert original image to QPixmap for preview and comparison
            QImage originalQImage = cvMatToQImage(m_originalCapturedImage);
            QPixmap originalPixmap = QPixmap::fromImage(originalQImage);
            
            // Apply same scaling to original image
            QPixmap scaledOriginalPixmap = originalPixmap.scaled(
                labelSize,
                Qt::KeepAspectRatioByExpanding,
                Qt::FastTransformation
            );
            
            // Apply person scaling if needed
            if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
                if (m_segmentationEnabledInCapture && ((m_useBackgroundTemplate &&
                    !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                    qDebug() << "Person-only scaling preserved in original output";
                } else {
                    QSize originalSize = scaledOriginalPixmap.size();
                    int newWidth = qRound(originalSize.width() * m_personScaleFactor);
                    int newHeight = qRound(originalSize.height() * m_personScaleFactor);
                    scaledOriginalPixmap = scaledOriginalPixmap.scaled(
                        newWidth, newHeight,
                        Qt::KeepAspectRatio,
                        Qt::FastTransformation
                    );
                }
            }
            
            // FIRST: Send original image to loading page for background preview
            qDebug() << "STATIC: Sending original image to loading page for background";
            emit imageCapturedForLoading(scaledOriginalPixmap);
            
            // THEN: Show loading UI with original image background
            qDebug() << "STATIC: Showing loading UI with original image background";
        emit showLoadingPage();
        
        // START: Progress simulation for static processing
        emit videoProcessingProgress(0);
        
        // PROGRESS: Simulate processing stages with realistic timing
        QTimer::singleShot(200, [this]() {
            emit videoProcessingProgress(25);
                qDebug() << "STATIC: Processing progress 25%";
        });
        
        QTimer::singleShot(600, [this]() {
            emit videoProcessingProgress(50);
                qDebug() << "STATIC: Processing progress 50%";
        });
        
        QTimer::singleShot(1000, [this]() {
            emit videoProcessingProgress(75);
                qDebug() << "STATIC: Processing progress 75%";
        });
        
        QTimer::singleShot(1400, [this]() {
            emit videoProcessingProgress(90);
                qDebug() << "STATIC: Processing progress 90%";
        });
        
            // FINALLY: Send processed image to final output page (after processing simulation)
            QTimer::singleShot(1800, [this, scaledOriginalPixmap]() {
            emit videoProcessingProgress(100);
                qDebug() << "STATIC: Processing complete - sending to final output";
                emit imageCapturedWithComparison(m_capturedImage, scaledOriginalPixmap);
                    emit showFinalOutputPage();
            });
            
            qDebug() << "Emitted static image with loading UI flow - corrected and original versions";
        } else {
            // No comparison available - send to loading page first, then final page
            qDebug() << "STATIC: Sending single image to loading page";
        emit imageCapturedForLoading(m_capturedImage);
        
            qDebug() << "STATIC: Showing loading UI";
        emit showLoadingPage();
        
            // Send to final output page with progress simulation
        // START: Progress simulation for static processing
        emit videoProcessingProgress(0);
        
        // PROGRESS: Simulate processing stages with realistic timing
        QTimer::singleShot(200, [this]() {
            emit videoProcessingProgress(25);
                qDebug() << "STATIC: Processing progress 25%";
        });
        
        QTimer::singleShot(600, [this]() {
            emit videoProcessingProgress(50);
                qDebug() << "STATIC: Processing progress 50%";
        });
        
        QTimer::singleShot(1000, [this]() {
            emit videoProcessingProgress(75);
                qDebug() << "STATIC: Processing progress 75%";
        });
        
        QTimer::singleShot(1400, [this]() {
            emit videoProcessingProgress(90);
                qDebug() << "STATIC: Processing progress 90%";
        });
        
        // Send to final output page after processing simulation
        QTimer::singleShot(1800, [this]() {
            emit videoProcessingProgress(100);
                qDebug() << "STATIC: Processing complete - sending single image to final output";
            emit imageCaptured(m_capturedImage);
            emit showFinalOutputPage();
        });
        
            qDebug() << "Emitted single image with loading UI flow";
        }
        
        qDebug() << "Image captured (includes background template and segmentation).";
        qDebug() << "Captured image size:" << m_capturedImage.size() << "Original size:" << cameraPixmap.size();
    } else {
        qWarning() << "Failed to capture image: original camera image is empty.";
        QMessageBox::warning(this, "Capture Failed", "No camera feed available to capture an image.");
    }
}


void Capture::setCaptureMode(CaptureMode mode)
{
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to:" << static_cast<int>(mode);
}

void Capture::setVideoTemplate(const VideoTemplate &templateData)
{
    m_currentVideoTemplate = templateData;
    qDebug() << "VIDEO TEMPLATE SET:" << templateData.name;
    qDebug() << "  - Duration:" << templateData.durationSeconds << "seconds";
    qDebug() << "  - Recording will automatically stop after" << templateData.durationSeconds << "seconds";

    // Reset frame counter to ensure smooth initial processing
    frameCount = 0;

    // Ensure segmentation is properly initialized
    if (m_segmentationEnabledInCapture) {
        qDebug() << "Segmentation enabled for template transition";
    }
}

void Capture::enableDynamicVideoBackground(const QString &videoPath)
{
    qDebug() << "enableDynamicVideoBackground called with path:" << videoPath;
    
    // Close previous if open
    if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.release();
    }
    // m_dynamicGpuReader removed - using CPU VideoCapture for OpenCL
    
    // Clean up the path and verify file exists
    QString cleanPath = QDir::cleanPath(videoPath);
    m_dynamicVideoPath = cleanPath;
    m_useDynamicVideoBackground = false;
    
    qDebug() << "Cleaned path:" << cleanPath;
    qDebug() << "File exists check:" << QFile::exists(cleanPath);
    
    if (!QFile::exists(cleanPath)) {
        qWarning() << "Video file does not exist:" << cleanPath;
        return;
    }

    bool opened = false;

    // CUDA VideoReader removed - using CPU VideoCapture (OpenCL doesn't have direct video decoding)

    // CPU fallback using multiple backends
    if (!opened) {
        std::vector<int> backends = {
            cv::CAP_MSMF,
            cv::CAP_FFMPEG,
            cv::CAP_DSHOW,
            cv::CAP_ANY
        };

        for (int backend : backends) {
            qDebug() << "Trying CPU backend:" << backend;
            m_dynamicVideoCap.open(cleanPath.toStdString(), backend);
            if (m_dynamicVideoCap.isOpened()) {
                opened = true;
                qDebug() << "Successfully opened video with CPU backend:" << backend;
                break;
            }
        }
    }

    if (!opened) {
        qWarning() << "Failed to open dynamic video with both GPU and CPU readers:" << cleanPath;
        return;
    }

    // AUTOMATIC DURATION DETECTION: Get video duration and update template
    double videoDurationSeconds = 0.0;
    if (m_dynamicVideoCap.isOpened()) {
        // Get total frame count and FPS to calculate duration
        double totalFrames = m_dynamicVideoCap.get(cv::CAP_PROP_FRAME_COUNT);
        m_videoFrameRate = m_dynamicVideoCap.get(cv::CAP_PROP_FPS);
        m_videoTotalFrames = static_cast<int>(totalFrames);
        if (m_videoFrameRate > 0 && totalFrames > 0) {
            videoDurationSeconds = totalFrames / m_videoFrameRate;
            qDebug() << "VIDEO DURATION DETECTED (CPU):" << videoDurationSeconds << "seconds";
            qDebug() << "  - Total frames:" << totalFrames;
            qDebug() << "  - Frame rate:" << m_videoFrameRate << "FPS";
        }
    } else if (false) { // CUDA VideoReader removed
        // Probe container FPS and frame count via a short-lived CPU reader to drive GPU playback at native speed
        double probedFps = 0.0;
        double probedTotal = 0.0;
        {
            cv::VideoCapture probe;
            // Prefer FFMPEG for accurate container metadata
            if (!probe.open(cleanPath.toStdString(), cv::CAP_FFMPEG)) {
                probe.open(cleanPath.toStdString(), cv::CAP_MSMF);
            }
            if (probe.isOpened()) {
                probedFps = probe.get(cv::CAP_PROP_FPS);
                probedTotal = probe.get(cv::CAP_PROP_FRAME_COUNT);
                probe.release();
            }
        }
        if (probedFps > 0.0) {
            m_videoFrameRate = probedFps;
            m_videoTotalFrames = static_cast<int>(probedTotal);
            if (probedTotal > 0.0) {
                videoDurationSeconds = probedTotal / probedFps;
                qDebug() << "VIDEO DURATION DETECTED (GPU probe):" << videoDurationSeconds << "seconds";
                qDebug() << "VIDEO FRAME COUNT:" << m_videoTotalFrames << "frames";
            }
            qDebug() << "NVDEC playback FPS set to native:" << m_videoFrameRate;
        } else {
            // Fallback if probe failed
            m_videoFrameRate = 30.0;
            m_videoTotalFrames = 0;
            qDebug() << "Using default FPS (30) for video reader; probe failed";
        }
    }

    // Update video template with detected duration
    if (videoDurationSeconds > 0) {
        QString templateName = QFileInfo(m_dynamicVideoPath).baseName();
        m_currentVideoTemplate = VideoTemplate(templateName, static_cast<int>(videoDurationSeconds));
        qDebug() << "RECORDING DURATION UPDATED:" << m_currentVideoTemplate.durationSeconds << "seconds";
        qDebug() << "  - Template name:" << m_currentVideoTemplate.name;
        qDebug() << "  - Recording will automatically stop when video template ends";
    } else {
        // Fallback to default duration if detection fails
        m_currentVideoTemplate = VideoTemplate("Dynamic Template", 10);
        qWarning() << "Could not detect video duration, using default 10 seconds";
    }

    // Phase 1: Detect video frame rate for synchronization
    if (m_dynamicVideoCap.isOpened()) {
        if (m_videoFrameRate <= 0) {
            m_videoFrameRate = 30.0; // Fallback to 30 FPS if detection fails
        }
        m_videoFrameInterval = qRound(1000.0 / m_videoFrameRate); // Convert to milliseconds
        
        // Ensure minimum responsiveness - don't go below 16ms (60 FPS max)
        if (m_videoFrameInterval < 16) {
            m_videoFrameInterval = 16;
        }
        
        qDebug() << "Video frame rate detected (CPU):" << m_videoFrameRate << "FPS, interval:" << m_videoFrameInterval << "ms";
    } else if (false) { // CUDA VideoReader removed
        // For GPU reader, use probed/native frame rate if available
        if (m_videoFrameRate <= 0) m_videoFrameRate = 30.0;
        m_videoFrameInterval = qRound(1000.0 / m_videoFrameRate);
        if (m_videoFrameInterval < 16) m_videoFrameInterval = 16;
        qDebug() << "Using NVDEC playback frame rate:" << m_videoFrameRate << "FPS, interval:" << m_videoFrameInterval << "ms";
    }

    // Prime first frame using available reader. Keep GPU path GPU-only to minimize CPU.
    cv::Mat first;
    bool frameRead = false;

    // CUDA VideoReader removed - using CPU VideoCapture

    if (!frameRead && m_dynamicVideoCap.isOpened()) {
        frameRead = m_dynamicVideoCap.read(first);
        if (frameRead && !first.empty()) {
            qDebug() << "First frame size (CPU):" << first.cols << "x" << first.rows;
        }
    }

    if (frameRead && !first.empty()) {
        if (!first.empty()) {
            m_dynamicVideoFrame = first.clone();
        }
        m_useDynamicVideoBackground = true;
        qDebug() << "Dynamic video background enabled:" << m_dynamicVideoPath;

        if (m_videoPlaybackTimer) {
            m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
            m_videoPlaybackTimer->start();
            m_videoPlaybackActive = true;
            qDebug() << "Video playback timer started with interval:" << m_videoFrameInterval << "ms";
        }
    } else {
        qWarning() << "Could not read first frame from dynamic background video:" << m_dynamicVideoPath;
        if (m_dynamicVideoCap.isOpened()) m_dynamicVideoCap.release();
        // m_dynamicGpuReader removed
    }

    // When using dynamic video, disable static background template and foreground overlay
    m_useBackgroundTemplate = false;
    m_selectedBackgroundTemplate.clear();
    if (overlayImageLabel) {
        overlayImageLabel->hide();
    }
    
    // Clear foreground path to prevent it from being passed to final output
    // This ensures foreground templates don't appear in final output when using dynamic templates
    if (foreground) {
        foreground->setSelectedForeground("");
        qDebug() << "Dynamic template enabled - foreground template cleared to prevent visibility in final output";
    }
}

void Capture::disableDynamicVideoBackground()
{
    // Phase 1: Stop video playback timer
    if (m_videoPlaybackTimer && m_videoPlaybackActive) {
        m_videoPlaybackTimer->stop();
        m_videoPlaybackActive = false;
        qDebug() << "Video playback timer stopped";
    }

    if (m_dynamicVideoCap.isOpened()) m_dynamicVideoCap.release();
    // m_dynamicGpuReader removed
    m_dynamicVideoFrame.release();
    // NOTE: Do NOT clear m_dynamicVideoPath here to preserve selection for restoration
    // m_dynamicVideoPath.clear(); 
    m_useDynamicVideoBackground = false;
    
    // Note: We don't restore the foreground template here automatically
    // as the user may have changed their selection while dynamic was active
    // The foreground template will be restored when the user navigates back to foreground selection
}

bool Capture::isDynamicVideoBackgroundEnabled() const
{
    return m_useDynamicVideoBackground;
}

void Capture::clearDynamicVideoPath()
{
    // Clear the stored dynamic video path to prevent auto-restoration
    m_dynamicVideoPath.clear();
    qDebug() << "Cleared dynamic video path for mode switching";
}
// Phase 1: Video Playback Timer Slot - Advances video frames at native frame rate
void Capture::onVideoPlaybackTimer()
{
    if (!m_useDynamicVideoBackground || !m_videoPlaybackActive) {
        return;
    }

    // THREAD SAFETY: Use tryLock to avoid blocking if processing is still ongoing
    if (!m_dynamicVideoMutex.tryLock()) {
        qDebug() << "Skipping frame advance - previous frame still processing";
        return; // Skip this frame to maintain timing
    }

    cv::Mat nextFrame;
    bool frameRead = false;

    // CUDA VideoReader removed - using CPU VideoCapture for OpenCL

    // CPU fallback
    if (!frameRead) {
        if (!m_dynamicVideoCap.isOpened()) {
            // Reopen with preferred backend
            if (!m_dynamicVideoPath.isEmpty()) {
                m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                if (!m_dynamicVideoCap.isOpened()) {
                    m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                }
            }
        }
        if (m_dynamicVideoCap.isOpened()) {
            frameRead = m_dynamicVideoCap.read(nextFrame);
            if (frameRead && !nextFrame.empty()) {
                double totalFrames = m_dynamicVideoCap.get(cv::CAP_PROP_FRAME_COUNT);
                double currentFrameIndex = m_dynamicVideoCap.get(cv::CAP_PROP_POS_FRAMES);
                if (totalFrames > 0 && currentFrameIndex >= totalFrames - 1) {
                    m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    if (!m_dynamicVideoCap.read(nextFrame) || nextFrame.empty()) {
                        frameRead = false;
                    }
                }
            }
        }
    }

    if (frameRead && !nextFrame.empty()) {
        m_dynamicVideoFrame = nextFrame.clone();
    }
    
    // THREAD SAFETY: Unlock mutex before returning
    m_dynamicVideoMutex.unlock();
}
// Reset dynamic video to start for re-recording
void Capture::resetDynamicVideoToStart()
{
    if (!m_useDynamicVideoBackground) {
        return;
    }

    // Stop the current video playback timer
    if (m_videoPlaybackTimer && m_videoPlaybackActive) {
        m_videoPlaybackTimer->stop();
        m_videoPlaybackActive = false;
    }

    // Reset video readers to beginning
    // CUDA VideoReader removed
    if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
        qDebug() << "CPU video reader reset to start";
    }

    // Read first frame to prime the system
    cv::Mat firstFrame;
    bool frameRead = false;

    // CUDA VideoReader removed - using CPU VideoCapture
    if (m_dynamicVideoCap.isOpened()) {
        if (m_dynamicVideoCap.read(firstFrame) && !firstFrame.empty()) {
            frameRead = true;
        }
    }

    if (frameRead && !firstFrame.empty()) {
        m_dynamicVideoFrame = firstFrame.clone();
        qDebug() << "Video reset to first frame for re-recording";
    }

    // Restart the video playback timer
    if (m_videoPlaybackTimer) {
        m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
        m_videoPlaybackTimer->start();
        m_videoPlaybackActive = true;
        qDebug() << "Video playback timer restarted after reset";
    }
}

// Phase 2A: GPU-Only Processing Initialization
void Capture::initializeGPUOnlyProcessing()
{
    m_gpuOnlyProcessingEnabled = false;
    m_gpuProcessingAvailable = false;

    // Check if OpenCL is available and GPU processing is supported
    if (m_useOpenCL && cv::ocl::useOpenCL()) {
        try {
            // Test GPU memory allocation (OpenCL)
            cv::UMat testMat(100, 100, CV_8UC3);
            if (!testMat.empty()) {
                m_gpuProcessingAvailable = true;
                m_gpuOnlyProcessingEnabled = true;

                // Initialize GPU buffers
                m_gpuVideoFrame.release();
                m_gpuSegmentedFrame.release();
                m_gpuPersonMask.release();
                m_gpuBackgroundFrame.release();

                qDebug() << "Phase 2A: GPU-only processing pipeline initialized successfully";
                qDebug() << "GPU memory available for video processing";
            }
        } catch (const cv::Exception& e) {
            qWarning() << "GPU-only processing initialization failed:" << e.what();
            m_gpuProcessingAvailable = false;
            m_gpuOnlyProcessingEnabled = false;
        }
    }

    if (!m_gpuProcessingAvailable) {
        qDebug() << "Phase 2A: GPU-only processing not available, using CPU fallback";
    }
}

bool Capture::isGPUOnlyProcessingAvailable() const
{
    return m_gpuProcessingAvailable && m_gpuOnlyProcessingEnabled;
}

// Phase 2A: GPU-Only Processing Pipeline
cv::Mat Capture::processFrameWithGPUOnlyPipeline(const cv::Mat &frame)
{
    if (frame.empty()) {
        return cv::Mat();
    }

    // Background model update removed - watershed doesn't need it
    m_personDetectionTimer.start();

    try {
        qDebug() << "Phase 2A: Using GPU-only processing pipeline";

        // Upload frame to GPU (OpenCL) - single transfer
        frame.copyTo(m_gpuVideoFrame);

        // WATERSHED SEGMENTATION MODE: Use GPU-accelerated watershed masking
        if (m_segmentationEnabledInCapture) {
            qDebug() << "Processing watershed segmentation with GPU acceleration";
            
            // VALIDATION: Ensure GPU frame is valid
            if (m_gpuVideoFrame.empty() || m_gpuVideoFrame.cols == 0 || m_gpuVideoFrame.rows == 0) {
                qWarning() << "GPU video frame is invalid for watershed segmentation, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // Create GPU-accelerated watershed mask with crash protection
            cv::UMat gpuPersonMask;
            try {
                gpuPersonMask = createWatershedPersonMaskGPU(m_gpuVideoFrame);
            } catch (const cv::Exception &e) {
                qWarning() << "GPU watershed mask creation failed:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            } catch (const std::exception &e) {
                qWarning() << "Exception in GPU watershed:" << e.what() << "- falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // VALIDATION: Ensure mask is valid
            if (gpuPersonMask.empty()) {
                qWarning() << "GPU watershed mask is empty, falling back to CPU";
                return processFrameWithUnifiedDetection(frame);
            }
            
            // Download mask to derive detections on CPU (for bounding boxes)
            cv::Mat personMask;
            try {
                gpuPersonMask.copyTo(personMask);
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
            
            qDebug() << "Derived" << detections.size() << "detections from watershed mask";
            
            // Use cleaned frame (with green spill removed) for GPU-only segmentation
            cv::Mat segmentedFrame;
            try {
                // Upload cleaned frame to GPU for segmentation
                cleanedFrame.copyTo(m_gpuVideoFrame);
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
            
            qDebug() << "GPU watershed processing completed successfully";
            return segmentedFrame;
        }

        // Optimized processing for 30 FPS with GPU (OpenCL)
        cv::UMat processFrame = m_gpuVideoFrame;
        if (frame.cols > 640) {
            double scale = 640.0 / frame.cols;
            cv::resize(m_gpuVideoFrame, processFrame, cv::Size(), scale, scale, cv::INTER_LINEAR);
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

// Enhanced Person Detection and Segmentation methods
void Capture::initializePersonDetection()
{
    qDebug() << "===== initializePersonDetection() CALLED =====";
    qDebug() << "Initializing Enhanced Person Detection and Segmentation...";

    // Initialize HOG detectors for person detection
    qDebug() << "===== CAPTURE INITIALIZATION STARTED =====";
    m_hogDetector.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    m_hogDetectorDaimler.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());

    // Initialize CPU HOG detector (CUDA HOG removed - using OpenCL/CPU)
    qDebug() << "===== STARTING HOG INITIALIZATION =====";

    // Check if OpenCL is available
    // CUDA HOG detector removed - using CPU HOG detector
    if (false) { // CUDA removed - OpenCL used instead
        try {
            // CUDA HOG detector removed - using CPU HOG detector
            qDebug() << "HOG detector initialized - using CPU (OpenCL available for other operations)";
        } catch (const cv::Exception& e) {
            qWarning() << "HOG initialization failed:" << e.what();
        }
    } else {
        qDebug() << "OpenCL not available for HOG initialization";
    }
    qDebug() << "===== HOG INITIALIZATION COMPLETE =====";

    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    // OPTIMIZATION: Only create if not already initialized to avoid recreating unnecessarily
    if (m_bgSubtractor.empty()) {
        m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
        qDebug() << "Background subtractor initialized in initializePersonDetection()";
    } else {
        qDebug() << "Background subtractor already initialized, skipping recreation";
    }

    //  Initialize GPU Memory Pool for optimized OpenCL operations
    if (!m_gpuMemoryPoolInitialized && cv::ocl::useOpenCL()) {
        try {
            qDebug() << " Initializing GPU Memory Pool for optimized OpenCL operations...";
            m_gpuMemoryPool.initialize(1280, 720); // Initialize with common camera resolution
            m_gpuMemoryPoolInitialized = true;
            qDebug() << "GPU Memory Pool initialized successfully";
        } catch (const cv::Exception& e) {
            qWarning() << " GPU Memory Pool initialization failed:" << e.what();
            m_gpuMemoryPoolInitialized = false;
        }
    }

    // Check if OpenCL is available for GPU acceleration
    try {
        if (cv::ocl::useOpenCL()) {
            cv::ocl::setUseOpenCL(true);
            m_useOpenCL = true;
            qDebug() << "OpenCL GPU acceleration enabled";
            
            cv::ocl::Context context = cv::ocl::Context::getDefault();
            if (!context.empty()) {
                size_t deviceCount = context.ndevices();
                qDebug() << "OpenCL devices found:" << deviceCount;
                if (deviceCount > 0) {
                    cv::ocl::Device device = context.device(0);
                    qDebug() << "OpenCL Device:" << device.name().c_str();
                    qDebug() << "Memory:" << device.globalMemSize() / (1024*1024) << "MB";
                    qDebug() << "OpenCL will be used for color conversion and resizing operations";
                }
            }
        } else {
            qDebug() << "OpenCL not available, using CPU";
            m_useOpenCL = false;
        }
    } catch (...) {
        qDebug() << "OpenCL initialization failed, using CPU";
        m_useOpenCL = false;
    }

    // Check if OpenCL is available for HOG detection (ALWAYS ENABLE FOR HOG)
    try {
        if (cv::ocl::useOpenCL()) {
            m_useGPU = true;
            qDebug() << "OpenCL GPU acceleration enabled for HOG detection";
            qDebug() << "OpenCL will be used for HOG detection (GPU acceleration)";

            // Force OpenCL usage
            cv::ocl::setUseOpenCL(true);

            // Test OpenCL with a simple operation
            cv::UMat testMat;
            testMat.create(100, 100, CV_8UC3);
            if (!testMat.empty()) {
                qDebug() << "OpenCL memory allocation test passed";

                // Test OpenCL with a simple operation
                cv::UMat testResult;
                cv::cvtColor(testMat, testResult, cv::COLOR_BGR2GRAY);
                qDebug() << "OpenCL color conversion test passed";

                // Pre-allocate GPU memory pools for better performance
                qDebug() << "Pre-allocating GPU memory pools...";
                try {
                    // Pre-allocate common frame sizes for GPU operations
                    cv::UMat gpuFramePool1, gpuFramePool2, gpuFramePool3;
                    gpuFramePool1.create(720, 1280, CV_8UC3);  // Common camera resolution
                    gpuFramePool2.create(480, 640, CV_8UC3);   // Smaller processing size
                    gpuFramePool3.create(360, 640, CV_8UC1);   // Grayscale processing

                    qDebug() << "GPU memory pools pre-allocated successfully";
                    qDebug() << "  - Frame pool 1: 1280x720 (RGB)";
                    qDebug() << "  - Frame pool 2: 640x480 (RGB)";
                    qDebug() << "  - Frame pool 3: 640x360 (Grayscale)";
                } catch (const cv::Exception& e) {
                    qWarning() << "GPU memory pool allocation failed:" << e.what();
                }
            }

            // OpenCL device info not available in this OpenCV version
            qDebug() << "OpenCL GPU acceleration ready for HOG detection";

        } else {
            qDebug() << "OpenCL not available for HOG, will use CPU";
            m_useGPU = false;
        }
    } catch (...) {
        qDebug() << "OpenCL initialization failed for HOG, will use CPU";
        m_useGPU = false;
    }

    // Check if OpenCL is available for AMD GPU acceleration (FALLBACK)
    if (!m_useOpenCL) {
        try {
            if (cv::ocl::useOpenCL()) {
                m_useGPU = true;
                qDebug() << "OpenCL GPU acceleration enabled for AMD GPU (fallback)";
                qDebug() << "Using UMat for GPU memory management";
            } else {
                qDebug() << "OpenCL not available, using CPU";
                m_useGPU = false;
            }
        } catch (...) {
            qDebug() << "OpenCL initialization failed, using CPU";
            m_useGPU = false;
        }
    }

    // Initialize async processing
    m_personDetectionWatcher = new QFutureWatcher<cv::Mat>(this);
    connect(m_personDetectionWatcher, &QFutureWatcher<cv::Mat>::finished,
            this, &Capture::onPersonDetectionFinished);

    // OpenCL is already initialized above - no additional setup needed

    qDebug() << "Enhanced Person Detection and Segmentation initialized successfully";
    qDebug() << "GPU Priority: OpenCL > CPU (fallback)";
}

void Capture::adjustRect(cv::Rect &r) const
{
    // Ensure the detection rectangle covers the full person: do not shrink.
    // Keeping the original detector rectangle preserves full-body coverage.
    // No-op for performance and coverage.
    (void)r; // Suppress unused parameter warning
}

// CUDA HOG detector functions completely removed - using CPU HOG detector (runClassicHogPass) instead

std::vector<cv::Rect> Capture::runClassicHogPass(const cv::Mat &frame)
{
    std::vector<cv::Rect> combined;

    if (frame.empty()) {
        return combined;
    }

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

    std::vector<cv::Rect> defaultDetections;
    m_hogDetector.detectMultiScale(resized, defaultDetections, 0.0, cv::Size(8, 8), cv::Size(), 1.05, 2, false);

    std::vector<cv::Rect> daimlerDetections;
    m_hogDetectorDaimler.detectMultiScale(resized, daimlerDetections, 0.0, cv::Size(8, 8), cv::Size(), 1.05, 2, false);

    auto upscale = [](std::vector<cv::Rect> &rects) {
        for (auto &rect : rects) {
            rect.x = cvRound(rect.x * 2.0);
            rect.y = cvRound(rect.y * 2.0);
            rect.width = cvRound(rect.width * 2.0);
            rect.height = cvRound(rect.height * 2.0);
        }
    };

    upscale(defaultDetections);
    upscale(daimlerDetections);

    combined.insert(combined.end(), defaultDetections.begin(), defaultDetections.end());
    combined.insert(combined.end(), daimlerDetections.begin(), daimlerDetections.end());

    return combined;
}

std::vector<cv::Rect> Capture::nonMaximumSuppression(const std::vector<cv::Rect> &detections,
                                                     double overlapThreshold)
{
    if (detections.empty()) {
        return {};
    }

    std::vector<cv::Rect> boxes = detections;
    std::vector<cv::Rect> result;
    result.reserve(boxes.size());

    std::sort(boxes.begin(), boxes.end(), [](const cv::Rect &a, const cv::Rect &b) {
        return a.area() > b.area();
    });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        const cv::Rect &a = boxes[i];
        result.push_back(a);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            const cv::Rect &b = boxes[j];
            const int intersectionArea = (a & b).area();
            const int unionArea = a.area() + b.area() - intersectionArea;

            if (unionArea <= 0) {
                continue;
            }

            const double overlap = static_cast<double>(intersectionArea) / static_cast<double>(unionArea);
            if (overlap > overlapThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

std::vector<cv::Rect> Capture::detectPeople(const cv::Mat &frame)
{
    std::vector<cv::Rect> detections;

    // CUDA HOG detector removed - using CPU HOG detector
    detections = runClassicHogPass(frame);

    // Adjust rectangles
    for (auto &rect : detections) {
        adjustRect(rect);
    }

    // Non-maximum suppression
    detections = nonMaximumSuppression(detections, 0.6);
    // Merge near-duplicates to ensure one box per person without heavy grouping
    detections = enforceOneBoxPerPerson(detections);

    // Lightweight temporal smoothing for stability
    detections = smoothDetections(detections);

    return detections;
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

    // Use watershed segmentation for person detection
    if (m_segmentationEnabledInCapture) {
        cv::Mat personMask;
        try {
            personMask = createWatershedPersonMask(frame);
        } catch (const cv::Exception& e) {
            qWarning() << "Exception in createWatershedPersonMask:" << e.what();
            personMask = cv::Mat();
        } catch (...) {
            qWarning() << "Unknown exception in createWatershedPersonMask";
            personMask = cv::Mat();
        }
        
        // Validate mask before using it
        if (personMask.empty() || personMask.rows != frame.rows || personMask.cols != frame.cols) {
            qWarning() << "Invalid watershed mask, using empty detections";
            std::vector<cv::Rect> emptyDetections;
            m_lastDetections = emptyDetections;
            cv::Mat segmentedFrame = createSegmentedFrame(frame, emptyDetections);
            m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
            m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;
            return segmentedFrame;
        }
        
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
            // Use dynamic video background - check if timer is active or use fallback
            try {
                // THREAD SAFETY: Lock mutex for safe video frame access
                QMutexLocker locker(&m_dynamicVideoMutex);
                
                if (!m_dynamicVideoFrame.empty()) {
                    cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    qDebug() << "Successfully using video frame for segmentation - frame size:" << m_dynamicVideoFrame.cols << "x" << m_dynamicVideoFrame.rows;
                    qDebug() << "Segmented frame size:" << segmentedFrame.cols << "x" << segmentedFrame.rows;
                } else {
                    // Fallback: read frame synchronously if timer hasn't advanced yet or frame is empty
                    cv::Mat nextBg;
                    
                    // Use CPU video reader (OpenCL doesn't have direct video decoding)
                    if (m_dynamicVideoCap.isOpened()) {
                        if (!m_dynamicVideoCap.read(nextBg) || nextBg.empty()) {
                            m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            m_dynamicVideoCap.read(nextBg);
                        }
                    } else {
                        // Video capture not opened - try to reopen it
                        if (!m_dynamicVideoPath.isEmpty()) {
                            qDebug() << "Video capture not opened, attempting to reopen:" << m_dynamicVideoPath;
                            m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                            if (!m_dynamicVideoCap.isOpened()) {
                                m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                            }
                            if (m_dynamicVideoCap.isOpened()) {
                                m_dynamicVideoCap.read(nextBg);
                            }
                        }
                    }
                    
                    if (!nextBg.empty()) {
                        cv::resize(nextBg, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                        m_dynamicVideoFrame = nextBg.clone(); // Store the frame for next time
                        qDebug() << "Fallback: Successfully read video frame for segmentation";
                        
                        // Ensure timer is started if it's not already running
                        if (!m_videoPlaybackActive && m_videoPlaybackTimer) {
                            m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
                            m_videoPlaybackTimer->start();
                            m_videoPlaybackActive = true;
                            qDebug() << "Started video playback timer from fallback";
                        }
                    } else {
                        segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                        qWarning() << "Fallback: Failed to read video frame - using black background. Video path:" << m_dynamicVideoPath;
                        qWarning() << "Video capture opened:" << m_dynamicVideoCap.isOpened();
                    }
                }
            } catch (const cv::Exception &e) {
                qWarning() << "CPU segmentation crashed:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            } catch (const std::exception &e) {
                qWarning() << "Exception in CPU segmentation:" << e.what() << "- using black background";
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else {
            // Not using dynamic video background - use template or black background
            qDebug() << "Dynamic video background not enabled - using template or black background";
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
            // Ensure segmentedFrame is initialized before using it
            if (segmentedFrame.empty()) {
                // Fallback: initialize with black background if not set
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                qWarning() << "segmentedFrame was empty, initialized with black background";
            }
            
            // ALWAYS use watershed segmentation
            cv::Mat personMask;
            try {
                personMask = createWatershedPersonMask(frame);
            } catch (const cv::Exception& e) {
                qWarning() << "Exception in createWatershedPersonMask (createSegmentedFrame):" << e.what();
                personMask = cv::Mat();
            } catch (...) {
                qWarning() << "Unknown exception in createWatershedPersonMask (createSegmentedFrame)";
                personMask = cv::Mat();
            }

            // Validate mask before using it
            if (personMask.empty()) {
                qDebug() << "Watershed mask is empty, returning background only";
                return segmentedFrame;
            }
            
            // Validate mask size matches frame size
            if (personMask.rows != frame.rows || personMask.cols != frame.cols) {
                qWarning() << "Watershed mask size mismatch in createSegmentedFrame: mask=" 
                          << personMask.cols << "x" << personMask.rows 
                          << ", frame=" << frame.cols << "x" << frame.rows 
                          << " - returning background only";
                return segmentedFrame;
            }

            int nonZeroPixels = cv::countNonZero(personMask);
            qDebug() << "Watershed person mask non-zero:" << nonZeroPixels;

            // If mask is empty, just return the background
            if (nonZeroPixels == 0) {
                qDebug() << "Watershed mask has no non-zero pixels, returning background only";
                return segmentedFrame;
            }

            // Validate mask size matches frame size before using it
            if (personMask.rows != frame.rows || personMask.cols != frame.cols) {
                qWarning() << "Watershed mask size mismatch: mask=" << personMask.cols << "x" << personMask.rows 
                          << ", frame=" << frame.cols << "x" << frame.rows << " - returning background only";
                return segmentedFrame;
            }

            // Apply mask to extract person from camera frame
            cv::Mat personRegion;
            try {
                frame.copyTo(personRegion, personMask);
                if (personRegion.empty()) {
                    qWarning() << "Failed to extract person region from frame - returning background only";
                    return segmentedFrame;
                }
            } catch (const cv::Exception& e) {
                qWarning() << "Exception while extracting person region:" << e.what() << " - returning background only";
                return segmentedFrame;
            } catch (...) {
                qWarning() << "Unknown exception while extracting person region - returning background only";
                return segmentedFrame;
            }

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
                // Validate segmentedFrame before using its size
                if (segmentedFrame.empty()) {
                    qWarning() << "segmentedFrame is empty, cannot scale person - returning background only";
                    return segmentedFrame;
                }
                
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
                        yOffset + actualScaledSize.height <= backgroundSize.height &&
                        scaledPersonRegion.cols == scaledPersonMask.cols &&
                        scaledPersonRegion.rows == scaledPersonMask.rows) {
                        
                        try {
                            // Validate rect is within bounds
                            if (xOffset < 0 || yOffset < 0 || 
                                xOffset + actualScaledSize.width > segmentedFrame.cols ||
                                yOffset + actualScaledSize.height > segmentedFrame.rows) {
                                qWarning() << " CRASH PREVENTION: ROI rect out of bounds - skipping compositing";
                            } else {
                                cv::Rect backgroundRect(cv::Point(xOffset, yOffset), actualScaledSize);
                                cv::Mat backgroundROI = segmentedFrame(backgroundRect);
                                
                                // Validate ROI and scaled region sizes match
                                if (backgroundROI.rows == scaledPersonRegion.rows && 
                                    backgroundROI.cols == scaledPersonRegion.cols &&
                                    backgroundROI.rows == scaledPersonMask.rows &&
                                    backgroundROI.cols == scaledPersonMask.cols) {
                                    scaledPersonRegion.copyTo(backgroundROI, scaledPersonMask);
                                    qDebug() << " COMPOSITING: Successfully composited scaled person at offset" << xOffset << "," << yOffset;
                                } else {
                                    qWarning() << " CRASH PREVENTION: Size mismatch in ROI compositing - skipping";
                                }
                            }
                        } catch (const cv::Exception& e) {
                            qWarning() << " CRASH PREVENTION: Compositing failed:" << e.what() << "- skipping compositing";
                        } catch (...) {
                            qWarning() << " CRASH PREVENTION: Unknown exception in compositing - skipping";
                        }
                    } else {
                        qWarning() << " CRASH PREVENTION: Invalid offsets or size mismatch - skipping compositing";
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
            cv::Rect adjustedRect = detection;
            adjustRect(adjustedRect);
            cv::rectangle(displayFrame, adjustedRect.tl(), adjustedRect.br(), cv::Scalar(0, 255, 0), 3);

            qDebug() << "Rectangle" << i << "at" << adjustedRect.x << adjustedRect.y << adjustedRect.width << "x" << adjustedRect.height;
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
                    cv::resize(m_dynamicGpuFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.copyTo(segmentedFrame);
                    qDebug() << "RECORDING: Using GPU frame for background";
                } else if (!m_dynamicVideoFrame.empty() && m_dynamicVideoFrame.cols > 0 && m_dynamicVideoFrame.rows > 0) {
                    m_dynamicVideoFrame.copyTo(m_gpuBackgroundFrame);
                    cv::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.copyTo(segmentedFrame);
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
            // Phase 2A: GPU-only video background processing
            try {
                // THREAD SAFETY: Lock mutex for safe GPU frame access
                QMutexLocker locker(&m_dynamicVideoMutex);
                
                if (!m_dynamicGpuFrame.empty()) {
                    // Already on GPU from NVDEC, avoid CPU upload
                    cv::resize(m_dynamicGpuFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.copyTo(segmentedFrame);
                    qDebug() << "Using NVDEC GPU frame for segmentation - size:" << m_dynamicGpuFrame.cols << "x" << m_dynamicGpuFrame.rows;
                } else if (!m_dynamicVideoFrame.empty()) {
                    // Fallback: CPU frame upload
                    m_dynamicVideoFrame.copyTo(m_gpuBackgroundFrame);
                    cv::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.copyTo(segmentedFrame);
                    qDebug() << "Fallback CPU frame upload for segmentation - size:" << m_dynamicVideoFrame.cols << "x" << m_dynamicVideoFrame.rows;
                } else {
                    // Try to read frame synchronously if empty
                    cv::Mat nextBg;
                    if (m_dynamicVideoCap.isOpened()) {
                        if (!m_dynamicVideoCap.read(nextBg) || nextBg.empty()) {
                            m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            m_dynamicVideoCap.read(nextBg);
                        }
                    } else if (!m_dynamicVideoPath.isEmpty()) {
                        // Try to reopen
                        m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_MSMF);
                        if (!m_dynamicVideoCap.isOpened()) {
                            m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
                        }
                        if (m_dynamicVideoCap.isOpened()) {
                            m_dynamicVideoCap.read(nextBg);
                        }
                    }
                    
                    if (!nextBg.empty()) {
                        m_dynamicVideoFrame = nextBg.clone();
                        m_dynamicVideoFrame.copyTo(m_gpuBackgroundFrame);
                        cv::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                        m_gpuSegmentedFrame.copyTo(segmentedFrame);
                        qDebug() << "GPU: Successfully read video frame synchronously";
                        
                        // Ensure timer is started if it's not already running
                        if (!m_videoPlaybackActive && m_videoPlaybackTimer) {
                            m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
                            m_videoPlaybackTimer->start();
                            m_videoPlaybackActive = true;
                            qDebug() << "Started video playback timer from GPU fallback";
                        }
                    } else {
                        segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                        qWarning() << "GPU: Dynamic video frame is empty - using black background. Video path:" << m_dynamicVideoPath;
                    }
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
                // Upload template to GPU (OpenCL)
                cachedBackgroundTemplate.copyTo(m_gpuBackgroundFrame);

                // Resize on GPU
                cv::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);

                // Download result
                m_gpuSegmentedFrame.copyTo(segmentedFrame);
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

    if (m_useOpenCL) {
        try {
            // Upload ROI to GPU
            cv::UMat gpu_roi;
            roi.copyTo(gpu_roi);

            // CRASH PREVENTION: Validate ROI has 3 channels before BGR2GRAY conversion
            if (roi.empty() || roi.channels() != 3) {
                qWarning() << "Invalid ROI for GPU processing: empty or not 3 channels";
                return roiMask; // Return empty mask
            }

            // Convert to grayscale on GPU
            cv::UMat gpu_gray;
            cv::cvtColor(gpu_roi, gpu_gray, cv::COLOR_BGR2GRAY);

            // Apply Gaussian blur on GPU using OpenCL
            cv::UMat gpu_blurred;
            cv::GaussianBlur(gpu_gray, gpu_blurred, cv::Size(5, 5), 0);

            // OpenCL-accelerated Canny edge detection
            cv::UMat gpu_edges;
            cv::Canny(gpu_blurred, gpu_edges, 15, 45);

            // OpenCL-accelerated morphological dilation
            cv::UMat gpu_dilated;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::dilate(gpu_edges, gpu_dilated, kernel);

            // Download result back to CPU
            gpu_dilated.copyTo(edges);

            qDebug() << "GPU-accelerated edge detection applied";

        } catch (const cv::Exception& e) {
            qWarning() << "OpenCL edge detection failed, falling back to CPU:" << e.what();
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
            
            if (m_useOpenCL) {
                try {
                    // GPU-accelerated absolute difference
                    cv::UMat gpu_roi, gpu_ref, gpu_diff;
                    roi.copyTo(gpu_roi);
                    refResized.copyTo(gpu_ref);
                    
                    cv::absdiff(gpu_roi, gpu_ref, gpu_diff);
                    
                    // Convert to grayscale and threshold
                    cv::UMat gpu_gray;
                    cv::cvtColor(gpu_diff, gpu_gray, cv::COLOR_BGR2GRAY);
                    
                    cv::UMat gpu_mask;
                    cv::threshold(gpu_gray, gpu_mask, 30, 255, cv::THRESH_BINARY);
                    
                    gpu_mask.copyTo(fgMask);
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU (OpenCL)
                cv::UMat gpu_fgMask;
                fgMask.copyTo(gpu_fgMask);

                // Create morphological kernels
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

                // GPU-accelerated morphological operations (OpenCL)
                cv::morphologyEx(gpu_fgMask, gpu_fgMask, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(gpu_fgMask, gpu_fgMask, cv::MORPH_CLOSE, kernel);
                cv::dilate(gpu_fgMask, gpu_fgMask, kernel_dilate);

                // Download result back to CPU
                gpu_fgMask.copyTo(fgMask);

                qDebug() << "GPU-accelerated morphological operations applied";

            } catch (const cv::Exception& e) {
                qWarning() << "OpenCL morphological operations failed, falling back to CPU:" << e.what();
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

        if (m_useOpenCL) {
            try {
                // Upload ROI to GPU
                cv::UMat gpu_roi;
                roi.copyTo(gpu_roi);

                // Convert to HSV on GPU
                cv::UMat gpu_hsv;
                cv::cvtColor(gpu_roi, gpu_hsv, cv::COLOR_BGR2HSV);

                // Create masks for skin-like colors and non-background colors on GPU
                cv::UMat gpu_skinMask, gpu_colorMask;
                // Widened skin range and relaxed saturation/value to better capture varied tones/lighting
                cv::inRange(gpu_hsv, cv::Scalar(0, 10, 40), cv::Scalar(25, 255, 255), gpu_skinMask);
                // Broader general color mask with relaxed S/V to include darker/low-saturation clothing
                cv::inRange(gpu_hsv, cv::Scalar(0, 15, 35), cv::Scalar(180, 255, 255), gpu_colorMask);

                // Combine masks on GPU using bitwise_or
                cv::UMat gpu_combinedMask;
                cv::bitwise_or(gpu_skinMask, gpu_colorMask, gpu_combinedMask);

                // Download result back to CPU
                gpu_combinedMask.copyTo(combinedMask);

                qDebug() << "GPU-accelerated color segmentation applied";

            } catch (const cv::Exception& e) {
                qWarning() << "OpenCL color segmentation failed, falling back to CPU:" << e.what();
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU
                cv::UMat gpu_combinedMask;
                combinedMask.copyTo(gpu_combinedMask);

                // Create morphological kernel
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

                // GPU-accelerated morphological operations (OpenCL)
                cv::morphologyEx(gpu_combinedMask, gpu_combinedMask, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(gpu_combinedMask, gpu_combinedMask, cv::MORPH_CLOSE, kernel);

                // Download result back to CPU
                gpu_combinedMask.copyTo(combinedMask);

                qDebug() << "GPU-accelerated color morphological operations applied";

            } catch (const cv::Exception& e) {
                qWarning() << "OpenCL color morphological operations failed, falling back to CPU:" << e.what();
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU
                cv::UMat gpu_roiMask;
                roiMask.copyTo(gpu_roiMask);

                // Create morphological kernels
                cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

                // GPU-accelerated morphological operations (OpenCL)
                cv::morphologyEx(gpu_roiMask, gpu_roiMask, cv::MORPH_CLOSE, kernel_clean);
                cv::dilate(gpu_roiMask, gpu_roiMask, kernel_dilate);

                // Download result back to CPU
                gpu_roiMask.copyTo(roiMask);

                qDebug() << "GPU-accelerated final morphological cleanup applied";

            } catch (const cv::Exception& e) {
                qWarning() << "OpenCL final morphological cleanup failed, falling back to CPU:" << e.what();
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
cv::Mat Capture::enhancedSilhouetteSegmentGPUOnly(const cv::UMat &gpuFrame, const cv::Rect &detection)
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
    cv::UMat gpuRoi = gpuFrame(expandedRect);
        cv::UMat gpuRoiMask(gpuRoi.size(), CV_8UC1, cv::Scalar(0));

        // CRASH PREVENTION: Validate gpuRoi has 3 channels before BGR2GRAY conversion
        if (gpuRoi.empty() || gpuRoi.channels() != 3) {
            qWarning() << "Invalid gpuRoi for GPU processing: empty or not 3 channels";
            return cv::Mat::zeros(gpuFrame.size(), CV_8UC1); // Return empty mask
        }
        
        // Use standard GPU processing without memory pool
    cv::UMat gpuGray, gpuEdges;
    cv::cvtColor(gpuRoi, gpuGray, cv::COLOR_BGR2GRAY);

        // OpenCL Canny edge detection
        cv::Canny(gpuGray, gpuEdges, 50, 150);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(gpuEdges, gpuRoiMask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(gpuRoiMask, gpuRoiMask, cv::MORPH_OPEN, kernel);
        cv::dilate(gpuRoiMask, gpuRoiMask, kernel);

        cv::UMat gpuConnectedMask;
        cv::threshold(gpuRoiMask, gpuConnectedMask, 127, 255, cv::THRESH_BINARY);
        cv::morphologyEx(gpuConnectedMask, gpuConnectedMask, cv::MORPH_CLOSE, kernel);

        cv::Mat kernel_final = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(gpuConnectedMask, gpuConnectedMask, cv::MORPH_CLOSE, kernel_final);

        cv::Mat finalMask;
        gpuConnectedMask.copyTo(finalMask);

        cv::UMat gpuFullMask(gpuFrame.size(), CV_8UC1, cv::Scalar(0));
        cv::UMat gpuFinalMask;
        finalMask.copyTo(gpuFinalMask);
        gpuFinalMask.copyTo(gpuFullMask(expandedRect));

        cv::Mat fullMask;
        gpuFullMask.copyTo(fullMask);

        qDebug() << " Phase 2A: Standard GPU processing completed (memory pool not available)";
        return fullMask;
    }

    // Extract ROI on GPU using memory pool
    cv::UMat gpuRoi = gpuFrame(expandedRect);
    cv::UMat& gpuRoiMask = m_gpuMemoryPool.getNextSegmentationBuffer();
    gpuRoiMask.create(gpuRoi.size(), CV_8UC1);
    gpuRoiMask.setTo(cv::Scalar(0));

    // OpenCL doesn't use explicit streams like CUDA - operations are automatically parallelized

    // CRASH PREVENTION: Validate gpuRoi has 3 channels before BGR2GRAY conversion
    if (gpuRoi.empty() || gpuRoi.channels() != 3) {
        qWarning() << "Invalid gpuRoi for GPU memory pool processing: empty or not 3 channels";
        return cv::Mat::zeros(gpuFrame.size(), CV_8UC1); // Return empty mask
    }
    
    // Step 1: GPU Color Conversion (async)
    cv::UMat& gpuGray = m_gpuMemoryPool.getNextTempBuffer();
    cv::UMat& gpuEdges = m_gpuMemoryPool.getNextDetectionBuffer();
    cv::cvtColor(gpuRoi, gpuGray, cv::COLOR_BGR2GRAY);

    // Step 2: GPU Canny Edge Detection (OpenCL)
    cv::Canny(gpuGray, gpuEdges, 50, 150);

    // Step 3: GPU Morphological Operations (OpenCL)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // Apply GPU morphological pipeline (OpenCL)
    cv::morphologyEx(gpuEdges, gpuRoiMask, cv::MORPH_CLOSE, kernel);      // Close gaps
    cv::morphologyEx(gpuRoiMask, gpuRoiMask, cv::MORPH_OPEN, kernel);     // Remove noise
    cv::dilate(gpuRoiMask, gpuRoiMask, kernel);   // Expand regions

    // Step 4: GPU-accelerated area-based filtering (async)
    cv::UMat& gpuConnectedMask = m_gpuMemoryPool.getNextSegmentationBuffer();

    // Create a mask for large connected regions (person-like areas) - async
    cv::threshold(gpuRoiMask, gpuConnectedMask, 127, 255, cv::THRESH_BINARY);

    // Apply additional GPU morphological cleanup (OpenCL)
    cv::morphologyEx(gpuConnectedMask, gpuConnectedMask, cv::MORPH_CLOSE, kernel);

    // Step 5: Final GPU morphological cleanup (OpenCL)
    cv::Mat kernel_final = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(gpuConnectedMask, gpuConnectedMask, cv::MORPH_CLOSE, kernel_final);

    // Step 6: Single download at the end (minimize GPU-CPU transfers)
    cv::Mat finalMask;
    gpuConnectedMask.copyTo(finalMask);

    //  GPU-OPTIMIZED: Create full-size mask directly on GPU
    cv::UMat& gpuFullMask = m_gpuMemoryPool.getNextFrameBuffer();
    gpuFullMask.create(gpuFrame.size(), CV_8UC1);
    gpuFullMask.setTo(cv::Scalar(0));

    // Copy the processed ROI back to the full-size mask on GPU (OpenCL)
    cv::UMat gpuFinalMask;
    finalMask.copyTo(gpuFinalMask);
    gpuFinalMask.copyTo(gpuFullMask(expandedRect));

    // Single download at the very end
    cv::Mat fullMask;
    gpuFullMask.copyTo(fullMask);

    qDebug() << " Phase 2A: GPU MEMORY POOL + ASYNC STREAMS silhouette segmentation completed";

    return fullMask;
}

// Phase 2A: GPU Result Validation
void Capture::validateGPUResults(const cv::Mat &gpuResult, const cv::Mat &cpuResult)
{
    if (gpuResult.empty() || cpuResult.empty()) {
        qWarning() << "Phase 2A: GPU/CPU result validation failed - empty results";
        return;
    }

    if (gpuResult.size() != cpuResult.size() || gpuResult.type() != cpuResult.type()) {
        qWarning() << "Phase 2A: GPU/CPU result validation failed - size/type mismatch";
        return;
    }

    // Compare results (allow small differences due to floating-point precision)
    cv::Mat diff;
    cv::absdiff(gpuResult, cpuResult, diff);
    double maxDiff = cv::norm(diff, cv::NORM_INF);

    if (maxDiff > 5.0) { // Allow small differences
        qWarning() << "Phase 2A: GPU/CPU result validation failed - max difference:" << maxDiff;
    } else {
        qDebug() << "Phase 2A: GPU/CPU result validation passed - max difference:" << maxDiff;
    }
}

void Capture::onPersonDetectionFinished()
{
    if (m_personDetectionWatcher && m_personDetectionWatcher->isFinished()) {
        try {
            cv::Mat result = m_personDetectionWatcher->result();
            if (!result.empty()) {
                QMutexLocker locker(&m_personDetectionMutex);

                //  NO REAL-TIME LIGHTING: Store result without lighting correction
                // Lighting will ONLY be applied in post-processing after recording
                m_lastSegmentedFrame = result.clone();

                // Update GPU utilization flags
                if (m_useOpenCL) {
                    m_gpuUtilized = true;
                } else if (m_useGPU) {
                    m_gpuUtilized = true;
                    // m_cudaUtilized removed - using OpenCL
                }

                qDebug() << "Person detection finished - segmented frame updated, size:"
                         << result.cols << "x" << result.rows;
            } else {
                qDebug() << "Person detection finished but result empty";
            }
        } catch (const std::exception& e) {
            qWarning() << "Exception in person detection finished callback:" << e.what();
        }
    } else {
        qDebug() << "Person detection watcher not finished or null";
    }
}

// Enhanced Person Detection and Segmentation Control Methods
void Capture::setShowPersonDetection(bool show)
{
    m_segmentationEnabledInCapture = show;
    qDebug() << "Person detection display set to:" << show << "(segmentation enabled:" << m_segmentationEnabledInCapture << ")";
}

bool Capture::getShowPersonDetection() const
{
    return m_segmentationEnabledInCapture;
}

void Capture::setPersonDetectionConfidenceThreshold(double threshold)
{
    // This could be used to adjust HOG detector parameters
    qDebug() << "Person detection confidence threshold set to:" << threshold;
}

double Capture::getPersonDetectionConfidenceThreshold() const
{
    return 0.0; // Default threshold
}

void Capture::setSystemMonitor(SystemMonitor* monitor)
{
    m_systemMonitor = monitor;
}

void Capture::togglePersonDetection()
{
    // Toggle segmentation on/off
    if (m_segmentationEnabledInCapture) {
        m_segmentationEnabledInCapture = false;
        qDebug() << "Segmentation DISABLED via button";
        
        // Clear any cached segmentation data
        m_lastSegmentedFrame = cv::Mat();
        m_lastDetections.clear();
        
        // Reset GPU utilization flags
        m_gpuUtilized = false;
        } else {
        m_segmentationEnabledInCapture = true;
        qDebug() << "Segmentation ENABLED via button";
        }

    updateDebugDisplay();
}



double Capture::getPersonDetectionProcessingTime() const
{
    return m_lastPersonDetectionTime;
}

bool Capture::isGPUAvailable() const
{
    return m_useGPU;
}

bool Capture::isOpenCLAvailable() const
{
    return m_useOpenCL && cv::ocl::useOpenCL();
}

// --- Watershed segmentation API (no configuration needed) ---

cv::Mat Capture::getMotionMask(const cv::Mat &frame)
{
    cv::Mat fgMask;

    // Check if static reference image(s) are available - use them for subtraction
    if (!m_subtractionReferenceImage.empty() || !m_subtractionReferenceImage2.empty()) {
        cv::Mat refResized;
        
        // If both reference images are available, blend them
        if (!m_subtractionReferenceImage.empty() && !m_subtractionReferenceImage2.empty()) {
            cv::Mat ref1, ref2;
            if (m_subtractionReferenceImage.size() != frame.size()) {
                cv::resize(m_subtractionReferenceImage, ref1, frame.size(), 0, 0, cv::INTER_LINEAR);
            } else {
                ref1 = m_subtractionReferenceImage;
            }
            if (m_subtractionReferenceImage2.size() != frame.size()) {
                cv::resize(m_subtractionReferenceImage2, ref2, frame.size(), 0, 0, cv::INTER_LINEAR);
            } else {
                ref2 = m_subtractionReferenceImage2;
            }
            
            // Blend the two reference images: weight * ref2 + (1-weight) * ref1
            double alpha = m_subtractionBlendWeight;
            double beta = 1.0 - alpha;
            cv::addWeighted(ref1, beta, ref2, alpha, 0.0, refResized);
        } else if (!m_subtractionReferenceImage.empty()) {
            // Use only first reference image
            if (m_subtractionReferenceImage.size() != frame.size()) {
                cv::resize(m_subtractionReferenceImage, refResized, frame.size(), 0, 0, cv::INTER_LINEAR);
            } else {
                refResized = m_subtractionReferenceImage;
            }
        } else {
            // Use only second reference image
            if (m_subtractionReferenceImage2.size() != frame.size()) {
                cv::resize(m_subtractionReferenceImage2, refResized, frame.size(), 0, 0, cv::INTER_LINEAR);
            } else {
                refResized = m_subtractionReferenceImage2;
            }
        }
        
        if (m_useOpenCL) {
            try {
                // GPU-accelerated absolute difference
                cv::UMat gpu_frame, gpu_ref, gpu_diff;
                frame.copyTo(gpu_frame);
                refResized.copyTo(gpu_ref);
                
                cv::absdiff(gpu_frame, gpu_ref, gpu_diff);
                
                // CRASH PREVENTION: Validate diff has 3 channels before BGR2GRAY conversion
                if (gpu_diff.empty() || gpu_diff.channels() != 3) {
                    qWarning() << "Invalid diff for GPU processing: empty or not 3 channels";
                    // Fallback to CPU
                    cv::Mat diff;
                    cv::absdiff(frame, refResized, diff);
                    if (diff.channels() == 3) {
                        cv::Mat gray;
                        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
                        cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
                    } else {
                        qWarning() << "Diff is not 3 channels, using empty mask";
                        fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    }
                    return fgMask; // Return empty mask
                }
                
                // CRASH PREVENTION: Validate gpu_diff has 3 channels before BGR2GRAY conversion
                if (gpu_diff.empty() || gpu_diff.channels() != 3) {
                    qWarning() << "Invalid gpu_diff for GPU processing: empty or not 3 channels";
                    fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    return fgMask; // Return empty mask
                }
                
                // Convert to grayscale and threshold
                cv::UMat gpu_gray;
                cv::cvtColor(gpu_diff, gpu_gray, cv::COLOR_BGR2GRAY);
                
                cv::UMat gpu_mask;
                cv::threshold(gpu_gray, gpu_mask, 30, 255, cv::THRESH_BINARY);
                
                gpu_mask.copyTo(fgMask);
            } catch (...) {
                // Fallback to CPU
                cv::Mat diff;
                cv::absdiff(frame, refResized, diff);
                // CRASH PREVENTION: Validate diff has 3 channels before BGR2GRAY conversion
                if (diff.empty() || diff.channels() != 3) {
                    qWarning() << "Invalid diff for CPU fallback: empty or not 3 channels";
                    fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                    return fgMask; // Return empty mask
                }
                cv::Mat gray;
                cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
                cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
            }
        } else {
            // CPU static reference subtraction
            cv::Mat diff;
            cv::absdiff(frame, refResized, diff);
            // CRASH PREVENTION: Validate diff has 3 channels before BGR2GRAY conversion
            if (diff.empty() || diff.channels() != 3) {
                qWarning() << "Invalid diff for CPU processing: empty or not 3 channels";
                fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
                return fgMask;
            }
            cv::Mat gray;
            cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
            cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
        }
        
        // Apply morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
        
        return fgMask;
    }

    // Fallback to MOG2 background subtractor if no static reference
    if (m_useOpenCL) {
        // OpenCL-accelerated background subtraction (using CPU BackgroundSubtractorMOG2 with OpenCL UMat)
        try {
            // Upload to GPU (OpenCL)
            cv::UMat gpu_frame;
            frame.copyTo(gpu_frame);

            // Use CPU background subtractor with OpenCL UMat (OpenCV automatically uses OpenCL when available)
            // Background subtractor works with UMat automatically

            // OpenCL-accelerated background subtraction (using CPU BackgroundSubtractorMOG2 with UMat)
            cv::UMat gpu_fgmask;
            m_bgSubtractor->apply(gpu_frame, gpu_fgmask, -1);

            // Download result to CPU
            gpu_fgmask.copyTo(fgMask);

            // Apply morphological operations on CPU
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);

        } catch (...) {
            // Fallback to CPU if OpenCL fails
            m_bgSubtractor->apply(frame, fgMask);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
        }
    } else {
        // CPU fallback
        m_bgSubtractor->apply(frame, fgMask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);
    }

    return fgMask;
}

// WATERSHED SEGMENTATION: Marker-based segmentation algorithm
// FAST & ACCURATE: Works for both green AND teal/cyan backdrops
cv::Mat Capture::createWatershedPersonMask(const cv::Mat &frame) const
{
    if (frame.empty()) return cv::Mat();

    try {
        // Validate input frame
        if (frame.empty() || frame.rows <= 0 || frame.cols <= 0) {
            qWarning() << "Watershed: Invalid input frame";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Convert to grayscale for processing
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (gray.empty() || gray.rows != frame.rows || gray.cols != frame.cols) {
            qWarning() << "Watershed: Failed to convert to grayscale";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Apply Gaussian blur to reduce noise
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        if (blurred.empty()) {
            qWarning() << "Watershed: Failed to apply Gaussian blur";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Apply threshold to get binary image
        cv::Mat thresh;
        cv::threshold(blurred, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
        if (thresh.empty()) {
            qWarning() << "Watershed: Failed to apply threshold";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Noise removal using morphological operations
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat opening;
        cv::morphologyEx(thresh, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
        if (opening.empty()) {
            qWarning() << "Watershed: Failed morphological opening";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Sure background area (dilate to get background)
        cv::Mat sureBg;
        cv::dilate(opening, sureBg, kernel, cv::Point(-1, -1), 3);
        if (sureBg.empty()) {
            qWarning() << "Watershed: Failed dilation";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Finding sure foreground area using distance transform
        cv::Mat distTransform;
        cv::distanceTransform(opening, distTransform, cv::DIST_L2, 5);
        if (distTransform.empty()) {
            qWarning() << "Watershed: Failed distance transform";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Normalize distance transform for thresholding
        cv::Mat distNorm;
        cv::normalize(distTransform, distNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        if (distNorm.empty()) {
            qWarning() << "Watershed: Failed normalization";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Threshold to get sure foreground (center of objects)
        cv::Mat sureFg;
        cv::threshold(distNorm, sureFg, 0.4 * 255, 255, cv::THRESH_BINARY);
        sureFg.convertTo(sureFg, CV_8UC1);
        if (sureFg.empty()) {
            qWarning() << "Watershed: Failed to create sure foreground";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Find unknown region (boundary between foreground and background)
        cv::Mat unknown;
        cv::subtract(sureBg, sureFg, unknown);
        if (unknown.empty()) {
            qWarning() << "Watershed: Failed to create unknown region";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Create markers for watershed
        cv::Mat markers;
        int numComponents = cv::connectedComponents(sureFg, markers);
        
        // Check if we have any foreground components
        if (numComponents <= 1) {
            // No foreground detected, return empty mask
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Convert markers to 32-bit signed integer for watershed
        cv::Mat markers32s;
        markers.convertTo(markers32s, CV_32S);
        
        // Validate Mat sizes before processing
        if (markers32s.empty() || markers32s.rows != frame.rows || markers32s.cols != frame.cols) {
            qWarning() << "Watershed: Invalid markers32s size, returning empty mask";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        if (unknown.empty() || unknown.rows != frame.rows || unknown.cols != frame.cols) {
            qWarning() << "Watershed: Invalid unknown mask size, returning empty mask";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Add 1 to all labels so that sure background is 1, not 0
        markers32s = markers32s + cv::Scalar(1);
        
        // Mark the unknown region with 0 (use direct comparison to avoid Mat creation issues)
        // Safe row-by-row access
        try {
            for (int y = 0; y < markers32s.rows && y < unknown.rows; y++) {
                const uchar* unknownRow = unknown.ptr<uchar>(y);
                int* markerRow = markers32s.ptr<int>(y);
                
                int cols = qMin(markers32s.cols, unknown.cols);
                for (int x = 0; x < cols; x++) {
                    if (unknownRow[x] == 255) {
                        markerRow[x] = 0;
                    }
                }
            }
        } catch (const cv::Exception& e) {
            qWarning() << "Watershed: Exception while marking unknown region:" << e.what();
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        } catch (...) {
            qWarning() << "Watershed: Unknown exception while marking unknown region";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Apply watershed algorithm (watershed modifies the markers in place)
        cv::Mat frameCopy = frame.clone();
        if (frameCopy.empty()) {
            qWarning() << "Watershed: Failed to clone frame";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Validate frameCopy and markers32s before watershed
        if (frameCopy.rows != markers32s.rows || frameCopy.cols != markers32s.cols) {
            qWarning() << "Watershed: Size mismatch between frameCopy and markers32s";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        try {
            cv::watershed(frameCopy, markers32s);
        } catch (const cv::Exception& e) {
            qWarning() << "Watershed: Exception in watershed algorithm:" << e.what();
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        } catch (...) {
            qWarning() << "Watershed: Unknown exception in watershed algorithm";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Create mask: foreground regions (markers > 1, and not -1 which is boundary)
        cv::Mat personMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        // Watershed marks boundaries as -1, foreground as > 1
        // Iterate through markers to create mask (more reliable than compare for CV_32S)
        try {
            for (int y = 0; y < markers32s.rows && y < personMask.rows; y++) {
                const int* markerRow = markers32s.ptr<int>(y);
                uchar* maskRow = personMask.ptr<uchar>(y);
                
                int cols = qMin(markers32s.cols, personMask.cols);
                for (int x = 0; x < cols; x++) {
                    if (markerRow[x] > 1) {
                        maskRow[x] = 255;
                    }
                }
            }
        } catch (const cv::Exception& e) {
            qWarning() << "Watershed: Exception while creating mask:" << e.what();
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        } catch (...) {
            qWarning() << "Watershed: Unknown exception while creating mask";
            return cv::Mat::zeros(frame.size(), CV_8UC1);
        }
        
        // Refine mask with morphological operations
        try {
            cv::Mat refineKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            if (!refineKernel.empty()) {
                cv::morphologyEx(personMask, personMask, cv::MORPH_CLOSE, refineKernel);
                cv::morphologyEx(personMask, personMask, cv::MORPH_OPEN, refineKernel);
            }
            
            // Fill holes in the mask using a better approach
            if (!personMask.empty()) {
                cv::Mat maskFilled = personMask.clone();
                if (!maskFilled.empty()) {
                    cv::floodFill(maskFilled, cv::Mat(), cv::Point(0, 0), cv::Scalar(255));
                    cv::bitwise_not(maskFilled, maskFilled);
                    personMask = personMask | maskFilled;
                }
            }
        } catch (const cv::Exception& e) {
            qWarning() << "Watershed: Exception in morphological operations:" << e.what();
            // Return mask as-is if refinement fails
        } catch (...) {
            qWarning() << "Watershed: Unknown exception in morphological operations";
            // Return mask as-is if refinement fails
        }
        
        return personMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Error in createWatershedPersonMask:" << e.what();
        return cv::Mat();
    }
}

//  GPU-ACCELERATED WATERSHED MASKING with Optimized Memory Management
cv::UMat Capture::createWatershedPersonMaskGPU(const cv::UMat &gpuFrame) const
{
    cv::UMat emptyMask;
    if (gpuFrame.empty()) {
        qWarning() << "GPU frame is empty, cannot create watershed mask";
        return emptyMask;
    }

    try {
        // Download to CPU for watershed (watershed algorithm not fully GPU-accelerated in OpenCV)
        cv::Mat cpuFrame;
        gpuFrame.copyTo(cpuFrame);
        
        // Use CPU watershed implementation
        cv::Mat cpuMask = createWatershedPersonMask(cpuFrame);
        
        // Upload back to GPU
        cv::UMat gpuPersonMask;
        cpuMask.copyTo(gpuPersonMask);
        
        return gpuPersonMask;

    } catch (const cv::Exception &e) {
        qWarning() << "GPU watershed masking failed:" << e.what() << "- returning empty mask";
        return emptyMask;
    } catch (const std::exception &e) {
        qWarning() << "Exception in GPU watershed masking:" << e.what();
        return emptyMask;
    } catch (...) {
        qWarning() << "Unknown exception in GPU watershed masking";
        return emptyMask;
    }
}

// All green screen helper methods removed - using watershed algorithm instead
// Removed methods: refineGreenScreenMaskWithContours, applyTemporalMaskSmoothing, 
// refineWithGrabCut, applyDistanceBasedRefinement, createTrimap, customGuidedFilter,
// extractPersonWithAlphaMatting, removeGreenSpillGPU

// Derive bounding boxes from a binary person mask
std::vector<cv::Rect> Capture::deriveDetectionsFromMask(const cv::Mat &mask) const
{
    std::vector<cv::Rect> detections;
    if (mask.empty() || mask.rows <= 0 || mask.cols <= 0) {
        return detections;
    }

    std::vector<std::vector<cv::Point>> contours;
    try {
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    } catch (const cv::Exception& e) {
        qWarning() << "Exception in findContours:" << e.what();
        return detections;
    } catch (...) {
        qWarning() << "Unknown exception in findContours";
        return detections;
    }

    // GPU MEMORY PROTECTION: Maximum detection size to prevent GPU memory overflow
    const int MAX_DETECTION_WIDTH = std::min(1920, mask.cols);
    const int MAX_DETECTION_HEIGHT = std::min(1080, mask.rows);
    const int MIN_DETECTION_AREA = 1000; // Minimum area to filter noise

    for (const auto &c : contours) {
        cv::Rect r = cv::boundingRect(c);
        
        // Filter by minimum area
        if (r.area() < MIN_DETECTION_AREA) continue;
        
        // CLAMP detection rectangle to safe bounds
        r.x = std::max(0, r.x);
        r.y = std::max(0, r.y);
        r.width = std::min(r.width, std::min(MAX_DETECTION_WIDTH, mask.cols - r.x));
        r.height = std::min(r.height, std::min(MAX_DETECTION_HEIGHT, mask.rows - r.y));
        
        // Validate final rectangle
        if (r.width > 0 && r.height > 0 && r.area() >= MIN_DETECTION_AREA) {
            detections.push_back(r);
        }
    }

    // Prefer the largest contours (likely persons)
    std::sort(detections.begin(), detections.end(), [](const cv::Rect &a, const cv::Rect &b){ return a.area() > b.area(); });
    if (detections.size() > 3) detections.resize(3);
    
    qDebug() << "Derived" << detections.size() << "valid detections from mask";
    return detections;
}

std::vector<cv::Rect> Capture::filterDetectionsByMotion(const std::vector<cv::Rect> &detections, const cv::Mat &motionMask, double overlapThreshold) const
{
    std::vector<cv::Rect> filtered;

    for (const auto& rect : detections) {
        // Validate rectangle bounds to prevent crashes
        if (rect.x < 0 || rect.y < 0 ||
            rect.width <= 0 || rect.height <= 0 ||
            rect.x + rect.width > motionMask.cols ||
            rect.y + rect.height > motionMask.rows) {
            continue; // Skip invalid rectangles
        }

        // Create ROI for this detection
        cv::Mat roi = motionMask(rect);

        // Calculate percentage of motion pixels in the detection area
        int motionPixels = cv::countNonZero(roi);
        double motionRatio = (double)motionPixels / (roi.rows * roi.cols);

        // Keep detection if there's significant motion (more than the overlap threshold)
        if (motionRatio > overlapThreshold) {
            filtered.push_back(rect);
        }
    }

    return filtered;
}

// Hand Detection Method Implementations
// Hand detection completely removed - all method implementations removed

// New method to safely enable processing modes after camera is stable
void Capture::enableProcessingModes()
{
    // Only enable heavy processing modes after camera has been running for a while
    if (frameCount > 50) {
        qDebug() << "Camera stable, enabling processing modes";
        // You can enable specific modes here if needed
    }
}
// Method to disable heavy processing modes for non-capture pages
void Capture::disableProcessingModes()
{
    qDebug() << "Disabling heavy processing modes for non-capture pages";

    // Hand detection removed

    // Disable segmentation outside capture interface
    disableSegmentationOutsideCapture();

    // Reset processing timers
    m_personDetectionTimer.restart();

    qDebug() << "Heavy processing modes disabled - camera continues running";
}

// Loading camera label management methods
void Capture::showLoadingCameraLabel()
{
    // Clear the video label to show black screen instead of previous frame
    if (ui->videoLabel) {
        ui->videoLabel->clear();
        ui->videoLabel->setText(""); // Ensure no text is displayed
        qDebug() << "Video label cleared to black screen";
    }
    
    if (loadingCameraLabel) {
        // Set basic size and position for visibility
        loadingCameraLabel->setFixedSize(500, 120); // Increased height from 100 to 120

        // Center the label properly in the middle of the screen
        int x = (width() - 500) / 2; // Center horizontally
        int y = (height() - 120) / 2; // Center vertically
        loadingCameraLabel->move(x, y);

        loadingCameraLabel->show();
        loadingCameraLabel->raise(); // Bring to front
        qDebug() << "Loading camera label shown centered at position:" << x << "," << y;
    }
}



void Capture::hideLoadingCameraLabel()
{
    if (loadingCameraLabel) {
        loadingCameraLabel->hide();
        qDebug() << "Loading camera label hidden";
    }
}

// Loading label management (thread-safe)
void Capture::handleFirstFrame()
{
    // This method runs in the main thread (thread-safe)
    qDebug() << "handleFirstFrame() called in thread:" << QThread::currentThread();

    // Hide the loading camera label when first frame is received
    hideLoadingCameraLabel();

    //  Initialize GPU Memory Pool when first frame is received
    if (!m_gpuMemoryPoolInitialized && cv::ocl::useOpenCL()) {
        try {
            qDebug() << " Initializing GPU Memory Pool on first frame...";
            m_gpuMemoryPool.initialize(1280, 720); // Initialize with common camera resolution
            m_gpuMemoryPoolInitialized = true;
            qDebug() << "GPU Memory Pool initialized successfully on first frame";
        } catch (const cv::Exception& e) {
            qWarning() << " GPU Memory Pool initialization failed on first frame:" << e.what();
            m_gpuMemoryPoolInitialized = false;
        }
    }

    // Mark camera as initialized for the first time
    if (!m_cameraFirstInitialized) {
        m_cameraFirstInitialized = true;
        qDebug() << "Camera first initialization complete - loading label hidden permanently";
    } else {
        qDebug() << "Camera frame received (not first initialization)";
    }
}

// Segmentation Control Methods for Capture Interface
void Capture::enableSegmentationInCapture()
{
    qDebug() << "Enabling segmentation for capture interface";
    m_segmentationEnabledInCapture = true;

    // Debug dynamic video background state
    qDebug() << "Dynamic video background state:";
    qDebug() << "  - m_useDynamicVideoBackground:" << m_useDynamicVideoBackground;
    qDebug() << "  - m_videoPlaybackActive:" << m_videoPlaybackActive;
    qDebug() << "  - m_dynamicVideoPath:" << m_dynamicVideoPath;
    qDebug() << "  - m_dynamicVideoFrame empty:" << m_dynamicVideoFrame.empty();

    // If we have a dynamic video background but playback is not active, restart it
    if (m_useDynamicVideoBackground && !m_videoPlaybackActive && !m_dynamicVideoPath.isEmpty()) {
        qDebug() << "Dynamic video background detected but playback not active - restarting video playback";
        
        // Restart video playback timer
        if (m_videoPlaybackTimer && m_videoFrameInterval > 0) {
            m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
            m_videoPlaybackTimer->start();
            m_videoPlaybackActive = true;
            qDebug() << "Video playback timer restarted with interval:" << m_videoFrameInterval << "ms";
        }
        
        // If we don't have a current frame, try to read the first frame
        if (m_dynamicVideoFrame.empty()) {
            cv::Mat firstFrame;
            bool frameRead = false;

            // CUDA VideoReader removed - using CPU VideoCapture
            if (m_dynamicVideoCap.isOpened()) {
                if (m_dynamicVideoCap.read(firstFrame) && !firstFrame.empty()) {
                    frameRead = true;
                }
            }

            if (frameRead && !firstFrame.empty()) {
                m_dynamicVideoFrame = firstFrame.clone();
                qDebug() << "Successfully loaded first frame for segmentation display";
            } else {
                qWarning() << "Failed to load first frame for segmentation display";
            }
        }
    }

    // Enable segmentation by default
    m_segmentationEnabledInCapture = true;
    qDebug() << "Segmentation enabled by default for capture interface";

    // Clear any previous segmentation results to force new processing
    m_lastSegmentedFrame = cv::Mat();
    m_lastDetections.clear();

    // Update UI to reflect the current state
    updateDebugDisplay();
}
void Capture::disableSegmentationOutsideCapture()
{
    qDebug() << "Disabling segmentation outside capture interface";

    // Disable segmentation
    m_segmentationEnabledInCapture = false;

    // Clear any cached segmentation data
    m_lastSegmentedFrame = cv::Mat();
    m_lastDetections.clear();

    // Reset GPU utilization flags
    m_gpuUtilized = false;

    // Update UI
    updateDebugDisplay();

    qDebug() << "Segmentation disabled";
}

void Capture::restoreSegmentationState()
{
    qDebug() << "Restoring segmentation state for capture interface";

    // Enable segmentation by default when returning to capture interface
    m_segmentationEnabledInCapture = true;
    qDebug() << "Segmentation enabled by default";

    updateDebugDisplay();
}
bool Capture::isSegmentationEnabledInCapture() const
{
    return m_segmentationEnabledInCapture;
}

// Background Template Control Methods
void Capture::setSelectedBackgroundTemplate(const QString &path)
{
    m_selectedBackgroundTemplate = path;
    m_useBackgroundTemplate = !path.isEmpty();
    qDebug() << "Background template set to:" << path << "Use template:" << m_useBackgroundTemplate;
    
    // Clear cached template background to force reload with new template
    m_lastTemplateBackground = cv::Mat();
    qDebug() << "Cleared cached template background to force reload";
    
    // Automatically set the reference template for lighting correction
    if (m_useBackgroundTemplate && !path.isEmpty()) {
        qDebug() << "Setting reference template for lighting correction...";
        setReferenceTemplate(path);
        qDebug() << "Reference template automatically set for lighting correction";
        
        // VERIFY it was set
        if (m_lightingCorrector) {
            cv::Mat refTemplate = m_lightingCorrector->getReferenceTemplate();
            qDebug() << "VERIFICATION: Reference template is" << (refTemplate.empty() ? "EMPTY " : "SET ");
            if (!refTemplate.empty()) {
                qDebug() << "Reference template size:" << refTemplate.cols << "x" << refTemplate.rows;
            }
        }
    }
}

QString Capture::getSelectedBackgroundTemplate() const
{
    return m_selectedBackgroundTemplate;
}

// Video Template Duration Control Methods
void Capture::setVideoTemplateDuration(int durationSeconds)
{
    if (durationSeconds > 0) {
        m_currentVideoTemplate.durationSeconds = durationSeconds;
        qDebug() << "VIDEO TEMPLATE DURATION UPDATED:" << durationSeconds << "seconds";
        qDebug() << "  - Template name:" << m_currentVideoTemplate.name;
        qDebug() << "  - Recording will automatically stop after" << durationSeconds << "seconds";
    } else {
        qWarning() << "Invalid duration specified:" << durationSeconds << "seconds (must be > 0)";
    }
}

int Capture::getVideoTemplateDuration() const
{
    return m_currentVideoTemplate.durationSeconds;
}

//  GPU MEMORY POOL IMPLEMENTATION

GPUMemoryPool::GPUMemoryPool()
    : currentFrameBuffer(0)
    , currentSegBuffer(0)
    , currentDetBuffer(0)
    , currentTempBuffer(0)
    , currentGuidedFilterBuffer(0)
    , currentBoxFilterBuffer(0)
    , currentEdgeBlurBuffer(0)
    , currentEdgeDetectionBuffer(0)
    , initialized(false)
    , poolWidth(0)
    , poolHeight(0)
{
    qDebug() << " GPU Memory Pool: Constructor called (OpenCL)";
}

GPUMemoryPool::~GPUMemoryPool()
{
    qDebug() << " GPU Memory Pool: Destructor called";
    release();
}
void GPUMemoryPool::initialize(int width, int height)
{
    if (initialized && poolWidth == width && poolHeight == height) {
        qDebug() << " GPU Memory Pool: Already initialized with correct dimensions";
        return;
    }

    qDebug() << " GPU Memory Pool: Initializing with dimensions" << width << "x" << height;

    try {
        // Release existing resources
        release();

        // Initialize frame buffers (triple buffering) - using OpenCL UMat
        for (int i = 0; i < 3; ++i) {
            gpuFrameBuffers[i].create(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Frame buffer" << i << "allocated (OpenCL)";
        }

        // Initialize segmentation buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuSegmentationBuffers[i].create(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Segmentation buffer" << i << "allocated (OpenCL)";
        }

        // Initialize detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuDetectionBuffers[i].create(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Detection buffer" << i << "allocated (OpenCL)";
        }

        // Initialize temporary buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuTempBuffers[i].create(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Temp buffer" << i << "allocated (OpenCL)";
        }

        //  Initialize guided filtering buffers (quad buffering for complex operations)
        for (int i = 0; i < 4; ++i) {
            gpuGuidedFilterBuffers[i].create(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Guided filter buffer" << i << "allocated (OpenCL)";
        }

        // Initialize box filter buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuBoxFilterBuffers[i].create(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Box filter buffer" << i << "allocated (OpenCL)";
        }

        //  Initialize edge blurring buffers (triple buffering for complex operations)
        for (int i = 0; i < 3; ++i) {
            gpuEdgeBlurBuffers[i].create(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Edge blur buffer" << i << "allocated (OpenCL)";
        }

        // Initialize edge detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuEdgeDetectionBuffers[i].create(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Edge detection buffer" << i << "allocated (OpenCL)";
        }

        // Create reusable morphology kernel (OpenCL uses regular OpenCV operations)
        morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

        qDebug() << " GPU Memory Pool: OpenCL buffers initialized successfully";

        // Update state
        poolWidth = width;
        poolHeight = height;
        initialized = true;

        qDebug() << " GPU Memory Pool: Initialization completed successfully";

    } catch (const cv::Exception& e) {
        qWarning() << " GPU Memory Pool: Initialization failed:" << e.what();
        release();
    }
}

cv::UMat& GPUMemoryPool::getNextFrameBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuFrameBuffers[currentFrameBuffer];
    currentFrameBuffer = (currentFrameBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::UMat& GPUMemoryPool::getNextSegmentationBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuSegmentationBuffers[currentSegBuffer];
    currentSegBuffer = (currentSegBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::UMat& GPUMemoryPool::getNextDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuDetectionBuffers[currentDetBuffer];
    currentDetBuffer = (currentDetBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::UMat& GPUMemoryPool::getNextTempBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuTempBuffers[currentTempBuffer];
    currentTempBuffer = (currentTempBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Guided Filtering buffer access methods
cv::UMat& GPUMemoryPool::getNextGuidedFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuGuidedFilterBuffers[currentGuidedFilterBuffer];
    currentGuidedFilterBuffer = (currentGuidedFilterBuffer + 1) % 4; // Quad buffering
    return buffer;
}

cv::UMat& GPUMemoryPool::getNextBoxFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuBoxFilterBuffers[currentBoxFilterBuffer];
    currentBoxFilterBuffer = (currentBoxFilterBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Edge Blurring buffer access methods
cv::UMat& GPUMemoryPool::getNextEdgeBlurBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuEdgeBlurBuffers[currentEdgeBlurBuffer];
    currentEdgeBlurBuffer = (currentEdgeBlurBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::UMat& GPUMemoryPool::getNextEdgeDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::UMat emptyBuffer;
        return emptyBuffer;
    }

    cv::UMat& buffer = gpuEdgeDetectionBuffers[currentEdgeDetectionBuffer];
    currentEdgeDetectionBuffer = (currentEdgeDetectionBuffer + 1) % 2; // Double buffering
    return buffer;
}

void GPUMemoryPool::release()
{
    if (!initialized) {
        return;
    }

    qDebug() << " GPU Memory Pool: Releasing resources";

    // Release GPU buffers
    for (int i = 0; i < 3; ++i) {
        gpuFrameBuffers[i].release();
    }

    for (int i = 0; i < 2; ++i) {
        gpuSegmentationBuffers[i].release();
        gpuDetectionBuffers[i].release();
        gpuTempBuffers[i].release();
        gpuBoxFilterBuffers[i].release();
    }

    // Release guided filtering buffers
    for (int i = 0; i < 4; ++i) {
        gpuGuidedFilterBuffers[i].release();
    }

    // Release edge blurring buffers
    for (int i = 0; i < 3; ++i) {
        gpuEdgeBlurBuffers[i].release();
    }

    for (int i = 0; i < 2; ++i) {
        gpuEdgeDetectionBuffers[i].release();
    }

    // Clear morphology kernel (OpenCL uses regular OpenCV operations)
    morphKernel.release();

    // Reset state
    initialized = false;
    poolWidth = 0;
    poolHeight = 0;
    currentFrameBuffer = 0;
    currentSegBuffer = 0;
    currentDetBuffer = 0;
    currentTempBuffer = 0;

    qDebug() << " GPU Memory Pool: Resources released";
}

void GPUMemoryPool::resetBuffers()
{
    if (!initialized) {
        return;
    }

    qDebug() << " GPU Memory Pool: Resetting buffer indices";

    currentFrameBuffer = 0;
    currentSegBuffer = 0;
    currentDetBuffer = 0;
    currentTempBuffer = 0;
    currentGuidedFilterBuffer = 0;
    currentBoxFilterBuffer = 0;
    currentEdgeBlurBuffer = 0;
    currentEdgeDetectionBuffer = 0;
}

//  ASYNCHRONOUS RECORDING SYSTEM IMPLEMENTATION

void Capture::initializeRecordingSystem()
{
    qDebug() << " ASYNC RECORDING: Initializing recording system...";

    try {
        // Create recording thread
        if (!m_recordingThread) {
            m_recordingThread = new QThread(this);
            m_recordingThread->setObjectName("RecordingThread");
        }

        // Create recording frame timer
        if (!m_recordingFrameTimer) {
            m_recordingFrameTimer = new QTimer();
            m_recordingFrameTimer->setObjectName("RecordingFrameTimer");
            m_recordingFrameTimer->moveToThread(m_recordingThread);
            connect(m_recordingFrameTimer, &QTimer::timeout, this, &Capture::processRecordingFrame, Qt::QueuedConnection);
        }

        // OpenCL doesn't use explicit streams - operations are automatically parallelized

        // Initialize GPU recording buffer (OpenCL)
        if (cv::ocl::useOpenCL()) {
            m_recordingGpuBuffer = cv::UMat(720, 1280, CV_8UC3);
            qDebug() << " ASYNC RECORDING: GPU recording buffer initialized";
        }

        // Start recording thread
        m_recordingThread->start();
        m_recordingThreadActive = true;

        // Start processing timer
        m_recordingFrameTimer->start(16); // 60 FPS processing rate

        qDebug() << " ASYNC RECORDING: Recording system initialized successfully";

    } catch (const std::exception& e) {
        qWarning() << " ASYNC RECORDING: Initialization failed:" << e.what();
        cleanupRecordingSystem();
    }
}

void Capture::cleanupRecordingSystem()
{
    qDebug() << " ASYNC RECORDING: Cleaning up recording system...";

    // Stop processing timer
    if (m_recordingFrameTimer) {
        m_recordingFrameTimer->stop();
    }

    // Stop and cleanup recording thread
    if (m_recordingThread && m_recordingThreadActive) {
        m_recordingThread->quit();
        m_recordingThread->wait(1000); // Wait up to 1 second
        m_recordingThreadActive = false;
    }

    // Clear frame queue
    {
        QMutexLocker locker(&m_recordingMutex);
        m_recordingFrameQueue.clear();
    }

    // Release GPU resources
    m_recordingGpuBuffer.release();

    qDebug() << " ASYNC RECORDING: Recording system cleaned up";
}
void Capture::queueFrameForRecording(const cv::Mat &frame)
{
    if (!m_recordingThreadActive) {
        return;
    }

    // Queue frame for asynchronous processing
    {
        QMutexLocker locker(&m_recordingMutex);

        // Limit queue size to prevent memory issues
        if (m_recordingFrameQueue.size() < 10) {
            m_recordingFrameQueue.enqueue(frame.clone());
            qDebug() << " ASYNC RECORDING: Frame queued, queue size:" << m_recordingFrameQueue.size();
        } else {
            qWarning() << " ASYNC RECORDING: Queue full, dropping frame";
        }
    }
}

//  ASYNC VIDEO PROCESSING COMPLETION HANDLER
void Capture::onVideoProcessingFinished()
{
    qDebug() << " Video processing finished in background thread";
    
    // CRASH PREVENTION: Check if watcher is valid
    if (!m_lightingWatcher) {
        qWarning() << " Lighting watcher is null in completion handler";
        // Fallback: send original frames
        if (!m_recordedFrames.isEmpty()) {
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
        }
        return;
    }
    
    // CRASH PREVENTION: Check if future was cancelled or failed
    if (m_lightingWatcher->isCanceled()) {
        qWarning() << " Video processing was cancelled";
        // Fallback: send original frames
        if (!m_recordedFrames.isEmpty()) {
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
        }
        return;
    }
    
    try {
        // CRASH PREVENTION: Check if future is finished before accessing result
        if (!m_lightingWatcher->isFinished()) {
            qWarning() << " Future is not finished yet, waiting...";
            m_lightingWatcher->waitForFinished(); // Wait for completion
            if (!m_lightingWatcher->isFinished()) {
                qWarning() << " Future did not finish";
                if (!m_recordedFrames.isEmpty()) {
                    emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
                    emit showFinalOutputPage();
                }
                return;
            }
        }
        
        // Get the processed frames from the background thread
        QList<QPixmap> processedFrames;
        try {
            processedFrames = m_lightingWatcher->result();
        } catch (const QException& e) {
            qWarning() << " Exception retrieving processed frames:" << e.what();
            throw; // Re-throw to be caught by outer catch
        }
        
        // CRASH PREVENTION: Validate processed frames
        if (processedFrames.isEmpty()) {
            qWarning() << " Processed frames list is empty, using original frames";
            if (!m_recordedFrames.isEmpty()) {
                emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
                emit showFinalOutputPage();
            }
            return;
        }
        
        qDebug() << " DIRECT CAPTURE RECORDING: Processing complete";
        qDebug() << "Original frames:" << m_originalRecordedFrames.size() 
                 << "Processed frames:" << processedFrames.size();
        
        // CRASH PREVENTION: Validate original frames before emitting
        if (m_originalRecordedFrames.isEmpty()) {
            qWarning() << " Original recorded frames are empty, sending processed frames only";
            emit videoRecorded(processedFrames, m_adjustedRecordingFPS);
        } else {
            // Send processed frames to final output page
            emit videoRecordedWithComparison(processedFrames, m_originalRecordedFrames, m_adjustedRecordingFPS);
        }
        
        //  FINAL STEP: Show final output page after all processing is complete
        emit showFinalOutputPage();
        qDebug() << " DIRECT CAPTURE RECORDING: Showing final output page";
        
    } catch (const QException& e) {
        qWarning() << " QException retrieving processed frames:" << e.what();
        // Fallback: send original frames
        if (!m_recordedFrames.isEmpty()) {
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
        }
    } catch (const std::exception& e) {
        qWarning() << " Error retrieving processed frames:" << e.what();
        // Fallback: send original frames
        if (!m_recordedFrames.isEmpty()) {
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
        }
    } catch (...) {
        qWarning() << " Unknown exception in onVideoProcessingFinished";
        // Fallback: send original frames
        if (!m_recordedFrames.isEmpty()) {
            emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
        }
    }
}

void Capture::processRecordingFrame()
{
    // This method is no longer needed since we're capturing display directly
    // Keeping it for future expansion if needed
    qDebug() << " ASYNC RECORDING: Process recording frame called (not used in direct capture mode)";
}

QPixmap Capture::processFrameForRecordingGPU(const cv::Mat &frame)
{
    QPixmap result;

    try {
        //  GPU-ACCELERATED FRAME PROCESSING

        // Upload frame to GPU
        cv::UMat gpuFrame;
        frame.copyTo(gpuFrame);

        // Get target size
        QSize labelSize = m_cachedLabelSize.isValid() ? m_cachedLabelSize : QSize(1280, 720);

        // GPU-accelerated scaling if needed
        cv::UMat gpuScaled;
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Check if we're in segmentation mode with background template or dynamic video background
            if (m_segmentationEnabledInCapture && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode, just fit to label
                cv::resize(gpuFrame, gpuScaled, cv::Size(labelSize.width(), labelSize.height()), 0, 0, cv::INTER_LINEAR);
            } else {
                // Apply frame scaling for other modes
                int newWidth = qRound(frame.cols * m_personScaleFactor);
                int newHeight = qRound(frame.rows * m_personScaleFactor);
                
                //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
                newWidth = qMax(1, newWidth);
                newHeight = qMax(1, newHeight);
                
                qDebug() << " GPU RECORDING: Scaling frame to" << newWidth << "x" << newHeight << "with factor" << m_personScaleFactor;
                cv::resize(gpuFrame, gpuScaled, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
            }
        } else {
            // No scaling needed, just fit to label
            cv::resize(gpuFrame, gpuScaled, cv::Size(labelSize.width(), labelSize.height()), 0, 0, cv::INTER_LINEAR);
        }

        // Download processed frame
        cv::Mat processedFrame;
        gpuScaled.copyTo(processedFrame);

        // OpenCL synchronization is automatic

        // Convert to QPixmap
        QImage qImage = cvMatToQImage(processedFrame);
        result = QPixmap::fromImage(qImage);

        qDebug() << " ASYNC RECORDING: GPU frame processing completed";

    } catch (const cv::Exception& e) {
        qWarning() << " ASYNC RECORDING: GPU processing failed:" << e.what();

        // Fallback to CPU processing
        QImage qImage = cvMatToQImage(frame);
        result = QPixmap::fromImage(qImage);

        // Apply scaling on CPU as fallback
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            QSize labelSize = m_cachedLabelSize.isValid() ? m_cachedLabelSize : QSize(1280, 720);
            result = result.scaled(labelSize, Qt::KeepAspectRatioByExpanding, Qt::FastTransformation);
        }
    }

    return result;
}
// Resource Management Methods
void Capture::cleanupResources()
{
    qDebug() << "Capture::cleanupResources - Cleaning up resources when leaving capture page";
    
    // Stop all timers
    if (m_videoPlaybackTimer && m_videoPlaybackTimer->isActive()) {
        m_videoPlaybackTimer->stop();
        qDebug() << "Stopped video playback timer";
    }
    
    if (recordTimer && recordTimer->isActive()) {
        recordTimer->stop();
        qDebug() << "Stopped record timer";
    }
    
    if (recordingFrameTimer && recordingFrameTimer->isActive()) {
        recordingFrameTimer->stop();
        qDebug() << "Stopped recording frame timer";
    }
    
    if (debugUpdateTimer && debugUpdateTimer->isActive()) {
        debugUpdateTimer->stop();
        qDebug() << "Stopped debug update timer";
    }
    
    // Stop recording if active
    if (m_isRecording) {
        stopRecording();
        qDebug() << "Stopped active recording";
    }
    
    // Disable all processing modes
    disableProcessingModes();
    disableSegmentationOutsideCapture();
    
    // Release video resources
    disableDynamicVideoBackground();
    
    // Release GPU memory
    if (m_gpuMemoryPoolInitialized) {
        m_gpuMemoryPool.release();
        m_gpuMemoryPoolInitialized = false;
        qDebug() << "Released GPU memory pool";
    }
    
    // Clear frame buffers
    m_currentFrame.release();
    m_lastSegmentedFrame.release();
    m_dynamicVideoFrame.release();
    m_dynamicGpuFrame.release();
    m_gpuVideoFrame.release();
    m_gpuSegmentedFrame.release();
    m_gpuPersonMask.release();
    m_gpuBackgroundFrame.release();
    m_recordingGpuBuffer.release();
    
    // Clear detection results
    m_lastDetections.clear();
    // Hand detection removed
    
    qDebug() << "Capture::cleanupResources - Resource cleanup completed";
}

void Capture::initializeResources()
{
    qDebug() << " Capture::initializeResources - Initializing resources when entering capture page";
    
    // Initialize GPU memory pool if available
    if (isGPUOnlyProcessingAvailable() && !m_gpuMemoryPoolInitialized) {
        m_gpuMemoryPool.initialize(1280, 720); // Default resolution
        m_gpuMemoryPoolInitialized = true;
        qDebug() << " Initialized GPU memory pool";
    }
    
    // Initialize person detection
    initializePersonDetection();
    
    // Initialize hand detection
    // Hand detection removed
    
    // Start debug update timer
    if (debugUpdateTimer) {
        debugUpdateTimer->start(1000); // Update every second
        qDebug() << " Started debug update timer";
    }
    
    qDebug() << " Capture::initializeResources - Resource initialization completed";
}
// ============================================================================
// LIGHTING CORRECTION IMPLEMENTATION
// ============================================================================

void Capture::initializeLightingCorrection()
{
    qDebug() << "Initializing lighting correction system";
    
    try {
        // Create lighting corrector instance
        m_lightingCorrector = new LightingCorrector();
        
        // Initialize the lighting corrector
        if (m_lightingCorrector->initialize()) {
            qDebug() << "Lighting correction system initialized successfully";
            qDebug() << "GPU acceleration:" << (m_lightingCorrector->isGPUAvailable() ? "Available" : "Not available");
         } else {
            qWarning() << "Lighting correction initialization failed";
            delete m_lightingCorrector;
            m_lightingCorrector = nullptr;
        }
        
    } catch (const std::exception& e) {
        qWarning() << "Lighting correction initialization failed:" << e.what();
        if (m_lightingCorrector) {
            delete m_lightingCorrector;
            m_lightingCorrector = nullptr;
        }
    }
}
bool Capture::isGPULightingAvailable() const
{
    return m_lightingCorrector ? m_lightingCorrector->isGPUAvailable() : false;
}

void Capture::setReferenceTemplate(const QString &templatePath)
{
    if (m_lightingCorrector) {
        QString resolvedPath = resolveTemplatePath(templatePath);
        if (!resolvedPath.isEmpty()) {
            if (m_lightingCorrector->setReferenceTemplate(resolvedPath)) {
                qDebug() << "Reference template set for lighting correction:" << resolvedPath;
            } else {
                qWarning() << "Failed to set reference template from resolved path:" << resolvedPath;
            }
        } else {
            qWarning() << "Could not resolve reference template path:" << templatePath;
        }
    }
}

void Capture::setSubtractionReferenceImage(const QString &imagePath)
{
    if (imagePath.isEmpty()) {
        m_subtractionReferenceImage.release();
        qDebug() << "Subtraction reference image cleared";
        return;
    }
    
    // Try to resolve the path
    QString resolvedPath = resolveTemplatePath(imagePath);
    if (resolvedPath.isEmpty()) {
        // Try direct path
        if (QFile::exists(imagePath)) {
            resolvedPath = imagePath;
        } else {
            qWarning() << "Could not resolve subtraction reference image path:" << imagePath;
            m_subtractionReferenceImage.release();
            return;
        }
    }
    
    // Load the reference image
    cv::Mat refImage = cv::imread(resolvedPath.toStdString());
    if (!refImage.empty()) {
        m_subtractionReferenceImage = refImage;
        qDebug() << "Subtraction reference image loaded from:" << resolvedPath
                 << "Size:" << m_subtractionReferenceImage.cols << "x" << m_subtractionReferenceImage.rows;
    } else {
        qWarning() << "Failed to load subtraction reference image from:" << resolvedPath;
        m_subtractionReferenceImage.release();
    }
}

void Capture::setSubtractionReferenceImage2(const QString &imagePath)
{
    if (imagePath.isEmpty()) {
        m_subtractionReferenceImage2.release();
        qDebug() << "Subtraction reference image 2 cleared";
        return;
    }
    
    // Try to resolve the path
    QString resolvedPath = resolveTemplatePath(imagePath);
    if (resolvedPath.isEmpty()) {
        // Try direct path
        if (QFile::exists(imagePath)) {
            resolvedPath = imagePath;
        } else {
            qWarning() << "Could not resolve subtraction reference image 2 path:" << imagePath;
            m_subtractionReferenceImage2.release();
            return;
        }
    }
    
    // Load the reference image
    cv::Mat refImage = cv::imread(resolvedPath.toStdString());
    if (!refImage.empty()) {
        m_subtractionReferenceImage2 = refImage;
        qDebug() << "Subtraction reference image 2 loaded from:" << resolvedPath
                 << "Size:" << m_subtractionReferenceImage2.cols << "x" << m_subtractionReferenceImage2.rows;
    } else {
        qWarning() << "Failed to load subtraction reference image 2 from:" << resolvedPath;
        m_subtractionReferenceImage2.release();
    }
}

void Capture::setSubtractionReferenceBlendWeight(double weight)
{
    m_subtractionBlendWeight = std::max(0.0, std::min(1.0, weight));
    qDebug() << "Subtraction reference blend weight set to:" << m_subtractionBlendWeight;
}

cv::Mat Capture::applyPostProcessingLighting()
{
    qDebug() << "POST-PROCESSING: Apply lighting to raw person data and re-composite";
    
    // Check if we have raw person data
    if (m_lastRawPersonRegion.empty() || m_lastRawPersonMask.empty()) {
        qWarning() << "No raw person data available, returning original segmented frame";
        return m_lastSegmentedFrame.clone();
    }
    
    // Start from a clean background template/dynamic video frame (no person composited yet)
    cv::Mat result;
    cv::Mat cleanBackground;
    if (!m_lastTemplateBackground.empty()) {
        cleanBackground = m_lastTemplateBackground.clone();
        qDebug() << "POST-PROCESSING: Using cached template background";
    } else if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
        // Check if this is bg6.png (white background special case)
        if (m_selectedBackgroundTemplate.contains("bg6.png")) {
            // Create white background instead of loading a file
            cleanBackground = cv::Mat(m_lastSegmentedFrame.size(), m_lastSegmentedFrame.type(), cv::Scalar(255, 255, 255));
            qDebug() << "POST-PROCESSING: Created white background for bg6.png";
        } else {
            QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
            cv::Mat bg = cv::imread(resolvedPath.toStdString());
            if (!bg.empty()) {
                cv::resize(bg, cleanBackground, m_lastSegmentedFrame.size());
                qDebug() << "POST-PROCESSING: Loaded background template from" << resolvedPath;
            } else {
                qWarning() << "POST-PROCESSING: Failed to load background from" << resolvedPath;
            }
        }
    }
    if (cleanBackground.empty()) {
        // Fallback to a blank frame matching the output size if no cached template available
        cleanBackground = cv::Mat::zeros(m_lastSegmentedFrame.size(), m_lastSegmentedFrame.type());
        qDebug() << "POST-PROCESSING: Using black background (fallback)";
    }
    result = cleanBackground.clone();
    
    // Apply lighting to the raw person region (post-processing as in original)
    cv::Mat lightingCorrectedPerson = applyLightingToRawPersonRegion(m_lastRawPersonRegion, m_lastRawPersonMask);
    
    // Scale the lighting-corrected person respecting the person scale factor (same as original segmentation)
    cv::Mat scaledPerson, scaledMask;
    cv::Size backgroundSize = result.size();
    cv::Size scaledPersonSize;
    
    if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
        int scaledWidth = static_cast<int>(backgroundSize.width * m_personScaleFactor + 0.5);
        int scaledHeight = static_cast<int>(backgroundSize.height * m_personScaleFactor + 0.5);
        scaledWidth = qMax(1, scaledWidth);
        scaledHeight = qMax(1, scaledHeight);
        scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
        qDebug() << "POST-PROCESSING: Scaling person to" << scaledWidth << "x" << scaledHeight << "with factor" << m_personScaleFactor;
    } else {
        scaledPersonSize = backgroundSize;
    }
    
    cv::resize(lightingCorrectedPerson, scaledPerson, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
    cv::resize(m_lastRawPersonMask, scaledMask, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
    
    // Calculate centered offset for placing the scaled person
    cv::Size actualScaledSize(scaledPerson.cols, scaledPerson.rows);
    int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
    int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;
    
    // If person is scaled down, we need to place it on a full-size canvas at the centered position
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
            
            qDebug() << "POST-PROCESSING: Placed scaled person at offset" << xOffset << "," << yOffset;
        } else {
            qWarning() << "POST-PROCESSING: Invalid offset, using direct copy";
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
    
    // Soft-edge alpha blend only around the person (robust feather, background untouched)
    try {
        // Ensure binary mask 0/255
        cv::Mat binMask;
        cv::threshold(scaledMask, binMask, 127, 255, cv::THRESH_BINARY);

        // First: shrink mask slightly to avoid fringe, then hard-copy interior
        cv::Mat interiorMask;
        cv::erode(binMask, interiorMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // ~2px shrink
        scaledPerson.copyTo(result, interiorMask);

        // Use clean template/dynamic background for edge blending
        cv::Mat backgroundFrame = cleanBackground;

        //  OpenCL-Accelerated Guided image filtering to refine a soft alpha only on a thin edge ring
        // Guidance is the current output (result) which already has person hard-copied
        const int gfRadius = 8; // window size (reduced for better performance)
        const float gfEps = 1e-2f; // regularization (increased for better performance)
        
        // Use GPU memory pool for optimized guided filtering (OpenCL)
        cv::Mat alphaFloat = guidedFilterGrayAlphaOpenCLOptimized(result, binMask, gfRadius, gfEps, m_gpuMemoryPool);
        
        //  ENHANCED: Apply edge blurring to create smooth transitions between background and segmented object
        const float edgeBlurRadius = 3.0f; // Increased blur radius for better background-object transition
        cv::Mat edgeBlurredPerson = applyEdgeBlurringOpenCL(scaledPerson, binMask, backgroundFrame, edgeBlurRadius, m_gpuMemoryPool);
        if (!edgeBlurredPerson.empty()) {
            scaledPerson = edgeBlurredPerson;
            qDebug() << "STATIC MODE: Applied OpenCL edge blurring with radius" << edgeBlurRadius;
        } else {
            // Fallback to alternative method if OpenCL fails
            edgeBlurredPerson = applyEdgeBlurringAlternative(scaledPerson, binMask, edgeBlurRadius);
            if (!edgeBlurredPerson.empty()) {
                scaledPerson = edgeBlurredPerson;
                qDebug() << "STATIC MODE: Applied alternative edge blurring with radius" << edgeBlurRadius;
            }
        }
        
        // Build thin inner/outer rings around the boundary for localized updates only
        cv::Mat inner, outer, ringInner, ringOuter;
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*1+1, 2*1+1))); // shrink by ~1px for inner ring
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*4+1, 2*4+1))); // expand by ~4px for outer ring
        cv::subtract(binMask, inner, ringInner);   // just inside the boundary
        cv::subtract(outer, binMask, ringOuter);   // just outside the boundary
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);  // full person interior remains 1
        alphaFloat.setTo(0.0f, outer == 0); // outside remains 0
        // Strongly bias ring blend toward template to eliminate colored outlines
        alphaFloat = alphaFloat * 0.3f;

        // Optional de-spill strictly on edge ring (disabled to avoid hue shifts)
        // To enable, reduce saturation multiplicatively only on the ring to prevent tinting.
        //{
        //    cv::Mat ring; cv::subtract(outer, inner, ring);
        //    if (cv::countNonZero(ring) > 0) {
        //        cv::Mat hsv; cv::cvtColor(scaledPerson, hsv, cv::COLOR_BGR2HSV);
        //        std::vector<cv::Mat> ch; cv::split(hsv, ch);
        //        cv::Mat satScaled; cv::multiply(ch[1], 0.6, satScaled, 1.0, ch[1].type());
        //        satScaled.copyTo(ch[1], ring);
        //        cv::merge(ch, hsv);
        //        cv::cvtColor(hsv, scaledPerson, cv::COLOR_HSV2BGR);
        //    }
        //}

        // Composite only where outer>0 to avoid touching background (use original colors)
        cv::Mat personF, bgF; scaledPerson.convertTo(personF, CV_32F); backgroundFrame.convertTo(bgF, CV_32F);
        std::vector<cv::Mat> a3 = {alphaFloat, alphaFloat, alphaFloat};
        cv::Mat alpha3; cv::merge(a3, alpha3);
        // Inner ring: solve for decontaminated foreground using matting equation, then composite
        // F_clean = (I - (1 - alpha) * B) / max(alpha, eps)
        cv::Mat alphaSafe;
        cv::max(alpha3, 0.05f, alphaSafe); // avoid division by very small alpha
        cv::Mat Fclean = (personF - bgF.mul(1.0f - alpha3)).mul(1.0f / alphaSafe);
        cv::Mat compF = Fclean.mul(alpha3) + bgF.mul(1.0f - alpha3);
        cv::Mat out8u; compF.convertTo(out8u, CV_8U);
        out8u.copyTo(result, ringInner);

        // Outer ring: copy template directly to eliminate any colored outline
        backgroundFrame.copyTo(result, ringOuter);
        
        //  FINAL EDGE BLURRING: Apply edge blurring to the final composite result
        const float finalEdgeBlurRadius = 4.0f; // Stronger blur for final result
        cv::Mat finalEdgeBlurred = applyEdgeBlurringOpenCL(result, binMask, cleanBackground, finalEdgeBlurRadius, m_gpuMemoryPool);
        if (!finalEdgeBlurred.empty()) {
            result = finalEdgeBlurred;
            qDebug() << "STATIC MODE: Applied final OpenCL edge blurring to composite result with radius" << finalEdgeBlurRadius;
        } else {
            // Fallback to alternative method if OpenCL fails
            finalEdgeBlurred = applyEdgeBlurringAlternative(result, binMask, finalEdgeBlurRadius);
            if (!finalEdgeBlurred.empty()) {
                result = finalEdgeBlurred;
                qDebug() << "STATIC MODE: Applied final alternative edge blurring to composite result with radius" << finalEdgeBlurRadius;
            }
        }
    } catch (const cv::Exception &e) {
        qWarning() << "Soft-edge blend failed:" << e.what();
        scaledPerson.copyTo(result, scaledMask);
    }
    
    // Save debug images
    cv::imwrite("debug_post_original_segmented.png", m_lastSegmentedFrame);
    cv::imwrite("debug_post_lighting_corrected_person.png", lightingCorrectedPerson);
    cv::imwrite("debug_post_final_result.png", result);
    qDebug() << "POST-PROCESSING: Applied lighting to person and re-composited";
    qDebug() << "Debug images saved: post_original_segmented, post_lighting_corrected_person, post_final_result";
    
    return result;
}
cv::Mat Capture::applyLightingToRawPersonRegion(const cv::Mat &personRegion, const cv::Mat &personMask)
{
    qDebug() << "RAW PERSON APPROACH: Apply lighting to extracted person region only";
    
    //  CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs - returning empty mat";
        return cv::Mat();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch between person region and mask - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.type() != CV_8UC3) {
        qWarning() << "Invalid person region format - returning original";
        return personRegion.clone();
    }
    
    if (personMask.type() != CV_8UC1) {
        qWarning() << "Invalid mask format - returning original";
        return personRegion.clone();
    }
    
    // Start with exact copy of person region
    cv::Mat result;
    try {
        result = personRegion.clone();
    } catch (const std::exception& e) {
        qWarning() << "Failed to clone person region:" << e.what();
        return cv::Mat();
    }
    
    //  CRASH PREVENTION: Check lighting corrector availability
    if (!m_lightingCorrector) {
        qWarning() << "No lighting corrector available - returning original";
        return result;
    }
    
    try {
        // Get template reference for color matching
        cv::Mat templateRef = m_lightingCorrector->getReferenceTemplate();
        if (templateRef.empty()) {
            qWarning() << "No template reference, applying subtle lighting correction";
            // Apply subtle lighting correction to make person blend better
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    if (y < personMask.rows && x < personMask.cols && 
                        personMask.at<uchar>(y, x) > 0) {  // Person pixel
                        cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                        // SUBTLE CHANGES FOR NATURAL BLENDING:
                        pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 1.1);  // Slightly brighter blue
                        pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 1.05); // Slightly brighter green
                        pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 1.08); // Slightly brighter red
                    }
                }
            }
        } else {
            // Apply template-based color matching
            cv::resize(templateRef, templateRef, personRegion.size());
            
            // Convert to LAB for color matching
            cv::Mat personLab, templateLab;
            cv::cvtColor(personRegion, personLab, cv::COLOR_BGR2Lab);
            cv::cvtColor(templateRef, templateLab, cv::COLOR_BGR2Lab);
            
            // Calculate template statistics
            cv::Scalar templateMean, templateStd;
            cv::meanStdDev(templateLab, templateMean, templateStd);
            
            // Apply color matching to person region
            cv::Mat resultLab = personLab.clone();
            std::vector<cv::Mat> channels;
            cv::split(resultLab, channels);
            
            // Apply template color matching for natural blending
            // Calculate person statistics for comparison
            cv::Scalar personMean, personStd;
            cv::meanStdDev(personLab, personMean, personStd);
            
            // Adjust person lighting to match template characteristics
            for (int c = 0; c < 3; c++) {
                // Calculate the difference between template and person
                double lightingDiff = templateMean[c] - personMean[c];
                
                // Apply subtle adjustment (only 15% of the difference for natural blending)
                channels[c] = channels[c] + lightingDiff * 0.15;
            }
            
            // Additional brightness adjustment for better blending
            // If template is brighter, slightly brighten the person
            double brightnessDiff = templateMean[0] - personMean[0]; // L channel
            if (brightnessDiff > 0) {
                channels[0] = channels[0] + brightnessDiff * 0.1; // Slight brightness boost
            }
            
            cv::merge(channels, resultLab);
            cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
            
            // Apply mask to ensure only person pixels are affected
            cv::Mat maskedResult;
            result.copyTo(maskedResult, personMask);
            personRegion.copyTo(maskedResult, ~personMask);
            result = maskedResult;
        }
                
                // Save debug images (safely)
                try {
                    cv::imwrite("debug_raw_person_original.png", personRegion);
                    cv::imwrite("debug_raw_person_mask.png", personMask);
                    cv::imwrite("debug_raw_person_result.png", result);
            qDebug() << "RAW PERSON APPROACH: Applied lighting to person region only";
                    qDebug() << "Debug images saved: raw_person_original, raw_person_mask, raw_person_result";
                } catch (const std::exception& e) {
                    qWarning() << "Failed to save debug images:" << e.what();
                }
                
    } catch (const std::exception& e) {
        qWarning() << "Exception in lighting correction:" << e.what() << "- returning original";
                return personRegion.clone();
    }
    
                return result;
}

cv::Mat Capture::createPersonMaskFromSegmentedFrame(const cv::Mat &segmentedFrame)
{
    try {
        // CRASH PREVENTION: Validate segmentedFrame has 3 channels before BGR2GRAY conversion
        if (segmentedFrame.empty() || segmentedFrame.channels() != 3) {
            qWarning() << "Invalid segmentedFrame for mask creation: empty or not 3 channels, channels:" 
                       << (segmentedFrame.empty() ? 0 : segmentedFrame.channels());
            return cv::Mat::zeros(segmentedFrame.empty() ? cv::Size(640, 480) : segmentedFrame.size(), CV_8UC1);
        }
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(segmentedFrame, gray, cv::COLOR_BGR2GRAY);
        
        // OPTIMIZED: Lower threshold to preserve black clothing pixels (0-30 grayscale range)
        cv::Mat mask;
        cv::threshold(gray, mask, 5, 255, cv::THRESH_BINARY);
        
        // FAST HOLE FILLING: Use morphological operations only (no flood fill - too slow)
        // Multiple passes with increasing kernel sizes to fill holes in black regions
        cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel1);
        
        cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel2);
        
        // Remove small protrusions
        cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, openKernel);
        
        // Apply Gaussian blur for smooth edges
        cv::GaussianBlur(mask, mask, cv::Size(15, 15), 0);
        
        return mask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Failed to create person mask:" << e.what();
        return cv::Mat::zeros(segmentedFrame.size(), CV_8UC1);
    }
}
//  DYNAMIC VIDEO PROCESSING MOVED TO capture_dynamic.cpp
//  See src/core/capture_dynamic.cpp for processRecordedVideoWithLighting()

//  Async Lighting Processing System for non-blocking video processing
void Capture::initializeAsyncLightingSystem()
{
    qDebug() << " Initializing async lighting system for non-blocking video processing";
    
    // Initialize the video processing watcher
    m_lightingWatcher = new QFutureWatcher<QList<QPixmap>>(this);
    
    // Connect the watcher to handle completion
    connect(m_lightingWatcher, &QFutureWatcher<QList<QPixmap>>::finished,
            this, &Capture::onVideoProcessingFinished);
    
    qDebug() << " Async lighting system initialized successfully";
}

void Capture::cleanupAsyncLightingSystem()
{
    qDebug() << " Cleaning up async lighting system";
    
    // Cancel any ongoing processing
    if (m_lightingWatcher) {
        m_lightingWatcher->cancel();
        m_lightingWatcher->waitForFinished();
        delete m_lightingWatcher;
        m_lightingWatcher = nullptr;
    }
    
    qDebug() << " Async lighting system cleaned up";
}
//  DYNAMIC VIDEO PROCESSING MOVED TO capture_dynamic.cpp
//  See src/core/capture_dynamic.cpp for:
//    - processRecordedVideoWithLighting()
//    - applyDynamicFrameEdgeBlending()
//    - applyFastEdgeBlendingForVideo()
//    - applySimpleDynamicCompositing()

//  NEW: Lightweight Segmented Frame Creation for Recording Performance
cv::Mat Capture::createLightweightSegmentedFrame(const cv::Mat &frame)
{
    // Fast segmentation without heavy processing for recording performance
    if (frame.empty()) {
        return frame;
    }
    
    // Use simple center detection for recording performance
    cv::Rect centerRect(frame.cols * 0.2, frame.rows * 0.1, 
                       frame.cols * 0.6, frame.rows * 0.8);
    
    cv::Mat result = frame.clone();
    
    // Apply background if available
    if (m_useDynamicVideoBackground && !m_dynamicVideoFrame.empty()) {
        // Resize dynamic background to match frame
        cv::Mat bgResized;
        cv::resize(m_dynamicVideoFrame, bgResized, cv::Size(frame.cols, frame.rows));
        
        // Simple mask-based compositing
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::rectangle(mask, centerRect, cv::Scalar(255), -1);
        
        // Apply gaussian blur for smoother edges
        cv::GaussianBlur(mask, mask, cv::Size(21, 21), 10);
        
        // Composite
        bgResized.copyTo(result);
        frame.copyTo(result, mask);
    } else if (m_useBackgroundTemplate && !m_selectedTemplate.empty()) {
        // Use static background template
        cv::Mat bgResized;
        cv::resize(m_selectedTemplate, bgResized, cv::Size(frame.cols, frame.rows));
        
        // Simple mask-based compositing
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::rectangle(mask, centerRect, cv::Scalar(255), -1);
        
        // Apply gaussian blur for smoother edges
        cv::GaussianBlur(mask, mask, cv::Size(21, 21), 10);
        
        // Composite
        bgResized.copyTo(result);
        frame.copyTo(result, mask);
    }
    
    return result;
}

//  Performance Control Methods
//  REMOVED: Real-time lighting methods - not needed for post-processing only mode

//  OpenCL-Accelerated Guided Filter for Edge-Blending (Memory Pool Optimized)
// GPU-optimized guided filtering that maintains FPS and quality using pre-allocated buffers
cv::Mat guidedFilterGrayAlphaOpenCLOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                                 GPUMemoryPool &memoryPool)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    // Check OpenCL availability
    if (!cv::ocl::useOpenCL()) {
        qWarning() << "OpenCL not available, falling back to CPU guided filter";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }

    try {
        //  Performance monitoring for guided filtering
        QElapsedTimer guidedFilterTimer;
        guidedFilterTimer.start();
        
        // Get pre-allocated GPU buffers from memory pool (optimized buffer usage)
        cv::UMat& gpuGuide = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuMask = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuI = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuP = memoryPool.getNextGuidedFilterBuffer();
        
        // Upload to GPU (OpenCL)
        guideBGR.copyTo(gpuGuide);
        hardMask.copyTo(gpuMask);
        
        // Convert guide to grayscale on GPU if needed
        if (guideBGR.channels() == 3) {
            cv::cvtColor(gpuGuide, gpuI, cv::COLOR_BGR2GRAY);
        } else {
            gpuI = gpuGuide;
        }
        
        // Convert to float32 on GPU
        gpuI.convertTo(gpuI, CV_32F, 1.0f / 255.0f);
        if (hardMask.type() != CV_32F) {
            gpuMask.convertTo(gpuP, CV_32F, 1.0f / 255.0f);
        } else {
            gpuP = gpuMask;
        }
        
        // Get additional buffers from memory pool
        cv::UMat& gpuMeanI = memoryPool.getNextBoxFilterBuffer();
        cv::UMat& gpuMeanP = memoryPool.getNextBoxFilterBuffer();
        cv::UMat& gpuCorrI = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuCorrIp = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuVarI = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuCovIp = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuA = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuB = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuMeanA = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuMeanB = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuQ = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuAlpha = memoryPool.getNextGuidedFilterBuffer();
        
        // Use box filter for GPU (OpenCL)
        cv::boxFilter(gpuI, gpuMeanI, CV_32F, cv::Size(radius, radius));
        cv::boxFilter(gpuP, gpuMeanP, CV_32F, cv::Size(radius, radius));
        
        // Compute I*I and I*P on GPU
        cv::UMat& gpuISquared = memoryPool.getNextGuidedFilterBuffer();
        cv::UMat& gpuIP = memoryPool.getNextGuidedFilterBuffer();
        cv::multiply(gpuI, gpuI, gpuISquared);
        cv::multiply(gpuI, gpuP, gpuIP);
        
        cv::boxFilter(gpuISquared, gpuCorrI, CV_32F, cv::Size(radius, radius));
        cv::boxFilter(gpuIP, gpuCorrIp, CV_32F, cv::Size(radius, radius));
        
        // Step 2: Compute variance and covariance on GPU
        cv::multiply(gpuMeanI, gpuMeanI, gpuVarI);
        cv::subtract(gpuCorrI, gpuVarI, gpuVarI);
        
        cv::multiply(gpuMeanI, gpuMeanP, gpuCovIp);
        cv::subtract(gpuCorrIp, gpuCovIp, gpuCovIp);
        
        // Step 3: Compute coefficients a and b on GPU
        cv::UMat& gpuEps = memoryPool.getNextGuidedFilterBuffer();
        cv::Mat epsMat = cv::Mat::ones(gpuVarI.size(), CV_32F) * eps;
        epsMat.copyTo(gpuEps);
        cv::add(gpuVarI, gpuEps, gpuVarI);
        cv::divide(gpuCovIp, gpuVarI, gpuA);
        
        cv::multiply(gpuA, gpuMeanI, gpuB);
        cv::subtract(gpuMeanP, gpuB, gpuB);
        
        // Step 4: Compute mean of coefficients on GPU
        cv::boxFilter(gpuA, gpuMeanA, CV_32F, cv::Size(radius, radius));
        cv::boxFilter(gpuB, gpuMeanB, CV_32F, cv::Size(radius, radius));
        
        // Step 5: Compute final result on GPU
        cv::multiply(gpuMeanA, gpuI, gpuQ);
        cv::add(gpuQ, gpuMeanB, gpuQ);
        
        // Clamp result to [0,1] on GPU
        cv::threshold(gpuQ, gpuAlpha, 0.0f, 0.0f, cv::THRESH_TOZERO);
        cv::threshold(gpuAlpha, gpuAlpha, 1.0f, 1.0f, cv::THRESH_TRUNC);
        
        // Download result back to CPU
        cv::Mat result;
        gpuAlpha.copyTo(result);
        
        //  Performance monitoring - log guided filtering time
        qint64 guidedFilterTime = guidedFilterTimer.elapsed();
        if (guidedFilterTime > 5) { // Only log if it takes more than 5ms
            qDebug() << "OpenCL Guided Filter Performance:" << guidedFilterTime << "ms for" 
                     << guideBGR.cols << "x" << guideBGR.rows << "image";
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "OpenCL guided filter failed:" << e.what() << "- falling back to CPU";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }
}

// CPU fallback for guided filtering (original implementation)
static cv::Mat guidedFilterGrayAlphaCPU(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    cv::Mat I8, I, p;
    if (guideBGR.channels() == 3) {
        cv::cvtColor(guideBGR, I8, cv::COLOR_BGR2GRAY);
    } else {
        I8 = guideBGR;
    }
    I8.convertTo(I, CV_32F, 1.0f / 255.0f);
    if (hardMask.type() != CV_32F) {
        hardMask.convertTo(p, CV_32F, 1.0f / 255.0f);
    } else {
        p = hardMask;
    }

    cv::Mat mean_I, mean_p, corr_I, corr_Ip;
    cv::boxFilter(I, mean_I, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(p, mean_p, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(I), corr_I, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(p), corr_Ip, CV_32F, cv::Size(radius, radius));

    cv::Mat var_I = corr_I - mean_I.mul(mean_I);
    cv::Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(b, mean_b, CV_32F, cv::Size(radius, radius));

    cv::Mat q = mean_a.mul(I) + mean_b;
    cv::Mat alpha; cv::min(cv::max(q, 0.0f), 1.0f, alpha);
    return alpha;
}
//  OpenCL-Accelerated Edge Blurring for Enhanced Edge-Blending
// GPU-optimized edge blurring that mixes background template with segmented object edges
cv::Mat applyEdgeBlurringOpenCL(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    // Check OpenCL availability
    if (!cv::ocl::useOpenCL()) {
        qWarning() << "OpenCL not available for edge blurring, falling back to CPU";
        return applyEdgeBlurringCPU(segmentedObject, objectMask, backgroundTemplate, blurRadius);
    }

    try {
        //  Performance monitoring for edge blurring
        QElapsedTimer edgeBlurTimer;
        edgeBlurTimer.start();

        // Get pre-allocated GPU buffers from memory pool
        cv::UMat& gpuObject = memoryPool.getNextEdgeBlurBuffer();
        cv::UMat& gpuMask = memoryPool.getNextEdgeDetectionBuffer();
        cv::UMat& gpuBackground = memoryPool.getNextEdgeBlurBuffer();
        cv::UMat& gpuBlurred = memoryPool.getNextEdgeBlurBuffer();
        cv::UMat& gpuResult = memoryPool.getNextEdgeBlurBuffer();

        // Upload to GPU (OpenCL)
        segmentedObject.copyTo(gpuObject);
        objectMask.copyTo(gpuMask);
        backgroundTemplate.copyTo(gpuBackground);

        // Convert mask to grayscale if needed
        if (objectMask.channels() == 3) {
            cv::cvtColor(gpuMask, gpuMask, cv::COLOR_BGR2GRAY);
        }

        // Step 1: Create transition zone by dilating the mask outward
        cv::UMat gpuDilatedMask;
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*static_cast<int>(blurRadius)+1, 2*static_cast<int>(blurRadius)+1));
        cv::dilate(gpuMask, gpuDilatedMask, dilateKernel);

        // Step 2: Create transition zone by subtracting original mask from dilated mask
        cv::UMat gpuTransitionZone;
        cv::subtract(gpuDilatedMask, gpuMask, gpuTransitionZone);

        // Step 3: Create inner edge zone by eroding the mask
        cv::UMat gpuErodedMask;
        cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(gpuMask, gpuErodedMask, erodeKernel);

        // Step 4: Create inner edge zone by subtracting eroded mask from original mask
        cv::UMat gpuInnerEdgeZone;
        cv::subtract(gpuMask, gpuErodedMask, gpuInnerEdgeZone);

        // Step 5: Combine transition zone and inner edge zone for comprehensive edge blurring
        cv::UMat gpuCombinedEdgeZone;
        cv::bitwise_or(gpuTransitionZone, gpuInnerEdgeZone, gpuCombinedEdgeZone);

        // Step 6: Apply Gaussian blur to both object and background
        cv::GaussianBlur(gpuObject, gpuBlurred, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);
        
        cv::UMat gpuBlurredBackground;
        cv::GaussianBlur(gpuBackground, gpuBlurredBackground, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);

        // Step 7: Create mixed background-object blend for edge zones
        cv::UMat gpuMixedBlend;
        cv::addWeighted(gpuBlurred, 0.6f, gpuBlurredBackground, 0.4f, 0, gpuMixedBlend);

        // Step 8: Apply smooth blending using the combined edge zone
        // Copy original object to result
        gpuObject.copyTo(gpuResult);
        
        // Apply mixed background-object blend in the combined edge zone
        gpuMixedBlend.copyTo(gpuResult, gpuCombinedEdgeZone);

        // Download result back to CPU
        cv::Mat result;
        gpuResult.copyTo(result);

        //  Performance monitoring - log edge blurring time
        qint64 edgeBlurTime = edgeBlurTimer.elapsed();
        if (edgeBlurTime > 3) { // Only log if it takes more than 3ms
            qDebug() << "OpenCL Edge Blur Performance:" << edgeBlurTime << "ms for" 
                     << segmentedObject.cols << "x" << segmentedObject.rows << "image, radius:" << blurRadius;
        }

        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "OpenCL edge blurring failed:" << e.what() << "- falling back to CPU";
        return applyEdgeBlurringCPU(segmentedObject, objectMask, backgroundTemplate, blurRadius);
    }
}

// CPU fallback for edge blurring
static cv::Mat applyEdgeBlurringCPU(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    try {
        // Convert mask to grayscale if needed
        cv::Mat mask;
        if (objectMask.channels() == 3) {
            cv::cvtColor(objectMask, mask, cv::COLOR_BGR2GRAY);
        } else {
            mask = objectMask.clone();
        }

        // Step 1: Create transition zone by dilating the mask outward
        cv::Mat dilatedMask;
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
            cv::Size(2*static_cast<int>(blurRadius)+1, 2*static_cast<int>(blurRadius)+1));
        cv::dilate(mask, dilatedMask, dilateKernel);

        // Step 2: Create transition zone by subtracting original mask from dilated mask
        cv::Mat transitionZone;
        cv::subtract(dilatedMask, mask, transitionZone);

        // Step 3: Create inner edge zone by eroding the mask
        cv::Mat erodedMask;
        cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mask, erodedMask, erodeKernel);

        // Step 4: Create inner edge zone by subtracting eroded mask from original mask
        cv::Mat innerEdgeZone;
        cv::subtract(mask, erodedMask, innerEdgeZone);

        // Step 5: Combine transition zone and inner edge zone for comprehensive edge blurring
        cv::Mat combinedEdgeZone;
        cv::bitwise_or(transitionZone, innerEdgeZone, combinedEdgeZone);

        // Step 6: Apply Gaussian blur to both object and background
        cv::Mat blurred;
        cv::GaussianBlur(segmentedObject, blurred, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);
        
        cv::Mat blurredBackground;
        cv::GaussianBlur(backgroundTemplate, blurredBackground, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);

        // Step 7: Create mixed background-object blend for edge zones
        cv::Mat mixedBlend;
        cv::addWeighted(blurred, 0.6f, blurredBackground, 0.4f, 0, mixedBlend);

        // Step 8: Apply smooth blending using the combined edge zone
        cv::Mat result = segmentedObject.clone();
        
        // Apply mixed background-object blend in the combined edge zone
        mixedBlend.copyTo(result, combinedEdgeZone);

        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "CPU edge blurring failed:" << e.what() << "- returning original";
        return segmentedObject.clone();
    }
}
//  Alternative Edge Blurring Method using Distance Transform
// This method uses distance transform to create smooth edge transitions
// Made non-static so it can be used by capture_dynamic.cpp
cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    try {
        // Convert mask to grayscale if needed
        cv::Mat mask;
        if (objectMask.channels() == 3) {
            cv::cvtColor(objectMask, mask, cv::COLOR_BGR2GRAY);
        } else {
            mask = objectMask.clone();
        }

        // Step 1: Create distance transform from mask boundary
        cv::Mat distTransform;
        cv::distanceTransform(mask, distTransform, cv::DIST_L2, 5);
        
        // Step 2: Normalize distance transform to [0, 1] range
        cv::Mat normalizedDist;
        cv::normalize(distTransform, normalizedDist, 0, 1.0, cv::NORM_MINMAX, CV_32F);
        
        // Step 3: Create edge mask by thresholding distance transform
        cv::Mat edgeMask;
        float threshold = blurRadius / 10.0f; // Adjust threshold based on blur radius
        cv::threshold(normalizedDist, edgeMask, threshold, 1.0, cv::THRESH_BINARY);
        edgeMask.convertTo(edgeMask, CV_8U, 255.0f);
        
        // Step 4: Apply Gaussian blur to the entire object
        cv::Mat blurred;
        cv::GaussianBlur(segmentedObject, blurred, cv::Size(0, 0), blurRadius, blurRadius);
        
        // Step 5: Blend using distance-based alpha
        cv::Mat result = segmentedObject.clone();
        
        // Create alpha mask from distance transform
        cv::Mat alphaMask;
        normalizedDist.convertTo(alphaMask, CV_8U, 255.0f);
        
        // Apply blending only in edge regions
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (edgeMask.at<uchar>(y, x) > 0) {
                    float alpha = normalizedDist.at<float>(y, x);
                    cv::Vec3b original = result.at<cv::Vec3b>(y, x);
                    cv::Vec3b blurred_pixel = blurred.at<cv::Vec3b>(y, x);
                    
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uchar>(original[0] * (1.0f - alpha) + blurred_pixel[0] * alpha),
                        static_cast<uchar>(original[1] * (1.0f - alpha) + blurred_pixel[1] * alpha),
                        static_cast<uchar>(original[2] * (1.0f - alpha) + blurred_pixel[2] * alpha)
                    );
                }
            }
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Alternative edge blurring failed:" << e.what() << "- returning original";
        return segmentedObject.clone();
    }
}

QString Capture::resolveTemplatePath(const QString &templatePath)
{
    if (templatePath.isEmpty()) {
        return QString();
    }
    
    // Try multiple candidate paths to resolve the template path
    QStringList candidates;
    candidates << templatePath
               << QDir::currentPath() + "/" + templatePath
               << QCoreApplication::applicationDirPath() + "/" + templatePath
               << QCoreApplication::applicationDirPath() + "/../" + templatePath
               << QCoreApplication::applicationDirPath() + "/../../" + templatePath
               << "../" + templatePath
               << "../../" + templatePath
               << "../../../" + templatePath;

    // Find the first existing path
    for (const QString &candidate : candidates) {
        if (QFile::exists(candidate)) {
            // Only show debug message for new paths to avoid spam
            static QSet<QString> resolvedPaths;
            if (!resolvedPaths.contains(templatePath)) {
                qDebug() << "Template path resolved:" << templatePath << "-> " << candidate;
                resolvedPaths.insert(templatePath);
            }
            return candidate;
        }
    }
    
    qWarning() << "Template path could not be resolved:" << templatePath;
    qWarning() << "Tried paths:";
    for (const QString &candidate : candidates) {
        qWarning() << "    -" << candidate;
    }
    
    return QString(); // Return empty string if no path found
}


