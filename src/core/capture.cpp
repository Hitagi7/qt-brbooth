#include "core/capture.h"
#include "core/camera.h"
#include "core/capture_edge_blending.h"
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
#include <QList>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafilters.hpp> // Added for CUDA filter functions
#include <opencv2/cudaarithm.hpp> // Added for CUDA arithmetic operations (inRange, bitwise_or)
#include <QtConcurrent/QtConcurrent>
#include <QThreadPool>
#include <QMutexLocker>
#include <chrono>
#include <QFutureWatcher>
#include "algorithms/lighting_correction/lighting_corrector.h"
#include "core/system_monitor.h"

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
    , m_segmentedFrameCounter(0)
    , m_personDetectionMutex()
    , m_personDetectionTimer()
    , m_bgSubtractor()
    , m_subtractionReferenceImage()
    , m_subtractionReferenceImage2()
    , m_subtractionBlendWeight(0.5)  // Default: equal blend
    , m_useGPU(false)
    , m_useCUDA(false)
    , m_gpuUtilized(false)
    , m_systemMonitor(nullptr)
    , m_cudaUtilized(false)
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
    , m_dynamicGpuReader()
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
    , m_firstFrameHandled(false)
    , m_recordingThread(nullptr)
    , m_recordingFrameTimer(nullptr)
    , m_recordingMutex()
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , m_recordingThreadActive(false)
    , m_recordingFrameQueue()
    , m_recordingStream()
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_recordingGpuBuffer()
    , m_cachedPixmap(640, 480)
    // Lighting Correction Member
    , m_lightingCorrector(nullptr)
    //  Simplified Lighting Processing (POST-PROCESSING ONLY)
    , m_lightingProcessingThread(nullptr)
    , m_lightingWatcher(nullptr)
    // Lighting Comparison Storage
    , m_originalCapturedImage()
    , m_lightingCorrectedImage()
    , m_hasLightingComparison(false)
    , m_hasVideoLightingComparison(false)
    , m_recordedPersonScaleFactor(1.0) // Initialize to default scale (100%)
    , m_bgModelInitialized(false)
    , m_bgHueMean(60.0)
    , m_bgHueStd(10.0)
    , m_bgSatMean(120.0)
    , m_bgSatStd(20.0)
    , m_bgValMean(120.0)
    , m_bgValStd(20.0)
    , m_bgCbMean(90.0)
    , m_bgCbStd(10.0)
    , m_bgCrMean(120.0)
    , m_bgCrStd(10.0)
    , m_bgRedMean(60.0)
    , m_bgGreenMean(150.0)
    , m_bgBlueMean(60.0)
    , m_bgRedStd(20.0)
    , m_bgGreenStd(20.0)
    , m_bgBlueStd(20.0)
    , m_bgColorInvCov(cv::Matx33d::eye())
    , m_bgColorInvCovReady(false)

{
    // Initialize cv::Ptr members to nullptr (cannot be default-constructed)
    m_greenScreenCannyDetector = nullptr;
    m_greenScreenMorphOpen = nullptr;
    m_greenScreenMorphClose = nullptr;
    m_greenScreenGaussianBlur = nullptr;
    
    ui->setupUi(this);
    // Dynamic video background defaults
    m_useDynamicVideoBackground = false;
    m_dynamicVideoPath.clear();
    if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.release();
    }
    m_dynamicVideoFrame.release();
    m_dynamicGpuReader.release();
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

    // Green-screen defaults (robust to both ring/ceiling light scenarios)
    m_greenScreenEnabled = true;  // ENABLED BY DEFAULT - pure green removal only
    // Typical green in OpenCV HSV: H=[35, 85], allow wider tolerance for lighting shifts
    m_greenHueMin = 30;  // Target true greens, not dark teals
    m_greenHueMax = 95;  // Limit to avoid catching person's dark teal/greenish colors
    m_greenSatMin = 30;  // Higher saturation to target actual green screen, not person's dark colors
    m_greenValMin = 40;  // Higher brightness to avoid dark greenish colors on person
    // Morphological cleanup sizes (pixels)
    m_greenMaskOpen = 3;
    m_greenMaskClose = 7;
    // Temporal mask smoothing
    m_greenScreenMaskStableCount = 0;

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

    m_captureReady = true;  // Start with capture ready



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
    // Measure actual processing time for capture page FPS calculation
    QElapsedTimer frameTimer;
    frameTimer.start();
    static QElapsedTimer processingFpsTimer;
    static bool processingFpsTimerInitialized = false;
    if (!processingFpsTimerInitialized) {
        processingFpsTimer.start();
        processingFpsTimerInitialized = true;
    }

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

    // Track unique frames for accurate FPS measurement (frame-to-frame timing)
    static int lastDisplayedFrameCounter = 0;
    bool shouldMeasureFPS = false;

    // Check if we have processed segmentation results to display
    if (m_segmentationEnabledInCapture && !m_lastSegmentedFrame.empty()) {
        // Convert the processed OpenCV frame back to QImage for display
        displayImage = cvMatToQImage(m_lastSegmentedFrame);
        qDebug() << "Displaying processed segmentation frame";
        
        // Detect new unique frame - only measure FPS when frame counter changes
        // (when segmentation finishes and updates m_segmentedFrameCounter)
        if (m_segmentedFrameCounter != lastDisplayedFrameCounter) {
            shouldMeasureFPS = true;
            lastDisplayedFrameCounter = m_segmentedFrameCounter;
        }
    } else {
        // Use original camera image - measure FPS for every camera frame
        displayImage = image;
        shouldMeasureFPS = true;
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
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = frameTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Calculate current FPS using frame-to-frame timing method (MOST ACCURATE)
    // Only measure when a new unique frame is displayed
    if (shouldMeasureFPS) {
        // Frame-to-frame timing measurement
        static QElapsedTimer lastFrameTimer;
        static bool lastFrameTimerInitialized = false;
        
        if (!lastFrameTimerInitialized) {
            lastFrameTimer.start();
            lastFrameTimerInitialized = true;
        }
        
        // Get time since last unique frame was displayed
        qint64 frameTimeMs = lastFrameTimer.restart(); // Get time since last frame, then restart
        
        // Skip first frame measurement (it's inaccurate - measures time since timer start, not frame-to-frame)
        static bool firstFrameMeasured = false;
        if (!firstFrameMeasured) {
            firstFrameMeasured = true;
            // Don't return - continue to initialize the buffer, just skip this measurement
        } else if (frameTimeMs > 0) {
            // Only process valid frame times (between 5ms and 1000ms to avoid outliers)
            // 5ms minimum = 200 FPS max (reasonable cap)
            // 1000ms maximum = 1 FPS min (prevents division by zero and handles slow frames)
            if (frameTimeMs >= 5 && frameTimeMs <= 1000) {
                // Convert milliseconds to instant FPS
                double currentInstantFPS = 1000.0 / static_cast<double>(frameTimeMs);
                
                // Rolling window storage (circular buffer of 30 values)
                static const int FPS_WINDOW_SIZE = 30;
                static double fpsHistory[FPS_WINDOW_SIZE] = {0.0};
                static int fpsHistoryIndex = 0;
                
                // Store instant FPS measurement in circular buffer
                fpsHistory[fpsHistoryIndex] = currentInstantFPS;
                fpsHistoryIndex = (fpsHistoryIndex + 1) % FPS_WINDOW_SIZE;
                
                // Calculate average of last 30 FPS measurements for smooth reading
                // Filter outliers: only include FPS values between 1 and 200
                double sum = 0.0;
                int validFrames = 0;
                for (int i = 0; i < FPS_WINDOW_SIZE; ++i) {
                    if (fpsHistory[i] > 0.0 && fpsHistory[i] >= 1.0 && fpsHistory[i] <= 200.0) {
                        sum += fpsHistory[i];
                        validFrames++;
                    }
                }
                
                if (validFrames > 0) {
                    double currentAverageFPS = sum / static_cast<double>(validFrames);
                    m_currentFPS = static_cast<int>(std::round(currentAverageFPS));
                    
                    // Update system monitor with accurate frame-to-frame FPS
        if (m_systemMonitor) {
                        qDebug() << "Capture: Updating SystemMonitor with Frame-to-Frame FPS:" << m_currentFPS 
                                 << "instant=" << QString::number(currentInstantFPS, 'f', 1)
                                 << "avg=" << QString::number(currentAverageFPS, 'f', 1)
                                 << "frameTime=" << frameTimeMs << "ms"
                                 << "Pointer:" << (void*)m_systemMonitor;
                        try {
                            m_systemMonitor->updateFPS(static_cast<double>(m_currentFPS));
                qDebug() << "Capture: SystemMonitor FPS update completed successfully";
            } catch (const std::exception& e) {
                qDebug() << "Capture: Exception during FPS update:" << e.what();
            } catch (...) {
                qDebug() << "Capture: Unknown exception during FPS update";
            }
        } else {
            qDebug() << "Capture: SystemMonitor is NULL, cannot update FPS:" << m_currentFPS;
                    }
                }
            }
        }
    }

    // Print performance stats every 30 frames (approximately every second at 30 FPS)
    if (frameCount % 30 == 0) {
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
    qDebug() << "Capture Page System Processing FPS:" << QString::number(m_currentFPS, 'f', 1) << "FPS";
    qDebug() << "Camera Feed Rate (measured over " << frameCount << " frames):" << QString::number(measuredFPS, 'f', 1) << "FPS";
    qDebug() << "Avg processing time per frame (measured over " << frameCount << " frames):" << avgLoopTime << "ms";
    qDebug() << "Frame processing efficiency:" << (avgLoopTime < 16.67 ? "GOOD" : "NEEDS OPTIMIZATION");
    qDebug() << "Person Detection Enabled:" << (m_segmentationEnabledInCapture ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "Unified Detection Enabled:" << (m_segmentationEnabledInCapture ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "GPU Acceleration:" << (m_useGPU ? "YES (OpenCL)" : "NO (CPU)");
    qDebug() << "GPU Utilized:" << (m_gpuUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "CUDA Acceleration:" << (m_useCUDA ? "YES (CUDA)" : "NO (CPU)");
    qDebug() << "CUDA Utilized:" << (m_cudaUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "Person Detection FPS:" << (m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
    qDebug() << "Unified Detection FPS:" << (m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
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

    // Start the countdown
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

    qDebug() << "Debug display setup complete - FPS, GPU, and CUDA status should be visible";
}

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

    m_captureReady = true;

    // Reset segmentation state for capture interface
    enableSegmentationInCapture();
    qDebug() << "Segmentation reset for capture interface";

    // BUG FIX: Don't reset capture mode - preserve user's mode selection (static/dynamic)
    // The mode should only be changed when user explicitly selects a different template type
    qDebug() << "Preserving capture mode:" << (m_currentCaptureMode == VideoRecordMode ? "VideoRecordMode" : "ImageCaptureMode");

    // Reset video recording state (but keep the mode)
    m_recordedFrames.clear();
    m_originalRecordedFrames.clear();
    m_hasVideoLightingComparison = false;
    m_recordedSeconds = 0;

    // Reset dynamic video background to start from beginning
    if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
        resetDynamicVideoToStart();
        qDebug() << "Dynamic video reset to start for re-recording";
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
                    qDebug() << "Debug display SHOWN - FPS, GPU, and CUDA status visible";
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
                m_cudaUtilized = false;
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
    // Just enable segmentation after a short delay
    QTimer::singleShot(100, [this]() {
        enableSegmentationInCapture();
        qDebug() << "Segmentation ENABLED for capture interface";

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
    qDebug() << "Capture widget hidden - OPTIMIZED camera shutdown";

    // Disable segmentation when leaving capture page
    disableSegmentationOutsideCapture();
    qDebug() << "Segmentation DISABLED outside capture interface";

    // Note: Camera is now controlled by the main page change handler in brbooth.cpp
    // This prevents lag when returning to capture page
}


void Capture::updateDebugDisplay()
{
    // Debug output to verify the method is being called
    static int updateCount = 0;
    updateCount++;
    
        qDebug() << "updateDebugDisplay #" << updateCount;
    
    if (updateCount % 10 == 0) { // Log every 5 seconds (10 updates * 500ms)
        qDebug() << "Debug display update #" << updateCount << "FPS:" << m_currentFPS << "GPU:" << m_useGPU << "CUDA:" << m_useCUDA;
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
    if (!m_dynamicGpuReader.empty()) {
        m_dynamicGpuReader.release();
    }
    
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

    // Prefer GPU NVDEC via cudacodec if CUDA is enabled and available
    if (m_useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            m_dynamicGpuReader = cv::cudacodec::createVideoReader(cleanPath.toStdString());
            if (!m_dynamicGpuReader.empty()) {
                opened = true;
                qDebug() << "Using CUDA VideoReader (NVDEC) for dynamic video background";
            }
        } catch (const cv::Exception &e) {
            qWarning() << "CUDA VideoReader unavailable, falling back to CPU VideoCapture:" << e.what();
            m_dynamicGpuReader.release();
        }
    }

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
    } else if (!m_dynamicGpuReader.empty()) {
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
            qDebug() << "Using default FPS (30) for CUDA reader; probe failed";
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
    } else if (!m_dynamicGpuReader.empty()) {
        // For GPU reader, use probed/native frame rate if available
        if (m_videoFrameRate <= 0) m_videoFrameRate = 30.0;
        m_videoFrameInterval = qRound(1000.0 / m_videoFrameRate);
        if (m_videoFrameInterval < 16) m_videoFrameInterval = 16;
        qDebug() << "Using NVDEC playback frame rate:" << m_videoFrameRate << "FPS, interval:" << m_videoFrameInterval << "ms";
    }

    // Prime first frame using available reader. Keep GPU path GPU-only to minimize CPU.
    cv::Mat first;
    bool frameRead = false;

    if (!m_dynamicGpuReader.empty()) {
        try {
            cv::cuda::GpuMat gpu;
            if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
                if (gpu.type() == CV_8UC4) {
                    cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                }
                m_dynamicGpuFrame = gpu; // keep GPU copy for GPU compositing paths
                // Avoid download when in segmentation mode; only maintain a CPU copy if needed elsewhere
                if (!isGPUOnlyProcessingAvailable()) {
                    gpu.download(first);
                    frameRead = !first.empty();
                } else {
                    frameRead = true;
                }
            }
        } catch (const cv::Exception &e) {
            qWarning() << "CUDA reader failed to read first frame, falling back to CPU:" << e.what();
            m_dynamicGpuReader.release();
        }
    }

    if (!frameRead && m_dynamicVideoCap.isOpened()) {
        frameRead = m_dynamicVideoCap.read(first);
        if (frameRead && !first.empty()) {
            qDebug() << "First frame size (CPU):" << first.cols << "x" << first.rows;
        }
    }

    if (frameRead && (!m_dynamicGpuReader.empty() || !first.empty())) {
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
        m_dynamicGpuReader.release();
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
    m_dynamicGpuReader.release();
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

    // Try GPU reader first (keep data on GPU to minimize CPU usage)
    if (!m_dynamicGpuReader.empty()) {
        try {
            cv::cuda::GpuMat gpu;
            if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
                if (gpu.type() == CV_8UC4) {
                    cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                }
                m_dynamicGpuFrame = gpu; // keep latest GPU frame
                if (!isGPUOnlyProcessingAvailable()) {
                    gpu.download(nextFrame);
                    frameRead = !nextFrame.empty();
                } else {
                    frameRead = true; // GPU-only path does not require CPU copy
                }
            } else {
                // Attempt soft restart of GPU reader
                m_dynamicGpuReader.release();
                m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
                cv::cuda::GpuMat gpuRetry;
                if (!m_dynamicGpuReader.empty() && m_dynamicGpuReader->nextFrame(gpuRetry) && !gpuRetry.empty()) {
                    if (gpuRetry.type() == CV_8UC4) {
                        cv::cuda::cvtColor(gpuRetry, gpuRetry, cv::COLOR_BGRA2BGR);
                    }
                    m_dynamicGpuFrame = gpuRetry;
                    if (!isGPUOnlyProcessingAvailable()) {
                        gpuRetry.download(nextFrame);
                        frameRead = !nextFrame.empty();
                    } else {
                        frameRead = true;
                    }
                }
            }
        } catch (const cv::Exception &e) {
            qWarning() << "CUDA reader failed during timer; switching to CPU:" << e.what();
            m_dynamicGpuReader.release();
        } catch (const std::exception &e) {
            qWarning() << "Exception in GPU video reading:" << e.what();
            m_dynamicGpuReader.release();
        }
    }

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
    if (!m_dynamicGpuReader.empty()) {
        try {
            m_dynamicGpuReader.release();
            m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
            qDebug() << "GPU video reader reset to start";
        } catch (...) {
            qWarning() << "Failed to reset GPU video reader";
        }
    } else if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
        qDebug() << "CPU video reader reset to start";
    }

    // Read first frame to prime the system
    cv::Mat firstFrame;
    bool frameRead = false;

    if (!m_dynamicGpuReader.empty()) {
        cv::cuda::GpuMat gpu;
        if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
            cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
            gpu.download(firstFrame);
            frameRead = true;
        }
    } else if (m_dynamicVideoCap.isOpened()) {
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

    // Check if CUDA is available and GPU processing is supported
    if (m_useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            // Test GPU memory allocation
            cv::cuda::GpuMat testMat(100, 100, CV_8UC3);
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

// Enhanced Person Detection and Segmentation methods
void Capture::initializePersonDetection()
{
    qDebug() << "===== initializePersonDetection() CALLED =====";
    qDebug() << "Initializing Enhanced Person Detection and Segmentation...";

    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    // OPTIMIZATION: Only create if not already initialized to avoid recreating unnecessarily
    if (m_bgSubtractor.empty()) {
        m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
        qDebug() << "Background subtractor initialized in initializePersonDetection()";
    } else {
        qDebug() << "Background subtractor already initialized, skipping recreation";
    }

    //  Initialize GPU Memory Pool for optimized CUDA operations
    if (!m_gpuMemoryPoolInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            qDebug() << " Initializing GPU Memory Pool for optimized CUDA operations...";
            m_gpuMemoryPool.initialize(1280, 720); // Initialize with common camera resolution
            m_gpuMemoryPoolInitialized = true;
            qDebug() << "GPU Memory Pool initialized successfully";
        } catch (const cv::Exception& e) {
            qWarning() << " GPU Memory Pool initialization failed:" << e.what();
            m_gpuMemoryPoolInitialized = false;
        }
    }

    // Check if CUDA is available for NVIDIA GPU acceleration (PRIORITY)
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            m_useCUDA = true;
            qDebug() << "CUDA GPU acceleration enabled for NVIDIA GPU (PRIORITY)";
            qDebug() << "CUDA devices found:" << cv::cuda::getCudaEnabledDeviceCount();

            // Get CUDA device info
            cv::cuda::DeviceInfo deviceInfo(0);
            if (deviceInfo.isCompatible()) {
                qDebug() << "CUDA Device:" << deviceInfo.name();
                qDebug() << "Memory:" << deviceInfo.totalMemory() / (1024*1024) << "MB";
                qDebug() << "Compute Capability:" << deviceInfo.majorVersion() << "." << deviceInfo.minorVersion();
                qDebug() << "CUDA will be used for color conversion and resizing operations";

                // Pre-allocate CUDA GPU memory pools for better performance
                qDebug() << "Pre-allocating CUDA GPU memory pools...";
                try {
                    // Pre-allocate common frame sizes for CUDA operations
                    cv::cuda::GpuMat cudaFramePool1, cudaFramePool2, cudaFramePool3;
                    cudaFramePool1.create(720, 1280, CV_8UC3);  // Common camera resolution
                    cudaFramePool2.create(480, 640, CV_8UC3);   // Smaller processing size
                    cudaFramePool3.create(360, 640, CV_8UC1);   // Grayscale processing

                    qDebug() << "CUDA GPU memory pools pre-allocated successfully";
                    qDebug() << "  - CUDA Frame pool 1: 1280x720 (RGB)";
                    qDebug() << "  - CUDA Frame pool 2: 640x480 (RGB)";
                    qDebug() << "  - CUDA Frame pool 3: 640x360 (Grayscale)";

                    // Set CUDA device for optimal performance
                    cv::cuda::setDevice(0);
                    qDebug() << "CUDA device 0 set for optimal performance";

                } catch (const cv::Exception& e) {
                    qWarning() << "CUDA GPU memory pool allocation failed:" << e.what();
                }
            }
        } else {
            qDebug() << "CUDA not available, checking OpenCL";
            m_useCUDA = false;
        }
    } catch (...) {
        qDebug() << "CUDA initialization failed, checking OpenCL";
        m_useCUDA = false;
    }


    // Initialize async processing
    m_personDetectionWatcher = new QFutureWatcher<cv::Mat>(this);
    connect(m_personDetectionWatcher, &QFutureWatcher<cv::Mat>::finished,
            this, &Capture::onPersonDetectionFinished);

    // Set CUDA device for optimal performance
    if (m_useCUDA) {
        try {
            // Check CUDA device availability before setting
            int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
            if (deviceCount > 0) {
                cv::cuda::setDevice(0);
                qDebug() << "CUDA device 0 set for optimal performance";

                // Test CUDA memory allocation
                cv::cuda::GpuMat testMat;
                testMat.create(100, 100, CV_8UC3);
                if (testMat.empty()) {
                    throw cv::Exception(0, "CUDA memory allocation test failed", "", "", 0);
                }
                qDebug() << "CUDA memory allocation test passed";
            } else {
                qWarning() << "No CUDA devices available, disabling CUDA";
                m_useCUDA = false;
            }
        } catch (const cv::Exception& e) {
            qWarning() << "CUDA initialization failed:" << e.what() << "Disabling CUDA";
            m_useCUDA = false;
        } catch (...) {
            qWarning() << "Unknown CUDA initialization error, disabling CUDA";
            m_useCUDA = false;
        }
    }

    qDebug() << "Enhanced Person Detection and Segmentation initialized successfully";
    qDebug() << "GPU Priority: CUDA (NVIDIA) > OpenCL (AMD) > CPU (fallback)";
}

// Segmentation functions moved to capture_segmentation.cpp

// Phase 2A: GPU Result Validation
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
                
                // Increment frame counter to signal new unique frame is ready
                // This is used for accurate FPS measurement (frame-to-frame timing)
                m_segmentedFrameCounter++;

                // Update GPU utilization flags
                if (m_useCUDA) {
                    m_cudaUtilized = true;
                    m_gpuUtilized = false;
                } else if (m_useGPU) {
                    m_gpuUtilized = true;
                    m_cudaUtilized = false;
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
    qDebug() << "Capture::setSystemMonitor() called with pointer:" << (void*)monitor;
    m_systemMonitor = monitor;
}

void Capture::togglePersonDetection()
{
    // Toggle segmentation on/off
    m_segmentationEnabledInCapture = !m_segmentationEnabledInCapture;
    qDebug() << "Person detection toggled:" << m_segmentationEnabledInCapture;
}

double Capture::getPersonDetectionProcessingTime() const
{
    return m_lastPersonDetectionTime;
}

bool Capture::isGPUAvailable() const
{
    return m_useGPU;
}

bool Capture::isCUDAAvailable() const
{
    return m_useCUDA;
}

void Capture::enableProcessingModes()
{
    // Only enable heavy processing modes after camera has been running for a while
    qDebug() << "Enabling heavy processing modes";
    }

void Capture::disableProcessingModes()
{
    qDebug() << "Disabling heavy processing modes for non-capture pages";
}

void Capture::showLoadingCameraLabel()
{
    // Clear the video label to show black screen instead of previous frame
    if (ui && ui->videoLabel) {
        ui->videoLabel->clear();
        ui->videoLabel->setText("Loading camera...");
        ui->videoLabel->setStyleSheet("background-color: black; color: white; font-size: 24px;");
    }
}

void Capture::hideLoadingCameraLabel()
{
    if (loadingCameraLabel) {
        loadingCameraLabel->hide();
    }
}

void Capture::handleFirstFrame()
{
    // This method runs in the main thread (thread-safe)
    if (!m_firstFrameHandled) {
        m_firstFrameHandled = true;
        qDebug() << "First frame received - initializing processing";
        
        // Initialize GPU processing after first frame
        if (m_useCUDA || m_useGPU) {
            initializeGPUOnlyProcessing();
        }
        
        // Initialize person detection
        initializePersonDetection();
    }
}

void Capture::enableSegmentationInCapture()
{
    qDebug() << "Enabling segmentation for capture interface";
    m_segmentationEnabledInCapture = true;
}

void Capture::disableSegmentationOutsideCapture()
{
    qDebug() << "Disabling segmentation outside capture interface";
    m_segmentationEnabledInCapture = false;
}

void Capture::restoreSegmentationState()
{
    qDebug() << "Restoring segmentation state for capture interface";
    // Restore previous segmentation state if needed
}

bool Capture::isSegmentationEnabledInCapture() const
{
    return m_segmentationEnabledInCapture;
}

void Capture::setSelectedBackgroundTemplate(const QString &path)
{
    m_selectedBackgroundTemplate = path;
    m_useBackgroundTemplate = !path.isEmpty();
    qDebug() << "Background template set to:" << path << "useBackgroundTemplate:" << m_useBackgroundTemplate;
}

QString Capture::getSelectedBackgroundTemplate() const
{
    return m_selectedBackgroundTemplate;
}

void Capture::setVideoTemplateDuration(int durationSeconds)
{
    if (durationSeconds > 0) {
        m_currentVideoTemplate.durationSeconds = durationSeconds;
        qDebug() << "Video template duration set to:" << durationSeconds << "seconds";
    }
}

int Capture::getVideoTemplateDuration() const
{
    return m_currentVideoTemplate.durationSeconds;
}

void Capture::initializeRecordingSystem()
{
    qDebug() << " ASYNC RECORDING: Initializing recording system...";
}

void Capture::cleanupRecordingSystem()
{
    qDebug() << " ASYNC RECORDING: Cleaning up recording system...";
}

void Capture::queueFrameForRecording(const cv::Mat &frame)
{
    if (!m_recordingThreadActive) {
        return;
    }
}

void Capture::onVideoProcessingFinished()
{
    qDebug() << " Video processing finished in background thread";
}

void Capture::processRecordingFrame()
{
    // This method is no longer needed since we're capturing display directly
}

void Capture::cleanupResources()
{
    qDebug() << "Capture::cleanupResources - Cleaning up resources when leaving capture page";
}

void Capture::initializeResources()
{
    qDebug() << " Capture::initializeResources - Initializing resources when entering capture page";
}

void Capture::initializeLightingCorrection()
{
    qDebug() << "Initializing lighting correction system";
}

bool Capture::isGPULightingAvailable() const
{
    return m_lightingCorrector ? m_lightingCorrector->isGPUAvailable() : false;
}

void Capture::setReferenceTemplate(const QString &templatePath)
{
    if (m_lightingCorrector) {
        m_lightingCorrector->setReferenceTemplate(templatePath);
    }
}

void Capture::setSubtractionReferenceImage(const QString &imagePath)
{
    if (imagePath.isEmpty()) {
        m_subtractionReferenceImage = cv::Mat();
            return;
        }
    }
    
void Capture::setSubtractionReferenceImage2(const QString &imagePath)
{
    if (imagePath.isEmpty()) {
        m_subtractionReferenceImage2 = cv::Mat();
        return;
    }
}

void Capture::setSubtractionReferenceBlendWeight(double weight)
{
    m_subtractionBlendWeight = std::max(0.0, std::min(1.0, weight));
}

// Lighting correction functions moved to capture_lighting.cpp
// Edge blending functions moved to capture_edge_blending.cpp

// Remote changes (duplicate functions) - keeping local version above
// The following functions were already moved to separate files:
// - applyPostProcessingLighting() -> capture_lighting.cpp
// - applyLightingToRawPersonRegion() -> capture_lighting.cpp
// - extractPersonMaskFromSegmentedFrame() -> (helper function)
// Removing duplicate code from remote branch:

/*
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

        //  CUDA-Accelerated Guided image filtering to refine a soft alpha only on a thin edge ring
        // Guidance is the current output (result) which already has person hard-copied
        const int gfRadius = 8; // window size (reduced for better performance)
        const float gfEps = 1e-2f; // regularization (increased for better performance)
        
        // Use GPU memory pool stream and buffers for optimized guided filtering
        cv::cuda::Stream& guidedFilterStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, binMask, gfRadius, gfEps, m_gpuMemoryPool, guidedFilterStream);
        
        //  ENHANCED: Apply edge blurring to create smooth transitions between background and segmented object
        const float edgeBlurRadius = 3.0f; // Increased blur radius for better background-object transition
        cv::Mat edgeBlurredPerson = applyEdgeBlurringCUDA(scaledPerson, binMask, backgroundFrame, edgeBlurRadius, m_gpuMemoryPool, guidedFilterStream);
        if (!edgeBlurredPerson.empty()) {
            scaledPerson = edgeBlurredPerson;
            qDebug() << "STATIC MODE: Applied CUDA edge blurring with radius" << edgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
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
        cv::cuda::Stream& finalStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat finalEdgeBlurred = applyEdgeBlurringCUDA(result, binMask, cleanBackground, finalEdgeBlurRadius, m_gpuMemoryPool, finalStream);
        if (!finalEdgeBlurred.empty()) {
            result = finalEdgeBlurred;
            qDebug() << "STATIC MODE: Applied final CUDA edge blurring to composite result with radius" << finalEdgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
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
//  Performance Control Methods
//  REMOVED: Real-time lighting methods - not needed for post-processing only mode

//  CUDA-Accelerated Guided Filter for Edge-Blending (Memory Pool Optimized)
// GPU-optimized guided filtering that maintains FPS and quality using pre-allocated buffers
// Edge blending functions moved to capture_edge_blending.cpp

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

