#include "core/capture.h"
#include "core/camera.h"
#include "ui/foreground.h"
#include "ui_capture.h"
#include "algorithms/hand_detection/hand_detector.h"
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
#include <opencv2/cudabgsegm.hpp> // Added for cv::cuda::BackgroundSubtractorMOG2
#include <opencv2/cudafilters.hpp> // Added for CUDA filter functions
#include <opencv2/cudaarithm.hpp> // Added for CUDA arithmetic operations (inRange, bitwise_or)
#include <QtConcurrent/QtConcurrent>
#include <QMutexLocker>
#include <chrono>
#include <QFutureWatcher>
#include "algorithms/lighting_correction/lighting_corrector.h"
#include "core/system_monitor.h"

//  Forward declarations for guided filtering functions
static cv::Mat guidedFilterGrayAlphaCPU(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps);
static cv::Mat guidedFilterGrayAlphaCUDA(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, cv::cuda::Stream &stream = cv::cuda::Stream::Null());
static cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, GPUMemoryPool &memoryPool, cv::cuda::Stream &stream = cv::cuda::Stream::Null());

//  Forward declarations for edge blurring functions
static cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, GPUMemoryPool &memoryPool, cv::cuda::Stream &stream = cv::cuda::Stream::Null());
static cv::Mat applyEdgeBlurringCPU(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius);
static cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

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
    , m_useCUDA(false)
    , m_gpuUtilized(false)
    , m_cudaUtilized(false)
    , m_handDetector(new HandDetector())
    , m_personDetectionWatcher(nullptr)
    , m_showHandDetection(true)
    , m_lastDetections()
    // , m_tfliteModelLoaded(false)
    , m_handDetectionEnabled(false)
    , m_handDetectionMutex()
    , m_handDetectionTimer()
    , m_lastHandDetectionTime(0.0)
    , m_handDetectionFPS(0)
    , m_lastHandDetections()
    , m_systemMonitor(nullptr)
    , m_handDetectionFuture()
    , m_handDetectionWatcher(nullptr)
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
    , m_cudaHogDetector()
    , m_gpuMemoryPool()
    , m_gpuMemoryPoolInitialized(false)
    , m_recordingThread(nullptr)
    , m_recordingFrameTimer(nullptr)
    , m_recordingMutex()
    , m_recordingFrameQueue()
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , m_recordingThreadActive(false)
    , m_recordingStream()
    , handDetectionLabel(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_recordingGpuBuffer()
    , m_cachedPixmap(640, 480)
    , m_cudaHogScales{0.5, 0.75}
    , m_cudaHogHitThresholdPrimary(0.0)
    , m_cudaHogHitThresholdSecondary(-0.2)
    , m_cudaHogWinStridePrimary(8, 8)
    , m_cudaHogWinStrideSecondary(16, 16)
    , m_detectionNmsOverlap(0.35)
    , m_detectionMotionOverlap(0.12)
    , m_smoothingHoldFrames(5)
    , m_smoothingHoldCounter(0)
    , m_detectionSkipInterval(2)
    , m_detectionSkipCounter(0)
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

    // Initialize Hand Detection (disabled by default)
    qDebug() << "Constructor: Setting m_handDetectionEnabled = false";
    m_handDetectionEnabled = false;
    qDebug() << "Constructor: Before initializeHandDetection(), m_handDetectionEnabled = " << m_handDetectionEnabled;
    initializeHandDetection();
    qDebug() << "Constructor: After initializeHandDetection(), m_handDetectionEnabled = " << m_handDetectionEnabled;
    // Ensure it stays disabled after initialization
    m_handDetectionEnabled = false;
    qDebug() << "Constructor: Final setting, m_handDetectionEnabled = " << m_handDetectionEnabled;
    qDebug() << "Constructor: Hand detection is DISABLED by default as requested";
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

    // Initialize async processing for hand detection
    m_handDetectionWatcher = new QFutureWatcher<QList<HandDetection>>(this);
    connect(m_handDetectionWatcher, &QFutureWatcher<QList<HandDetection>>::finished,
            this, &Capture::onHandDetectionFinished);

    // Connect hand detection signal to slot for thread-safe UI updates
    connect(this, &Capture::handTriggeredCapture, this, &Capture::onHandTriggeredCapture);

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
    if (m_handDetector) {
        disconnect(m_handDetector, nullptr, this, nullptr);
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
    if (handDetectionLabel){ delete handDetectionLabel; handDetectionLabel = nullptr; }

    // Clean up hand detector
    if (m_handDetector){ delete m_handDetector; m_handDetector = nullptr; }
    
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

        // Process hand detection in background (non-blocking) - only after initial frames
        if (m_handDetectionEnabled && frameCount > 30 && !m_handDetectionWatcher->isRunning()) {
            QMutexLocker locker(&m_handDetectionMutex);
            m_currentFrame = qImageToCvMat(image);

            // Process hand detection in background thread
            QFuture<QList<HandDetection>> future = QtConcurrent::run([this]() {
                return m_handDetector->detect(m_currentFrame);
            });

            // Set the future to the watcher for async processing
            m_handDetectionWatcher->setFuture(future);
        }
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = frameTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Calculate current FPS for real-time display (thread-safe)
    static QElapsedTimer fpsTimer;
    static int fpsFrameCount = 0;
    static bool fpsTimerInitialized = false;

    if (!fpsTimerInitialized) {
        fpsTimer.start();
        fpsTimerInitialized = true;
    }
    fpsFrameCount++;

    // Adaptive FPS calculation based on actual camera FPS (30-60 FPS range)
    int targetFPS = qMax(30, qMin(60, static_cast<int>(m_actualCameraFPS)));
    if (fpsFrameCount >= targetFPS) { // Update FPS every targetFPS frames
        double fpsDuration = fpsTimer.elapsed() / 1000.0;
        m_currentFPS = fpsDuration > 0 ? fpsFrameCount / fpsDuration : targetFPS;
        fpsFrameCount = 0;
        fpsTimer.start();
    }

    // Print performance stats every targetFPS frames (aligned with actual camera FPS)
    if (frameCount % targetFPS == 0) {
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
    qDebug() << "CUDA Acceleration:" << (m_useCUDA ? "YES (CUDA)" : "NO (CPU)");
    qDebug() << "CUDA Utilized:" << (m_cudaUtilized ? "ACTIVE" : "IDLE");
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

    //  CAPTURE EXACTLY WHAT'S DISPLAYED ON SCREEN (with video template)
    QPixmap currentDisplayPixmap;

    // Get the current display from the video label (what user actually sees)
    if (ui->videoLabel) {
        QPixmap labelPixmap = ui->videoLabel->pixmap();
        if (!labelPixmap.isNull()) {
            currentDisplayPixmap = labelPixmap;
            qDebug() << " DIRECT CAPTURE: Capturing current display from video label";
        } else {
            qDebug() << " DIRECT CAPTURE: Video label pixmap is null, using fallback";
        }
            } else {
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

    // Keep hand detection state as set by user
        if (m_handDetector) {
            m_handDetector->resetGestureState();
    }

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

                // Reset capture button for next capture and re-enable hand detection
                ui->capture->setEnabled(true);
                enableHandDetection(true);
                qDebug() << "Capture completed - hand detection re-enabled for next capture";
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
    qDebug() << "setupDebugDisplay called - m_handDetectionEnabled:" << m_handDetectionEnabled;
    
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
        qDebug() << "Force refresh - m_handDetectionEnabled:" << m_handDetectionEnabled;
        updateDebugDisplay();
    });

    qDebug() << "Debug display setup complete - FPS, GPU, and CUDA status should be visible";
    qDebug() << "Final hand detection state:" << m_handDetectionEnabled;
}

void Capture::enableHandDetectionForCapture()
{
    // Don't force hand detection to be enabled - respect user preference
    // enableHandDetection(true);  // Removed - let user control hand detection
    qDebug() << "enableHandDetectionForCapture called - hand detection state:" << m_handDetectionEnabled;
    enableSegmentationInCapture();
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

    // Reset hand detection state (keep user preference)
    m_captureReady = true;
    if (m_handDetector) {
        m_handDetector->resetGestureState();
        qDebug() << "Hand detection state reset";
    }

    // Reset segmentation state for capture interface
    enableSegmentationInCapture();
    qDebug() << "Segmentation reset for capture interface";

    // Reset all detection state
    m_lastHandDetections.clear();
    m_handDetectionFPS = 0.0;
    m_lastHandDetectionTime = 0.0;

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
            // Toggle hand detection
            if (m_handDetectionEnabled) {
                enableHandDetection(false);
                qDebug() << "Hand Detection DISABLED via 'H' key";
            } else {
                enableHandDetection(true);
                qDebug() << "Hand Detection ENABLED via 'H' key";
            }
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
        if (m_handDetector) {
            m_handDetector->resetGestureState();
        }
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

void Capture::drawHandBoundingBoxes(cv::Mat &/*frame*/, const QList<HandDetection> &detections)
{
    // REMOVED: No more bounding box drawing to avoid conflicts with segmentation
    // Only show gesture status in console for debugging
    for (const auto& detection : detections) {
        if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
            // Check if capture should be triggered - automatically start countdown when hand closed
            if (m_handDetector->shouldTriggerCapture()) {
                qDebug() << "HAND CLOSED DETECTED! Automatically triggering capture...";
                qDebug() << "Segmentation enabled:" << m_segmentationEnabledInCapture;

                // Emit signal to trigger capture in main thread (thread-safe)
                emit handTriggeredCapture();
            }

            // Show gesture status in console only
            bool isOpen = m_handDetector->isHandOpen(detection.landmarks);
            bool isClosed = m_handDetector->isHandClosed(detection.landmarks);
            double closureRatio = m_handDetector->calculateHandClosureRatio(detection.landmarks);

            // Update hand state for trigger logic
            m_handDetector->updateHandState(isClosed);

            if (isOpen || isClosed) {
                QString gestureStatus = isOpen ? "OPEN" : "CLOSED";
                qDebug() << "Hand detected - Gesture:" << gestureStatus
                         << "Confidence:" << static_cast<int>(detection.confidence * 100) << "%"
                         << "Closure ratio:" << closureRatio;
            }
        }
    }
}



void Capture::updateDebugDisplay()
{
    // Debug output to verify the method is being called
    static int updateCount = 0;
    updateCount++;
    
    // Always log hand detection state for debugging
    qDebug() << "updateDebugDisplay #" << updateCount << "- m_handDetectionEnabled:" << m_handDetectionEnabled;
    
    if (updateCount % 10 == 0) { // Log every 5 seconds (10 updates * 500ms)
        qDebug() << "Debug display update #" << updateCount << "FPS:" << m_currentFPS << "GPU:" << m_useGPU << "CUDA:" << m_useCUDA;
    }

    if (debugLabel) {
        QString peopleDetected = QString::number(m_lastDetections.size());
        QString segmentationStatus = m_segmentationEnabledInCapture ? "ON" : "OFF";
        QString handStatus = m_handDetectionEnabled ? "ON" : "OFF";
        QString aiFPS = m_segmentationEnabledInCapture ? QString::number(m_personDetectionFPS, 'f', 1) : "0.0";
        
        // Debug: Log the exact values being used
        static int debugCount = 0;
        if (debugCount < 5) {
            qDebug() << "DEBUG DISPLAY VALUES:";
            qDebug() << "  - m_handDetectionEnabled:" << m_handDetectionEnabled;
            qDebug() << "  - handStatus string:" << handStatus;
            qDebug() << "  - segmentationStatus:" << segmentationStatus;
            debugCount++;
        }
        
        QString debugInfo = QString("FPS: %1 | People: %2 | Seg: %3 | Hand: %4 | Person FPS: %5")
                           .arg(QString::number(m_currentFPS, 'f', 1))
                           .arg(peopleDetected)
                           .arg(segmentationStatus)
                           .arg(handStatus)
                           .arg(aiFPS);
        debugLabel->setText(debugInfo);
        
        // Debug output to verify hand detection state
        static int debugCount2 = 0;
        if (debugCount2 < 10) { // Log first 10 times to see startup behavior
            qDebug() << "Debug display - m_handDetectionEnabled:" << m_handDetectionEnabled << "Display:" << handStatus;
            debugCount2++;
        }
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
    if (m_segmentationEnabledInCapture && !m_bgSubtractor) {
        qWarning() << "Background subtractor not initialized, initializing now...";
        m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
        if (!m_bgSubtractor) {
            qWarning() << "Failed to create background subtractor!";
            QMessageBox::warning(this, "Recording Error", "Failed to initialize segmentation system. Please restart the application.");
            ui->capture->setEnabled(true);
            return;
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
    
    if (m_recordedFrames.isEmpty()) {
        qWarning() << " No recorded frames available for post-processing";
        return;
    }
    
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
            QList<QPixmap> processedFrames = processRecordedVideoWithLighting(m_recordedFrames, m_adjustedRecordingFPS);
            emit videoRecordedWithComparison(processedFrames, m_originalRecordedFrames, m_adjustedRecordingFPS);
            emit showFinalOutputPage();
            return;
        }
        
        // Run processing in background thread using QtConcurrent
        QFuture<QList<QPixmap>> future = QtConcurrent::run(
            [this]() {
                return processRecordedVideoWithLighting(m_recordedFrames, m_adjustedRecordingFPS);
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

// Phase 2A: GPU-Only Processing Pipeline
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

// Enhanced Person Detection and Segmentation methods
void Capture::initializePersonDetection()
{
    qDebug() << "===== initializePersonDetection() CALLED =====";
    qDebug() << "Initializing Enhanced Person Detection and Segmentation...";

    // Initialize HOG detectors for person detection
    qDebug() << "===== CAPTURE INITIALIZATION STARTED =====";
    m_hogDetector.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    m_hogDetectorDaimler.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());

    // Initialize CUDA HOG detector for GPU acceleration
    qDebug() << "===== STARTING CUDA HOG INITIALIZATION =====";

    // Check if CUDA is available
    int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
    qDebug() << "CUDA devices found:" << cudaDevices;

    if (cudaDevices > 0) {
        try {
            qDebug() << "Creating CUDA HOG detector...";
            // Create CUDA HOG with default people detector
            m_cudaHogDetector = cv::cuda::HOG::create(
                cv::Size(64, 128),  // win_size
                cv::Size(16, 16),   // block_size
                cv::Size(8, 8),     // block_stride
                cv::Size(8, 8),     // cell_size
                9                   // nbins
            );

            if (!m_cudaHogDetector.empty()) {
                qDebug() << "CUDA HOG detector created successfully";
                m_cudaHogDetector->setSVMDetector(m_cudaHogDetector->getDefaultPeopleDetector());
                qDebug() << "CUDA HOG detector ready for GPU acceleration";
            } else {
                qWarning() << "CUDA HOG creation failed - detector is empty";
                m_cudaHogDetector = nullptr;
            }
        } catch (const cv::Exception& e) {
            qWarning() << "CUDA HOG initialization failed:" << e.what();
            m_cudaHogDetector = nullptr;
        }
    } else {
        qDebug() << "CUDA not available for HOG initialization";
        m_cudaHogDetector = nullptr;
    }
    qDebug() << "===== FINAL CUDA HOG INITIALIZATION CHECK =====";
    qDebug() << "CUDA HOG detector pointer:" << m_cudaHogDetector.get();
    qDebug() << "CUDA HOG detector empty:" << (m_cudaHogDetector && m_cudaHogDetector.empty() ? "yes" : "no");

    if (m_cudaHogDetector && !m_cudaHogDetector.empty()) {
        qDebug() << "CUDA HOG detector successfully initialized and ready!";
        m_useCUDA = true; // Ensure CUDA is enabled
    } else {
        qWarning() << "CUDA HOG detector initialization failed or not available";
        m_cudaHogDetector = nullptr;
    }
    qDebug() << "===== CUDA HOG INITIALIZATION COMPLETE =====";

    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);

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
    if (!m_useCUDA) {
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

void Capture::adjustRect(cv::Rect &r) const
{
    // Ensure the detection rectangle covers the full person: do not shrink.
    // Keeping the original detector rectangle preserves full-body coverage.
    // No-op for performance and coverage.
    (void)r; // Suppress unused parameter warning
}

std::vector<cv::Rect> Capture::runCudaHogPass(const cv::Mat &frame,
                                             double resizeScale,
                                             double hitThreshold,
                                             const cv::Size &winStride)
{
    std::vector<cv::Rect> detections;

    if (!m_useCUDA || !m_cudaHogDetector || m_cudaHogDetector->empty() || frame.empty()) {
        return detections;
    }

    if (resizeScale <= 0.0) {
        return detections;
    }

    try {
        cv::cuda::GpuMat gpuFrame;
        gpuFrame.upload(frame);

        cv::cuda::GpuMat gpuGray;
        cv::cuda::cvtColor(gpuFrame, gpuGray, cv::COLOR_BGR2GRAY);

        cv::Size targetSize(cvRound(frame.cols * resizeScale), cvRound(frame.rows * resizeScale));
        if (targetSize.width < 128 || targetSize.height < 256) {
            targetSize.width = std::max(targetSize.width, 128);
            targetSize.height = std::max(targetSize.height, 256);
        }

        cv::cuda::GpuMat gpuResized;
        cv::cuda::resize(gpuGray, gpuResized, targetSize, 0, 0, cv::INTER_LINEAR);

        m_cudaHogDetector->setHitThreshold(hitThreshold);
        // Validate winStride against block stride (8x8) and window size (64x128)
        cv::Size validatedStride = winStride;
        const cv::Size blockStride(8, 8);
        const cv::Size winSize(64, 128);
        if (validatedStride.width <= 0 || validatedStride.height <= 0 ||
            (validatedStride.width % blockStride.width) != 0 ||
            (validatedStride.height % blockStride.height) != 0 ||
            validatedStride.width > winSize.width ||
            validatedStride.height > winSize.height) {
            // Prefer 16x16 for speed; otherwise fallback to 8x8
            validatedStride = cv::Size(16, 16);
        }
        m_cudaHogDetector->setWinStride(validatedStride);

        std::vector<cv::Rect> found;
        m_cudaHogDetector->detectMultiScale(gpuResized, found);

        const double invScale = 1.0 / resizeScale;
        for (auto &rect : found) {
            rect.x = cvRound(rect.x * invScale);
            rect.y = cvRound(rect.y * invScale);
            rect.width = cvRound(rect.width * invScale);
            rect.height = cvRound(rect.height * invScale);
            detections.push_back(rect);
        }

    } catch (const cv::Exception &e) {
        qWarning() << "CUDA HOG pass failed:" << e.what();
    }

    return detections;
}

std::vector<cv::Rect> Capture::runCudaHogMultiPass(const cv::Mat &frame)
{
    std::vector<cv::Rect> combined;

    for (size_t i = 0; i < m_cudaHogScales.size(); ++i) {
        const double scale = m_cudaHogScales[i];
        const double hitThreshold = (i == 0) ? m_cudaHogHitThresholdPrimary : m_cudaHogHitThresholdSecondary;
        const cv::Size &stride = (i == 0) ? m_cudaHogWinStridePrimary : m_cudaHogWinStrideSecondary;

        std::vector<cv::Rect> passDetections = runCudaHogPass(frame, scale, hitThreshold, stride);
        combined.insert(combined.end(), passDetections.begin(), passDetections.end());

        // Early exit only if primary pass already found at least 2 detections
        if (i == 0 && passDetections.size() >= 2) {
            break;
        }
    }

    // If CUDA pass failed or found nothing, fall back to classic HOG
    if (combined.empty()) {
        try {
            std::vector<cv::Rect> cpuDetections = runClassicHogPass(frame);
            combined.insert(combined.end(), cpuDetections.begin(), cpuDetections.end());
        } catch (...) {
            // Keep empty if CPU also fails
        }
    }

    return combined;
}

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

    if (m_useCUDA && m_cudaHogDetector && !m_cudaHogDetector->empty()) {
        detections = runCudaHogMultiPass(frame);
    } else {
        detections = runClassicHogPass(frame);
    }

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
        } else if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
            // Phase 1: Use pre-advanced video frame from timer instead of reading synchronously
            try {
                // THREAD SAFETY: Lock mutex for safe video frame access
                QMutexLocker locker(&m_dynamicVideoMutex);
                
                if (!m_dynamicVideoFrame.empty()) {
                    cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    qDebug() << "Successfully using video frame for segmentation - frame size:" << m_dynamicVideoFrame.cols << "x" << m_dynamicVideoFrame.rows;
                    qDebug() << "Segmented frame size:" << segmentedFrame.cols << "x" << segmentedFrame.rows;
                } else {
                    // Fallback: read frame synchronously if timer hasn't advanced yet
                    cv::Mat nextBg;
                    
                    // Use CPU video reader (skip CUDA reader since it's failing)
                    if (m_dynamicVideoCap.isOpened()) {
                        if (!m_dynamicVideoCap.read(nextBg) || nextBg.empty()) {
                            m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            m_dynamicVideoCap.read(nextBg);
                        }
                    }
                    
                    if (!nextBg.empty()) {
                        cv::resize(nextBg, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                        m_dynamicVideoFrame = segmentedFrame.clone();
                        qDebug() << "Fallback: Successfully read video frame for segmentation";
                    } else {
                        segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                        qWarning() << "Fallback: Failed to read video frame - using black background";
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
                        yOffset + actualScaledSize.height <= backgroundSize.height &&
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
        } else if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
            // Phase 2A: GPU-only video background processing
            try {
                // THREAD SAFETY: Lock mutex for safe GPU frame access
                QMutexLocker locker(&m_dynamicVideoMutex);
                
                if (!m_dynamicGpuFrame.empty()) {
                    // Already on GPU from NVDEC, avoid CPU upload
                    cv::cuda::resize(m_dynamicGpuFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.download(segmentedFrame);
                    qDebug() << "Using NVDEC GPU frame for segmentation - size:" << m_dynamicGpuFrame.cols << "x" << m_dynamicGpuFrame.rows;
                } else if (!m_dynamicVideoFrame.empty()) {
                    // Fallback: CPU frame upload
                    m_gpuBackgroundFrame.upload(m_dynamicVideoFrame);
                    cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_gpuSegmentedFrame.download(segmentedFrame);
                    qDebug() << "Fallback CPU frame upload for segmentation - size:" << m_dynamicVideoFrame.cols << "x" << m_dynamicVideoFrame.rows;
                } else {
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                    qWarning() << "Dynamic video frame is empty - using black background";
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
        m_cudaUtilized = false;
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

bool Capture::isCUDAAvailable() const
{
    return m_useCUDA;
}

// --- Green-screen configuration API ---
void Capture::setGreenScreenEnabled(bool enabled)
{
    m_greenScreenEnabled = enabled;
}

bool Capture::isGreenScreenEnabled() const
{
    return m_greenScreenEnabled;
}

void Capture::setGreenHueRange(int hueMin, int hueMax)
{
    m_greenHueMin = std::max(0, std::min(179, hueMin));
    m_greenHueMax = std::max(0, std::min(179, hueMax));
}

void Capture::setGreenSaturationMin(int sMin)
{
    m_greenSatMin = std::max(0, std::min(255, sMin));
}

void Capture::setGreenValueMin(int vMin)
{
    m_greenValMin = std::max(0, std::min(255, vMin));
}

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
        
        if (m_useCUDA) {
            try {
                // GPU-accelerated absolute difference
                cv::cuda::GpuMat gpu_frame, gpu_ref, gpu_diff;
                gpu_frame.upload(frame);
                gpu_ref.upload(refResized);
                
                cv::cuda::absdiff(gpu_frame, gpu_ref, gpu_diff);
                
                // Convert to grayscale and threshold
                cv::cuda::GpuMat gpu_gray;
                cv::cuda::cvtColor(gpu_diff, gpu_gray, cv::COLOR_BGR2GRAY);
                
                cv::cuda::GpuMat gpu_mask;
                cv::cuda::threshold(gpu_gray, gpu_mask, 30, 255, cv::THRESH_BINARY);
                
                gpu_mask.download(fgMask);
            } catch (...) {
                // Fallback to CPU
                cv::Mat diff;
                cv::absdiff(frame, refResized, diff);
                cv::Mat gray;
                cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
                cv::threshold(gray, fgMask, 30, 255, cv::THRESH_BINARY);
            }
        } else {
            // CPU static reference subtraction
            cv::Mat diff;
            cv::absdiff(frame, refResized, diff);
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
    if (m_useCUDA) {
        // CUDA-accelerated background subtraction using cv::cuda::BackgroundSubtractorMOG2
        try {
            // Upload to GPU
            cv::cuda::GpuMat gpu_frame;
            gpu_frame.upload(frame);

            // Create CUDA background subtractor if not already created
            static cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> cuda_bg_subtractor;
            if (cuda_bg_subtractor.empty()) {
                cuda_bg_subtractor = cv::cuda::createBackgroundSubtractorMOG2(500, 16, false);
            }

            // CUDA-accelerated background subtraction
            cv::cuda::GpuMat gpu_fgmask;
            cuda_bg_subtractor->apply(gpu_frame, gpu_fgmask, -1);

            // Download result to CPU
            gpu_fgmask.download(fgMask);

            // Apply morphological operations on CPU (OpenCV CUDA limitation)
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);

        } catch (...) {
            // Fallback to CPU if CUDA fails
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

void Capture::updateGreenBackgroundModel(const cv::Mat &frame) const
{
    if (frame.empty() || frame.channels() != 3) {
        return;
    }

    const int rawBorderX = std::max(6, frame.cols / 24);
    const int rawBorderY = std::max(6, frame.rows / 24);
    const int borderX = std::min(rawBorderX, frame.cols);
    const int borderY = std::min(rawBorderY, frame.rows);

    if (borderX <= 0 || borderY <= 0) {
        return;
    }

    cv::Mat sampleMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    const cv::Rect topRect(0, 0, frame.cols, borderY);
    const cv::Rect bottomRect(0, std::max(0, frame.rows - borderY), frame.cols, borderY);
    const cv::Rect leftRect(0, 0, borderX, frame.rows);
    const cv::Rect rightRect(std::max(0, frame.cols - borderX), 0, borderX, frame.rows);

    cv::rectangle(sampleMask, topRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, bottomRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, leftRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, rightRect, cv::Scalar(255), cv::FILLED);

    const int samplePixels = cv::countNonZero(sampleMask);
    if (samplePixels < (frame.rows + frame.cols)) {
        return;
    }

    cv::Mat hsv, ycrcb;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Scalar hsvMean, hsvStd;
    cv::Scalar ycrcbMean, ycrcbStd;
    cv::Scalar bgrMean, bgrStd;

    cv::meanStdDev(hsv, hsvMean, hsvStd, sampleMask);
    cv::meanStdDev(ycrcb, ycrcbMean, ycrcbStd, sampleMask);
    cv::meanStdDev(frame, bgrMean, bgrStd, sampleMask);

    cv::Mat sampleData(samplePixels, 3, CV_32F);
    int idx = 0;
    for (int y = 0; y < frame.rows; ++y) {
        const uchar *maskPtr = sampleMask.ptr<uchar>(y);
        const cv::Vec3b *pixPtr = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            if (!maskPtr[x]) continue;
            float *row = sampleData.ptr<float>(idx++);
            row[0] = static_cast<float>(pixPtr[x][0]);
            row[1] = static_cast<float>(pixPtr[x][1]);
            row[2] = static_cast<float>(pixPtr[x][2]);
        }
    }
    if (idx > 3) {
        cv::Mat cropped = sampleData.rowRange(0, idx);
        cv::Mat cov, meanRow;
        cv::calcCovarMatrix(cropped, cov, meanRow,
                            cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
        cv::Mat covDouble;
        cov.convertTo(covDouble, CV_64F);
        covDouble += cv::Mat::eye(3, 3, CV_64F) * 1e-3;
        cv::Mat invCov;
        if (cv::invert(covDouble, invCov, cv::DECOMP_SVD)) {
            m_bgColorInvCov(0, 0) = invCov.at<double>(0, 0);
            m_bgColorInvCov(0, 1) = invCov.at<double>(0, 1);
            m_bgColorInvCov(0, 2) = invCov.at<double>(0, 2);
            m_bgColorInvCov(1, 0) = invCov.at<double>(1, 0);
            m_bgColorInvCov(1, 1) = invCov.at<double>(1, 1);
            m_bgColorInvCov(1, 2) = invCov.at<double>(1, 2);
            m_bgColorInvCov(2, 0) = invCov.at<double>(2, 0);
            m_bgColorInvCov(2, 1) = invCov.at<double>(2, 1);
            m_bgColorInvCov(2, 2) = invCov.at<double>(2, 2);
            m_bgColorInvCovReady = true;
        }
    }

    const bool alreadyInitialized = m_bgModelInitialized;
    auto blendValue = [alreadyInitialized](double current, double measurement) {
        if (!alreadyInitialized) return measurement;
        return 0.85 * current + 0.15 * measurement;
    };

    m_bgHueMean = blendValue(m_bgHueMean, hsvMean[0]);
    m_bgHueStd = blendValue(m_bgHueStd, std::max(1.0, hsvStd[0]));
    m_bgSatMean = blendValue(m_bgSatMean, hsvMean[1]);
    m_bgSatStd = blendValue(m_bgSatStd, std::max(1.0, hsvStd[1]));
    m_bgValMean = blendValue(m_bgValMean, hsvMean[2]);
    m_bgValStd = blendValue(m_bgValStd, std::max(1.0, hsvStd[2]));

    m_bgCbMean = blendValue(m_bgCbMean, ycrcbMean[2]);
    m_bgCbStd = blendValue(m_bgCbStd, std::max(1.0, ycrcbStd[2]));
    m_bgCrMean = blendValue(m_bgCrMean, ycrcbMean[1]);
    m_bgCrStd = blendValue(m_bgCrStd, std::max(1.0, ycrcbStd[1]));

    m_bgBlueMean = blendValue(m_bgBlueMean, bgrMean[0]);
    m_bgGreenMean = blendValue(m_bgGreenMean, bgrMean[1]);
    m_bgRedMean = blendValue(m_bgRedMean, bgrMean[2]);
    m_bgBlueStd = blendValue(m_bgBlueStd, std::max(4.0, bgrStd[0]));
    m_bgGreenStd = blendValue(m_bgGreenStd, std::max(4.0, bgrStd[1]));
    m_bgRedStd = blendValue(m_bgRedStd, std::max(4.0, bgrStd[2]));

    m_bgModelInitialized = true;
}

Capture::AdaptiveGreenThresholds Capture::computeAdaptiveGreenThresholds() const
{
    AdaptiveGreenThresholds thresholds;

    auto clampHue = [](int value) {
        return std::max(0, std::min(179, value));
    };
    auto clampByte = [](int value) {
        return std::max(0, std::min(255, value));
    };

    if (!m_bgModelInitialized) {
        thresholds.hueMin = clampHue(m_greenHueMin);
        thresholds.hueMax = clampHue(m_greenHueMax);
        thresholds.strictSatMin = clampByte(m_greenSatMin);
        thresholds.relaxedSatMin = clampByte(std::max(10, m_greenSatMin - 10));
        thresholds.strictValMin = clampByte(m_greenValMin);
        thresholds.relaxedValMin = clampByte(std::max(10, m_greenValMin - 10));
        thresholds.darkSatMin = clampByte(std::max(5, m_greenSatMin - 10));
        thresholds.darkValMax = clampByte(m_greenValMin + 50);
        thresholds.cbMin = 50.0;
        thresholds.cbMax = 150.0;
        thresholds.crMax = 150.0;
        thresholds.greenDelta = 8.0;
        thresholds.greenRatioMin = 0.45;
        thresholds.lumaMin = 45.0;
        thresholds.probabilityThreshold = 0.55;
        thresholds.guardValueMax = 150;
        thresholds.guardSatMax = 90;
        thresholds.edgeGuardMin = 45.0;
        thresholds.hueGuardPadding = 6;
        return thresholds;
    }

    const double hueStd = std::max(4.0, m_bgHueStd);
    const double satStd = std::max(4.0, m_bgSatStd);
    const double valStd = std::max(4.0, m_bgValStd);
    const double cbStd = std::max(2.5, m_bgCbStd);
    const double crStd = std::max(2.5, m_bgCrStd);

    const int huePadding = static_cast<int>(std::round(2.5 * hueStd)) + 4;
    const int relaxedSatAmount = static_cast<int>(std::round(1.9 * satStd)) + 5;
    const int relaxedValAmount = static_cast<int>(std::round(1.6 * valStd)) + 5;

    thresholds.hueMin = clampHue(static_cast<int>(std::round(m_bgHueMean)) - huePadding);
    thresholds.hueMax = clampHue(static_cast<int>(std::round(m_bgHueMean)) + huePadding);
    thresholds.strictSatMin = clampByte(static_cast<int>(std::round(m_bgSatMean - 0.6 * satStd)));
    thresholds.relaxedSatMin = clampByte(std::max(18, static_cast<int>(std::round(m_bgSatMean - relaxedSatAmount))));
    thresholds.strictValMin = clampByte(static_cast<int>(std::round(m_bgValMean - 0.6 * valStd)));
    thresholds.relaxedValMin = clampByte(std::max(18, static_cast<int>(std::round(m_bgValMean - relaxedValAmount))));
    thresholds.darkSatMin = clampByte(std::max(5, static_cast<int>(std::round(m_bgSatMean - 0.8 * satStd))));
    thresholds.darkValMax = clampByte(static_cast<int>(std::round(m_bgValMean + 2.2 * valStd)));

    const double cbRange = 2.2 * cbStd + 6.0;
    thresholds.cbMin = std::max(0.0, m_bgCbMean - cbRange);
    thresholds.cbMax = std::min(255.0, m_bgCbMean + cbRange);
    thresholds.crMax = std::min(255.0, m_bgCrMean + 2.4 * crStd + 6.0);

    const double greenDominance = m_bgGreenMean - std::max(m_bgRedMean, m_bgBlueMean);
    thresholds.greenDelta = std::max(4.0, greenDominance * 0.35 + 6.0);
    const double sumRGB = std::max(1.0, m_bgRedMean + m_bgGreenMean + m_bgBlueMean);
    const double bgRatio = m_bgGreenMean / sumRGB;
    thresholds.greenRatioMin = std::clamp(bgRatio - 0.05, 0.35, 0.8);
    thresholds.lumaMin = std::max(25.0, m_bgValMean - 1.2 * valStd);
    thresholds.probabilityThreshold = 0.58;
    thresholds.guardValueMax = clampByte(static_cast<int>(std::round(std::min(200.0, thresholds.lumaMin + 70.0))));
    thresholds.guardSatMax = clampByte(std::max(25, static_cast<int>(std::round(m_bgSatMean - 0.3 * satStd))));
    thresholds.edgeGuardMin = std::max(35.0, 55.0 - 0.25 * valStd);
    thresholds.hueGuardPadding = 8;
    auto invVariance = [](double stdVal) {
        const double boundedStd = std::max(5.0, stdVal);
        return 1.0 / (boundedStd * boundedStd + 50.0);
    };
    thresholds.invVarB = invVariance(m_bgBlueStd);
    thresholds.invVarG = invVariance(m_bgGreenStd);
    thresholds.invVarR = invVariance(m_bgRedStd);
    const double avgStd = (m_bgBlueStd + m_bgGreenStd + m_bgRedStd) / 3.0;
    thresholds.colorDistanceThreshold = std::max(2.5, std::min(4.5, 1.2 + avgStd * 0.08));
    thresholds.colorGuardThreshold = thresholds.colorDistanceThreshold + 1.6;

    return thresholds;
}

// AGGRESSIVE GREEN REMOVAL: Remove all green pixels including boundary pixels
// FAST & ACCURATE: Works for both green AND teal/cyan backdrops
cv::Mat Capture::createGreenScreenPersonMask(const cv::Mat &frame) const
{
    if (frame.empty()) return cv::Mat();

    try {
        // Split BGR channels
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        cv::Mat green = channels[1];
        cv::Mat red = channels[2];

        // For green AND cyan/teal screens: just check that green dominates red
        // Green: G > R, G > B  
        // Cyan/Teal: G > R, B > R, G ≈ B
        // Both have GREEN > RED, so just check G - R
        // Dark clothing has similar R, G, B values (G - R is small)
        cv::Mat greenDominantR;
        cv::subtract(green, red, greenDominantR);  // G - R

        // G - R > 15 means green or teal background (red is low relative to green)
        // Higher threshold (15) to preserve dark clothing
        cv::Mat greenMask;
        cv::threshold(greenDominantR, greenMask, 15, 255, cv::THRESH_BINARY);

        // Invert: NOT green/teal = person
        cv::Mat personMask;
        cv::bitwise_not(greenMask, personMask);

        return personMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Error in createGreenScreenPersonMask:" << e.what();
        return cv::Mat();
    }
}

//  GPU-ACCELERATED GREEN SCREEN MASKING with Optimized Memory Management
cv::cuda::GpuMat Capture::createGreenScreenPersonMaskGPU(const cv::cuda::GpuMat &gpuFrame) const
{
    cv::cuda::GpuMat emptyMask;
    if (gpuFrame.empty()) {
        qWarning() << "GPU frame is empty, cannot create green screen mask";
        return emptyMask;
    }

    if (!m_bgModelInitialized) {
        try {
            cv::Mat fallbackFrame;
            gpuFrame.download(fallbackFrame);
            updateGreenBackgroundModel(fallbackFrame);
        } catch (const cv::Exception &) {
            // Ignore download errors; thresholds fallback to defaults
        }
    }

    try {
        //  INITIALIZE CACHED FILTERS (once only, reused for all frames)
        static bool filtersInitialized = false;
        if (!filtersInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                // Cache filters to avoid recreating on every frame
                const_cast<Capture*>(this)->m_greenScreenCannyDetector = cv::cuda::createCannyEdgeDetector(30, 90);
                const_cast<Capture*>(this)->m_greenScreenGaussianBlur = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(5, 5), 1.0);
                filtersInitialized = true;
                qDebug() << "GPU green screen filters initialized and cached";
            } catch (const cv::Exception &e) {
                qWarning() << "Failed to initialize GPU green screen filters:" << e.what();
                filtersInitialized = false;
            }
        }

        // GPU: FAST & ACCURATE - Works for both green AND teal/cyan backdrops
        // Split BGR channels
        std::vector<cv::cuda::GpuMat> channels(3);
        cv::cuda::split(gpuFrame, channels);
        cv::cuda::GpuMat green = channels[1];
        cv::cuda::GpuMat red = channels[2];

        // For green AND cyan/teal screens: just check that green dominates red
        // Green: G > R, G > B  |  Cyan/Teal: G > R, B > R, G ≈ B
        // Both have GREEN > RED, so just check G - R
        cv::cuda::GpuMat greenDominantR;
        cv::cuda::subtract(green, red, greenDominantR);  // G - R

        // G - R > 15 means green or teal background (red is low relative to green)
        // Higher threshold (15) to preserve dark clothing
        cv::cuda::GpuMat greenMask;
        cv::cuda::threshold(greenDominantR, greenMask, 15, 255, cv::THRESH_BINARY);

        // Invert: NOT green/teal = person
        cv::cuda::GpuMat gpuPersonMask;
        cv::cuda::bitwise_not(greenMask, gpuPersonMask);

        return gpuPersonMask;

    } catch (const cv::Exception &e) {
        qWarning() << "GPU green screen masking failed:" << e.what() << "- returning empty mask";
        return emptyMask;
    } catch (const std::exception &e) {
        qWarning() << "Exception in GPU green screen masking:" << e.what();
        return emptyMask;
    } catch (...) {
        qWarning() << "Unknown exception in GPU green screen masking";
        return emptyMask;
    }
}

// CONTOUR-BASED MASK REFINEMENT - Remove all fragments, keep only person
cv::Mat Capture::refineGreenScreenMaskWithContours(const cv::Mat &mask, int minArea) const
{
    if (mask.empty()) return cv::Mat();

    try {
        cv::Mat refinedMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        
        // Find all contours in the mask
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat maskCopy = mask.clone();
        cv::findContours(maskCopy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            qDebug() << "No contours found in green screen mask";
            return refinedMask;
        }
        
        // Sort contours by area (largest first)
        std::vector<std::pair<int, double>> contourAreas;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            contourAreas.push_back({static_cast<int>(i), area});
        }
        std::sort(contourAreas.begin(), contourAreas.end(), 
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });
        
        // Keep only the largest contours (person) that meet minimum area
        int keptContours = 0;
        const int MAX_PERSONS = 3; // Support up to 3 people
        
        for (size_t i = 0; i < std::min(contourAreas.size(), (size_t)MAX_PERSONS); i++) {
            int idx = contourAreas[i].first;
            double area = contourAreas[i].second;
            
            if (area >= minArea) {
                // STRATEGY 1: Draw filled contour (aggressive - includes all pixels)
                cv::drawContours(refinedMask, contours, idx, cv::Scalar(255), cv::FILLED);
                keptContours++;
                qDebug() << "Kept contour" << i << "with area:" << area;
            } else {
                qDebug() << "Rejected contour" << i << "with area:" << area << "(too small)";
            }
        }
        
        if (keptContours == 0) {
            qWarning() << "No valid contours found - all below minimum area" << minArea;
            return refinedMask;
        }
        
        // STRATEGY 2: Apply convex hull to smooth boundaries and remove concave artifacts
        // ENHANCED: Less aggressive - use union instead of intersection to preserve black regions
        cv::Mat hullMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        for (size_t i = 0; i < std::min(contourAreas.size(), (size_t)MAX_PERSONS); i++) {
            int idx = contourAreas[i].first;
            double area = contourAreas[i].second;
            
            if (area >= minArea) {
                std::vector<cv::Point> hull;
                cv::convexHull(contours[idx], hull);
                
                // Only apply convex hull if it doesn't expand area too much (< 30% increase)
                // Increased threshold to be less aggressive with black clothing
                double hullArea = cv::contourArea(hull);
                if (hullArea < area * 1.3) {
                    std::vector<std::vector<cv::Point>> hullContours = {hull};
                    cv::drawContours(hullMask, hullContours, 0, cv::Scalar(255), cv::FILLED);
                }
            }
        }
        
        // ENHANCED: Use union instead of intersection to preserve black regions with holes
        // This prevents removal of valid person pixels in black clothing areas
        if (!hullMask.empty() && cv::countNonZero(hullMask) > 0) {
            cv::bitwise_or(refinedMask, hullMask, refinedMask);
        }
        
        // STRATEGY 3: Final morphological cleanup with larger kernel to fill holes in black regions
        // ENHANCED: Use larger kernel to fill larger holes that may appear in black clothing
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(refinedMask, refinedMask, cv::MORPH_CLOSE, kernel); // Fill small/medium holes
        cv::morphologyEx(refinedMask, refinedMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));  // Remove small protrusions
        
        // Additional pass with even larger kernel for very large holes (e.g., in black clothing)
        cv::Mat largeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        cv::morphologyEx(refinedMask, refinedMask, cv::MORPH_CLOSE, largeKernel);
        
        qDebug() << "Contour refinement complete - kept" << keptContours << "contours";
        return refinedMask;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Contour refinement failed:" << e.what();
        return mask.clone();
    } catch (const std::exception &e) {
        qWarning() << "Exception in contour refinement:" << e.what();
        return mask.clone();
    }
}

// TEMPORAL MASK SMOOTHING - Ensure consistency across frames
cv::Mat Capture::applyTemporalMaskSmoothing(const cv::Mat &currentMask) const
{
    if (currentMask.empty()) return cv::Mat();

    // First frame or after reset
    if (m_lastGreenScreenMask.empty() || m_lastGreenScreenMask.size() != currentMask.size()) {
        m_lastGreenScreenMask = currentMask.clone();
        m_greenScreenMaskStableCount = 0;
        return currentMask.clone();
    }

    try {
        // Calculate difference between current and last mask
        cv::Mat diff;
        cv::absdiff(currentMask, m_lastGreenScreenMask, diff);
        int diffPixels = cv::countNonZero(diff);
        double diffRatio = (double)diffPixels / (currentMask.rows * currentMask.cols);
        
        // If masks are very similar (< 5% difference), blend them
        cv::Mat smoothedMask;
        if (diffRatio < 0.05) {
            // Blend current with last mask (70% current, 30% last) for stability
            cv::addWeighted(currentMask, 0.7, m_lastGreenScreenMask, 0.3, 0, smoothedMask, CV_8U);
            cv::threshold(smoothedMask, smoothedMask, 127, 255, cv::THRESH_BINARY);
            m_greenScreenMaskStableCount++;
        } else if (diffRatio < 0.15) {
            // Medium change - blend with more weight to current (80% current, 20% last)
            cv::addWeighted(currentMask, 0.8, m_lastGreenScreenMask, 0.2, 0, smoothedMask, CV_8U);
            cv::threshold(smoothedMask, smoothedMask, 127, 255, cv::THRESH_BINARY);
            m_greenScreenMaskStableCount = std::max(0, m_greenScreenMaskStableCount - 1);
        } else {
            // Large change - use current mask directly (new person or movement)
            smoothedMask = currentMask.clone();
            m_greenScreenMaskStableCount = 0;
        }
        
        // Update last mask for next frame
        m_lastGreenScreenMask = smoothedMask.clone();
        
        return smoothedMask;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Temporal smoothing failed:" << e.what();
        m_lastGreenScreenMask = currentMask.clone();
        return currentMask.clone();
    } catch (const std::exception &e) {
        qWarning() << "Exception in temporal smoothing:" << e.what();
        m_lastGreenScreenMask = currentMask.clone();
        return currentMask.clone();
    }
}

// GRABCUT REFINEMENT - Advanced foreground/background separation
cv::Mat Capture::refineWithGrabCut(const cv::Mat &frame, const cv::Mat &initialMask) const
{
    if (frame.empty() || initialMask.empty()) {
        qWarning() << "GrabCut: Empty input";
        return initialMask.clone();
    }

    // VALIDATION: Check frame and mask dimensions match
    if (frame.size() != initialMask.size()) {
        qWarning() << "GrabCut: Size mismatch - frame:" << frame.cols << "x" << frame.rows 
                   << "mask:" << initialMask.cols << "x" << initialMask.rows;
        return initialMask.clone();
    }

    // VALIDATION: Check mask has sufficient non-zero pixels
    int nonZeroCount = cv::countNonZero(initialMask);
    if (nonZeroCount < 1000) {
        qWarning() << "GrabCut: Insufficient mask pixels:" << nonZeroCount;
        return initialMask.clone();
    }

    try {
        qDebug() << "Applying GrabCut refinement for precise segmentation";
        
        // Create GrabCut mask from initial mask
        // GC_BGD=0 (definite background), GC_FGD=1 (definite foreground)
        // GC_PR_BGD=2 (probably background), GC_PR_FGD=3 (probably foreground)
        cv::Mat grabCutMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        
        // SAFE EROSION: Check if mask is large enough before eroding
        cv::Mat definiteFG;
        cv::Rect maskBounds = cv::boundingRect(initialMask);
        if (maskBounds.width > 30 && maskBounds.height > 30) {
            cv::erode(initialMask, definiteFG, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
        } else {
            // Mask too small for erosion, use smaller kernel
            cv::erode(initialMask, definiteFG, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        }
        
        // VALIDATION: Ensure we still have some foreground pixels
        if (cv::countNonZero(definiteFG) < 100) {
            qWarning() << "GrabCut: Eroded mask too small, using original";
            return initialMask.clone();
        }
        
        grabCutMask.setTo(cv::GC_FGD, definiteFG);
        
        // Dilate mask to get probable foreground region
        cv::Mat probableFG;
        cv::dilate(initialMask, probableFG, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
        cv::Mat unknownRegion = probableFG - definiteFG;
        grabCutMask.setTo(cv::GC_PR_FGD, unknownRegion);
        
        // Everything else is definite background
        cv::Mat bgMask = (grabCutMask == cv::GC_BGD);
        grabCutMask.setTo(cv::GC_BGD, bgMask);
        
        // Apply GrabCut algorithm (ONLY 1 iteration for speed and stability)
        cv::Mat bgModel, fgModel;
        cv::Rect roi = cv::boundingRect(initialMask);
        
        // VALIDATION: Ensure ROI is valid and reasonable
        roi.x = std::max(0, roi.x - 10);
        roi.y = std::max(0, roi.y - 10);
        roi.width = std::min(frame.cols - roi.x, roi.width + 20);
        roi.height = std::min(frame.rows - roi.y, roi.height + 20);
        
        if (roi.width > 50 && roi.height > 50 && 
            roi.x >= 0 && roi.y >= 0 && 
            roi.x + roi.width <= frame.cols && 
            roi.y + roi.height <= frame.rows) {
            
            // CRITICAL: Use only 1 iteration to prevent crashes
            cv::grabCut(frame, grabCutMask, roi, bgModel, fgModel, 1, cv::GC_INIT_WITH_MASK);
            
            // Extract foreground
            cv::Mat refinedMask = (grabCutMask == cv::GC_FGD) | (grabCutMask == cv::GC_PR_FGD);
            refinedMask.convertTo(refinedMask, CV_8U, 255);
            
            // VALIDATION: Ensure output is reasonable
            int refinedCount = cv::countNonZero(refinedMask);
            if (refinedCount > 500 && refinedCount < frame.rows * frame.cols * 0.9) {
                qDebug() << "GrabCut refinement complete";
                return refinedMask;
            } else {
                qWarning() << "GrabCut produced invalid result, using original";
                return initialMask.clone();
            }
        } else {
            qWarning() << "Invalid ROI for GrabCut:" << roi.x << roi.y << roi.width << roi.height;
            return initialMask.clone();
        }
        
    } catch (const cv::Exception &e) {
        qWarning() << "GrabCut OpenCV exception:" << e.what();
        return initialMask.clone();
    } catch (const std::exception &e) {
        qWarning() << "GrabCut std exception:" << e.what();
        return initialMask.clone();
    } catch (...) {
        qWarning() << "GrabCut unknown exception";
        return initialMask.clone();
    }
}

// DISTANCE-BASED REFINEMENT - Ensure no green pixels near person edges
cv::Mat Capture::applyDistanceBasedRefinement(const cv::Mat &frame, const cv::Mat &mask) const
{
    if (frame.empty() || mask.empty()) {
        return mask.clone();
    }

    try {
        qDebug() << "Applying distance-based refinement to remove edge artifacts";
        
        // Convert to HSV for green detection
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        
        // Create strict green mask (more aggressive than chroma key)
        cv::Scalar lowerGreen(m_greenHueMin, m_greenSatMin + 40, m_greenValMin + 30);  // Much stricter: only bright, saturated greens
        cv::Scalar upperGreen(m_greenHueMax, 255, 255);
        cv::Mat strictGreenMask;
        cv::inRange(hsv, lowerGreen, upperGreen, strictGreenMask);
        
        // Calculate distance transform from green pixels
        cv::Mat greenDist;
        cv::distanceTransform(~strictGreenMask, greenDist, cv::DIST_L2, 5);
        
        // Normalize distance to [0, 1]
        cv::Mat normalizedDist;
        cv::normalize(greenDist, normalizedDist, 0, 1.0, cv::NORM_MINMAX, CV_32F);
        
        // Create safety buffer: pixels must be at least 10 pixels away from green
        cv::Mat safetyMask;
        cv::threshold(normalizedDist, safetyMask, 0.05, 1.0, cv::THRESH_BINARY);
        safetyMask.convertTo(safetyMask, CV_8U, 255);
        
        // Apply safety mask to person mask
        cv::Mat refinedMask;
        cv::bitwise_and(mask, safetyMask, refinedMask);
        
        // Fill any holes created by the safety buffer
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(refinedMask, refinedMask, cv::MORPH_CLOSE, kernel);
        
        qDebug() << "Distance-based refinement complete";
        return refinedMask;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Distance refinement failed:" << e.what();
        return mask.clone();
    } catch (const std::exception &e) {
        qWarning() << "Exception in distance refinement:" << e.what();
        return mask.clone();
    }
}

// CREATE TRIMAP - Three-zone map for alpha matting
cv::Mat Capture::createTrimap(const cv::Mat &mask, int erodeSize, int dilateSize) const
{
    if (mask.empty()) {
        return cv::Mat();
    }

    try {
        cv::Mat trimap = cv::Mat::zeros(mask.size(), CV_8UC1);
        
        // Definite foreground (eroded mask) = 255
        cv::Mat definiteFG;
        cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erodeSize, erodeSize));
        cv::erode(mask, definiteFG, erodeKernel);
        trimap.setTo(255, definiteFG);
        
        // Definite background (inverse of dilated mask) = 0
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilateSize, dilateSize));
        cv::Mat dilatedMask;
        cv::dilate(mask, dilatedMask, dilateKernel);
        cv::Mat definiteBG = (dilatedMask == 0);
        trimap.setTo(0, definiteBG);
        
        // Unknown region (between eroded and dilated) = 128
        cv::Mat unknownRegion = (trimap != 0) & (trimap != 255);
        trimap.setTo(128, unknownRegion);
        
        return trimap;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Trimap creation failed:" << e.what();
        return mask.clone();
    }
}

// CUSTOM GUIDED FILTER IMPLEMENTATION - Edge-preserving smoothing without ximgproc
cv::Mat Capture::customGuidedFilter(const cv::Mat &guide, const cv::Mat &src, int radius, double eps) const
{
    // Convert to float
    cv::Mat guideFloat, srcFloat;
    guide.convertTo(guideFloat, CV_32F);
    src.convertTo(srcFloat, CV_32F);
    
    // Mean of guide and source
    cv::Mat meanI, meanP;
    cv::boxFilter(guideFloat, meanI, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(srcFloat, meanP, CV_32F, cv::Size(radius, radius));
    
    // Correlation and variance
    cv::Mat corrI, corrIp;
    cv::boxFilter(guideFloat.mul(guideFloat), corrI, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(guideFloat.mul(srcFloat), corrIp, CV_32F, cv::Size(radius, radius));
    
    cv::Mat varI = corrI - meanI.mul(meanI);
    cv::Mat covIp = corrIp - meanI.mul(meanP);
    
    // Linear coefficients a and b
    cv::Mat a = covIp / (varI + eps);
    cv::Mat b = meanP - a.mul(meanI);
    
    // Mean of a and b
    cv::Mat meanA, meanB;
    cv::boxFilter(a, meanA, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(b, meanB, CV_32F, cv::Size(radius, radius));
    
    // Output
    return meanA.mul(guideFloat) + meanB;
}

// ALPHA MATTING - Extract precise alpha channel for natural edges
cv::Mat Capture::extractPersonWithAlphaMatting(const cv::Mat &frame, const cv::Mat &trimap) const
{
    if (frame.empty() || trimap.empty()) {
        return cv::Mat();
    }

    try {
        qDebug() << "Applying alpha matting for natural edge extraction";
        
        // Simple but effective alpha matting using custom guided filter
        cv::Mat alpha = trimap.clone();
        alpha.convertTo(alpha, CV_32F, 1.0/255.0);
        
        // Convert frame to grayscale for guidance
        cv::Mat frameGray;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        } else {
            frameGray = frame.clone();
        }
        
        // Apply custom guided filter for smooth alpha matte
        cv::Mat guidedAlpha = customGuidedFilter(frameGray, alpha, 8, 1e-6);
        
        // Threshold to get clean binary mask
        cv::Mat refinedMask;
        cv::threshold(guidedAlpha, refinedMask, 0.5, 1.0, cv::THRESH_BINARY);
        refinedMask.convertTo(refinedMask, CV_8U, 255);
        
        // Additional cleanup with morphology
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(refinedMask, refinedMask, cv::MORPH_OPEN, kernel);
        
        qDebug() << "Alpha matting complete";
        return refinedMask;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Alpha matting failed:" << e.what();
        // Fallback: convert trimap to binary mask
        cv::Mat fallback;
        cv::threshold(trimap, fallback, 127, 255, cv::THRESH_BINARY);
        return fallback;
    } catch (const std::exception &e) {
        qWarning() << "Exception in alpha matting:" << e.what();
        cv::Mat fallback;
        cv::threshold(trimap, fallback, 127, 255, cv::THRESH_BINARY);
        return fallback;
    }
}

// GPU-ACCELERATED GREEN SPILL REMOVAL - Remove green tint from person pixels
cv::cuda::GpuMat Capture::removeGreenSpillGPU(const cv::cuda::GpuMat &gpuFrame, const cv::cuda::GpuMat &gpuMask) const
{
    cv::cuda::GpuMat result;
    if (gpuFrame.empty() || gpuMask.empty()) {
        return gpuFrame.clone();
    }

    try {
        // Convert to HSV for color correction
        cv::cuda::GpuMat gpuHSV;
        cv::cuda::cvtColor(gpuFrame, gpuHSV, cv::COLOR_BGR2HSV);

        // Split HSV channels
        std::vector<cv::cuda::GpuMat> hsvChannels(3);
        cv::cuda::split(gpuHSV, hsvChannels);

        // Create a desaturation map based on green hue proximity
        // Pixels closer to green hue will be more desaturated
        cv::cuda::GpuMat hueChannel = hsvChannels[0].clone();
        
        // Create desaturation mask for greenish pixels (narrower range to preserve person's colors)
        cv::cuda::GpuMat greenishMask1, greenishMask2;
        cv::cuda::threshold(hueChannel, greenishMask1, m_greenHueMin, 255, cv::THRESH_BINARY);  // No extra margin
        cv::cuda::threshold(hueChannel, greenishMask2, m_greenHueMax, 255, cv::THRESH_BINARY_INV);  // No extra margin
        cv::cuda::GpuMat greenishRange;
        cv::cuda::bitwise_and(greenishMask1, greenishMask2, greenishRange);
        
        // Desaturate pixels in green hue range
        cv::cuda::GpuMat satChannel = hsvChannels[1].clone();
        cv::cuda::GpuMat desaturated;
        cv::cuda::multiply(satChannel, cv::Scalar(0.3), desaturated, 1.0, satChannel.type()); // Reduce saturation to 30%
        
        // Apply desaturation only to greenish pixels within person mask
        cv::cuda::GpuMat spillMask;
        cv::cuda::bitwise_and(greenishRange, gpuMask, spillMask);
        desaturated.copyTo(satChannel, spillMask);
        
        // Merge back
        hsvChannels[1] = satChannel;
        cv::cuda::merge(hsvChannels, gpuHSV);
        
        // Convert back to BGR
        cv::cuda::cvtColor(gpuHSV, result, cv::COLOR_HSV2BGR);
        
        // Also apply color correction in BGR space to remove green channel dominance
        std::vector<cv::cuda::GpuMat> bgrChannels(3);
        cv::cuda::split(result, bgrChannels);
        
        // Reduce green channel where spill is detected
        cv::cuda::GpuMat reducedGreen;
        cv::cuda::multiply(bgrChannels[1], cv::Scalar(0.85), reducedGreen, 1.0, bgrChannels[1].type());
        reducedGreen.copyTo(bgrChannels[1], spillMask);
        
        // Slightly boost blue and red to compensate
        cv::cuda::GpuMat boostedBlue, boostedRed;
        cv::cuda::multiply(bgrChannels[0], cv::Scalar(1.08), boostedBlue, 1.0, bgrChannels[0].type());
        cv::cuda::multiply(bgrChannels[2], cv::Scalar(1.08), boostedRed, 1.0, bgrChannels[2].type());
        boostedBlue.copyTo(bgrChannels[0], spillMask);
        boostedRed.copyTo(bgrChannels[2], spillMask);
        
        cv::cuda::merge(bgrChannels, result);
        
        // Synchronize
        cv::cuda::Stream::Null().waitForCompletion();
        
        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "GPU green spill removal failed:" << e.what();
        return gpuFrame.clone();
    } catch (const std::exception &e) {
        qWarning() << "Exception in GPU green spill removal:" << e.what();
        return gpuFrame.clone();
    }
}

// Derive bounding boxes from a binary person mask
std::vector<cv::Rect> Capture::deriveDetectionsFromMask(const cv::Mat &mask) const
{
    std::vector<cv::Rect> detections;
    if (mask.empty()) return detections;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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
void Capture::processFrameWithHandDetection(const cv::Mat &frame)
{
    if (!m_handDetector || !m_handDetector->isInitialized()) {
        return;
    }

    m_handDetectionTimer.start();

    // Process hand detection using the hand_detector.h/.cpp system
    QList<HandDetection> detections = m_handDetector->detect(frame);

    // Store results with mutex protection
    {
        QMutexLocker locker(&m_handDetectionMutex);
        m_lastHandDetections = detections;
        m_lastHandDetectionTime = m_handDetectionTimer.elapsed() / 1000.0;
    }

    // Process hand detection results for trigger logic (only if enabled)
    if (!m_handDetectionEnabled) {
        return; // Skip processing if hand detection is disabled
    }

    for (const auto& detection : detections) {
        if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
            // Check if capture should be triggered - automatically start countdown when hand closed
            if (m_handDetector->shouldTriggerCapture()) {
                qDebug() << "HAND CLOSED DETECTED! Automatically triggering capture...";
                qDebug() << "Segmentation enabled:" << m_segmentationEnabledInCapture;

                // Emit signal to trigger capture in main thread (thread-safe)
                emit handTriggeredCapture();
            }

            // Show gesture status in console only
            bool isOpen = m_handDetector->isHandOpen(detection.landmarks);
            bool isClosed = m_handDetector->isHandClosed(detection.landmarks);
            double closureRatio = m_handDetector->calculateHandClosureRatio(detection.landmarks);

            // Update hand state for trigger logic
            m_handDetector->updateHandState(isClosed);

            if (isOpen || isClosed) {
                QString gestureStatus = isOpen ? "OPEN" : "CLOSED";
                qDebug() << "Hand detected - Gesture:" << gestureStatus
                         << "Confidence:" << static_cast<int>(detection.confidence * 100) << "%"
                         << "Closure ratio:" << closureRatio;
            }
        }
    }

    // Update FPS calculation
    static int handDetectionFrameCount = 0;
    static QElapsedTimer handDetectionFPSTimer;

    if (handDetectionFrameCount == 0) {
        handDetectionFPSTimer.start();
    }
    handDetectionFrameCount++;

    if (handDetectionFrameCount >= 30) { // Update every 30 frames
        double duration = handDetectionFPSTimer.elapsed() / 1000.0;
        m_handDetectionFPS = duration > 0 ? handDetectionFrameCount / duration : 0;
        handDetectionFrameCount = 0;
        handDetectionFPSTimer.start();
    }
}

void Capture::initializeHandDetection()
{
    if (!m_handDetector) {
        m_handDetector = new HandDetector();
    }

    if (m_handDetector && !m_handDetector->isInitialized()) {
        bool success = m_handDetector->initialize();
        if (success) {
            qDebug() << "Hand detection initialized successfully";
            qDebug() << "initializeHandDetection: Before setting false, m_handDetectionEnabled = " << m_handDetectionEnabled;
            m_handDetectionEnabled = false; // Disabled by default
            qDebug() << "initializeHandDetection: After setting false, m_handDetectionEnabled = " << m_handDetectionEnabled;
        } else {
            qDebug() << "Failed to initialize hand detection";
            qDebug() << "initializeHandDetection: Before setting false (failed), m_handDetectionEnabled = " << m_handDetectionEnabled;
            m_handDetectionEnabled = false;
            qDebug() << "initializeHandDetection: After setting false (failed), m_handDetectionEnabled = " << m_handDetectionEnabled;
        }
    }
}

void Capture::startHandTriggeredCountdown()
{
    // This slot runs in the main thread and safely updates UI elements
    if (!countdownTimer || !countdownTimer->isActive()) {
        qDebug() << "Starting countdown from hand detection signal...";
        
        // CRITICAL: Temporarily disable hand detection to prevent loop during countdown
        // (will be re-enabled after countdown completes)
        enableHandDetection(false);
        qDebug() << "🚫 Hand detection temporarily DISABLED during countdown";
        
        // Start 5-second countdown for hand-triggered capture (same as button press)
        ui->capture->setEnabled(false);
        countdownValue = 5; // 5 second countdown
        countdownLabel->setText(QString::number(countdownValue));
        countdownLabel->show();
        countdownLabel->raise(); // Bring to front
        countdownTimer->start(1000); // 1 second intervals
        qDebug() << "5-second countdown automatically started by hand detection!";
    } else {
        qDebug() << "Countdown already active, ignoring hand trigger";
    }
}

void Capture::onHandTriggeredCapture()
{
    // This slot runs in the main thread and safely handles hand detection triggers
    qDebug() << "Hand triggered capture signal received in main thread";
    startHandTriggeredCountdown();
}

void Capture::onHandDetectionFinished()
{
    // This slot runs in the main thread when hand detection completes
    if (!m_handDetectionWatcher || !m_handDetectionEnabled) {
        return;
    }

    // Get the detection results from the watcher
    QList<HandDetection> detections = m_handDetectionWatcher->result();
    
    // Store detections for debug display
    {
        QMutexLocker locker(&m_handDetectionMutex);
        m_lastHandDetections = detections;
    }

    // Update accuracy tracking for system monitor
    if (m_systemMonitor && !detections.isEmpty()) {
        // Use the highest confidence detection as accuracy metric
        double maxConfidence = 0.0;
        for (const auto& detection : detections) {
            if (detection.confidence > maxConfidence) {
                maxConfidence = detection.confidence;
            }
        }
        m_systemMonitor->updateAccuracy(maxConfidence);
    }

    // FIST GESTURE TRIGGER LOGIC
    // Track consecutive frames where a fist is detected
    static int closedFistFrameCount = 0;
    static bool alreadyTriggered = false;
    
    bool fistDetectedThisFrame = false;
    
    // Debug: Show what we got
    if (!detections.isEmpty()) {
        qDebug() << "📦 Received" << detections.size() << "FIST detection(s) | Confidence threshold:" << m_handDetector->getConfidenceThreshold();
    }
    
    // ALL detections are already FISTS (filtered in detectHandGestures)
    // Just check if any detection has sufficient confidence
    for (const auto& detection : detections) {
        qDebug() << "Checking FIST - Type:" << detection.handType 
                 << "| Confidence:" << detection.confidence 
                 << "| isClosed:" << detection.isClosed;
        
        if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
            // This is a valid FIST with good confidence!
            fistDetectedThisFrame = true;
            qDebug() << "VALID FIST CONFIRMED! Confidence:" << detection.confidence;
            break; // Found a valid fist, no need to check other detections
        } else {
            qDebug() << "Fist detected but confidence too low:" << detection.confidence << "<" << m_handDetector->getConfidenceThreshold();
        }
    }
    
    // Update frame counter based on detection
    if (fistDetectedThisFrame) {
        closedFistFrameCount++;
        qDebug() << "Fist frame count:" << closedFistFrameCount << "/ 2 required";
        
        // Trigger capture after 2 frames (prevents false triggers while still fast)
        if (closedFistFrameCount >= 2 && !alreadyTriggered) {
            qDebug() << "FIST TRIGGER! 2 consecutive frames - starting capture!";
            alreadyTriggered = true;
            emit handTriggeredCapture();
        }
    } else {
        // Reset counter if no fist detected
        if (closedFistFrameCount > 0) {
            qDebug() << "Fist lost - resetting counter from" << closedFistFrameCount;
        }
        closedFistFrameCount = 0;
        alreadyTriggered = false; // Allow new triggers
    }
}
void Capture::enableHandDetection(bool enable)
{
    m_handDetectionEnabled = enable;
    qDebug() << "FIST DETECTION" << (enable ? "ENABLED" : "DISABLED") << "- Only detecting FIST gestures!";
    qDebug() << "enableHandDetection called with enable=" << enable << "from:" << Q_FUNC_INFO;

    if (enable) {
        // Enable hand detection
        if (m_handDetector && !m_handDetector->isInitialized()) {
            initializeHandDetection();
        }
        if (m_handDetector) {
            m_handDetector->resetGestureState();
            qDebug() << "FIST detector ready - Make a FIST to trigger capture!";
        }
    } else {
        // Disable hand detection
        if (m_handDetector) {
            m_handDetector->resetGestureState();
            qDebug() << "FIST detection disabled";
        }
        // Clear any pending detections
        m_lastHandDetections.clear();
    }
}

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

    // Disable hand detection
    m_handDetectionEnabled = false;

    // Disable segmentation outside capture interface
    disableSegmentationOutsideCapture();

    // Clear any pending detections
    m_lastHandDetections.clear();

    // Reset processing timers
    m_personDetectionTimer.restart();
    m_handDetectionTimer.restart();

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
    if (!m_gpuMemoryPoolInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
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
    m_cudaUtilized = false;

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
    : morphCloseFilter()
    , morphOpenFilter()
    , morphDilateFilter()
    , cannyDetector()
    , detectionStream()
    , segmentationStream()
    , compositionStream()
    , currentFrameBuffer(0)
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
    qDebug() << " GPU Memory Pool: Constructor called";
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

        // Initialize frame buffers (triple buffering)
        for (int i = 0; i < 3; ++i) {
            gpuFrameBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Frame buffer" << i << "allocated";
        }

        // Initialize segmentation buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuSegmentationBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Segmentation buffer" << i << "allocated";
        }

        // Initialize detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuDetectionBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Detection buffer" << i << "allocated";
        }

        // Initialize temporary buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuTempBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Temp buffer" << i << "allocated";
        }

        //  Initialize guided filtering buffers (quad buffering for complex operations)
        for (int i = 0; i < 4; ++i) {
            gpuGuidedFilterBuffers[i] = cv::cuda::GpuMat(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Guided filter buffer" << i << "allocated";
        }

        // Initialize box filter buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuBoxFilterBuffers[i] = cv::cuda::GpuMat(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Box filter buffer" << i << "allocated";
        }

        //  Initialize edge blurring buffers (triple buffering for complex operations)
        for (int i = 0; i < 3; ++i) {
            gpuEdgeBlurBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Edge blur buffer" << i << "allocated";
        }

        // Initialize edge detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuEdgeDetectionBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Edge detection buffer" << i << "allocated";
        }

        // Create reusable CUDA filters (create once, use many times)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        morphCloseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel);
        morphOpenFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel);
        morphDilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
        cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);

        qDebug() << " GPU Memory Pool: CUDA filters created successfully";

        // Initialize CUDA streams for parallel processing
        detectionStream = cv::cuda::Stream();
        segmentationStream = cv::cuda::Stream();
        compositionStream = cv::cuda::Stream();

        qDebug() << " GPU Memory Pool: CUDA streams initialized";

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

cv::cuda::GpuMat& GPUMemoryPool::getNextFrameBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuFrameBuffers[currentFrameBuffer];
    currentFrameBuffer = (currentFrameBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextSegmentationBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuSegmentationBuffers[currentSegBuffer];
    currentSegBuffer = (currentSegBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuDetectionBuffers[currentDetBuffer];
    currentDetBuffer = (currentDetBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextTempBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuTempBuffers[currentTempBuffer];
    currentTempBuffer = (currentTempBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Guided Filtering buffer access methods
cv::cuda::GpuMat& GPUMemoryPool::getNextGuidedFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuGuidedFilterBuffers[currentGuidedFilterBuffer];
    currentGuidedFilterBuffer = (currentGuidedFilterBuffer + 1) % 4; // Quad buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextBoxFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuBoxFilterBuffers[currentBoxFilterBuffer];
    currentBoxFilterBuffer = (currentBoxFilterBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Edge Blurring buffer access methods
cv::cuda::GpuMat& GPUMemoryPool::getNextEdgeBlurBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuEdgeBlurBuffers[currentEdgeBlurBuffer];
    currentEdgeBlurBuffer = (currentEdgeBlurBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextEdgeDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuEdgeDetectionBuffers[currentEdgeDetectionBuffer];
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

    // Release CUDA filters
    morphCloseFilter.release();
    morphOpenFilter.release();
    morphDilateFilter.release();
    cannyDetector.release();

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

        // Initialize CUDA recording stream
        m_recordingStream = cv::cuda::Stream();

        // Initialize GPU recording buffer
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            m_recordingGpuBuffer = cv::cuda::GpuMat(720, 1280, CV_8UC3);
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
    
    // Check if watcher is valid
    if (!m_lightingWatcher) {
        qWarning() << " Lighting watcher is null in completion handler";
        return;
    }
    
    try {
        // Get the processed frames from the background thread
        QList<QPixmap> processedFrames = m_lightingWatcher->result();
        
        qDebug() << " DIRECT CAPTURE RECORDING: Processing complete";
        qDebug() << "Original frames:" << m_originalRecordedFrames.size() 
                 << "Processed frames:" << processedFrames.size();
        
        // Send processed frames to final output page
        emit videoRecordedWithComparison(processedFrames, m_originalRecordedFrames, m_adjustedRecordingFPS);
        
        //  FINAL STEP: Show final output page after all processing is complete
        emit showFinalOutputPage();
        qDebug() << " DIRECT CAPTURE RECORDING: Showing final output page";
        
    } catch (const std::exception& e) {
        qWarning() << " Error retrieving processed frames:" << e.what();
        // Fallback: send original frames
        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
        emit showFinalOutputPage();
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
        cv::cuda::GpuMat gpuFrame;
        gpuFrame.upload(frame, m_recordingStream);

        // Get target size
        QSize labelSize = m_cachedLabelSize.isValid() ? m_cachedLabelSize : QSize(1280, 720);

        // GPU-accelerated scaling if needed
        cv::cuda::GpuMat gpuScaled;
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Check if we're in segmentation mode with background template or dynamic video background
            if (m_segmentationEnabledInCapture && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode, just fit to label
                cv::cuda::resize(gpuFrame, gpuScaled, cv::Size(labelSize.width(), labelSize.height()), 0, 0, cv::INTER_LINEAR, m_recordingStream);
            } else {
                // Apply frame scaling for other modes
                int newWidth = qRound(frame.cols * m_personScaleFactor);
                int newHeight = qRound(frame.rows * m_personScaleFactor);
                
                //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
                newWidth = qMax(1, newWidth);
                newHeight = qMax(1, newHeight);
                
                qDebug() << " GPU RECORDING: Scaling frame to" << newWidth << "x" << newHeight << "with factor" << m_personScaleFactor;
                cv::cuda::resize(gpuFrame, gpuScaled, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR, m_recordingStream);
            }
        } else {
            // No scaling needed, just fit to label
            cv::cuda::resize(gpuFrame, gpuScaled, cv::Size(labelSize.width(), labelSize.height()), 0, 0, cv::INTER_LINEAR, m_recordingStream);
        }

        // Download processed frame
        cv::Mat processedFrame;
        gpuScaled.download(processedFrame, m_recordingStream);

        // Wait for GPU operations to complete
        m_recordingStream.waitForCompletion();

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
    m_lastHandDetections.clear();
    
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
    initializeHandDetection();
    
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

// Forward declaration to ensure availability before use
static cv::Mat guidedFilterGrayAlpha(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps);
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
QList<QPixmap> Capture::processRecordedVideoWithLighting(const QList<QPixmap> &inputFrames, double fps)
{
    Q_UNUSED(fps); // FPS parameter kept for future use but not currently needed
    
    //  ENHANCED: Apply edge-blending lighting similar to static mode
    QList<QPixmap> outputFrames;
    outputFrames.reserve(inputFrames.size());

    const int total = inputFrames.size();
    qDebug() << "Starting enhanced post-processing with edge blending for" << total << "frames";
    
    //  CRASH PREVENTION: Check if lighting corrector is properly initialized
    bool lightingAvailable = (m_lightingCorrector != nullptr);
    qDebug() << "Lighting corrector available:" << lightingAvailable;
    
    //  CRASH PREVENTION: If no lighting available, return frames as-is
    if (!lightingAvailable) {
        qDebug() << "No lighting correction available, returning original frames";
        return inputFrames;
    }
    
    for (int i = 0; i < total; ++i) {
        // Update progress to loading UI
        int pct = (total > 0) ? ((i * 100) / total) : 100;
        emit videoProcessingProgress(pct);

        try {
            //  CRASH PREVENTION: Validate frame before processing
            if (i >= inputFrames.size() || inputFrames.at(i).isNull()) {
                qWarning() << "Invalid frame at index" << i << "- using black frame";
                outputFrames.append(QPixmap(640, 480));
                continue;
            }

            // Get current frame
            QPixmap currentFrame = inputFrames.at(i);
            
            // Convert to cv::Mat for processing
            QImage frameImage = currentFrame.toImage().convertToFormat(QImage::Format_BGR888);
            if (frameImage.isNull()) {
                qWarning() << "Failed to convert frame" << i << "to QImage - using original";
                outputFrames.append(currentFrame);
                continue;
            }

            cv::Mat composedFrame(frameImage.height(), frameImage.width(), CV_8UC3,
                                  const_cast<uchar*>(frameImage.bits()), frameImage.bytesPerLine());
            
            if (composedFrame.empty()) {
                qWarning() << "Failed to convert frame" << i << "to cv::Mat - using original";
                outputFrames.append(currentFrame);
                continue;
            }

            cv::Mat composedCopy = composedFrame.clone();
            cv::Mat finalFrame;

            //  ENHANCED: Apply edge blending if raw person data is available for this frame
            bool hasRawPersonData = (i < m_recordedRawPersonRegions.size() && 
                                     i < m_recordedRawPersonMasks.size() &&
                                     !m_recordedRawPersonRegions[i].empty() &&
                                     !m_recordedRawPersonMasks[i].empty());

            if (hasRawPersonData) {
                // Apply advanced edge blending similar to static mode
                try {
                    finalFrame = applyDynamicFrameEdgeBlending(composedCopy, 
                                                               m_recordedRawPersonRegions[i],
                                                               m_recordedRawPersonMasks[i],
                                                               i < m_recordedBackgroundFrames.size() ? 
                                                               m_recordedBackgroundFrames[i] : cv::Mat());
                    
                    if (finalFrame.empty()) {
                        qWarning() << "Edge blending returned empty result for frame" << i << "- using global correction";
                        finalFrame = m_lightingCorrector->applyGlobalLightingCorrection(composedCopy);
                    }
                } catch (const std::exception& e) {
                    qWarning() << "Edge blending failed for frame" << i << ":" << e.what() << "- using global correction";
                    finalFrame = m_lightingCorrector->applyGlobalLightingCorrection(composedCopy);
                }
            } else {
                // Fallback to comprehensive global lighting correction (same as static mode)
                qDebug() << "ENHANCED: Applying full global lighting correction (same as static mode) for frame" << i;
                try {
                    finalFrame = m_lightingCorrector->applyGlobalLightingCorrection(composedCopy);
                    if (finalFrame.empty()) {
                        qWarning() << "Global lighting correction returned empty result for frame" << i;
                        finalFrame = composedCopy;
                    }
                } catch (const std::exception& e) {
                    qWarning() << "Global lighting correction failed for frame" << i << ":" << e.what();
                    finalFrame = composedCopy;
                }
            }

            // Convert back to QPixmap
            QImage outImage = cvMatToQImage(finalFrame);
            if (outImage.isNull()) {
                qWarning() << "Failed to convert processed frame" << i << "back to QImage - using original";
                outputFrames.append(currentFrame);
            } else {
                outputFrames.append(QPixmap::fromImage(outImage));
            }

        } catch (const std::exception& e) {
            qWarning() << "Exception processing frame" << i << ":" << e.what() << "- using original frame";
            if (i < inputFrames.size()) {
                outputFrames.append(inputFrames.at(i));
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
//  NEW: Dynamic Frame Edge Blending (similar to static mode)
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
        
        // Apply lighting correction to the raw person region (same as static mode - single lighting pass)
        cv::Mat lightingCorrectedPerson = applyLightingToRawPersonRegion(rawPersonRegion, rawPersonMask);
        qDebug() << "DYNAMIC: Applied raw person lighting correction (matching static mode)";
        
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

        //  CUDA-Accelerated Guided image filtering to refine a soft alpha only on a thin edge ring
        const int gfRadius = 8; // window size (reduced for better performance)
        const float gfEps = 1e-2f; // regularization (increased for better performance)
        
        // Use GPU memory pool stream and buffers for optimized guided filtering
        cv::cuda::Stream& guidedFilterStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, binMask, gfRadius, gfEps, m_gpuMemoryPool, guidedFilterStream);
        
        //  ENHANCED: Apply edge blurring to create smooth transitions between background and segmented object
        const float edgeBlurRadius = 3.0f; // Increased blur radius for better background-object transition
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
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*1+1, 2*1+1))); // shrink by ~1px for inner ring
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*4+1, 2*4+1))); // expand by ~4px for outer ring
        cv::subtract(binMask, inner, ringInner);   // just inside the boundary
        cv::subtract(outer, binMask, ringOuter);   // just outside the boundary
        
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);  // full person interior remains 1
        alphaFloat.setTo(0.0f, outer == 0); // outside remains 0
        // Strongly bias ring blend toward template to eliminate colored outlines
        alphaFloat = alphaFloat * 0.3f;

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
        const float finalEdgeBlurRadius = 4.0f; // Stronger blur for final result
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

// Scaffold: Guided filter-based feather alpha builder (stub)
// Currently returns normalized hard mask; swap in a true guided filter later if needed
static cv::Mat buildGuidedFeatherAlphaStub(const cv::Mat &guideBGR, const cv::Mat &hardMask)
{
    Q_UNUSED(guideBGR);
    cv::Mat alphaFloat;
    if (hardMask.empty()) return alphaFloat;
    hardMask.convertTo(alphaFloat, CV_32F, 1.0f / 255.0f);
    return alphaFloat;
}

//  CUDA-Accelerated Guided Filter for Edge-Blending (Memory Pool Optimized)
// GPU-optimized guided filtering that maintains FPS and quality using pre-allocated buffers
static cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                                 GPUMemoryPool &memoryPool, cv::cuda::Stream &stream)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    // Check CUDA availability
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        qWarning() << "CUDA not available, falling back to CPU guided filter";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }

    try {
        //  Performance monitoring for guided filtering
        QElapsedTimer guidedFilterTimer;
        guidedFilterTimer.start();
        
        // Get pre-allocated GPU buffers from memory pool (optimized buffer usage)
        cv::cuda::GpuMat& gpuGuide = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMask = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuP = memoryPool.getNextGuidedFilterBuffer();
        
        // Upload to GPU with stream
        gpuGuide.upload(guideBGR, stream);
        gpuMask.upload(hardMask, stream);
        
        // Convert guide to grayscale on GPU if needed
        if (guideBGR.channels() == 3) {
            cv::cuda::cvtColor(gpuGuide, gpuI, cv::COLOR_BGR2GRAY, 0, stream);
        } else {
            gpuI = gpuGuide;
        }
        
        // Convert to float32 on GPU
        gpuI.convertTo(gpuI, CV_32F, 1.0f / 255.0f, stream);
        if (hardMask.type() != CV_32F) {
            gpuMask.convertTo(gpuP, CV_32F, 1.0f / 255.0f, stream);
        } else {
            gpuP = gpuMask;
        }
        
        // Get additional buffers from memory pool
        cv::cuda::GpuMat& gpuMeanI = memoryPool.getNextBoxFilterBuffer();
        cv::cuda::GpuMat& gpuMeanP = memoryPool.getNextBoxFilterBuffer();
        cv::cuda::GpuMat& gpuCorrI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuCorrIp = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuVarI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuCovIp = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuA = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuB = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMeanA = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMeanB = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuQ = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuAlpha = memoryPool.getNextGuidedFilterBuffer();
        
        // Create box filter for GPU (reuse existing filter if available)
        cv::Ptr<cv::cuda::Filter> boxFilter = cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(radius, radius));
        
        // Step 1: Compute means and correlations on GPU
        boxFilter->apply(gpuI, gpuMeanI, stream);
        boxFilter->apply(gpuP, gpuMeanP, stream);
        
        // Compute I*I and I*P on GPU
        cv::cuda::GpuMat& gpuISquared = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuIP = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::multiply(gpuI, gpuI, gpuISquared, 1.0, -1, stream);
        cv::cuda::multiply(gpuI, gpuP, gpuIP, 1.0, -1, stream);
        
        boxFilter->apply(gpuISquared, gpuCorrI, stream);
        boxFilter->apply(gpuIP, gpuCorrIp, stream);
        
        // Step 2: Compute variance and covariance on GPU
        cv::cuda::multiply(gpuMeanI, gpuMeanI, gpuVarI, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrI, gpuVarI, gpuVarI, cv::noArray(), -1, stream);
        
        cv::cuda::multiply(gpuMeanI, gpuMeanP, gpuCovIp, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrIp, gpuCovIp, gpuCovIp, cv::noArray(), -1, stream);
        
        // Step 3: Compute coefficients a and b on GPU
        cv::cuda::GpuMat& gpuEps = memoryPool.getNextGuidedFilterBuffer();
        gpuEps.upload(cv::Mat::ones(gpuVarI.size(), CV_32F) * eps, stream);
        cv::cuda::add(gpuVarI, gpuEps, gpuVarI, cv::noArray(), -1, stream);
        cv::cuda::divide(gpuCovIp, gpuVarI, gpuA, 1.0, -1, stream);
        
        cv::cuda::multiply(gpuA, gpuMeanI, gpuB, 1.0, -1, stream);
        cv::cuda::subtract(gpuMeanP, gpuB, gpuB, cv::noArray(), -1, stream);
        
        // Step 4: Compute mean of coefficients on GPU
        boxFilter->apply(gpuA, gpuMeanA, stream);
        boxFilter->apply(gpuB, gpuMeanB, stream);
        
        // Step 5: Compute final result on GPU
        cv::cuda::multiply(gpuMeanA, gpuI, gpuQ, 1.0, -1, stream);
        cv::cuda::add(gpuQ, gpuMeanB, gpuQ, cv::noArray(), -1, stream);
        
        // Clamp result to [0,1] on GPU
        cv::cuda::threshold(gpuQ, gpuAlpha, 0.0f, 0.0f, cv::THRESH_TOZERO, stream);
        cv::cuda::threshold(gpuAlpha, gpuAlpha, 1.0f, 1.0f, cv::THRESH_TRUNC, stream);
        
        // Download result back to CPU
        cv::Mat result;
        gpuAlpha.download(result, stream);
        stream.waitForCompletion();
        
        //  Performance monitoring - log guided filtering time
        qint64 guidedFilterTime = guidedFilterTimer.elapsed();
        if (guidedFilterTime > 5) { // Only log if it takes more than 5ms
            qDebug() << "CUDA Guided Filter Performance:" << guidedFilterTime << "ms for" 
                     << guideBGR.cols << "x" << guideBGR.rows << "image";
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "CUDA guided filter failed:" << e.what() << "- falling back to CPU";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }
}
//  CUDA-Accelerated Guided Filter for Edge-Blending
// GPU-optimized guided filtering that maintains FPS and quality
static cv::Mat guidedFilterGrayAlphaCUDA(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                        cv::cuda::Stream &stream)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    // Check CUDA availability
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        qWarning() << "CUDA not available, falling back to CPU guided filter";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }

    try {
        // Convert input to GPU
        cv::cuda::GpuMat gpuGuide, gpuMask;
        cv::cuda::GpuMat gpuI, gpuP;
        
        // Upload to GPU with stream
        gpuGuide.upload(guideBGR, stream);
        gpuMask.upload(hardMask, stream);
        
        // Convert guide to grayscale on GPU if needed
        if (guideBGR.channels() == 3) {
            cv::cuda::cvtColor(gpuGuide, gpuI, cv::COLOR_BGR2GRAY, 0, stream);
        } else {
            gpuI = gpuGuide;
        }
        
        // Convert to float32 on GPU
        gpuI.convertTo(gpuI, CV_32F, 1.0f / 255.0f, stream);
        if (hardMask.type() != CV_32F) {
            gpuMask.convertTo(gpuP, CV_32F, 1.0f / 255.0f, stream);
        } else {
            gpuP = gpuMask;
        }
        
        // GPU buffers for guided filtering
        cv::cuda::GpuMat gpuMeanI, gpuMeanP, gpuCorrI, gpuCorrIp;
        cv::cuda::GpuMat gpuVarI, gpuCovIp, gpuA, gpuB;
        cv::cuda::GpuMat gpuMeanA, gpuMeanB, gpuQ, gpuAlpha;
        
        // Create box filter for GPU
        cv::Ptr<cv::cuda::Filter> boxFilter = cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(radius, radius));
        
        // Step 1: Compute means and correlations on GPU
        boxFilter->apply(gpuI, gpuMeanI, stream);
        boxFilter->apply(gpuP, gpuMeanP, stream);
        
        // Compute I*I and I*P on GPU
        cv::cuda::GpuMat gpuISquared, gpuIP;
        cv::cuda::multiply(gpuI, gpuI, gpuISquared, 1.0, -1, stream);
        cv::cuda::multiply(gpuI, gpuP, gpuIP, 1.0, -1, stream);
        
        boxFilter->apply(gpuISquared, gpuCorrI, stream);
        boxFilter->apply(gpuIP, gpuCorrIp, stream);
        
        // Step 2: Compute variance and covariance on GPU
        cv::cuda::multiply(gpuMeanI, gpuMeanI, gpuVarI, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrI, gpuVarI, gpuVarI, cv::noArray(), -1, stream);
        
        cv::cuda::multiply(gpuMeanI, gpuMeanP, gpuCovIp, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrIp, gpuCovIp, gpuCovIp, cv::noArray(), -1, stream);
        
        // Step 3: Compute coefficients a and b on GPU
        cv::cuda::GpuMat gpuEps;
        gpuEps.upload(cv::Mat::ones(gpuVarI.size(), CV_32F) * eps, stream);
        cv::cuda::add(gpuVarI, gpuEps, gpuVarI, cv::noArray(), -1, stream);
        cv::cuda::divide(gpuCovIp, gpuVarI, gpuA, 1.0, -1, stream);
        
        cv::cuda::multiply(gpuA, gpuMeanI, gpuB, 1.0, -1, stream);
        cv::cuda::subtract(gpuMeanP, gpuB, gpuB, cv::noArray(), -1, stream);
        
        // Step 4: Compute mean of coefficients on GPU
        boxFilter->apply(gpuA, gpuMeanA, stream);
        boxFilter->apply(gpuB, gpuMeanB, stream);
        
        // Step 5: Compute final result on GPU
        cv::cuda::multiply(gpuMeanA, gpuI, gpuQ, 1.0, -1, stream);
        cv::cuda::add(gpuQ, gpuMeanB, gpuQ, cv::noArray(), -1, stream);
        
        // Clamp result to [0,1] on GPU
        cv::cuda::threshold(gpuQ, gpuAlpha, 0.0f, 0.0f, cv::THRESH_TOZERO, stream);
        cv::cuda::threshold(gpuAlpha, gpuAlpha, 1.0f, 1.0f, cv::THRESH_TRUNC, stream);
        
        // Download result back to CPU
        cv::Mat result;
        gpuAlpha.download(result, stream);
        stream.waitForCompletion();
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "CUDA guided filter failed:" << e.what() << "- falling back to CPU";
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
//  CUDA-Accelerated Edge Blurring for Enhanced Edge-Blending
// GPU-optimized edge blurring that mixes background template with segmented object edges
static cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    // Check CUDA availability
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        qWarning() << "CUDA not available for edge blurring, falling back to CPU";
        return applyEdgeBlurringCPU(segmentedObject, objectMask, backgroundTemplate, blurRadius);
    }

    try {
        //  Performance monitoring for edge blurring
        QElapsedTimer edgeBlurTimer;
        edgeBlurTimer.start();

        // Get pre-allocated GPU buffers from memory pool
        cv::cuda::GpuMat& gpuObject = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuMask = memoryPool.getNextEdgeDetectionBuffer();
        cv::cuda::GpuMat& gpuBackground = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuBlurred = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuResult = memoryPool.getNextEdgeBlurBuffer();

        // Upload to GPU with stream
        gpuObject.upload(segmentedObject, stream);
        gpuMask.upload(objectMask, stream);
        gpuBackground.upload(backgroundTemplate, stream);

        // Convert mask to grayscale if needed
        if (objectMask.channels() == 3) {
            cv::cuda::cvtColor(gpuMask, gpuMask, cv::COLOR_BGR2GRAY, 0, stream);
        }

        // Step 1: Create transition zone by dilating the mask outward
        cv::cuda::GpuMat gpuDilatedMask;
        cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, 
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*static_cast<int>(blurRadius)+1, 2*static_cast<int>(blurRadius)+1))
        );
        dilateFilter->apply(gpuMask, gpuDilatedMask, stream);

        // Step 2: Create transition zone by subtracting original mask from dilated mask
        cv::cuda::GpuMat gpuTransitionZone;
        cv::cuda::subtract(gpuDilatedMask, gpuMask, gpuTransitionZone, cv::noArray(), -1, stream);

        // Step 3: Create inner edge zone by eroding the mask
        cv::cuda::GpuMat gpuErodedMask;
        cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, 
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))
        );
        erodeFilter->apply(gpuMask, gpuErodedMask, stream);

        // Step 4: Create inner edge zone by subtracting eroded mask from original mask
        cv::cuda::GpuMat gpuInnerEdgeZone;
        cv::cuda::subtract(gpuMask, gpuErodedMask, gpuInnerEdgeZone, cv::noArray(), -1, stream);

        // Step 5: Combine transition zone and inner edge zone for comprehensive edge blurring
        cv::cuda::GpuMat gpuCombinedEdgeZone;
        cv::cuda::bitwise_or(gpuTransitionZone, gpuInnerEdgeZone, gpuCombinedEdgeZone, cv::noArray(), stream);

        // Step 6: Apply Gaussian blur to both object and background
        cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(
            CV_8UC3, CV_8UC3, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f
        );
        gaussianFilter->apply(gpuObject, gpuBlurred, stream);
        
        cv::cuda::GpuMat gpuBlurredBackground;
        gaussianFilter->apply(gpuBackground, gpuBlurredBackground, stream);

        // Step 7: Create mixed background-object blend for edge zones
        cv::cuda::GpuMat gpuMixedBlend;
        cv::cuda::addWeighted(gpuBlurred, 0.6f, gpuBlurredBackground, 0.4f, 0, gpuMixedBlend, -1, stream);

        // Step 8: Apply smooth blending using the combined edge zone
        // Copy original object to result
        gpuObject.copyTo(gpuResult);
        
        // Apply mixed background-object blend in the combined edge zone
        gpuMixedBlend.copyTo(gpuResult, gpuCombinedEdgeZone);

        // Download result back to CPU
        cv::Mat result;
        gpuResult.download(result, stream);
        stream.waitForCompletion();

        //  Performance monitoring - log edge blurring time
        qint64 edgeBlurTime = edgeBlurTimer.elapsed();
        if (edgeBlurTime > 3) { // Only log if it takes more than 3ms
            qDebug() << "CUDA Edge Blur Performance:" << edgeBlurTime << "ms for" 
                     << segmentedObject.cols << "x" << segmentedObject.rows << "image, radius:" << blurRadius;
        }

        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "CUDA edge blurring failed:" << e.what() << "- falling back to CPU";
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
static cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius)
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

