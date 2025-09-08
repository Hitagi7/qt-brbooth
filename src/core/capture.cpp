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

Capture::Capture(QWidget *parent, Foreground *fg, Camera *existingCameraWorker, QThread *existingCameraThread)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , foreground(fg)
    , cameraThread(existingCameraThread)
    , cameraWorker(existingCameraWorker)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
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
    , m_displayMode(NormalMode)  // Start with normal mode to prevent freezing
    , m_personDetectionFPS(0)
    , m_lastPersonDetectionTime(0.0)
    , m_currentFrame()
    , m_lastSegmentedFrame()
    , m_personDetectionMutex()
    , m_personDetectionTimer()
    , m_hogDetector()
    , m_hogDetectorDaimler(cv::Size(48, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9)
    , m_bgSubtractor()
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
    , m_handDetectionFuture()
    , m_captureReady(false)
    , m_lastSegmentationMode(NormalMode)
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
    , fpsLabel(nullptr)
    , gpuStatusLabel(nullptr)
    , cudaStatusLabel(nullptr)
    , personDetectionLabel(nullptr)
    , personDetectionButton(nullptr)
    , personSegmentationLabel(nullptr)
    , personSegmentationButton(nullptr)
    , handDetectionLabel(nullptr)
    , handDetectionButton(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_recordingGpuBuffer()
    , m_cachedPixmap(640, 480)
    // Lighting Correction Member
    , m_lightingCorrector(nullptr)
    // Lighting Comparison Storage
    , m_originalCapturedImage()
    , m_lightingCorrectedImage()
    , m_hasLightingComparison(false)

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
    m_videoFrameRate = 30.0; // Default to 30 FPS
    m_videoFrameInterval = 33; // Default interval (1000ms / 30fps ≈ 33ms)
    m_videoPlaybackActive = false;

    // Connect video playback timer to slot
    connect(m_videoPlaybackTimer, &QTimer::timeout, this, &Capture::onVideoPlaybackTimer);

    // Phase 2A: Initialize GPU-only processing
    initializeGPUOnlyProcessing();
    
    // Initialize lighting correction system
    initializeLightingCorrection();

    setContentsMargins(0, 0, 0, 0);

    // Enable keyboard focus for this widget
    setFocusPolicy(Qt::StrongFocus);
    setFocus();

    // Setup Debug Display
    setupDebugDisplay();

    // Update button states
    updatePersonDetectionButton();
    // updateHandDetectionButton();

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
            qDebug() << "📹 First time camera initialization - showing loading label";
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

    // Initialize Hand Detection (enabled by default)
    initializeHandDetection();
    m_handDetectionEnabled = true;
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
    if (fpsLabel){ delete fpsLabel; fpsLabel = nullptr; }

    if (handDetectionLabel){ delete handDetectionLabel; handDetectionLabel = nullptr; }
    if (handDetectionButton){ delete handDetectionButton; handDetectionButton = nullptr; }

    // Clean up hand detector
    if (m_handDetector){ delete m_handDetector; m_handDetector = nullptr; }
    
    // Clean up lighting corrector
    if (m_lightingCorrector){ 
        m_lightingCorrector->cleanup();
        delete m_lightingCorrector; 
        m_lightingCorrector = nullptr; 
    }

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
    if ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) && !m_lastSegmentedFrame.empty()) {
        // Convert the processed OpenCV frame back to QImage for display
        displayImage = cvMatToQImage(m_lastSegmentedFrame);
        qDebug() << "🎯 Displaying processed segmentation frame";
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
            if (m_displayMode == SegmentationMode && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode or dynamic video mode, don't scale the entire frame
                // Person scaling is handled in createSegmentedFrame
                qDebug() << "🎯 Person-only scaling applied in segmentation mode (background template or dynamic video)";
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

                qDebug() << "🎯 Frame scaled to" << newWidth << "x" << newHeight
                         << "with factor" << m_personScaleFactor;
            }
        }

        ui->videoLabel->setPixmap(scaledPixmap);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->videoLabel->update();
    }

    // BACKGROUND PROCESSING: Move heavy work to separate threads (non-blocking)
    if (frameCount > 5 && frameCount % 3 == 0) { // Process every 3rd frame after frame 5
        // Process person detection in background (non-blocking) - only if segmentation is enabled
        if ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) && m_segmentationEnabledInCapture) {
            qDebug() << "🎯 Starting person detection processing - frame:" << frameCount << "mode:" << m_displayMode;
            QMutexLocker locker(&m_personDetectionMutex);
            m_currentFrame = qImageToCvMat(image);

            // Process unified detection in background thread
            QFuture<cv::Mat> future = QtConcurrent::run([this]() {
                return processFrameWithUnifiedDetection(m_currentFrame);
            });
            m_personDetectionWatcher->setFuture(future);
        }

        // Process hand detection in background (non-blocking) - only after initial frames
        if (m_handDetectionEnabled && frameCount > 30) {
            QMutexLocker locker(&m_handDetectionMutex);
            m_currentFrame = qImageToCvMat(image);

            // Process hand detection in background thread
            QFuture<QList<HandDetection>> future = QtConcurrent::run([this]() {
                return m_handDetector->detect(m_currentFrame);
            });

            // Store the future for later processing
            m_handDetectionFuture = future;
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
    qDebug() << "Person Detection Enabled:" << ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "Unified Detection Enabled:" << ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? "YES (ENABLED)" : "NO (DISABLED)");
    qDebug() << "GPU Acceleration:" << (m_useGPU ? "YES (OpenCL)" : "NO (CPU)");
    qDebug() << "GPU Utilization:" << (m_gpuUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "CUDA Acceleration:" << (m_useCUDA ? "YES (CUDA)" : "NO (CPU)");
    qDebug() << "CUDA Utilization:" << (m_cudaUtilized ? "ACTIVE" : "IDLE");
    qDebug() << "Person Detection FPS:" << ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
    qDebug() << "Unified Detection FPS:" << ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? QString::number(m_personDetectionFPS, 'f', 1) : "N/A (DISABLED)");
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

    // 🚀 CAPTURE EXACTLY WHAT'S DISPLAYED ON SCREEN (with video template)
    QPixmap currentDisplayPixmap;

    // Get the current display from the video label (what user actually sees)
    if (ui->videoLabel) {
        QPixmap labelPixmap = ui->videoLabel->pixmap();
        if (!labelPixmap.isNull()) {
            currentDisplayPixmap = labelPixmap;
            qDebug() << "🚀 DIRECT CAPTURE: Capturing current display from video label";
        } else {
            qDebug() << "🚀 DIRECT CAPTURE: Video label pixmap is null, using fallback";
        }
            } else {
        // Fallback: Get the appropriate frame to record
        cv::Mat frameToRecord;

        if ((m_displayMode == SegmentationMode || m_displayMode == RectangleMode) && !m_lastSegmentedFrame.empty()) {
            frameToRecord = m_lastSegmentedFrame.clone();
            qDebug() << "🚀 DIRECT CAPTURE: Fallback - using segmented frame";
        } else if (!m_originalCameraImage.isNull()) {
            frameToRecord = qImageToCvMat(m_originalCameraImage);
            qDebug() << "🚀 DIRECT CAPTURE: Fallback - using original frame";
        } else {
            qWarning() << "🚀 DIRECT CAPTURE: No frame available for recording";
            return;
        }

        // Convert to QPixmap for recording
        QImage qImage = cvMatToQImage(frameToRecord);
        currentDisplayPixmap = QPixmap::fromImage(qImage);
    }

    // Add the current display directly to recorded frames (no additional processing needed)
    m_recordedFrames.append(currentDisplayPixmap);
    qDebug() << "🚀 DIRECT CAPTURE: Display frame captured directly, total frames:" << m_recordedFrames.size();
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

    // Re-enable hand detection when recapture button is pressed
    if (!m_handDetectionEnabled) {
        m_handDetectionEnabled = true;
        if (m_handDetector) {
            m_handDetector->resetGestureState();
        }
        qDebug() << "🖐️ Hand detection RE-ENABLED by recapture button press";
        return; // Don't start countdown, just enable hand detection
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
        qDebug() << "🎬 Manual countdown started! 5 seconds to prepare...";
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
    if (countdownValue > 0) {
        countdownLabel->setText(QString::number(countdownValue));
    } else {
        countdownTimer->stop();
        countdownLabel->hide();

        if (m_currentCaptureMode == ImageCaptureMode) {
            performImageCapture();

            // Reset capture button for next capture (hand detection stays disabled until button press)
            ui->capture->setEnabled(true);
            qDebug() << "🎯 Capture completed - hand detection disabled until recapture button is pressed";
        } else if (m_currentCaptureMode == VideoRecordMode) {
            startRecording();
        }
    }
}

void Capture::updateRecordTimer()
{
    m_recordedSeconds++;

    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        qDebug() << "🎯 RECORDING COMPLETE: Reached video template duration ("
                 << m_currentVideoTemplate.durationSeconds << " seconds)";
        stopRecording();
    } else {
        // Show progress every 2 seconds or when near completion
        if (m_recordedSeconds % 2 == 0 ||
            m_recordedSeconds >= m_currentVideoTemplate.durationSeconds - 2) {
            qDebug() << "🎯 RECORDING PROGRESS:" << m_recordedSeconds << "/"
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
    // Create debug widget
    debugWidget = new QWidget(this);
    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 5px; }");

    QVBoxLayout *debugLayout = new QVBoxLayout(debugWidget);

    // Debug info label
    debugLabel = new QLabel("Initializing...", debugWidget);
    debugLabel->setStyleSheet("QLabel { color: white; font-size: 12px; font-weight: bold; }");
    debugLayout->addWidget(debugLabel);

    // FPS label
    fpsLabel = new QLabel("FPS: 0", debugWidget);
    fpsLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; }");
    debugLayout->addWidget(fpsLabel);

    // GPU Status label
    gpuStatusLabel = new QLabel("GPU: Checking...", debugWidget);
    gpuStatusLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 12px; }");
    debugLayout->addWidget(gpuStatusLabel);

    // CUDA Status label
    cudaStatusLabel = new QLabel("CUDA: Checking...", debugWidget);
    cudaStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
    debugLayout->addWidget(cudaStatusLabel);

    // Unified Detection label
    personDetectionLabel = new QLabel("Unified Detection: OFF", debugWidget);
    personDetectionLabel->setStyleSheet("QLabel { color: #ffaa00; font-size: 12px; }");
    debugLayout->addWidget(personDetectionLabel);

    // Unified Detection button
    personDetectionButton = new QPushButton("Enable Unified Detection", debugWidget);
    personDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #388e3c; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    connect(personDetectionButton, &QPushButton::clicked, this, &Capture::togglePersonDetection);
    debugLayout->addWidget(personDetectionButton);

    // Unified Detection Status label
    personSegmentationLabel = new QLabel("Detection & Segmentation: OFF", debugWidget);
    personSegmentationLabel->setStyleSheet("QLabel { color: #ff8800; font-size: 12px; }");
    debugLayout->addWidget(personSegmentationLabel);

    // Unified Detection Status button
    personSegmentationButton = new QPushButton("Toggle Unified Detection", debugWidget);
    personSegmentationButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #1976d2; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    connect(personSegmentationButton, &QPushButton::clicked, this, &Capture::togglePersonDetection);
    debugLayout->addWidget(personSegmentationButton);

    // handDetectionLabel = new QLabel("Hand Detection: OFF", debugWidget);
    // handDetectionLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 12px; }");
    // debugLayout->addWidget(handDetectionLabel);

    // handDetectionButton = new QPushButton("Disable Hand Detection", debugWidget);
    // handDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    // connect(handDetectionButton, &QPushButton::clicked, this, &Capture::toggleHandDetection);
    // debugLayout->addWidget(handDetectionButton);

    // Performance tips label
    QLabel *tipsLabel = new QLabel("Press 'S' to toggle detection\nPress 'G' to toggle segmentation/rectangles\nPress 'D' to hide/show\nPress 'P' for stats", debugWidget);
    tipsLabel->setStyleSheet("QLabel { color: #cccccc; font-size: 10px; font-style: italic; }");
    debugLayout->addWidget(tipsLabel);

    // Add debug widget to the main widget instead of videoLabel's layout
    debugWidget->setParent(this);
    debugWidget->move(10, 10); // Position in top-left corner
    debugWidget->resize(280, 350); // Larger size for better visibility
    debugWidget->raise(); // Ensure it's on top
    debugWidget->setVisible(true); // Make sure it's visible

    debugWidget->show(); // Show debug widget so user can enable segmentation and hand detection

    qDebug() << "Debug display setup complete - FPS, GPU, and CUDA status should be visible";
}

void Capture::enableHandDetectionForCapture()
{
    enableHandDetection(true);
    enableSegmentationInCapture();
}

void Capture::setCaptureReady(bool ready)
{
    m_captureReady = ready;
    qDebug() << "🎯 Capture ready state set to:" << ready;
}

bool Capture::isCaptureReady() const
{
    return m_captureReady;
}

void Capture::resetCapturePage()
{
    qDebug() << "🔄 COMPLETE CAPTURE PAGE RESET";

    // Reset all timers and countdown
    if (countdownTimer) {
        countdownTimer->stop();
        qDebug() << "⏱️ Countdown timer stopped";
    }
    if (recordTimer) {
        recordTimer->stop();
        qDebug() << "⏱️ Record timer stopped";
    }
    if (recordingFrameTimer) {
        recordingFrameTimer->stop();
        qDebug() << "⏱️ Recording frame timer stopped";
    }

    // Hide countdown label
    if (countdownLabel) {
        countdownLabel->hide();
        qDebug() << "📺 Countdown label hidden";
    }

    // Reset capture button
    ui->capture->setEnabled(true);
    qDebug() << "🔘 Capture button reset to enabled";

    // Reset hand detection completely
    m_handDetectionEnabled = true;
    m_captureReady = true;
    if (m_handDetector) {
        m_handDetector->resetGestureState();
        qDebug() << "🖐️ Hand detection completely reset";
    }

    // Reset segmentation state for capture interface
    enableSegmentationInCapture();
    qDebug() << "🎯 Segmentation reset for capture interface";

    // Reset all detection state
    m_lastHandDetections.clear();
    m_handDetectionFPS = 0.0;
    m_lastHandDetectionTime = 0.0;

    // Reset capture mode
    m_currentCaptureMode = ImageCaptureMode;

    // Reset video recording state
    m_recordedFrames.clear();
    m_recordedSeconds = 0;

    // Reset dynamic video background to start from beginning
    if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
        resetDynamicVideoToStart();
        qDebug() << "🎞️ Dynamic video reset to start for re-recording";
    }

    qDebug() << "✅ Capture page completely reset - all state cleared";
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

    // Center the status overlay when window is resized
    if (statusOverlay && statusOverlay->isVisible()) {
        int x = (width() - statusOverlay->width()) / 2;
        int y = (height() - statusOverlay->height()) / 2;
        statusOverlay->move(x, y);
    }





    // Ensure debug widget is visible and properly positioned
    if (debugWidget) {
        debugWidget->move(10, 10);
        debugWidget->raise();
        debugWidget->setVisible(true);
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
        case Qt::Key_L:
            // Toggle lighting correction
            setLightingCorrectionEnabled(!isLightingCorrectionEnabled());
            qDebug() << "🌟 Lighting correction toggled:" << (isLightingCorrectionEnabled() ? "ON" : "OFF");
            qDebug() << "🌟 Current display mode:" << m_displayMode;
            qDebug() << "🌟 Background template enabled:" << m_useBackgroundTemplate;
            qDebug() << "🌟 Template path:" << m_selectedBackgroundTemplate;
            break;
        case Qt::Key_S:
            // Only allow segmentation toggle if enabled in capture interface
            if (m_segmentationEnabledInCapture) {
                // Three-way toggle: Normal -> Rectangles -> Segmentation -> Normal
                switch (m_displayMode) {
                    case NormalMode:
                        setSegmentationMode(1); // RectangleMode
                        qDebug() << "Switched to RECTANGLE MODE (Original frame + Green rectangles)";
                        break;
                    case RectangleMode:
                        setSegmentationMode(2); // SegmentationMode
                        qDebug() << "Switched to SEGMENTATION MODE (Black background + Edge-based silhouettes)";
                        break;
                    case SegmentationMode:
                        setSegmentationMode(0); // NormalMode
                        qDebug() << "Switched to NORMAL MODE (Original camera view)";
                        break;
                }
            } else {
                qDebug() << "🎯 Segmentation toggle ignored - not enabled in capture interface";
                // Show status overlay to inform user
                if (statusOverlay) {
                    statusOverlay->setText("SEGMENTATION: DISABLED (Only available in capture interface)");
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
            }

            // Force immediate update
            qDebug() << "🎯 Current display mode:" << m_displayMode << "(0=Normal, 1=Rectangle, 2=Segmentation)";

            // Reset utilization when switching to normal mode
            if (m_displayMode == NormalMode) {
                m_gpuUtilized = false;
                m_cudaUtilized = false;
            }

            // Show prominent status overlay
            if (statusOverlay) {
                QString statusText;
                switch (m_displayMode) {
                    case NormalMode:
                        statusText = "NORMAL CAMERA VIEW: ENABLED";
                        break;
                    case RectangleMode:
                        statusText = "ORIGINAL FRAME + GREEN RECTANGLES: ENABLED";
                        break;
                    case SegmentationMode:
                        statusText = "BLACK BACKGROUND + EDGE-BASED SILHOUETTES: ENABLED";
                        break;
                }

                // Add segmentation state to status
                if (m_segmentationEnabledInCapture) {
                    statusText += " (Capture Interface)";
                } else {
                    statusText += " (Disabled - Capture Only)";
                }

                statusOverlay->setText(statusText);

                // Center the overlay
                statusOverlay->resize(statusOverlay->sizeHint());
                int x = (width() - statusOverlay->width()) / 2;
                int y = (height() - statusOverlay->height()) / 2;
                statusOverlay->move(x, y);

                statusOverlay->show();
                statusOverlay->raise();

                // Auto-hide after 2 seconds
                QTimer::singleShot(2000, [this]() {
                    if (statusOverlay) {
                        statusOverlay->hide();
                    }
                });
            }
            break;


            // Update button states
            updatePersonDetectionButton();


            // Force update debug display immediately
            updateDebugDisplay();

            // Show debug widget prominently when toggling
            if (debugWidget) {
                debugWidget->show();
                debugWidget->raise();
                debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.95); color: white; border-radius: 10px; border: 3px solid #00ff00; padding: 5px; }");

                // Show comprehensive status message
                if (debugLabel) {
                    QString status;
                    switch (m_displayMode) {
                        case NormalMode:
                            status = "NORMAL VIEW: ENABLED";
                            break;
                        case RectangleMode:
                            status = "ORIGINAL + GREEN RECTANGLES: ENABLED";
                            break;
                        case SegmentationMode:
                            status = "BLACK BG + EDGE SILHOUETTES: ENABLED";
                            break;
                    }
                    debugLabel->setText(status);
                    debugLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 16px; font-weight: bold; }");
                }

                // Make FPS label more prominent
                if (fpsLabel) {
                    fpsLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 14px; font-weight: bold; }");
                }

                // Make GPU status more prominent
                if (gpuStatusLabel) {
                    gpuStatusLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 14px; font-weight: bold; }");
                }

                // Make CUDA status more prominent
                if (cudaStatusLabel) {
                    cudaStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 14px; font-weight: bold; }");
                }

                // Make person detection label more prominent
                if (personDetectionLabel) {
                    personDetectionLabel->setStyleSheet("QLabel { color: #ffaa00; font-size: 14px; font-weight: bold; }");
                }

                // Make person segmentation label more prominent
                if (personSegmentationLabel) {
                    personSegmentationLabel->setStyleSheet("QLabel { color: #ff8800; font-size: 14px; font-weight: bold; }");
                }

                // Auto-hide the enhanced styling after 5 seconds (longer for better visibility)
                QTimer::singleShot(5000, [this]() {
                    if (debugWidget) {
                        debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 5px; }");
                        if (debugLabel) {
                            debugLabel->setStyleSheet("QLabel { color: white; font-size: 12px; font-weight: bold; }");
                        }
                        if (fpsLabel) {
                            fpsLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; }");
                        }
                        if (gpuStatusLabel) {
                            gpuStatusLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 12px; }");
                        }
                        if (cudaStatusLabel) {
                            cudaStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
                        }
                        if (personDetectionLabel) {
                            personDetectionLabel->setStyleSheet("QLabel { color: #ffaa00; font-size: 12px; }");
                        }
                        if (personSegmentationLabel) {
                            personSegmentationLabel->setStyleSheet("QLabel { color: #ff8800; font-size: 12px; }");
                        }
                    }
                });
            }

            // Show prominent status overlay
            if (statusOverlay) {
                QString statusText = ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ?
                    "PERSON DETECTION: ENABLED" :
                    "PERSON DETECTION: DISABLED");
                statusOverlay->setText(statusText);

                // Center the overlay
                statusOverlay->resize(statusOverlay->sizeHint());
                int x = (width() - statusOverlay->width()) / 2;
                int y = (height() - statusOverlay->height()) / 2;
                statusOverlay->move(x, y);

                statusOverlay->show();
                statusOverlay->raise();

                // Auto-hide after 2 seconds
                QTimer::singleShot(2000, [this]() {
                    if (statusOverlay) {
                        statusOverlay->hide();
                    }
                });
            }
            break;
        case Qt::Key_H:
            // Temporarily disabled hand detection toggle
            qDebug() << "Hand detection toggle disabled";
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
    // Just enable hand detection and segmentation after a short delay
    QTimer::singleShot(100, [this]() {
        m_handDetectionEnabled = true;
        if (m_handDetector) {
            m_handDetector->resetGestureState();
        }
        enableSegmentationInCapture();
        qDebug() << "🖐️ Hand detection ENABLED - close your hand to trigger capture automatically";
        qDebug() << "🎯 Segmentation ENABLED for capture interface";

        // Restore dynamic video background if a path was previously set
        if (!m_dynamicVideoPath.isEmpty() && !m_useDynamicVideoBackground) {
            qDebug() << "🎞️ Restoring dynamic video background:" << m_dynamicVideoPath;
            enableDynamicVideoBackground(m_dynamicVideoPath);
        }
    });
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    qDebug() << "Capture widget hidden - OPTIMIZED camera and hand detection shutdown";

    // Disable hand detection when page is hidden (but keep camera running for faster return)
    enableHandDetection(false);
    qDebug() << "🖐️ Hand detection DISABLED";

    // Disable segmentation when leaving capture page
    disableSegmentationOutsideCapture();
    qDebug() << "🎯 Segmentation DISABLED outside capture interface";

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
                qDebug() << "🎯 HAND CLOSED DETECTED! Automatically triggering capture...";
                qDebug() << "🎯 Current display mode:" << m_displayMode << "(0=Normal, 1=Rectangle, 2=Segmentation)";

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
    if (updateCount % 10 == 0) { // Log every 5 seconds (10 updates * 500ms)
        qDebug() << "Debug display update #" << updateCount << "FPS:" << m_currentFPS << "GPU:" << m_useGPU << "CUDA:" << m_useCUDA;
    }

    if (debugLabel) {
        QString peopleDetected = QString::number(m_lastDetections.size());
        QString modeText;
        QString segmentationStatus = m_segmentationEnabledInCapture ? "ENABLED" : "DISABLED";

        QString backgroundStatus = m_useBackgroundTemplate ? "TEMPLATE" : "BLACK";

        switch (m_displayMode) {
            case NormalMode:
                modeText = "NORMAL VIEW";
                break;
            case RectangleMode:
                modeText = "ORIGINAL + RECTANGLES";
                break;
            case SegmentationMode:
                modeText = QString("SEGMENTATION (%1 BG)").arg(backgroundStatus);
                break;
        }
        QString lightingStatus = isLightingCorrectionEnabled() ? 
            (isGPULightingAvailable() ? "GPU" : "CPU") : "OFF";
        QString debugInfo = QString("FPS: %1 | %2 | People: %3 | Segmentation: %4 | BG: %5 | Lighting: %6")
                           .arg(m_currentFPS)
                           .arg(modeText)
                           .arg(peopleDetected)
                           .arg(segmentationStatus)
                           .arg(backgroundStatus)
                           .arg(lightingStatus);
        debugLabel->setText(debugInfo);
    }

    if (fpsLabel) {
        fpsLabel->setText(QString("FPS: %1").arg(m_currentFPS));
    }

    if (gpuStatusLabel) {
        QString gpuStatus;
        if (m_gpuUtilized) {
            gpuStatus = "ACTIVE (OpenCL)";
        } else if (m_useGPU) {
            gpuStatus = "AVAILABLE (OpenCL)";
        } else {
            gpuStatus = "OFF (CPU)";
        }
        gpuStatusLabel->setText(QString("GPU: %1").arg(gpuStatus));

        // Change color based on utilization
        if (m_gpuUtilized) {
            gpuStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; font-weight: bold; }");
        } else if (m_useGPU) {
            gpuStatusLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 12px; }");
        } else {
            gpuStatusLabel->setStyleSheet("QLabel { color: #ff6666; font-size: 12px; }");
        }
    }

    if (handDetectionLabel) {
        QString handStatus = m_showHandDetection ? "ON (Tracking)" : "OFF";
        QString handTime = QString::number(m_lastHandDetectionTime * 1000, 'f', 1);
        QString detectorType = m_handDetector ? m_handDetector->getDetectorType() : "Unknown";
        QString avgTime = m_handDetector ? QString::number(m_handDetector->getAverageProcessingTime(), 'f', 1) : "0.0";
        handDetectionLabel->setText(QString("Hand Detection: %1 (%2ms) [%3] Avg: %4ms").arg(handStatus).arg(handTime).arg(detectorType).arg(avgTime));
    }

    if (cudaStatusLabel) {
        QString cudaStatus;
        if (m_cudaUtilized) {
            cudaStatus = "ACTIVE (CUDA GPU)";
        } else if (m_useCUDA) {
            cudaStatus = "AVAILABLE (CUDA)";
        } else {
            cudaStatus = "OFF (CPU)";
        }
        cudaStatusLabel->setText(QString("CUDA: %1").arg(cudaStatus));

        // Change color based on utilization
        if (m_cudaUtilized) {
            cudaStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; font-weight: bold; }");
        } else if (m_useCUDA) {
            cudaStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
        } else {
            cudaStatusLabel->setStyleSheet("QLabel { color: #ff6666; font-size: 12px; }");
        }
    }

    if (personDetectionLabel) {
        QString personStatus = ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? "ON" : "OFF");
        QString personTime = QString::number(m_lastPersonDetectionTime * 1000, 'f', 1);
        QString personFPS = QString::number(m_personDetectionFPS, 'f', 1);
        QString peopleCount = QString::number(m_lastDetections.size());
        personDetectionLabel->setText(QString("Unified Detection: %1 (%2ms, %3 FPS, %4 people)").arg(personStatus).arg(personTime).arg(personFPS).arg(peopleCount));
    }

    if (personSegmentationLabel) {
        QString segStatus = ((m_displayMode == RectangleMode || m_displayMode == SegmentationMode) ? "ON" : "OFF");
        QString segTime = QString::number(m_lastPersonDetectionTime * 1000, 'f', 1);
        QString segFPS = QString::number(m_personDetectionFPS, 'f', 1);
        QString peopleCount = QString::number(m_lastDetections.size());
        personSegmentationLabel->setText(QString("Detection & Segmentation: %1 (%2ms, %3 FPS, %4 people)").arg(segStatus).arg(segTime).arg(segFPS).arg(peopleCount));
    }

}

void Capture::startRecording()
{
    if (!cameraWorker->isCameraOpen()) {
        qWarning() << "Cannot start recording: Camera not opened by worker.";
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;

    // Use the original camera FPS for recording to maintain natural timing
    m_adjustedRecordingFPS = m_actualCameraFPS;

    qDebug() << "🚀 DIRECT CAPTURE RECORDING: Starting with FPS:" << m_adjustedRecordingFPS;
    qDebug() << "  - Scale factor:" << m_personScaleFactor;
    qDebug() << "  - Capturing exact display content";
    qDebug() << "  - Recording duration:" << m_currentVideoTemplate.durationSeconds << "seconds";
    qDebug() << "  - Video template:" << m_currentVideoTemplate.name;

    int frameIntervalMs = qMax(1, static_cast<int>(1000.0 / m_adjustedRecordingFPS));

    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
    qDebug() << "🚀 DIRECT CAPTURE RECORDING: Started at " + QString::number(m_adjustedRecordingFPS)
                    + " frames/sec (interval: " + QString::number(frameIntervalMs) + "ms)";

    // Pre-calculate label size for better performance during recording
    m_cachedLabelSize = ui->videoLabel->size();

    // Reset dynamic video to start when recording begins
    if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
        resetDynamicVideoToStart();
        qDebug() << "🎞️ Dynamic video reset to start for new recording";
    }
}

void Capture::stopRecording()
{
    if (!m_isRecording)
        return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;

    qDebug() << "🚀 DIRECT CAPTURE RECORDING: Stopped. Captured " + QString::number(m_recordedFrames.size())
                    + " frames.";

    if (!m_recordedFrames.isEmpty()) {
        qDebug() << "🚀 DIRECT CAPTURE RECORDING: Emitting video with FPS:" << m_adjustedRecordingFPS << "(base:" << m_actualCameraFPS << ")";

        // Emit the adjusted FPS to ensure playback matches the recording rate
        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
    }

    // Re-enable capture button for re-recording
    ui->capture->setEnabled(true);

    emit showFinalOutputPage();
    qDebug() << "🚀 DIRECT CAPTURE RECORDING: Stopped - capture button re-enabled for re-recording";
}

void Capture::performImageCapture()
{
    // Capture the processed frame that includes background template and segmentation
    if (!m_originalCameraImage.isNull()) {
        QPixmap cameraPixmap;
        QSize labelSize = ui->videoLabel->size();

        // Check if we have a processed segmented frame to capture
        if ((m_displayMode == SegmentationMode || m_displayMode == RectangleMode) && !m_lastSegmentedFrame.empty()) {
            // Store original segmented frame for comparison
            cv::Mat originalSegmentedFrame = m_lastSegmentedFrame.clone();
            
            // Apply person-only lighting correction using template reference
            cv::Mat lightingCorrectedFrame;
            qDebug() << "🌟 LIGHTING DEBUG - Segmentation mode detected";
            qDebug() << "🌟 LIGHTING DEBUG - Lighting enabled:" << isLightingCorrectionEnabled();
            qDebug() << "🌟 LIGHTING DEBUG - Background template enabled:" << m_useBackgroundTemplate;
            qDebug() << "🌟 LIGHTING DEBUG - Template path:" << m_selectedBackgroundTemplate;
            qDebug() << "🌟 LIGHTING DEBUG - Lighting corrector exists:" << (m_lightingCorrector != nullptr);
            
            // POST-PROCESSING: Apply lighting to raw person data and re-composite
            qDebug() << "🎯 POST-PROCESSING: Apply lighting to raw person data";
            lightingCorrectedFrame = applyPostProcessingLighting();
            qDebug() << "🎯 Post-processing lighting applied";
            
            // Store both versions for saving
            m_originalCapturedImage = originalSegmentedFrame;
            m_lightingCorrectedImage = lightingCorrectedFrame;
            m_hasLightingComparison = true;
            
            qDebug() << "🔥 FORCED: Stored both original and lighting-corrected versions for comparison";
            
            // Convert the processed OpenCV frame to QImage for capture
            QImage processedImage = cvMatToQImage(lightingCorrectedFrame);
            cameraPixmap = QPixmap::fromImage(processedImage);
            qDebug() << "🎯 Capturing processed segmented frame with background template and person lighting correction";
        } else {
            // For normal mode, apply global lighting correction if enabled
            cv::Mat originalFrame = qImageToCvMat(m_originalCameraImage);
            cv::Mat lightingCorrectedFrame;
            qDebug() << "🌟 LIGHTING DEBUG - Normal mode detected";
            qDebug() << "🌟 LIGHTING DEBUG - Lighting enabled:" << isLightingCorrectionEnabled();
            qDebug() << "🌟 LIGHTING DEBUG - Lighting corrector exists:" << (m_lightingCorrector != nullptr);
            
            if (isLightingCorrectionEnabled() && m_lightingCorrector) {
                lightingCorrectedFrame = m_lightingCorrector->applyGlobalLightingCorrection(originalFrame);
                qDebug() << "🎯 Applied global lighting correction (normal mode)";
            } else {
                lightingCorrectedFrame = originalFrame;
                qDebug() << "🎯 No lighting correction applied (normal mode)";
            }
            
            // Convert back to QImage
            QImage correctedImage = cvMatToQImage(lightingCorrectedFrame);
            cameraPixmap = QPixmap::fromImage(correctedImage);
            qDebug() << "🎯 Capturing original camera frame with lighting correction (normal mode)";
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
            if (m_displayMode == SegmentationMode && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode or dynamic video mode, don't scale the entire frame
                // Person scaling is already applied in createSegmentedFrame
                qDebug() << "🎯 Person-only scaling preserved in final output (background template or dynamic video mode)";
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

                qDebug() << "🎯 Frame scaled in final output to" << newWidth << "x" << newHeight
                         << "with factor" << m_personScaleFactor;
            }
        }

        m_capturedImage = scaledPixmap;
        
        // Emit appropriate signal based on whether we have comparison images
        if (m_hasLightingComparison && !m_originalCapturedImage.empty()) {
            // Convert original image to QPixmap for comparison
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
                if (m_displayMode == SegmentationMode && ((m_useBackgroundTemplate &&
                    !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                    qDebug() << "🎯 Person-only scaling preserved in original output";
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
            
            emit imageCapturedWithComparison(m_capturedImage, scaledOriginalPixmap);
            qDebug() << "🎯 Emitted comparison images - corrected and original versions";
        } else {
        emit imageCaptured(m_capturedImage);
            qDebug() << "🎯 Emitted single image (no lighting comparison)";
        }
        
        qDebug() << "Image captured (includes background template and segmentation).";
        qDebug() << "Captured image size:" << m_capturedImage.size() << "Original size:" << cameraPixmap.size();
    } else {
        qWarning() << "Failed to capture image: original camera image is empty.";
        QMessageBox::warning(this, "Capture Failed", "No camera feed available to capture an image.");
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

void Capture::setCaptureMode(CaptureMode mode)
{
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to:" << static_cast<int>(mode);
}

void Capture::setVideoTemplate(const VideoTemplate &templateData)
{
    m_currentVideoTemplate = templateData;
    qDebug() << "🎯 VIDEO TEMPLATE SET:" << templateData.name;
    qDebug() << "  - Duration:" << templateData.durationSeconds << "seconds";
    qDebug() << "  - Recording will automatically stop after" << templateData.durationSeconds << "seconds";

    // Reset frame counter to ensure smooth initial processing
    frameCount = 0;

    // Ensure we start in normal mode to prevent freezing
    if (m_displayMode != NormalMode) {
        m_displayMode = NormalMode;
        qDebug() << "Switched to normal mode to prevent freezing during template transition";
    }
}

void Capture::enableDynamicVideoBackground(const QString &videoPath)
{
    // Close previous if open
    if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.release();
    }
    if (!m_dynamicGpuReader.empty()) {
        m_dynamicGpuReader.release();
    }
    m_dynamicVideoPath = videoPath;
    m_useDynamicVideoBackground = false;

    bool opened = false;
    // Try GPU reader first (NVDEC)
    try {
        m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
        opened = !m_dynamicGpuReader.empty();
    } catch (const cv::Exception &e) {
        qWarning() << "GPU video reader unavailable:" << e.what();
        m_dynamicGpuReader.release();
    }
    if (!opened) {
        // Open using FFMPEG backend; if system has NVDEC via ffmpeg, it may use it
        m_dynamicVideoCap.open(m_dynamicVideoPath.toStdString(), cv::CAP_FFMPEG);
        opened = m_dynamicVideoCap.isOpened();
    }
    if (!opened) {
        qWarning() << "Failed to open dynamic video background:" << m_dynamicVideoPath;
        return;
    }

    // 🎯 AUTOMATIC DURATION DETECTION: Get video duration and update template
    double videoDurationSeconds = 0.0;
    if (m_dynamicVideoCap.isOpened()) {
        // Get total frame count and FPS to calculate duration
        double totalFrames = m_dynamicVideoCap.get(cv::CAP_PROP_FRAME_COUNT);
        m_videoFrameRate = m_dynamicVideoCap.get(cv::CAP_PROP_FPS);
        if (m_videoFrameRate > 0 && totalFrames > 0) {
            videoDurationSeconds = totalFrames / m_videoFrameRate;
            qDebug() << "🎯 VIDEO DURATION DETECTED:" << videoDurationSeconds << "seconds";
            qDebug() << "  - Total frames:" << totalFrames;
            qDebug() << "  - Frame rate:" << m_videoFrameRate << "FPS";
        }
    }

    // Update video template with detected duration
    if (videoDurationSeconds > 0) {
        QString templateName = QFileInfo(m_dynamicVideoPath).baseName();
        m_currentVideoTemplate = VideoTemplate(templateName, static_cast<int>(videoDurationSeconds));
        qDebug() << "🎯 RECORDING DURATION UPDATED:" << m_currentVideoTemplate.durationSeconds << "seconds";
        qDebug() << "  - Template name:" << m_currentVideoTemplate.name;
        qDebug() << "  - Recording will automatically stop when video template ends";
    } else {
        // Fallback to default duration if detection fails
        m_currentVideoTemplate = VideoTemplate("Dynamic Template", 10);
        qWarning() << "🎯 Could not detect video duration, using default 10 seconds";
    }

    // Phase 1: Detect video frame rate for synchronization
    if (m_dynamicVideoCap.isOpened()) {
        if (m_videoFrameRate <= 0) {
            m_videoFrameRate = 30.0; // Fallback to 30 FPS if detection fails
        }
        m_videoFrameInterval = qRound(1000.0 / m_videoFrameRate); // Convert to milliseconds
        qDebug() << "🎞️ Video frame rate detected:" << m_videoFrameRate << "FPS, interval:" << m_videoFrameInterval << "ms";
    } else if (!m_dynamicGpuReader.empty()) {
        // For GPU reader, use default frame rate (GPU readers don't always expose FPS)
        m_videoFrameRate = 30.0;
        m_videoFrameInterval = 33;
        qDebug() << "🎞️ Using default frame rate for GPU video reader:" << m_videoFrameRate << "FPS";
    }

    // Prime first frame
    cv::Mat first;
    if (!m_dynamicGpuReader.empty()) {
        cv::cuda::GpuMat gpu;
        if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
            cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
            gpu.download(first);
        }
    } else if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.read(first);
    }
    if (!first.empty()) {
        m_dynamicVideoFrame = first.clone();
        m_useDynamicVideoBackground = true;
        qDebug() << "🎞️ Dynamic video background enabled:" << m_dynamicVideoPath;

        // Phase 1: Start video playback timer for frame rate synchronization
        if (m_videoPlaybackTimer) {
            m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
            m_videoPlaybackTimer->start();
            m_videoPlaybackActive = true;
            qDebug() << "🎞️ Video playback timer started with interval:" << m_videoFrameInterval << "ms";
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
        qDebug() << "🎞️ Dynamic template enabled - foreground template cleared to prevent visibility in final output";
    }
}

void Capture::disableDynamicVideoBackground()
{
    // Phase 1: Stop video playback timer
    if (m_videoPlaybackTimer && m_videoPlaybackActive) {
        m_videoPlaybackTimer->stop();
        m_videoPlaybackActive = false;
        qDebug() << "🎞️ Video playback timer stopped";
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
    qDebug() << "🧹 Cleared dynamic video path for mode switching";
}

// Phase 1: Video Playback Timer Slot - Advances video frames at native frame rate
void Capture::onVideoPlaybackTimer()
{
    if (!m_useDynamicVideoBackground || !m_videoPlaybackActive) {
        return;
    }

    // Advance to next video frame at native frame rate
    cv::Mat nextFrame;
    bool frameRead = false;

    if (!m_dynamicGpuReader.empty()) {
        cv::cuda::GpuMat gpu;
        if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
            cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
            gpu.download(nextFrame);
            frameRead = true;
        } else {
            // Loop video - recreate reader
            try {
                m_dynamicGpuReader.release();
                m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
                if (m_dynamicGpuReader->nextFrame(gpu) && !gpu.empty()) {
                    cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                    gpu.download(nextFrame);
                    frameRead = true;
                }
            } catch (...) {
                qWarning() << "Failed to loop GPU video reader";
            }
        }
    } else if (m_dynamicVideoCap.isOpened()) {
        if (m_dynamicVideoCap.read(nextFrame) && !nextFrame.empty()) {
            frameRead = true;
        } else {
            // Loop video
            m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (m_dynamicVideoCap.read(nextFrame) && !nextFrame.empty()) {
                frameRead = true;
            }
        }
    }

    if (frameRead && !nextFrame.empty()) {
        // Store the new frame for use in segmentation
        m_dynamicVideoFrame = nextFrame.clone();
        qDebug() << "🎞️ Video frame advanced at native frame rate:" << m_videoFrameRate << "FPS";
    } else {
        qWarning() << "Failed to read next video frame in timer";
    }
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
            qDebug() << "🎞️ GPU video reader reset to start";
        } catch (...) {
            qWarning() << "Failed to reset GPU video reader";
        }
    } else if (m_dynamicVideoCap.isOpened()) {
        m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
        qDebug() << "🎞️ CPU video reader reset to start";
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
        qDebug() << "🎞️ Video reset to first frame for re-recording";
    }

    // Restart the video playback timer
    if (m_videoPlaybackTimer) {
        m_videoPlaybackTimer->setInterval(m_videoFrameInterval);
        m_videoPlaybackTimer->start();
        m_videoPlaybackActive = true;
        qDebug() << "🎞️ Video playback timer restarted after reset";
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

                qDebug() << "🎮 Phase 2A: GPU-only processing pipeline initialized successfully";
                qDebug() << "🎮 GPU memory available for video processing";
            }
        } catch (const cv::Exception& e) {
            qWarning() << "GPU-only processing initialization failed:" << e.what();
            m_gpuProcessingAvailable = false;
            m_gpuOnlyProcessingEnabled = false;
        }
    }

    if (!m_gpuProcessingAvailable) {
        qDebug() << "🎮 Phase 2A: GPU-only processing not available, using CPU fallback";
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

    m_personDetectionTimer.start();

    try {
        qDebug() << "🎮 Phase 2A: Using GPU-only processing pipeline";

        // Upload frame to GPU (single transfer)
        m_gpuVideoFrame.upload(frame);

        // Optimized processing for 30 FPS with GPU
        cv::cuda::GpuMat processFrame = m_gpuVideoFrame;
        if (frame.cols > 640) {
            double scale = 640.0 / frame.cols;
            cv::cuda::resize(m_gpuVideoFrame, processFrame, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }

        // Download for person detection (still need CPU for HOG)
        cv::Mat cpuProcessFrame;
        processFrame.download(cpuProcessFrame);

        // Detect people using enhanced detection (CPU-based HOG)
        std::vector<cv::Rect> found = detectPeople(cpuProcessFrame);

        // Scale results back if we resized the frame
        if (processFrame.cols != frame.cols) {
            double scale = (double)frame.cols / processFrame.cols;
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale);
                rect.y = cvRound(rect.y * scale);
                rect.width = cvRound(rect.width * scale);
                rect.height = cvRound(rect.height * scale);
            }
        }

        // Get motion mask for filtering (GPU-based)
        cv::Mat motionMask = getMotionMask(frame);

        // Filter detections by motion
        std::vector<cv::Rect> motionFiltered = filterByMotion(found, motionMask);

        // Store detections for UI display
        m_lastDetections = motionFiltered;

        // Create segmented frame with GPU-only processing
        cv::Mat segmentedFrame = createSegmentedFrameGPUOnly(frame, motionFiltered);

        // Update timing info
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;

        qDebug() << "🎮 Phase 2A: GPU-only processing completed successfully";

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
    qDebug() << "🎮 ===== initializePersonDetection() CALLED =====";
    qDebug() << "Initializing Enhanced Person Detection and Segmentation...";

    // Initialize HOG detectors for person detection
    qDebug() << "🎮 ===== CAPTURE INITIALIZATION STARTED =====";
    m_hogDetector.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    m_hogDetectorDaimler.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());

    // Initialize CUDA HOG detector for GPU acceleration
    qDebug() << "🎮 ===== STARTING CUDA HOG INITIALIZATION =====";

    // Check if CUDA is available
    int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
    qDebug() << "🎮 CUDA devices found:" << cudaDevices;

    if (cudaDevices > 0) {
        try {
            qDebug() << "🎮 Creating CUDA HOG detector...";
            // Create CUDA HOG with default people detector
            m_cudaHogDetector = cv::cuda::HOG::create(
                cv::Size(64, 128),  // win_size
                cv::Size(16, 16),   // block_size
                cv::Size(8, 8),     // block_stride
                cv::Size(8, 8),     // cell_size
                9                   // nbins
            );

            if (!m_cudaHogDetector.empty()) {
                qDebug() << "🎮 CUDA HOG detector created successfully";
                m_cudaHogDetector->setSVMDetector(m_cudaHogDetector->getDefaultPeopleDetector());
                qDebug() << "✅ CUDA HOG detector ready for GPU acceleration";
            } else {
                qWarning() << "⚠️ CUDA HOG creation failed - detector is empty";
                m_cudaHogDetector = nullptr;
            }
        } catch (const cv::Exception& e) {
            qWarning() << "⚠️ CUDA HOG initialization failed:" << e.what();
            m_cudaHogDetector = nullptr;
        }
    } else {
        qDebug() << "⚠️ CUDA not available for HOG initialization";
        m_cudaHogDetector = nullptr;
    }
    qDebug() << "🎮 ===== FINAL CUDA HOG INITIALIZATION CHECK =====";
    qDebug() << "🎮 CUDA HOG detector pointer:" << m_cudaHogDetector.get();
    qDebug() << "🎮 CUDA HOG detector empty:" << (m_cudaHogDetector && m_cudaHogDetector.empty() ? "yes" : "no");

    if (m_cudaHogDetector && !m_cudaHogDetector.empty()) {
        qDebug() << "✅ CUDA HOG detector successfully initialized and ready!";
        m_useCUDA = true; // Ensure CUDA is enabled
    } else {
        qWarning() << "⚠️ CUDA HOG detector initialization failed or not available";
        m_cudaHogDetector = nullptr;
    }
    qDebug() << "🎮 ===== CUDA HOG INITIALIZATION COMPLETE =====";

    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);

    // 🚀 Initialize GPU Memory Pool for optimized CUDA operations
    if (!m_gpuMemoryPoolInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            qDebug() << "🚀 Initializing GPU Memory Pool for optimized CUDA operations...";
            m_gpuMemoryPool.initialize(1280, 720); // Initialize with common camera resolution
            m_gpuMemoryPoolInitialized = true;
            qDebug() << "✅ GPU Memory Pool initialized successfully";
        } catch (const cv::Exception& e) {
            qWarning() << "🚀 GPU Memory Pool initialization failed:" << e.what();
            m_gpuMemoryPoolInitialized = false;
        }
    }

    // Check if CUDA is available for NVIDIA GPU acceleration (PRIORITY)
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            m_useCUDA = true;
            qDebug() << "🎮 CUDA GPU acceleration enabled for NVIDIA GPU (PRIORITY)";
            qDebug() << "CUDA devices found:" << cv::cuda::getCudaEnabledDeviceCount();

            // Get CUDA device info
            cv::cuda::DeviceInfo deviceInfo(0);
            if (deviceInfo.isCompatible()) {
                qDebug() << "CUDA Device:" << deviceInfo.name();
                qDebug() << "Memory:" << deviceInfo.totalMemory() / (1024*1024) << "MB";
                qDebug() << "Compute Capability:" << deviceInfo.majorVersion() << "." << deviceInfo.minorVersion();
                qDebug() << "CUDA will be used for color conversion and resizing operations";

                // Pre-allocate CUDA GPU memory pools for better performance
                qDebug() << "🎮 Pre-allocating CUDA GPU memory pools...";
                try {
                    // Pre-allocate common frame sizes for CUDA operations
                    cv::cuda::GpuMat cudaFramePool1, cudaFramePool2, cudaFramePool3;
                    cudaFramePool1.create(720, 1280, CV_8UC3);  // Common camera resolution
                    cudaFramePool2.create(480, 640, CV_8UC3);   // Smaller processing size
                    cudaFramePool3.create(360, 640, CV_8UC1);   // Grayscale processing

                    qDebug() << "✅ CUDA GPU memory pools pre-allocated successfully";
                    qDebug() << "  - CUDA Frame pool 1: 1280x720 (RGB)";
                    qDebug() << "  - CUDA Frame pool 2: 640x480 (RGB)";
                    qDebug() << "  - CUDA Frame pool 3: 640x360 (Grayscale)";

                    // Set CUDA device for optimal performance
                    cv::cuda::setDevice(0);
                    qDebug() << "CUDA device 0 set for optimal performance";

                } catch (const cv::Exception& e) {
                    qWarning() << "⚠️ CUDA GPU memory pool allocation failed:" << e.what();
                }
            }
        } else {
            qDebug() << "⚠️ CUDA not available, checking OpenCL";
            m_useCUDA = false;
        }
    } catch (...) {
        qDebug() << "⚠️ CUDA initialization failed, checking OpenCL";
        m_useCUDA = false;
    }

    // Check if OpenCL is available for HOG detection (ALWAYS ENABLE FOR HOG)
    try {
        if (cv::ocl::useOpenCL()) {
            m_useGPU = true;
            qDebug() << "🎮 OpenCL GPU acceleration enabled for HOG detection";
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
                qDebug() << "🎮 Pre-allocating GPU memory pools...";
                try {
                    // Pre-allocate common frame sizes for GPU operations
                    cv::UMat gpuFramePool1, gpuFramePool2, gpuFramePool3;
                    gpuFramePool1.create(720, 1280, CV_8UC3);  // Common camera resolution
                    gpuFramePool2.create(480, 640, CV_8UC3);   // Smaller processing size
                    gpuFramePool3.create(360, 640, CV_8UC1);   // Grayscale processing

                    qDebug() << "✅ GPU memory pools pre-allocated successfully";
                    qDebug() << "  - Frame pool 1: 1280x720 (RGB)";
                    qDebug() << "  - Frame pool 2: 640x480 (RGB)";
                    qDebug() << "  - Frame pool 3: 640x360 (Grayscale)";
                } catch (const cv::Exception& e) {
                    qWarning() << "⚠️ GPU memory pool allocation failed:" << e.what();
                }
            }

            // OpenCL device info not available in this OpenCV version
            qDebug() << "OpenCL GPU acceleration ready for HOG detection";

        } else {
            qDebug() << "⚠️ OpenCL not available for HOG, will use CPU";
            m_useGPU = false;
        }
    } catch (...) {
        qDebug() << "⚠️ OpenCL initialization failed for HOG, will use CPU";
        m_useGPU = false;
    }

    // Check if OpenCL is available for AMD GPU acceleration (FALLBACK)
    if (!m_useCUDA) {
        try {
            if (cv::ocl::useOpenCL()) {
                m_useGPU = true;
                qDebug() << "🎮 OpenCL GPU acceleration enabled for AMD GPU (fallback)";
                qDebug() << "Using UMat for GPU memory management";
            } else {
                qDebug() << "⚠️ OpenCL not available, using CPU";
                m_useGPU = false;
            }
        } catch (...) {
            qDebug() << "⚠️ OpenCL initialization failed, using CPU";
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

cv::Mat Capture::processFrameWithUnifiedDetection(const cv::Mat &frame)
{
    // Validate input frame
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
        qWarning() << "Invalid frame received, returning empty result";
        return cv::Mat::zeros(480, 640, CV_8UC3);
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

        // Detect people using enhanced detection
        std::vector<cv::Rect> found = detectPeople(processFrame);

        // Scale results back if we resized the frame
        if (processFrame.cols != frame.cols) {
            double scale = (double)frame.cols / processFrame.cols;
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale);
                rect.y = cvRound(rect.y * scale);
                rect.width = cvRound(rect.width * scale);
                rect.height = cvRound(rect.height * scale);
            }
        }

        // Get motion mask for filtering
        cv::Mat motionMask = getMotionMask(frame);

        // Filter detections by motion
        std::vector<cv::Rect> motionFiltered = filterByMotion(found, motionMask);

        // Store detections for UI display
        m_lastDetections = motionFiltered;

        // Create segmented frame with motion-filtered detections
        // qDebug() << "🎯 Creating segmented frame with" << motionFiltered.size() << "detections, display mode:" << m_displayMode;
        cv::Mat segmentedFrame = createSegmentedFrame(frame, motionFiltered);

        // Update timing info
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;

        // Log people detection for visibility (reduced frequency for performance)
        if (motionFiltered.size() > 0) {
            // qDebug() << "🎯 PEOPLE DETECTED:" << motionFiltered.size() << "person(s) in frame (motion filtered from" << found.size() << "detections)";
            // qDebug() << "🎯 Detection details:";
            // for (size_t i = 0; i < motionFiltered.size(); i++) {
            //     qDebug() << "🎯 Person" << i << "at" << motionFiltered[i].x << motionFiltered[i].y << motionFiltered[i].width << "x" << motionFiltered[i].height;
            // }
        } else {
            qDebug() << "⚠️ NO PEOPLE DETECTED in frame (total detections:" << found.size() << ")";

            // For testing: create a fake detection in the center of the frame if no detections found
            if (m_displayMode == SegmentationMode && frame.cols > 0 && frame.rows > 0) {
                qDebug() << "🎯 TESTING: Creating fake detection in center for segmentation testing";
                cv::Rect fakeDetection(frame.cols/4, frame.rows/4, frame.cols/2, frame.rows/2);
                motionFiltered.push_back(fakeDetection);
                qDebug() << "🎯 TESTING: Added fake detection at" << fakeDetection.x << fakeDetection.y << fakeDetection.width << "x" << fakeDetection.height;
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

    if (m_displayMode == SegmentationMode) {
        qDebug() << "🎯 SEGMENTATION MODE: Creating background + edge-based silhouettes";

        // Create background for edge-based segmentation
        cv::Mat segmentedFrame;

        // Use cached background template for performance
        static cv::Mat cachedBackgroundTemplate;
        static QString lastBackgroundPath;

        if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
            // Phase 1: Use pre-advanced video frame from timer instead of reading synchronously
            if (!m_dynamicVideoFrame.empty()) {
                cv::resize(m_dynamicVideoFrame, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                qDebug() << "🎞️ Using pre-advanced video frame for segmentation (Phase 1)";
            } else {
                // Fallback: read frame synchronously if timer hasn't advanced yet
                cv::Mat nextBg;
                if (!m_dynamicGpuReader.empty()) {
                    cv::cuda::GpuMat gpu;
                    if (!m_dynamicGpuReader->nextFrame(gpu) || gpu.empty()) {
                        // cudacodec doesn't expose random seek universally; recreate reader to loop
                        try {
                            m_dynamicGpuReader.release();
                            m_dynamicGpuReader = cv::cudacodec::createVideoReader(m_dynamicVideoPath.toStdString());
                            m_dynamicGpuReader->nextFrame(gpu);
                        } catch (...) {}
                    }
                    if (!gpu.empty()) {
                        cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGRA2BGR);
                        gpu.download(nextBg);
                    }
                }
                if (nextBg.empty() && m_dynamicVideoCap.isOpened()) {
                    if (!m_dynamicVideoCap.read(nextBg) || nextBg.empty()) {
                        m_dynamicVideoCap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        m_dynamicVideoCap.read(nextBg);
                    }
                }
                if (!nextBg.empty()) {
                    cv::resize(nextBg, segmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);
                    m_dynamicVideoFrame = segmentedFrame.clone();
                } else {
                    segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
                }
            }
        } else if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
            // Check if we need to reload the background template
            bool needReload = cachedBackgroundTemplate.empty() ||
                             lastBackgroundPath != m_selectedBackgroundTemplate;

            if (needReload) {
                qDebug() << "🎯 Loading background template:" << m_selectedBackgroundTemplate;

                // Check if this is image6 (white background special case)
                if (m_selectedBackgroundTemplate.contains("bg6.png")) {
                    // Create white background instead of loading a file
                    // OpenCV uses BGR format, so we need to set all channels to 255 for white
                    cachedBackgroundTemplate = cv::Mat(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
                    lastBackgroundPath = m_selectedBackgroundTemplate;
                    qDebug() << "🎯 White background created for image6, size:" << frame.cols << "x" << frame.rows;
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
                        qWarning() << "🎯 Background template not found in expected locations for request:" << requestedPath
                                   << "- falling back to black background";
                        cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                    } else {
                        // Load background image directly using OpenCV for performance
                        cv::Mat backgroundImage = cv::imread(resolvedPath.toStdString());
                        if (!backgroundImage.empty()) {
                            // Resize background to match frame size
                            cv::resize(backgroundImage, cachedBackgroundTemplate, frame.size(), 0, 0, cv::INTER_LINEAR);
                            lastBackgroundPath = m_selectedBackgroundTemplate;
                            qDebug() << "🎯 Background template loaded from" << resolvedPath
                                     << "and cached at" << frame.cols << "x" << frame.rows;
                        } else {
                            qWarning() << "🎯 Failed to decode background template from:" << resolvedPath
                                       << "- using black background";
                            cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                        }
                    }
                }
            }

            // Use cached background template
            segmentedFrame = cachedBackgroundTemplate.clone();
        } else {
            // Use black background (default) - same size as frame
            segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            qDebug() << "🎯 Using black background (no template selected)";
        }

        for (int i = 0; i < maxDetections; i++) {
            const auto& detection = detections[i];
            qDebug() << "🎯 Processing detection" << i << "at" << detection.x << detection.y << detection.width << "x" << detection.height;

            // Get enhanced edge-based segmentation mask for this person
            cv::Mat personMask = enhancedSilhouetteSegment(frame, detection);

            // Check if mask has any non-zero pixels
            int nonZeroPixels = cv::countNonZero(personMask);
            qDebug() << "🎯 Person mask has" << nonZeroPixels << "non-zero pixels";

            // Apply mask to extract person from camera frame
            cv::Mat personRegion;
            frame.copyTo(personRegion, personMask);
            
            // Store raw person data for post-processing (lighting will be applied after capture)
            m_lastRawPersonRegion = personRegion.clone();
            m_lastRawPersonMask = personMask.clone();
            
            // Store template background if using background template
            if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
                // Use cached template background if available, otherwise load it
                if (m_lastTemplateBackground.empty() || lastBackgroundPath != m_selectedBackgroundTemplate) {
                    QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
                    if (!resolvedPath.isEmpty()) {
                        cv::Mat templateBg = cv::imread(resolvedPath.toStdString());
                        if (!templateBg.empty()) {
                            cv::resize(templateBg, m_lastTemplateBackground, frame.size());
                        qDebug() << "🎯 Template background cached for post-processing from:" << resolvedPath;
                        } else {
                            qWarning() << "🎯 Failed to load template background from resolved path:" << resolvedPath;
                            m_lastTemplateBackground = cv::Mat(); // Clear cache
                        }
                    } else {
                        qWarning() << "🎯 Could not resolve template background path:" << m_selectedBackgroundTemplate;
                        m_lastTemplateBackground = cv::Mat(); // Clear cache
                    }
                }
                // m_lastTemplateBackground is now ready to use (either from cache or freshly loaded)
            }

            // Scale the person region with person-only scaling for background template mode and dynamic video mode
            cv::Mat scaledPersonRegion, scaledPersonMask;

            // Apply person-only scaling if we're using background template OR dynamic video background
            if ((m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground) {
                // Calculate scaled size for person based on background size and person scale factor
                cv::Size backgroundSize = segmentedFrame.size();
                cv::Size scaledPersonSize;

                if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
                    // Apply person scale factor
                    int scaledWidth = static_cast<int>(backgroundSize.width * m_personScaleFactor + 0.5);
                    int scaledHeight = static_cast<int>(backgroundSize.height * m_personScaleFactor + 0.5);
                    scaledPersonSize = cv::Size(scaledWidth, scaledHeight);

                    qDebug() << "🎯 Person scaled to" << scaledWidth << "x" << scaledHeight
                             << "with factor" << m_personScaleFactor;
                } else {
                    // No scaling needed, use background size
                    scaledPersonSize = backgroundSize;
                }

                // Scale person to the calculated size
                cv::resize(personRegion, scaledPersonRegion, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
                cv::resize(personMask, scaledPersonMask, scaledPersonSize, 0, 0, cv::INTER_LINEAR);

                // Calculate position to center the scaled person on the background
                int xOffset = (backgroundSize.width - scaledPersonSize.width) / 2;
                int yOffset = (backgroundSize.height - scaledPersonSize.height) / 2;

                // Ensure ROI is within background bounds
                if (xOffset >= 0 && yOffset >= 0 &&
                    xOffset + scaledPersonSize.width <= backgroundSize.width &&
                    yOffset + scaledPersonSize.height <= backgroundSize.height) {

                    // Create ROI for compositing using proper cv::Rect constructor
                    cv::Rect backgroundRect(cv::Point(xOffset, yOffset), scaledPersonSize);
                    cv::Rect personRect(cv::Point(0, 0), scaledPersonSize);

                    // Composite scaled person onto background at calculated position
                    cv::Mat backgroundROI = segmentedFrame(backgroundRect);
                    scaledPersonRegion(personRect).copyTo(backgroundROI, scaledPersonMask(personRect));
                } else {
                    // Fallback: composite at origin if scaling makes person too large
                    scaledPersonRegion.copyTo(segmentedFrame, scaledPersonMask);
                }
            } else {
                // For black background, scale to match frame size (original behavior)
                cv::resize(personRegion, scaledPersonRegion, segmentedFrame.size(), 0, 0, cv::INTER_LINEAR);
                cv::resize(personMask, scaledPersonMask, segmentedFrame.size(), 0, 0, cv::INTER_LINEAR);

                // Simple compositing: copy scaled person region directly to background where mask is non-zero
                scaledPersonRegion.copyTo(segmentedFrame, scaledPersonMask);
            }
        }

        qDebug() << "🎯 Segmentation complete, returning segmented frame";
        return segmentedFrame;
    } else if (m_displayMode == RectangleMode) {
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
    } else {
        // Normal mode - return original frame
        return frame.clone();
    }
}

// Phase 2A: GPU-Only Segmentation Frame Creation
cv::Mat Capture::createSegmentedFrameGPUOnly(const cv::Mat &frame, const std::vector<cv::Rect> &detections)
{
    // Process only first 3 detections for better performance
    int maxDetections = std::min(3, (int)detections.size());

    if (m_displayMode == SegmentationMode) {
        qDebug() << "🎮 Phase 2A: GPU-only segmentation frame creation";

        // Create background for edge-based segmentation
        cv::Mat segmentedFrame;

        // Use cached background template for performance
        static cv::Mat cachedBackgroundTemplate;
        static QString lastBackgroundPath;

        if (m_useDynamicVideoBackground && m_videoPlaybackActive) {
            // Phase 2A: GPU-only video background processing
            if (!m_dynamicVideoFrame.empty()) {
                // Upload video frame to GPU
                m_gpuBackgroundFrame.upload(m_dynamicVideoFrame);

                // Resize on GPU
                cv::cuda::resize(m_gpuBackgroundFrame, m_gpuSegmentedFrame, frame.size(), 0, 0, cv::INTER_LINEAR);

                // Download result
                m_gpuSegmentedFrame.download(segmentedFrame);
                qDebug() << "🎮 Phase 2A: GPU-only video background processing completed";
            } else {
                segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
            }
        } else if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
            // GPU-only background template processing
            if (lastBackgroundPath != m_selectedBackgroundTemplate) {
                QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
                if (!resolvedPath.isEmpty()) {
                    cachedBackgroundTemplate = cv::imread(resolvedPath.toStdString());
                    if (cachedBackgroundTemplate.empty()) {
                        qWarning() << "🎯 Failed to load background template from resolved path:" << resolvedPath;
                        cachedBackgroundTemplate = cv::Mat::zeros(frame.size(), frame.type());
                    } else {
                        // Only show success message once per template change
                        static QString lastLoggedTemplate;
                        if (lastLoggedTemplate != m_selectedBackgroundTemplate) {
                            qDebug() << "🎯 GPU: Background template loaded from resolved path:" << resolvedPath;
                            lastLoggedTemplate = m_selectedBackgroundTemplate;
                        }
                    }
                } else {
                    qWarning() << "🎯 GPU: Could not resolve background template path:" << m_selectedBackgroundTemplate;
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
            cv::Mat personSegment = enhancedSilhouetteSegmentGPUOnly(m_gpuVideoFrame, detections[i]);
            if (!personSegment.empty()) {
                // Composite person onto background
                cv::addWeighted(segmentedFrame, 1.0, personSegment, 1.0, 0.0, segmentedFrame);
            }
        }

        return segmentedFrame;

    } else if (m_displayMode == RectangleMode) {
        // Rectangle mode - draw rectangles on original frame
        cv::Mat result = frame.clone();
        for (int i = 0; i < maxDetections; i++) {
            cv::rectangle(result, detections[i], cv::Scalar(0, 255, 0), 2);
        }
        return result;

    } else {
        // Normal mode - return original frame
        return frame.clone();
    }
}

cv::Mat Capture::enhancedSilhouetteSegment(const cv::Mat &frame, const cv::Rect &detection)
{
    // Optimized frame skipping for GPU-accelerated segmentation - process every 4th frame
    static int frameCounter = 0;
    static double lastProcessingTime = 0.0;
    frameCounter++;

    // More aggressive skipping to prevent freezing
    bool shouldProcess = (frameCounter % 4 == 0); // Process every 4th frame by default

    // If processing is taking too long, skip even more frames
    if (lastProcessingTime > 20.0) { // Reduced threshold from 30ms to 20ms
        shouldProcess = (frameCounter % 6 == 0); // Process every 6th frame
    } else if (lastProcessingTime < 10.0) { // Reduced threshold from 15ms to 10ms
        shouldProcess = (frameCounter % 2 == 0); // Process every 2nd frame at most
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

    // qDebug() << "🎯 Starting enhanced silhouette segmentation for detection at" << detection.x << detection.y << detection.width << "x" << detection.height;

    // Person-focused silhouette segmentation with enhanced edge detection
    // Validate and clip detection rectangle to frame bounds
    qDebug() << "🎯 Frame size:" << frame.cols << "x" << frame.rows;
    qDebug() << "🎯 Original detection rectangle:" << detection.x << detection.y << detection.width << "x" << detection.height;

    // Create a clipped version of the detection rectangle
    cv::Rect clippedDetection = detection;

    // Clip to frame bounds
    clippedDetection.x = std::max(0, clippedDetection.x);
    clippedDetection.y = std::max(0, clippedDetection.y);
    clippedDetection.width = std::min(clippedDetection.width, frame.cols - clippedDetection.x);
    clippedDetection.height = std::min(clippedDetection.height, frame.rows - clippedDetection.y);

    qDebug() << "🎯 Clipped detection rectangle:" << clippedDetection.x << clippedDetection.y << clippedDetection.width << "x" << clippedDetection.height;

    // Check if the clipped rectangle is still valid
    if (clippedDetection.width <= 0 || clippedDetection.height <= 0) {
        qDebug() << "🎯 Clipped detection rectangle is invalid, returning empty mask";
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }

    // Create expanded rectangle for full body coverage
    cv::Rect expandedRect = clippedDetection;
    expandedRect.x = std::max(0, expandedRect.x - 25); // Larger expansion for full body
    expandedRect.y = std::max(0, expandedRect.y - 25);
    expandedRect.width = std::min(frame.cols - expandedRect.x, expandedRect.width + 50); // Larger expansion
    expandedRect.height = std::min(frame.rows - expandedRect.y, expandedRect.height + 50);

    qDebug() << "🎯 Expanded rectangle:" << expandedRect.x << expandedRect.y << expandedRect.width << "x" << expandedRect.height;

    // Validate expanded rectangle
    if (expandedRect.width <= 0 || expandedRect.height <= 0) {
        qDebug() << "🎯 Invalid expanded rectangle, returning empty mask";
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }

    // Create ROI for silhouette extraction
    cv::Mat roi = frame(expandedRect);
    cv::Mat roiMask = cv::Mat::zeros(roi.size(), CV_8UC1);

    qDebug() << "🎯 ROI created, size:" << roi.cols << "x" << roi.rows;

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

            qDebug() << "🎮 GPU-accelerated edge detection applied";

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

    qDebug() << "🎯 Found" << edgeContours.size() << "edge contours";

    // Filter contours based on person-like characteristics
    std::vector<std::vector<cv::Point>> validContours;
    cv::Point detectionCenter(expandedRect.width/2, expandedRect.height/2);

    // Only process edge contours if they exist
    if (!edgeContours.empty()) {
        qDebug() << "🎯 Filtering" << edgeContours.size() << "contours for person-like characteristics";

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

        qDebug() << "🎯 After filtering:" << validContours.size() << "valid contours";
    } else {
        qDebug() << "🎯 No edge contours found, skipping to background subtraction";
    }

    // If no valid edge contours found, use background subtraction approach
    if (validContours.empty()) {
        qDebug() << "🎯 No valid edge contours, trying background subtraction";
        // Use background subtraction for motion-based segmentation
        cv::Mat fgMask;
        m_bgSubtractor->apply(roi, fgMask);

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

                qDebug() << "🎮 GPU-accelerated morphological operations applied";

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
        qDebug() << "🎯 Background subtraction found" << validContours.size() << "contours";
    }

    // If still no valid contours, try color-based segmentation
    if (validContours.empty()) {
        qDebug() << "🎯 No contours from background subtraction, trying color-based segmentation";

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
                cv::cuda::inRange(gpu_hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), gpu_skinMask);
                cv::cuda::inRange(gpu_hsv, cv::Scalar(0, 30, 50), cv::Scalar(180, 255, 255), gpu_colorMask);

                // Combine masks on GPU using bitwise_or
                cv::cuda::GpuMat gpu_combinedMask;
                cv::cuda::bitwise_or(gpu_skinMask, gpu_colorMask, gpu_combinedMask);

                // Download result back to CPU
                gpu_combinedMask.download(combinedMask);

                qDebug() << "🎮 GPU-accelerated color segmentation applied";

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA color segmentation failed, falling back to CPU:" << e.what();
                // Fallback to CPU processing
                cv::Mat hsv;
                cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
                cv::Mat skinMask;
                cv::inRange(hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), skinMask);
                cv::Mat colorMask;
                cv::inRange(hsv, cv::Scalar(0, 30, 50), cv::Scalar(180, 255, 255), colorMask);
                cv::bitwise_or(skinMask, colorMask, combinedMask);
            }
        } else {
            // CPU fallback
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
            cv::Mat skinMask;
            cv::inRange(hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), skinMask);
            cv::Mat colorMask;
            cv::inRange(hsv, cv::Scalar(0, 30, 50), cv::Scalar(180, 255, 255), colorMask);
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

                qDebug() << "🎮 GPU-accelerated color morphological operations applied";

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
        qDebug() << "🎯 Color-based segmentation found" << validContours.size() << "contours";
    }

    // Create mask from valid contours
    if (!validContours.empty()) {
        qDebug() << "🎯 Creating mask from" << validContours.size() << "valid contours";
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

                qDebug() << "🎮 GPU-accelerated final morphological cleanup applied";

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
        qDebug() << "🎯 No valid contours found, creating empty mask";
    }

    // Create final mask for the entire frame
    cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    roiMask.copyTo(finalMask(expandedRect));

    int finalNonZeroPixels = cv::countNonZero(finalMask);
    qDebug() << "🎯 Enhanced silhouette segmentation complete, final mask has" << finalNonZeroPixels << "non-zero pixels";

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

    qDebug() << "🎮 Phase 2A: GPU-only silhouette segmentation";

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

    // 🚀 GPU MEMORY POOL OPTIMIZED PIPELINE - REUSABLE BUFFERS + ASYNC STREAMS

    // Check if GPU Memory Pool is available
    if (!m_gpuMemoryPoolInitialized || !m_gpuMemoryPool.isInitialized()) {
        qWarning() << "🚀 GPU Memory Pool not available, falling back to standard GPU processing";
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

        qDebug() << "🚀 Phase 2A: Standard GPU processing completed (memory pool not available)";
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

    // 🚀 SYNCHRONIZE STREAMS BEFORE DOWNLOAD
    detectionStream.waitForCompletion();
    segmentationStream.waitForCompletion();

    // Step 6: Single download at the end (minimize GPU-CPU transfers)
    cv::Mat finalMask;
    gpuConnectedMask.download(finalMask);

    // 🚀 GPU-OPTIMIZED: Create full-size mask directly on GPU
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

    qDebug() << "🚀 Phase 2A: GPU MEMORY POOL + ASYNC STREAMS silhouette segmentation completed";

    return fullMask;
}

// Phase 2A: GPU Result Validation
void Capture::validateGPUResults(const cv::Mat &gpuResult, const cv::Mat &cpuResult)
{
    if (gpuResult.empty() || cpuResult.empty()) {
        qWarning() << "🎮 Phase 2A: GPU/CPU result validation failed - empty results";
        return;
    }

    if (gpuResult.size() != cpuResult.size() || gpuResult.type() != cpuResult.type()) {
        qWarning() << "🎮 Phase 2A: GPU/CPU result validation failed - size/type mismatch";
        return;
    }

    // Compare results (allow small differences due to floating-point precision)
    cv::Mat diff;
    cv::absdiff(gpuResult, cpuResult, diff);
    double maxDiff = cv::norm(diff, cv::NORM_INF);

    if (maxDiff > 5.0) { // Allow small differences
        qWarning() << "🎮 Phase 2A: GPU/CPU result validation failed - max difference:" << maxDiff;
    } else {
        qDebug() << "🎮 Phase 2A: GPU/CPU result validation passed - max difference:" << maxDiff;
    }
}

void Capture::adjustRect(cv::Rect &r) const
{
    // The HOG detector returns slightly larger rectangles than the real objects,
    // so we slightly shrink the rectangles to get a nicer output.
    // EXACT from peopledetect_v1.cpp
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
}



std::vector<cv::Rect> Capture::detectPeople(const cv::Mat &frame)
{
    std::vector<cv::Rect> found;



    if (m_useCUDA) {
        // CUDA GPU-accelerated processing for NVIDIA GPU (PRIORITY)
        m_gpuUtilized = false;
        m_cudaUtilized = true;

        try {
            // Validate frame before CUDA operations
            if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
                throw cv::Exception(0, "Invalid frame for CUDA processing", "", "", 0);
            }

            // Validate input frame dimensions before GPU operations
            if (frame.rows <= 0 || frame.cols <= 0) {
                qWarning() << "🎮 Invalid frame dimensions for CUDA processing:" << frame.rows << "x" << frame.cols;
                found.clear();
                m_cudaUtilized = false;
                return found;
            }

            // Upload to CUDA GPU
            cv::cuda::GpuMat gpu_frame;
            gpu_frame.upload(frame);

            // Validate uploaded GPU frame
            if (gpu_frame.empty() || gpu_frame.rows <= 0 || gpu_frame.cols <= 0) {
                qWarning() << "🎮 Invalid GPU frame dimensions after upload:" << gpu_frame.rows << "x" << gpu_frame.cols;
                found.clear();
                m_cudaUtilized = false;
                return found;
            }

            // Convert to grayscale on GPU
            cv::cuda::GpuMat gpu_gray;
            cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);

            // Validate grayscale GPU frame
            if (gpu_gray.empty() || gpu_gray.rows <= 0 || gpu_gray.cols <= 0) {
                qWarning() << "🎮 Invalid grayscale GPU frame dimensions:" << gpu_gray.rows << "x" << gpu_gray.cols;
                found.clear();
                m_cudaUtilized = false;
                return found;
            }

            // Calculate resize dimensions for optimal detection accuracy (matching peopledetect_v1.cpp)
            int new_width = cvRound(gpu_gray.cols * 0.5); // 0.5x scale for better performance
            int new_height = cvRound(gpu_gray.rows * 0.5); // 0.5x scale for better performance

            // Ensure minimum dimensions for HOG detection (HOG needs at least 64x128)
            new_width = std::max(new_width, 128); // Minimum for HOG detection
            new_height = std::max(new_height, 256); // Minimum for HOG detection

            // Validate resize dimensions
            if (new_width <= 0 || new_height <= 0) {
                qWarning() << "🎮 Invalid resize dimensions:" << new_width << "x" << new_height;
                found.clear();
                m_cudaUtilized = false;
                return found;
            }

            qDebug() << "🎮 Resizing GPU matrix to:" << new_width << "x" << new_height;

            // Resize for optimal GPU performance (ensure minimum size for HOG)
            cv::cuda::GpuMat gpu_resized;
            cv::cuda::resize(gpu_gray, gpu_resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

            // Validate resized GPU matrix
            if (gpu_resized.empty() || gpu_resized.rows <= 0 || gpu_resized.cols <= 0) {
                qWarning() << "🎮 Invalid resized GPU matrix dimensions:" << gpu_resized.rows << "x" << gpu_resized.cols;
                found.clear();
                m_cudaUtilized = false;
                return found;
            }

            qDebug() << "🎮 Resized GPU matrix validated - size:" << gpu_resized.rows << "x" << gpu_resized.cols;

            // CUDA HOG detection
            if (m_cudaHogDetector && !m_cudaHogDetector.empty() && m_useCUDA) {
                try {
                    std::vector<cv::Rect> found_cuda;

                    // Simple CUDA HOG detection (working state)
                    m_cudaHogDetector->detectMultiScale(gpu_resized, found_cuda);

                    if (!found_cuda.empty()) {
                        found = found_cuda;
                        m_cudaUtilized = true;
                        qDebug() << "🎮 CUDA HOG detection SUCCESS - detected" << found_cuda.size() << "people";
                    } else {
                        qDebug() << "🎮 CUDA HOG completed but no people detected";
                        found.clear();
                    }

                } catch (const cv::Exception& e) {
                    qDebug() << "🎮 CUDA HOG error:" << e.what() << "falling back to CPU";
                    m_cudaUtilized = false;

                    // Fallback to CPU HOG detection (matching peopledetect_v1.cpp)
                    cv::Mat resized;
                    cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
                    m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

                    // Scale results back up to original size
                    double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0
                    for (auto& rect : found) {
                        rect.x = cvRound(rect.x * scale_factor);
                        rect.y = cvRound(rect.y * scale_factor);
                        rect.width = cvRound(rect.width * scale_factor);
                        rect.height = cvRound(rect.height * scale_factor);
                    }
                }
            } else {
                // CUDA HOG not available - check why
                if (!m_useCUDA) {
                    qDebug() << "🎮 CUDA not enabled, skipping CUDA HOG";
                } else if (!m_cudaHogDetector) {
                    qDebug() << "🎮 CUDA HOG detector not initialized";
                } else if (m_cudaHogDetector.empty()) {
                    qDebug() << "🎮 CUDA HOG detector is empty";
                }
                found.clear();
                m_cudaUtilized = false;
            }

            // Scale results back up to original size (CUDA HOG works on resized image)
            double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0 (matching peopledetect_v1.cpp)
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }

            qDebug() << "🎮 CUDA GPU: Color conversion + resize + HOG detection (GPU acceleration with CPU coordination)";

        } catch (const cv::Exception& e) {
            qWarning() << "🎮 CUDA processing error:" << e.what() << "falling back to CPU";
            m_cudaUtilized = false; // Switch to CPU

            // Fallback to CPU HOG detection (matching peopledetect_v1.cpp)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

            // Scale results back up to original size
            double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
        } catch (...) {
            qWarning() << "🎮 Unknown CUDA error, falling back to CPU";
            m_cudaUtilized = false; // Switch to CPU

            // Fallback to CPU HOG detection (matching peopledetect_v1.cpp)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

            // Scale results back up to original size
            double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
        }

    } else if (m_useGPU) {
        // OpenCL GPU-accelerated processing for AMD GPU (FALLBACK)
        m_gpuUtilized = true;
        m_cudaUtilized = false;

        try {
            // Upload to GPU using UMat
            cv::UMat gpu_frame;
            frame.copyTo(gpu_frame);

            // Convert to grayscale on GPU
            cv::UMat gpu_gray;
            cv::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);

            // Resize for optimal GPU performance (matching peopledetect_v1.cpp)
            cv::UMat gpu_resized;
            cv::resize(gpu_gray, gpu_resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

            // OpenCL-accelerated HOG detection (much faster than CPU)
            // Use UMat for GPU-accelerated detection
            std::vector<cv::Rect> found_umat;
            std::vector<double> weights;

            // Run HOG detection on GPU using UMat
            // Optimized parameters for better performance and accuracy
            m_hogDetector.detectMultiScale(gpu_resized, found_umat, weights, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

            // Copy results back to CPU
            found = found_umat;

            // Scale results back up to original size
            for (auto& rect : found) {
                rect.x *= 2; // 1/0.5 = 2
                rect.y *= 2;
                rect.width *= 2;
                rect.height *= 2;
            }

            qDebug() << "🎮 OpenCL GPU: Color conversion + resize + HOG detection (FULL GPU ACCELERATION)";

        } catch (const cv::Exception& e) {
            qWarning() << "OpenCL processing failed:" << e.what() << "Falling back to CPU";
            m_gpuUtilized = false;

            // Fallback to CPU processing (matching peopledetect_v1.cpp)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

            // Scale results back up to original size
            double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
        }

    } else {
            // CPU fallback (matching peopledetect_v1.cpp)
    m_gpuUtilized = false;
    m_cudaUtilized = false;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

        // Run detection with balanced speed/accuracy for 30 FPS
        m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);

        // Scale results back up to original size
        double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0
        for (auto& rect : found) {
            rect.x = cvRound(rect.x * scale_factor);
            rect.y = cvRound(rect.y * scale_factor);
            rect.width = cvRound(rect.width * scale_factor);
            rect.height = cvRound(rect.height * scale_factor);
        }

        qDebug() << "💻 CPU processing utilized (last resort)";
    }

    return found;
}

void Capture::onPersonDetectionFinished()
{
    if (m_personDetectionWatcher && m_personDetectionWatcher->isFinished()) {
        try {
            cv::Mat result = m_personDetectionWatcher->result();
            if (!result.empty()) {
                QMutexLocker locker(&m_personDetectionMutex);

                // Apply lighting correction only if template + mask exist
                if (m_lightingCorrector && !m_lastRawPersonMask.empty() && !m_selectedTemplate.empty()) {
                    cv::Mat maskToPass = m_lastRawPersonMask;
                    if (maskToPass.type() != CV_8U) maskToPass.convertTo(maskToPass, CV_8U);

                    m_lastSegmentedFrame = m_lightingCorrector->applyPersonLightingCorrection(
                        result, maskToPass, m_selectedTemplate);
                } else {
                    m_lastSegmentedFrame = result.clone();
                }

                // Update GPU utilization flags
                if (m_useCUDA) {
                    m_cudaUtilized = true;
                    m_gpuUtilized = false;
                } else if (m_useGPU) {
                    m_gpuUtilized = true;
                    m_cudaUtilized = false;
                }

                qDebug() << "✅ Person detection finished - segmented frame updated, size:"
                         << result.cols << "x" << result.rows;
            } else {
                qDebug() << "⚠️ Person detection finished but result empty";
            }
        } catch (const std::exception& e) {
            qWarning() << "Exception in person detection finished callback:" << e.what();
        }
    } else {
        qDebug() << "⚠️ Person detection watcher not finished or null";
    }
}

// Enhanced Person Detection and Segmentation Control Methods
void Capture::setShowPersonDetection(bool show)
{
    if (show) {
        m_displayMode = SegmentationMode;  // Default to segmentation when enabling
    } else {
        m_displayMode = NormalMode;
    }
    updatePersonDetectionButton();
    qDebug() << "Person detection display set to:" << show << "(mode:" << m_displayMode << ")";
}

bool Capture::getShowPersonDetection() const
{
    return (m_displayMode == RectangleMode || m_displayMode == SegmentationMode);
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

void Capture::togglePersonDetection()
{
    // Only allow toggling if segmentation is enabled in capture interface
    if (m_segmentationEnabledInCapture) {
        // Cycle through modes: Normal -> Rectangle -> Segmentation -> Normal
        switch (m_displayMode) {
            case NormalMode:
                setSegmentationMode(1); // RectangleMode
                break;
            case RectangleMode:
                setSegmentationMode(2); // SegmentationMode
                break;
            case SegmentationMode:
                setSegmentationMode(0); // NormalMode
                break;
        }

        qDebug() << "Person detection toggled to mode:" << m_displayMode;
    } else {
        qDebug() << "🎯 Person detection toggle ignored - segmentation not enabled in capture interface";
    }
}

void Capture::updatePersonDetectionButton()
{
    if (personDetectionButton) {
        QString buttonText;
        QString buttonStyle;

        if (m_segmentationEnabledInCapture) {
            // Show current mode when segmentation is enabled
            switch (m_displayMode) {
                case NormalMode:
                    buttonText = "Enable Detection (Press S)";
                    buttonStyle = "QPushButton { color: white; font-size: 12px; background-color: #388e3c; border: 1px solid white; padding: 5px; }";
                    break;
                case RectangleMode:
                    buttonText = "Switch to Segmentation (Press S)";
                    buttonStyle = "QPushButton { color: white; font-size: 12px; background-color: #1976d2; border: 1px solid white; padding: 5px; }";
                    break;
                case SegmentationMode:
                    buttonText = "Switch to Normal (Press S)";
                    buttonStyle = "QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; }";
                    break;
            }
        } else {
            // Show disabled state when segmentation is not enabled
            buttonText = "Segmentation: DISABLED (Capture Only)";
            buttonStyle = "QPushButton { color: #cccccc; font-size: 12px; background-color: #666666; border: 1px solid #999999; padding: 5px; }";
        }

        personDetectionButton->setText(buttonText);
        personDetectionButton->setStyleSheet(buttonStyle);
    }
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

cv::Mat Capture::getMotionMask(const cv::Mat &frame)
{
    cv::Mat fgMask;

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

std::vector<cv::Rect> Capture::filterByMotion(const std::vector<cv::Rect> &detections, const cv::Mat &/*motionMask*/)
{
    // Skip motion filtering for better FPS (motion filtering can be slow)
    // Return all detections for maximum FPS and accuracy
    return detections;

    /*
    // Uncomment the code below if you want motion filtering
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

        // Keep detection if there's significant motion (more than 10%) - matching peopledetect_v1.cpp
        if (motionRatio > 0.1) {
            filtered.push_back(rect);
        }
    }

    return filtered;
    */
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
                qDebug() << "🎯 HAND CLOSED DETECTED! Automatically triggering capture...";
                qDebug() << "🎯 Current display mode:" << m_displayMode << "(0=Normal, 1=Rectangle, 2=Segmentation)";

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
            qDebug() << "✅ Hand detection initialized successfully";
            m_handDetectionEnabled = true;
        } else {
            qDebug() << "❌ Failed to initialize hand detection";
            m_handDetectionEnabled = false;
        }
    }
}

void Capture::startHandTriggeredCountdown()
{
    // This slot runs in the main thread and safely updates UI elements
    if (!countdownTimer || !countdownTimer->isActive()) {
        qDebug() << "🎯 Starting countdown from hand detection signal...";
        // Start 5-second countdown for hand-triggered capture (same as button press)
        ui->capture->setEnabled(false);
        countdownValue = 5; // 5 second countdown
        countdownLabel->setText(QString::number(countdownValue));
        countdownLabel->show();
        countdownLabel->raise(); // Bring to front
        countdownTimer->start(1000); // 1 second intervals
        qDebug() << "🎯 5-second countdown automatically started by hand detection!";
    } else {
        qDebug() << "🎯 Countdown already active, ignoring hand trigger";
    }
}

void Capture::onHandTriggeredCapture()
{
    // This slot runs in the main thread and safely handles hand detection triggers
    qDebug() << "🎯 Hand triggered capture signal received in main thread";
    startHandTriggeredCountdown();
}

void Capture::enableHandDetection(bool enable)
{
    m_handDetectionEnabled = enable;
    qDebug() << "🖐️ Hand detection" << (enable ? "ENABLED" : "DISABLED");

    if (enable) {
        // Enable hand detection
        if (m_handDetector && !m_handDetector->isInitialized()) {
            initializeHandDetection();
        }
        if (m_handDetector) {
            m_handDetector->resetGestureState();
            qDebug() << "🖐️ Hand detection gesture state reset - ready for new detection";
        }
    } else {
        // Disable hand detection
        if (m_handDetector) {
            m_handDetector->resetGestureState();
            qDebug() << "🖐️ Hand detection gesture state reset - disabled";
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
    if (loadingCameraLabel) {
        // Set basic size and position for visibility
        loadingCameraLabel->setFixedSize(500, 120); // Increased height from 100 to 120

        // Center the label with the capture button (which is positioned more to the right)
        int x = (width() - 500) / 2 + 350; // Move right to align with capture button
        int y = (height() - 120) / 2; // Adjusted for new height
        loadingCameraLabel->move(x, y);

        loadingCameraLabel->show();
        loadingCameraLabel->raise(); // Bring to front
        qDebug() << "📹 Loading camera label shown at position:" << x << "," << y;
    }
}



void Capture::hideLoadingCameraLabel()
{
    if (loadingCameraLabel) {
        loadingCameraLabel->hide();
        qDebug() << "📹 Loading camera label hidden";
    }
}

// Loading label management (thread-safe)
void Capture::handleFirstFrame()
{
    // This method runs in the main thread (thread-safe)
    qDebug() << "🎯 handleFirstFrame() called in thread:" << QThread::currentThread();

    // Hide the loading camera label when first frame is received
    hideLoadingCameraLabel();

    // 🚀 Initialize GPU Memory Pool when first frame is received
    if (!m_gpuMemoryPoolInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        try {
            qDebug() << "🚀 Initializing GPU Memory Pool on first frame...";
            m_gpuMemoryPool.initialize(1280, 720); // Initialize with common camera resolution
            m_gpuMemoryPoolInitialized = true;
            qDebug() << "✅ GPU Memory Pool initialized successfully on first frame";
        } catch (const cv::Exception& e) {
            qWarning() << "🚀 GPU Memory Pool initialization failed on first frame:" << e.what();
            m_gpuMemoryPoolInitialized = false;
        }
    }

    // Mark camera as initialized for the first time
    if (!m_cameraFirstInitialized) {
        m_cameraFirstInitialized = true;
        qDebug() << "🎯 Camera first initialization complete - loading label hidden permanently";
    } else {
        qDebug() << "🎯 Camera frame received (not first initialization)";
    }
}

// Segmentation Control Methods for Capture Interface
void Capture::enableSegmentationInCapture()
{
    qDebug() << "🎯 Enabling segmentation for capture interface";
    m_segmentationEnabledInCapture = true;

    // Restore the last segmentation mode if it was different from normal
    if (m_lastSegmentationMode != NormalMode) {
        m_displayMode = m_lastSegmentationMode;
        qDebug() << "🎯 Restored segmentation mode:" << m_lastSegmentationMode;
    } else {
        // Default to normal mode for first-time capture page visits
        m_displayMode = NormalMode;
        qDebug() << "🎯 Using default normal mode for capture interface";
    }

    // Clear any previous segmentation results to force new processing
    m_lastSegmentedFrame = cv::Mat();
    m_lastDetections.clear();

    // Update UI to reflect the current state
    updatePersonDetectionButton();
    updateDebugDisplay();
}

void Capture::disableSegmentationOutsideCapture()
{
    qDebug() << "🎯 Disabling segmentation outside capture interface";

    // Store the current segmentation mode before disabling
    if (m_displayMode != NormalMode) {
        m_lastSegmentationMode = m_displayMode;
        qDebug() << "🎯 Stored segmentation mode:" << m_lastSegmentationMode << "for later restoration";
    }

    // Disable segmentation by switching to normal mode
    m_displayMode = NormalMode;
    m_segmentationEnabledInCapture = false;

    // Clear any cached segmentation data
    m_lastSegmentedFrame = cv::Mat();
    m_lastDetections.clear();

    // Reset GPU utilization flags
    m_gpuUtilized = false;
    m_cudaUtilized = false;

    // Update UI
    updatePersonDetectionButton();
    updateDebugDisplay();

    qDebug() << "🎯 Segmentation disabled - switched to normal mode";
}

void Capture::restoreSegmentationState()
{
    qDebug() << "🎯 Restoring segmentation state for capture interface";

    if (m_segmentationEnabledInCapture) {
        // If segmentation was enabled in capture, restore the last mode
        if (m_lastSegmentationMode != NormalMode) {
            m_displayMode = m_lastSegmentationMode;
            qDebug() << "🎯 Restored segmentation mode:" << m_lastSegmentationMode;
        } else {
            m_displayMode = NormalMode;
            qDebug() << "🎯 Using normal mode (no previous segmentation state)";
        }
    } else {
        // If segmentation was not enabled, stay in normal mode
        m_displayMode = NormalMode;
        qDebug() << "🎯 Segmentation not enabled in capture - staying in normal mode";
    }

    updatePersonDetectionButton();
    updateDebugDisplay();
}

bool Capture::isSegmentationEnabledInCapture() const
{
    return m_segmentationEnabledInCapture;
}

void Capture::setSegmentationMode(int mode)
{
    qDebug() << "🎯 Setting segmentation mode to:" << mode;

    // Only allow mode changes if segmentation is enabled in capture
    if (m_segmentationEnabledInCapture) {
        // Convert int to DisplayMode enum
        DisplayMode displayMode = static_cast<DisplayMode>(mode);
        m_displayMode = displayMode;

        // Store the mode for later restoration (if it's not normal mode)
        if (displayMode != NormalMode) {
            m_lastSegmentationMode = displayMode;
        }

        // Reset utilization when switching to normal mode
        if (displayMode == NormalMode) {
            m_gpuUtilized = false;
            m_cudaUtilized = false;
        }

        this->updatePersonDetectionButton();
        this->updateDebugDisplay();

        qDebug() << "🎯 Segmentation mode set to:" << displayMode;
    } else {
        qDebug() << "🎯 Cannot set segmentation mode - segmentation not enabled in capture interface";
    }
}

// Background Template Control Methods
void Capture::setSelectedBackgroundTemplate(const QString &path)
{
    m_selectedBackgroundTemplate = path;
    m_useBackgroundTemplate = !path.isEmpty();
    qDebug() << "🎯 Background template set to:" << path << "Use template:" << m_useBackgroundTemplate;
    
    // Automatically set the reference template for lighting correction
    if (m_useBackgroundTemplate && !path.isEmpty()) {
        setReferenceTemplate(path);
        qDebug() << "🌟 Reference template automatically set for lighting correction";
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
        qDebug() << "🎯 VIDEO TEMPLATE DURATION UPDATED:" << durationSeconds << "seconds";
        qDebug() << "  - Template name:" << m_currentVideoTemplate.name;
        qDebug() << "  - Recording will automatically stop after" << durationSeconds << "seconds";
    } else {
        qWarning() << "🎯 Invalid duration specified:" << durationSeconds << "seconds (must be > 0)";
    }
}

int Capture::getVideoTemplateDuration() const
{
    return m_currentVideoTemplate.durationSeconds;
}

// 🚀 GPU MEMORY POOL IMPLEMENTATION

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
    , initialized(false)
    , poolWidth(0)
    , poolHeight(0)
{
    qDebug() << "🚀 GPU Memory Pool: Constructor called";
}

GPUMemoryPool::~GPUMemoryPool()
{
    qDebug() << "🚀 GPU Memory Pool: Destructor called";
    release();
}

void GPUMemoryPool::initialize(int width, int height)
{
    if (initialized && poolWidth == width && poolHeight == height) {
        qDebug() << "🚀 GPU Memory Pool: Already initialized with correct dimensions";
        return;
    }

    qDebug() << "🚀 GPU Memory Pool: Initializing with dimensions" << width << "x" << height;

    try {
        // Release existing resources
        release();

        // Initialize frame buffers (triple buffering)
        for (int i = 0; i < 3; ++i) {
            gpuFrameBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            qDebug() << "🚀 GPU Memory Pool: Frame buffer" << i << "allocated";
        }

        // Initialize segmentation buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuSegmentationBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << "🚀 GPU Memory Pool: Segmentation buffer" << i << "allocated";
        }

        // Initialize detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuDetectionBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << "🚀 GPU Memory Pool: Detection buffer" << i << "allocated";
        }

        // Initialize temporary buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuTempBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << "🚀 GPU Memory Pool: Temp buffer" << i << "allocated";
        }

        // Create reusable CUDA filters (create once, use many times)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        morphCloseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel);
        morphOpenFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel);
        morphDilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
        cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);

        qDebug() << "🚀 GPU Memory Pool: CUDA filters created successfully";

        // Initialize CUDA streams for parallel processing
        detectionStream = cv::cuda::Stream();
        segmentationStream = cv::cuda::Stream();
        compositionStream = cv::cuda::Stream();

        qDebug() << "🚀 GPU Memory Pool: CUDA streams initialized";

        // Update state
        poolWidth = width;
        poolHeight = height;
        initialized = true;

        qDebug() << "🚀 GPU Memory Pool: Initialization completed successfully";

    } catch (const cv::Exception& e) {
        qWarning() << "🚀 GPU Memory Pool: Initialization failed:" << e.what();
        release();
    }
}

cv::cuda::GpuMat& GPUMemoryPool::getNextFrameBuffer()
{
    if (!initialized) {
        qWarning() << "🚀 GPU Memory Pool: Not initialized, returning empty buffer";
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
        qWarning() << "🚀 GPU Memory Pool: Not initialized, returning empty buffer";
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
        qWarning() << "🚀 GPU Memory Pool: Not initialized, returning empty buffer";
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
        qWarning() << "🚀 GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuTempBuffers[currentTempBuffer];
    currentTempBuffer = (currentTempBuffer + 1) % 2; // Double buffering
    return buffer;
}

void GPUMemoryPool::release()
{
    if (!initialized) {
        return;
    }

    qDebug() << "🚀 GPU Memory Pool: Releasing resources";

    // Release GPU buffers
    for (int i = 0; i < 3; ++i) {
        gpuFrameBuffers[i].release();
    }

    for (int i = 0; i < 2; ++i) {
        gpuSegmentationBuffers[i].release();
        gpuDetectionBuffers[i].release();
        gpuTempBuffers[i].release();
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

    qDebug() << "🚀 GPU Memory Pool: Resources released";
}

void GPUMemoryPool::resetBuffers()
{
    if (!initialized) {
        return;
    }

    qDebug() << "🚀 GPU Memory Pool: Resetting buffer indices";

    currentFrameBuffer = 0;
    currentSegBuffer = 0;
    currentDetBuffer = 0;
    currentTempBuffer = 0;
}

// 🚀 ASYNCHRONOUS RECORDING SYSTEM IMPLEMENTATION

void Capture::initializeRecordingSystem()
{
    qDebug() << "🚀 ASYNC RECORDING: Initializing recording system...";

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
            qDebug() << "🚀 ASYNC RECORDING: GPU recording buffer initialized";
        }

        // Start recording thread
        m_recordingThread->start();
        m_recordingThreadActive = true;

        // Start processing timer
        m_recordingFrameTimer->start(16); // 60 FPS processing rate

        qDebug() << "🚀 ASYNC RECORDING: Recording system initialized successfully";

    } catch (const std::exception& e) {
        qWarning() << "🚀 ASYNC RECORDING: Initialization failed:" << e.what();
        cleanupRecordingSystem();
    }
}

void Capture::cleanupRecordingSystem()
{
    qDebug() << "🚀 ASYNC RECORDING: Cleaning up recording system...";

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

    qDebug() << "🚀 ASYNC RECORDING: Recording system cleaned up";
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
            qDebug() << "🚀 ASYNC RECORDING: Frame queued, queue size:" << m_recordingFrameQueue.size();
        } else {
            qWarning() << "🚀 ASYNC RECORDING: Queue full, dropping frame";
        }
    }
}

void Capture::processRecordingFrame()
{
    // This method is no longer needed since we're capturing display directly
    // Keeping it for future expansion if needed
    qDebug() << "🚀 ASYNC RECORDING: Process recording frame called (not used in direct capture mode)";
}

QPixmap Capture::processFrameForRecordingGPU(const cv::Mat &frame)
{
    QPixmap result;

    try {
        // 🚀 GPU-ACCELERATED FRAME PROCESSING

        // Upload frame to GPU
        cv::cuda::GpuMat gpuFrame;
        gpuFrame.upload(frame, m_recordingStream);

        // Get target size
        QSize labelSize = m_cachedLabelSize.isValid() ? m_cachedLabelSize : QSize(1280, 720);

        // GPU-accelerated scaling if needed
        cv::cuda::GpuMat gpuScaled;
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Check if we're in segmentation mode with background template or dynamic video background
            if (m_displayMode == SegmentationMode && ((m_useBackgroundTemplate &&
                !m_selectedBackgroundTemplate.isEmpty()) || m_useDynamicVideoBackground)) {
                // For background template mode, just fit to label
                cv::cuda::resize(gpuFrame, gpuScaled, cv::Size(labelSize.width(), labelSize.height()), 0, 0, cv::INTER_LINEAR, m_recordingStream);
            } else {
                // Apply frame scaling for other modes
                int newWidth = qRound(frame.cols * m_personScaleFactor);
                int newHeight = qRound(frame.rows * m_personScaleFactor);
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

        qDebug() << "🚀 ASYNC RECORDING: GPU frame processing completed";

    } catch (const cv::Exception& e) {
        qWarning() << "🚀 ASYNC RECORDING: GPU processing failed:" << e.what();

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
    qDebug() << "🧹 Capture::cleanupResources - Cleaning up resources when leaving capture page";
    
    // Stop all timers
    if (m_videoPlaybackTimer && m_videoPlaybackTimer->isActive()) {
        m_videoPlaybackTimer->stop();
        qDebug() << "🧹 Stopped video playback timer";
    }
    
    if (recordTimer && recordTimer->isActive()) {
        recordTimer->stop();
        qDebug() << "🧹 Stopped record timer";
    }
    
    if (recordingFrameTimer && recordingFrameTimer->isActive()) {
        recordingFrameTimer->stop();
        qDebug() << "🧹 Stopped recording frame timer";
    }
    
    if (debugUpdateTimer && debugUpdateTimer->isActive()) {
        debugUpdateTimer->stop();
        qDebug() << "🧹 Stopped debug update timer";
    }
    
    // Stop recording if active
    if (m_isRecording) {
        stopRecording();
        qDebug() << "🧹 Stopped active recording";
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
        qDebug() << "🧹 Released GPU memory pool";
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
    
    qDebug() << "🧹 Capture::cleanupResources - Resource cleanup completed";
}

void Capture::initializeResources()
{
    qDebug() << "🚀 Capture::initializeResources - Initializing resources when entering capture page";
    
    // Initialize GPU memory pool if available
    if (isGPUOnlyProcessingAvailable() && !m_gpuMemoryPoolInitialized) {
        m_gpuMemoryPool.initialize(1280, 720); // Default resolution
        m_gpuMemoryPoolInitialized = true;
        qDebug() << "🚀 Initialized GPU memory pool";
    }
    
    // Initialize person detection
    initializePersonDetection();
    
    // Initialize hand detection
    initializeHandDetection();
    
    // Start debug update timer
    if (debugUpdateTimer) {
        debugUpdateTimer->start(1000); // Update every second
        qDebug() << "🚀 Started debug update timer";
    }
    
    qDebug() << "🚀 Capture::initializeResources - Resource initialization completed";
}
// ============================================================================
// LIGHTING CORRECTION IMPLEMENTATION
// ============================================================================

void Capture::initializeLightingCorrection()
{
    qDebug() << "🌟 Initializing lighting correction system";
    
    try {
        // Create lighting corrector instance
        m_lightingCorrector = new LightingCorrector();
        
        // Initialize the lighting corrector
        if (m_lightingCorrector->initialize()) {
            qDebug() << "🌟 Lighting correction system initialized successfully";
            qDebug() << "🌟 GPU acceleration:" << (m_lightingCorrector->isGPUAvailable() ? "Available" : "Not available");
        } else {
            qWarning() << "🌟 Lighting correction initialization failed";
            delete m_lightingCorrector;
            m_lightingCorrector = nullptr;
        }
        
    } catch (const std::exception& e) {
        qWarning() << "🌟 Lighting correction initialization failed:" << e.what();
        if (m_lightingCorrector) {
            delete m_lightingCorrector;
            m_lightingCorrector = nullptr;
        }
    }
}

void Capture::setLightingCorrectionEnabled(bool enabled)
{
    if (m_lightingCorrector) {
        m_lightingCorrector->setEnabled(enabled);
        qDebug() << "🌟 Lighting correction" << (enabled ? "enabled" : "disabled");
    }
}

bool Capture::isLightingCorrectionEnabled() const
{
    return m_lightingCorrector ? m_lightingCorrector->isEnabled() : false;
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
                qDebug() << "🌟 Reference template set for lighting correction:" << resolvedPath;
            } else {
                qWarning() << "🌟 Failed to set reference template from resolved path:" << resolvedPath;
            }
        } else {
            qWarning() << "🌟 Could not resolve reference template path:" << templatePath;
        }
    }
}

cv::Mat Capture::applyPersonLightingCorrection(const cv::Mat &inputImage, const cv::Mat &personMask)
{
    qDebug() << "🔥🔥🔥 applyPersonLightingCorrection CALLED!";
    qDebug() << "🔥 Input image size:" << inputImage.cols << "x" << inputImage.rows;
    qDebug() << "🔥 Person mask size:" << personMask.cols << "x" << personMask.rows;
    
    if (!m_lightingCorrector) {
        qWarning() << "🔥🔥🔥 LIGHTING CORRECTOR IS NULL!";
        return inputImage.clone();
    }
    
    if (!m_lightingCorrector->isEnabled()) {
        qWarning() << "🔥🔥🔥 LIGHTING CORRECTOR IS DISABLED!";
        return inputImage.clone();
    }
    
    qDebug() << "🔥 Lighting corrector exists and is enabled";
    
    // Get the reference template
    cv::Mat referenceTemplate = m_lightingCorrector->getReferenceTemplate();
    qDebug() << "🔥 Reference template size:" << referenceTemplate.cols << "x" << referenceTemplate.rows;
    
    if (referenceTemplate.empty()) {
        qWarning() << "🔥🔥🔥 REFERENCE TEMPLATE IS EMPTY!";
        return inputImage.clone();
    }
    
    qDebug() << "🔥 Calling lighting corrector->applyPersonLightingCorrection";
    cv::Mat result = m_lightingCorrector->applyPersonLightingCorrection(inputImage, personMask, referenceTemplate);
    qDebug() << "🔥 Returned from lighting corrector, result size:" << result.cols << "x" << result.rows;
    
    return result;
}


cv::Mat Capture::applyPostProcessingLighting()
{
    qDebug() << "🎯 POST-PROCESSING: Apply lighting to raw person data and re-composite";
    
    // Check if we have raw person data
    if (m_lastRawPersonRegion.empty() || m_lastRawPersonMask.empty()) {
        qWarning() << "🎯 No raw person data available, returning original segmented frame";
        return m_lastSegmentedFrame.clone();
    }
    
    // Start with the original segmented frame (which has the template background)
    cv::Mat result = m_lastSegmentedFrame.clone();
    
    // Apply lighting to the raw person region
    cv::Mat lightingCorrectedPerson = applyLightingToRawPersonRegion(m_lastRawPersonRegion, m_lastRawPersonMask);
    
    // Scale the lighting-corrected person to match the segmented frame size
    cv::Mat scaledPerson, scaledMask;
    cv::resize(lightingCorrectedPerson, scaledPerson, result.size());
    cv::resize(m_lastRawPersonMask, scaledMask, result.size());
    
    // Composite the lighting-corrected person onto the result
    // This ensures the background template is never modified
    scaledPerson.copyTo(result, scaledMask);
    
    // Save debug images
    cv::imwrite("debug_post_original_segmented.png", m_lastSegmentedFrame);
    cv::imwrite("debug_post_lighting_corrected_person.png", lightingCorrectedPerson);
    cv::imwrite("debug_post_final_result.png", result);
    qDebug() << "🎯 POST-PROCESSING: Applied lighting to person and re-composited";
    qDebug() << "🎯 Debug images saved: post_original_segmented, post_lighting_corrected_person, post_final_result";
    
    return result;
}

cv::Mat Capture::applyLightingToRawPersonRegion(const cv::Mat &personRegion, const cv::Mat &personMask)
{
    qDebug() << "🎯 RAW PERSON APPROACH: Apply lighting to extracted person region only";
    
    // Start with exact copy of person region
    cv::Mat result = personRegion.clone();
    
    // Get template reference for color matching
    cv::Mat templateRef = m_lightingCorrector ? m_lightingCorrector->getReferenceTemplate() : cv::Mat();
    if (templateRef.empty()) {
        qWarning() << "🎯 No template reference, applying subtle lighting correction";
        // Apply subtle lighting correction to make person blend better
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (personMask.at<uchar>(y, x) > 0) {  // Person pixel
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
    
    // Save debug images
    cv::imwrite("debug_raw_person_original.png", personRegion);
    cv::imwrite("debug_raw_person_mask.png", personMask);
    cv::imwrite("debug_raw_person_result.png", result);
    qDebug() << "🎯 RAW PERSON APPROACH: Applied lighting to person region only";
    qDebug() << "🎯 Debug images saved: raw_person_original, raw_person_mask, raw_person_result";
    
    return result;
}

cv::Mat Capture::createPersonMaskFromSegmentedFrame(const cv::Mat &segmentedFrame)
{
    try {
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(segmentedFrame, gray, cv::COLOR_BGR2GRAY);
        
        // Create mask where person pixels are non-black (not background)
        cv::Mat mask;
        cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);
        
        // Apply morphological operations to clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        // Apply Gaussian blur for smooth edges
        cv::GaussianBlur(mask, mask, cv::Size(15, 15), 0);
        
        return mask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 Failed to create person mask:" << e.what();
        return cv::Mat::zeros(segmentedFrame.size(), CV_8UC1);
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
                qDebug() << "🎯 Template path resolved:" << templatePath << "-> " << candidate;
                resolvedPaths.insert(templatePath);
            }
            return candidate;
        }
    }
    
    qWarning() << "🎯 Template path could not be resolved:" << templatePath;
    qWarning() << "🎯 Tried paths:";
    for (const QString &candidate : candidates) {
        qWarning() << "    -" << candidate;
    }
    
    return QString(); // Return empty string if no path found
}
