#include "core/capture.h"
#include "core/camera.h"
#include "core/amd_gpu_verifier.h"
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
#include <QMessageBox>
#include <QDateTime>
#include <QStackedLayout>
#include <QThread>
#include <QFileInfo>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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
    , m_useOpenCL(false)
    , m_useOpenGL(false)
    , m_gpuUtilized(false)
    , m_openclUtilized(false)
    , m_openglUtilized(false)
    , m_personDetectionWatcher(nullptr)
    , m_lastDetections()
    // , m_tfliteModelLoaded(false)
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , fpsLabel(nullptr)
    , gpuStatusLabel(nullptr)
    , openclStatusLabel(nullptr)
    , openglStatusLabel(nullptr)
    , personDetectionLabel(nullptr)
    , personDetectionButton(nullptr)
    , personSegmentationLabel(nullptr)
    , personSegmentationButton(nullptr)
    , handDetectionLabel(nullptr)
    , handDetectionButton(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_handDetector(new HandDetector())
    , m_showHandDetection(true)
    , m_handDetectionEnabled(false)
    , m_handDetectionMutex()
    , m_handDetectionTimer()
    , m_lastHandDetectionTime(0.0)
    , m_handDetectionFPS(0)
    , m_cachedPixmap(640, 480)
    , m_segmentationEnabledInCapture(false)
    , m_lastSegmentationMode(NormalMode)
{
    ui->setupUi(this);

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
        
        // Apply frame scaling if needed
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            QSize originalSize = scaledPixmap.size();
            int newWidth = qRound(originalSize.width() * m_personScaleFactor);
            int newHeight = qRound(originalSize.height() * m_personScaleFactor);
            
            scaledPixmap = scaledPixmap.scaled(
                newWidth, newHeight,
                Qt::KeepAspectRatio,
                Qt::FastTransformation
            );
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
    qDebug() << "OpenCL Acceleration:" << (m_useOpenCL ? "YES (OpenCL)" : "NO (CPU)");
    qDebug() << "OpenCL Utilization:" << (m_openclUtilized ? "ACTIVE" : "IDLE");
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

    // Capture the scaled frame exactly as it appears in the UI (ignoring foreground template)
    if (!m_originalCameraImage.isNull()) {
        QPixmap cameraPixmap = QPixmap::fromImage(m_originalCameraImage);

        // Use cached label size for better performance during recording
        QSize labelSize = m_cachedLabelSize.isValid() ? m_cachedLabelSize : ui->videoLabel->size();

        // Optimize scaling for recording - do both operations in one step if possible
        QPixmap scaledPixmap;
        
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            // Calculate final size directly to avoid double scaling
            QSize finalSize = labelSize;
            finalSize.setWidth(qRound(finalSize.width() * m_personScaleFactor));
            finalSize.setHeight(qRound(finalSize.height() * m_personScaleFactor));
            
            // Single scaling operation for better performance
            scaledPixmap = cameraPixmap.scaled(
                finalSize,
                Qt::KeepAspectRatio,
                Qt::FastTransformation
            );
        } else {
            // No person scaling needed, just fit to label
            scaledPixmap = cameraPixmap.scaled(
                labelSize,
                Qt::KeepAspectRatioByExpanding,
                Qt::FastTransformation
            );
        }
        
        m_recordedFrames.append(scaledPixmap);
    } else {
        qWarning() << "No original camera image available for recording frame.";
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
        stopRecording();
    }
    qDebug() << "Recording: " + QString::number(m_recordedSeconds) + " / "
                    + QString::number(m_currentVideoTemplate.durationSeconds) + "s";
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
    
    // OpenCL Status label
    openclStatusLabel = new QLabel("OpenCL: Checking...", debugWidget);
    openclStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
    debugLayout->addWidget(openclStatusLabel);
    
    // OpenGL Status label
    openglStatusLabel = new QLabel("OpenGL: Checking...", debugWidget);
    openglStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; }");
    debugLayout->addWidget(openglStatusLabel);
    
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
    
            qDebug() << "Debug display setup complete - FPS, GPU, and OpenGL status should be visible";
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
                    qDebug() << "Debug display SHOWN - FPS, GPU, and OpenGL status visible";
                } else {
                    debugWidget->setStyleSheet("QWidget { background-color: rgba(0, 0, 0, 0.8); color: white; border-radius: 5px; }");
                    qDebug() << "Debug display HIDDEN";
                }
            }
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
                m_openclUtilized = false;
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
                
                // Make OpenCL status more prominent
                if (openclStatusLabel) {
                    openclStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 14px; font-weight: bold; }");
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
                        if (openclStatusLabel) {
                            openclStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
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
        qDebug() << "Debug display update #" << updateCount << "FPS:" << m_currentFPS << "GPU:" << m_useGPU << "OpenCL:" << m_useOpenCL;
    }
    
    if (debugLabel) {
        QString peopleDetected = QString::number(m_lastDetections.size());
        QString modeText;
        QString segmentationStatus = m_segmentationEnabledInCapture ? "ENABLED" : "DISABLED";
        
        switch (m_displayMode) {
            case NormalMode:
                modeText = "NORMAL VIEW";
                break;
            case RectangleMode:
                modeText = "ORIGINAL + RECTANGLES";
                break;
            case SegmentationMode:
                modeText = "BLACK BG + EDGE SILHOUETTES";
                break;
        }
        QString debugInfo = QString("FPS: %1 | %2 | People: %3 | Segmentation: %4")
                           .arg(m_currentFPS)
                           .arg(modeText)
                           .arg(peopleDetected)
                           .arg(segmentationStatus);
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
    
    if (openclStatusLabel) {
        QString openclStatus;
        if (m_openclUtilized) {
            openclStatus = "ACTIVE (OpenCL)";
        } else if (m_useOpenCL) {
            openclStatus = "AVAILABLE (OpenCL)";
        } else {
            openclStatus = "OFF (CPU)";
        }
        openclStatusLabel->setText(QString("OpenCL: %1").arg(openclStatus));
        
        // Change color based on utilization
        if (m_openclUtilized) {
            openclStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; font-weight: bold; }");
        } else if (m_useOpenCL) {
            openclStatusLabel->setStyleSheet("QLabel { color: #ff00ff; font-size: 12px; }");
        } else {
            openclStatusLabel->setStyleSheet("QLabel { color: #ff6666; font-size: 12px; }");
        }
    }
    
    if (openglStatusLabel) {
        QString openglStatus;
        if (m_openglUtilized) {
            openglStatus = "ACTIVE (OpenGL)";
        } else if (m_useOpenGL) {
            openglStatus = "AVAILABLE (OpenGL)";
        } else {
            openglStatus = "OFF (CPU)";
        }
        openglStatusLabel->setText(QString("OpenGL: %1").arg(openglStatus));
        
        // Change color based on utilization
        if (m_openglUtilized) {
            openglStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; font-weight: bold; }");
        } else if (m_useOpenGL) {
            openglStatusLabel->setStyleSheet("QLabel { color: #00ff00; font-size: 12px; }");
        } else {
            openglStatusLabel->setStyleSheet("QLabel { color: #ff6666; font-size: 12px; }");
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
    // The scaling is applied to the frame content, not the timing
    m_adjustedRecordingFPS = m_actualCameraFPS;
    
    qDebug() << "Recording with original camera FPS:" << m_adjustedRecordingFPS;
    qDebug() << "  - Scale factor:" << m_personScaleFactor;
    qDebug() << "  - Frame scaling will be applied during capture, not timing";
    
    int frameIntervalMs = qMax(1, static_cast<int>(1000.0 / m_adjustedRecordingFPS));

    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
    qDebug() << "Recording started at adjusted FPS: " + QString::number(m_adjustedRecordingFPS)
                    + " frames/sec (interval: " + QString::number(frameIntervalMs) + "ms)";
    
    // Pre-calculate label size for better performance during recording
    m_cachedLabelSize = ui->videoLabel->size();
}

void Capture::stopRecording()
{
    if (!m_isRecording)
        return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;
    qDebug() << "Recording stopped. Captured " + QString::number(m_recordedFrames.size())
                    + " frames.";

    if (!m_recordedFrames.isEmpty()) {
        qDebug() << "Emitting video with adjusted FPS:" << m_adjustedRecordingFPS << "(base:" << m_actualCameraFPS << ")";
        
        // Emit the adjusted FPS to ensure playback matches the recording rate
        emit videoRecorded(m_recordedFrames, m_adjustedRecordingFPS);
    }
    
    emit showFinalOutputPage();
    qDebug() << "Recording stopped";
}

void Capture::performImageCapture()
{
    // Capture the scaled frame exactly as it appears in the UI (ignoring foreground template)
    if (!m_originalCameraImage.isNull()) {
        QPixmap cameraPixmap = QPixmap::fromImage(m_originalCameraImage);
        QSize labelSize = ui->videoLabel->size();

        // Apply the same scaling logic as the live display
        // First scale to fit the label - use FastTransformation for better performance
        QPixmap scaledPixmap = cameraPixmap.scaled(
            labelSize,
            Qt::KeepAspectRatioByExpanding,
            Qt::FastTransformation
        );

        // Apply person scaling if needed
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            QSize originalSize = scaledPixmap.size();
            int newWidth = qRound(originalSize.width() * m_personScaleFactor);
            int newHeight = qRound(originalSize.height() * m_personScaleFactor);
            
            scaledPixmap = scaledPixmap.scaled(
                newWidth, newHeight,
                Qt::KeepAspectRatio,
                Qt::FastTransformation
            );
        }
        
        m_capturedImage = scaledPixmap;
            emit imageCaptured(m_capturedImage);
        qDebug() << "Image captured (scaled frame exactly as displayed, no foreground compositing).";
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
    qDebug() << "Video template set:" << templateData.name << "Duration:" << templateData.durationSeconds;
    
    // Reset frame counter to ensure smooth initial processing
    frameCount = 0;
    
    // Ensure we start in normal mode to prevent freezing
    if (m_displayMode != NormalMode) {
        m_displayMode = NormalMode;
        qDebug() << "Switched to normal mode to prevent freezing during template transition";
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
    
    // Initialize OpenCL HOG detector for GPU acceleration
    qDebug() << "🎮 ===== STARTING OpenCL HOG INITIALIZATION =====";
    
    // Check if OpenCL is available
    if (cv::ocl::useOpenCL()) {
        try {
            qDebug() << "🎮 Creating OpenCL HOG detector...";
            // Create OpenCL HOG with default people detector
            m_openclHogDetector = cv::makePtr<cv::HOGDescriptor>(
                cv::Size(64, 128),  // win_size
                cv::Size(16, 16),   // block_size
                cv::Size(8, 8),     // block_stride
                cv::Size(8, 8),     // cell_size
                9                   // nbins
            );
            
            if (m_openclHogDetector) {
                qDebug() << "🎮 OpenCL HOG detector created successfully";
                m_openclHogDetector->setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
                qDebug() << "✅ OpenCL HOG detector ready for GPU acceleration";
            } else {
                qWarning() << "⚠️ OpenCL HOG creation failed - detector is empty";
                m_openclHogDetector = nullptr;
            }
        } catch (const cv::Exception& e) {
            qWarning() << "⚠️ OpenCL HOG initialization failed:" << e.what();
            m_openclHogDetector = nullptr;
        }
    } else {
        qDebug() << "⚠️ OpenCL not available for HOG initialization";
        m_openclHogDetector = nullptr;
    }
    qDebug() << "🎮 ===== FINAL OpenCL HOG INITIALIZATION CHECK =====";
    qDebug() << "🎮 OpenCL HOG detector pointer:" << m_openclHogDetector.get();
    qDebug() << "🎮 OpenCL HOG detector empty:" << (!m_openclHogDetector ? "yes" : "no");
    
    if (m_openclHogDetector) {
        qDebug() << "✅ OpenCL HOG detector successfully initialized and ready!";
        m_useOpenCL = true; // Ensure OpenCL is enabled
    } else {
        qWarning() << "⚠️ OpenCL HOG detector initialization failed or not available";
        m_openclHogDetector = nullptr;
    }
    qDebug() << "🎮 ===== OpenCL HOG INITIALIZATION COMPLETE =====";
    
    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
    
    // Initialize AMD GPU verification and OpenCL acceleration
    qDebug() << "🎮 ===== STARTING AMD GPU INITIALIZATION =====";
    
    // Initialize AMD GPU verification
    bool amdGPUAvailable = AMDGPUVerifier::initialize();
    
    if (amdGPUAvailable) {
        qDebug() << "🎮 AMD GPU detected and verified!";
        
        // Check OpenCL availability
        if (cv::ocl::useOpenCL()) {
            cv::ocl::setUseOpenCL(true);
            m_useOpenCL = true;
            m_useGPU = true;
            
            qDebug() << "🎮 OpenCL acceleration enabled for AMD GPU";
            
            // Get GPU info
            AMDGPUVerifier::GPUInfo gpuInfo = AMDGPUVerifier::getGPUInfo();
            qDebug() << "GPU Name:" << gpuInfo.name;
            qDebug() << "GPU Memory:" << gpuInfo.totalMemory / (1024*1024) << "MB";
            qDebug() << "Compute Units:" << gpuInfo.computeUnits;
            
            // Test OpenCL functionality
            if (AMDGPUVerifier::testOpenCLAcceleration()) {
                qDebug() << "✅ OpenCL acceleration test passed!";
                m_openclUtilized = true;
            } else {
                qDebug() << "⚠️ OpenCL acceleration test failed";
                m_openclUtilized = false;
            }
            
            // Test OpenGL functionality
            if (AMDGPUVerifier::testOpenGLAcceleration()) {
                qDebug() << "✅ OpenGL acceleration available";
                m_useOpenGL = true;
                m_openglUtilized = true;
            } else {
                qDebug() << "⚠️ OpenGL acceleration not available";
                m_useOpenGL = false;
                m_openglUtilized = false;
            }
            
        } else {
            qDebug() << "⚠️ OpenCL not available in OpenCV build";
            m_useOpenCL = false;
            m_useGPU = false;
        }
    } else {
        qDebug() << "⚠️ No AMD GPU found, using CPU fallback";
        m_useOpenCL = false;
        m_useGPU = false;
        m_useOpenGL = false;
    }
    
    qDebug() << "🎮 ===== AMD GPU INITIALIZATION COMPLETE =====";
    qDebug() << "OpenCL Available:" << m_useOpenCL;
    qDebug() << "OpenGL Available:" << m_useOpenGL;
    qDebug() << "GPU Available:" << m_useGPU;
    
    // Initialize background subtractor for motion detection (matching peopledetect_v1.cpp)
    m_bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
    
    // Check if OpenCL is available for AMD GPU acceleration
    if (m_useOpenCL) {
        qDebug() << "🎮 OpenCL GPU acceleration enabled for AMD GPU";
        
        // Pre-allocate OpenCL memory pools for better performance
        qDebug() << "🎮 Pre-allocating OpenCL memory pools...";
        try {
            // Pre-allocate common frame sizes for OpenCL operations using UMat
            cv::UMat openclFramePool1, openclFramePool2, openclFramePool3;
            openclFramePool1.create(720, 1280, CV_8UC3);  // Common camera resolution
            openclFramePool2.create(480, 640, CV_8UC3);   // Smaller processing size
            openclFramePool3.create(360, 640, CV_8UC1);   // Grayscale processing
            
            qDebug() << "✅ OpenCL memory pools pre-allocated successfully";
            qDebug() << "  - OpenCL Frame pool 1: 1280x720 (RGB)";
            qDebug() << "  - OpenCL Frame pool 2: 640x480 (RGB)";
            qDebug() << "  - OpenCL Frame pool 3: 640x360 (Grayscale)";
            
        } catch (const cv::Exception& e) {
            qWarning() << "⚠️ OpenCL memory pool allocation failed:" << e.what();
        }
    } else {
        qDebug() << "⚠️ OpenCL not available, using CPU processing";
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
    if (!m_useOpenCL) {
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
    
    // Set OpenCL device for optimal performance
    if (m_useOpenCL) {
        try {
            // Check OpenCL device availability before setting
            if (cv::ocl::haveOpenCL()) {
                cv::ocl::setUseOpenCL(true);
                qDebug() << "OpenCL enabled for optimal performance";
                
                // Test OpenCL memory allocation
                cv::UMat testMat;
                testMat.create(100, 100, CV_8UC3);
                if (testMat.empty()) {
                    throw cv::Exception(0, "OpenCL memory allocation test failed", "", "", 0);
                }
                qDebug() << "OpenCL memory allocation test passed";
            } else {
                qWarning() << "No OpenCL devices available, disabling OpenCL";
                m_useOpenCL = false;
            }
        } catch (const cv::Exception& e) {
            qWarning() << "OpenCL initialization failed:" << e.what() << "Disabling OpenCL";
            m_useOpenCL = false;
        } catch (...) {
            qWarning() << "Unknown OpenCL initialization error, disabling OpenCL";
            m_useOpenCL = false;
        }
    }
    
    qDebug() << "Enhanced Person Detection and Segmentation initialized successfully";
            qDebug() << "GPU Priority: OpenGL (AMD) > OpenCL (AMD) > CPU (fallback)";
}

cv::Mat Capture::processFrameWithUnifiedDetection(const cv::Mat &frame)
{
    // Validate input frame
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
        qWarning() << "Invalid frame received, returning empty result";
        return cv::Mat::zeros(480, 640, CV_8UC3);
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
        qDebug() << "🎯 SEGMENTATION MODE: Creating black background + edge-based silhouettes";
        // Create black background for edge-based segmentation
        cv::Mat segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
        
        for (int i = 0; i < maxDetections; i++) {
            const auto& detection = detections[i];
            qDebug() << "🎯 Processing detection" << i << "at" << detection.x << detection.y << detection.width << "x" << detection.height;
            
            // Get enhanced edge-based segmentation mask for this person
            cv::Mat personMask = enhancedSilhouetteSegment(frame, detection);
            
            // Check if mask has any non-zero pixels
            int nonZeroPixels = cv::countNonZero(personMask);
            qDebug() << "🎯 Person mask has" << nonZeroPixels << "non-zero pixels";
            
            // Apply mask to extract person
            cv::Mat personRegion;
            frame.copyTo(personRegion, personMask);
            
            // Add to segmented frame (black background + edge-based silhouettes)
            cv::add(segmentedFrame, personRegion, segmentedFrame);
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
    
    if (m_useOpenCL) {
        try {
            // Upload ROI to GPU using UMat
            cv::UMat gpu_roi;
            roi.copyTo(gpu_roi);
            
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
            
            qDebug() << "🎮 OpenCL-accelerated edge detection applied";
            
        } catch (const cv::Exception& e) {
            qWarning() << "OpenCL edge detection failed, falling back to CPU:" << e.what();
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU using UMat
                cv::UMat gpu_fgMask;
                fgMask.copyTo(gpu_fgMask);
                
                // Create morphological kernels
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                
                // GPU-accelerated morphological operations using OpenCL
                cv::UMat open_result, close_result, dilate_result;
                cv::morphologyEx(gpu_fgMask, open_result, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(open_result, close_result, cv::MORPH_CLOSE, kernel);
                cv::dilate(close_result, dilate_result, kernel_dilate);
                
                // Download result back to CPU
                dilate_result.copyTo(fgMask);
                
                qDebug() << "🎮 OpenCL-accelerated morphological operations applied";
                
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
        qDebug() << "🎯 Background subtraction found" << validContours.size() << "contours";
    }
    
    // If still no valid contours, try color-based segmentation
    if (validContours.empty()) {
        qDebug() << "🎯 No contours from background subtraction, trying color-based segmentation";
        
        // GPU-accelerated color space conversion and thresholding
        cv::Mat combinedMask;
        
        if (m_useOpenCL) {
            try {
                // Upload ROI to GPU using UMat
                cv::UMat gpu_roi;
                roi.copyTo(gpu_roi);
                
                // Convert to HSV on GPU
                cv::UMat gpu_hsv;
                cv::cvtColor(gpu_roi, gpu_hsv, cv::COLOR_BGR2HSV);
                
                // Create masks for skin-like colors and non-background colors on GPU
                cv::UMat gpu_skinMask, gpu_colorMask;
                cv::inRange(gpu_hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), gpu_skinMask);
                cv::inRange(gpu_hsv, cv::Scalar(0, 30, 50), cv::Scalar(180, 255, 255), gpu_colorMask);
                
                // Combine masks on GPU using bitwise_or
                cv::UMat gpu_combinedMask;
                cv::bitwise_or(gpu_skinMask, gpu_colorMask, gpu_combinedMask);
                
                // Download result back to CPU
                gpu_combinedMask.copyTo(combinedMask);
                
                qDebug() << "🎮 OpenCL-accelerated color segmentation applied";
                
            } catch (const cv::Exception& e) {
                qWarning() << "OpenCL color segmentation failed, falling back to CPU:" << e.what();
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU using UMat
                cv::UMat gpu_combinedMask;
                combinedMask.copyTo(gpu_combinedMask);
                
                // Create morphological kernel
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                
                // GPU-accelerated morphological operations using OpenCL
                cv::UMat open_result, close_result;
                cv::morphologyEx(gpu_combinedMask, open_result, cv::MORPH_OPEN, kernel);
                cv::morphologyEx(open_result, close_result, cv::MORPH_CLOSE, kernel);
                
                // Download result back to CPU
                close_result.copyTo(combinedMask);
                
                qDebug() << "🎮 OpenCL-accelerated color morphological operations applied";
                
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
        if (m_useOpenCL) {
            try {
                // Upload mask to GPU using UMat
                cv::UMat gpu_roiMask;
                roiMask.copyTo(gpu_roiMask);
                
                // Create morphological kernels
                cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
                
                // GPU-accelerated morphological operations using OpenCL
                cv::UMat close_result, dilate_result;
                cv::morphologyEx(gpu_roiMask, close_result, cv::MORPH_CLOSE, kernel_clean);
                cv::dilate(close_result, dilate_result, kernel_dilate);
                
                // Download result back to CPU
                dilate_result.copyTo(roiMask);
                
                qDebug() << "🎮 OpenCL-accelerated final morphological cleanup applied";
                
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
    

    
    if (m_useOpenCL) {
        // OpenCL GPU-accelerated processing for AMD GPU
        m_gpuUtilized = true;
        m_openclUtilized = true;
        
        try {
            // Validate frame before OpenCL operations
            if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
                throw cv::Exception(0, "Invalid frame for OpenCL processing", "", "", 0);
            }
            
            // Validate input frame dimensions before GPU operations
            if (frame.rows <= 0 || frame.cols <= 0) {
                qWarning() << "🎮 Invalid frame dimensions for OpenCL processing:" << frame.rows << "x" << frame.cols;
                found.clear();
                m_openclUtilized = false;
                return found;
            }
            
            // Upload to OpenCL GPU using UMat
            cv::UMat gpu_frame;
            frame.copyTo(gpu_frame);
            
            // Validate uploaded GPU frame
            if (gpu_frame.empty() || gpu_frame.rows <= 0 || gpu_frame.cols <= 0) {
                qWarning() << "🎮 Invalid GPU frame dimensions after upload:" << gpu_frame.rows << "x" << gpu_frame.cols;
                found.clear();
                m_openclUtilized = false;
                return found;
            }
            
            // Convert to grayscale on GPU
            cv::UMat gpu_gray;
            cv::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
            
            // Validate grayscale GPU frame
            if (gpu_gray.empty() || gpu_gray.rows <= 0 || gpu_gray.cols <= 0) {
                qWarning() << "🎮 Invalid grayscale GPU frame dimensions:" << gpu_gray.rows << "x" << gpu_gray.cols;
                found.clear();
                m_openclUtilized = false;
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
                m_openclUtilized = false;
                return found;
            }
            
            qDebug() << "🎮 Resizing GPU matrix to:" << new_width << "x" << new_height;
            
            // Resize for optimal GPU performance (ensure minimum size for HOG)
            cv::UMat gpu_resized;
            cv::resize(gpu_gray, gpu_resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
            
            // Validate resized GPU matrix
            if (gpu_resized.empty() || gpu_resized.rows <= 0 || gpu_resized.cols <= 0) {
                qWarning() << "🎮 Invalid resized GPU matrix dimensions:" << gpu_resized.rows << "x" << gpu_resized.cols;
                found.clear();
                m_openclUtilized = false;
                return found;
            }
            
            qDebug() << "🎮 Resized GPU matrix validated - size:" << gpu_resized.rows << "x" << gpu_resized.cols;
            
                         // OpenCL HOG detection
             if (m_openclHogDetector && m_useOpenCL) {
                try {
                    std::vector<cv::Rect> found_opencl;
                    
                    // Convert UMat to Mat for HOG detection
                    cv::Mat cpu_resized;
                    gpu_resized.copyTo(cpu_resized);
                    
                    // Simple OpenCL HOG detection (working state)
                    m_openclHogDetector->detectMultiScale(cpu_resized, found_opencl);
                    
                    if (!found_opencl.empty()) {
                        found = found_opencl;
                        m_openclUtilized = true;
                        qDebug() << "🎮 OpenCL HOG detection SUCCESS - detected" << found_opencl.size() << "people";
                    } else {
                        qDebug() << "🎮 OpenCL HOG completed but no people detected";
                        found.clear();
                    }
                    
                } catch (const cv::Exception& e) {
                    qDebug() << "🎮 OpenCL HOG error:" << e.what() << "falling back to CPU";
                    m_openclUtilized = false;
                    
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
                // OpenCL HOG not available - check why
                if (!m_useOpenCL) {
                    qDebug() << "🎮 OpenCL not enabled, skipping OpenCL HOG";
                } else if (!m_openclHogDetector) {
                    qDebug() << "🎮 OpenCL HOG detector not initialized";
                } else if (!m_openclHogDetector) {
                    qDebug() << "🎮 OpenCL HOG detector is null";
                }
                found.clear();
                m_openclUtilized = false;
            }
            
            // Scale results back up to original size (OpenGL HOG works on resized image)
            double scale_factor = 1.0 / 0.5; // 1/0.5 = 2.0 (matching peopledetect_v1.cpp)
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
            
            qDebug() << "🎮 OpenCL GPU: Color conversion + resize + HOG detection (GPU acceleration with CPU coordination)";
            
        } catch (const cv::Exception& e) {
            qWarning() << "🎮 OpenCL processing error:" << e.what() << "falling back to CPU";
            m_openclUtilized = false; // Switch to CPU
            
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
            qWarning() << "🎮 Unknown OpenCL error, falling back to CPU";
            m_openclUtilized = false; // Switch to CPU
            
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
        m_openclUtilized = false;
        
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
        m_openclUtilized = false;
    
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
                m_lastSegmentedFrame = result.clone();
                
                // Update GPU utilization status
                if (m_useOpenCL) {
                    m_openclUtilized = true;
                    m_gpuUtilized = false;
                } else if (m_useGPU) {
                    m_gpuUtilized = true;
                    m_openclUtilized = false;
                }
                
                qDebug() << "✅ Person detection processing completed - segmented frame updated, size:" << result.cols << "x" << result.rows;
            } else {
                qDebug() << "⚠️ Person detection processing completed but result is empty";
            }
        } catch (const std::exception& e) {
            qWarning() << "Exception in person detection finished callback:" << e.what();
        }
    } else {
        qDebug() << "⚠️ Person detection watcher is not finished or null";
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

bool Capture::isOpenCLAvailable() const
{
    return m_useOpenCL;
}

cv::Mat Capture::getMotionMask(const cv::Mat &frame)
{
    cv::Mat fgMask;
    
    if (m_useOpenCL) {
        // OpenCL-accelerated background subtraction using UMat
        try {
            // Upload to GPU using UMat
            cv::UMat gpu_frame;
            frame.copyTo(gpu_frame);
            
            // Create OpenCL background subtractor if not already created
            static cv::Ptr<cv::BackgroundSubtractorMOG2> opencl_bg_subtractor;
            if (opencl_bg_subtractor.empty()) {
                opencl_bg_subtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
            }
            
            // OpenCL-accelerated background subtraction
            cv::UMat gpu_fgmask;
            opencl_bg_subtractor->apply(gpu_frame, gpu_fgmask, -1);
            
            // Download result to CPU
            gpu_fgmask.copyTo(fgMask);
            
            // Apply morphological operations on CPU (OpenCV OpenCL limitation)
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
    m_openclUtilized = false;
    
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
            m_openclUtilized = false;
        }
        
        this->updatePersonDetectionButton();
        this->updateDebugDisplay();
        
        qDebug() << "🎯 Segmentation mode set to:" << displayMode;
    } else {
        qDebug() << "🎯 Cannot set segmentation mode - segmentation not enabled in capture interface";
    }
}





