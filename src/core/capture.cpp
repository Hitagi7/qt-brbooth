/*
 * Capture.cpp - Enhanced Person Detection with GPU Acceleration
 * 
 * OPTIMIZATIONS APPLIED:
 * 1. Fixed CUDA HOG stride alignment (win_stride must be divisible by block_stride)
 * 2. Optimized resize dimensions (0.6x scale for better accuracy vs performance)
 * 3. Improved HOG parameters for better detection sensitivity
 * 4. Added robust fallback from CUDA to CPU when errors occur
 * 5. Enhanced detection logging for debugging
 * 6. Enabled person detection by default for immediate testing
 * 
 * GPU Priority: CUDA (NVIDIA) > OpenCL (AMD) > CPU (fallback)
 * Target: 30 FPS with accurate people detection
 */

#include "core/capture.h"
#include "core/camera.h"
#include "ui/foreground.h"
#include "ui_capture.h"
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
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudabgsegm.hpp> // Added for cv::cuda::BackgroundSubtractorMOG2
#include <QtConcurrent/QtConcurrent>
#include <QMutexLocker>

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
    , cameraTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_actualCameraFPS(30.0)  // Default to 30 FPS
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)
    , m_recordedFrames()
    , m_capturedImage()
    , stackedLayout(nullptr)
    , loadingCameraLabel(nullptr)
    , videoLabelFPS(nullptr)
    , loopTimer()
    , totalTime(0)
    , frameCount(0)
    , frameTimer()
    , overlayImageLabel(nullptr)
    , m_personScaleFactor(1.0) // Initialize to 1.0 (normal size) - matches slider at 0
    // Unified Person Detection and Segmentation
    , m_displayMode(SegmentationMode)  // Start with segmentation mode by default
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
    , m_personDetectionWatcher(nullptr)
    , m_lastDetections()
    // , m_tfliteModelLoaded(false)
    , debugWidget(nullptr)
    , debugLabel(nullptr)
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
    // Temporarily disabled async processing and hand detection members
    // , m_segmentationWatcher(nullptr)
    // , m_processingAsync(false)
    // , m_asyncMutex()
    // , m_lastProcessedFrame()
    // , m_handDetector(new AdvancedHandDetector())
    // , m_showHandDetection(true)
    // , m_handDetectionMutex()
    // , m_handDetectionTimer()
    // , m_lastHandDetectionTime(0.0)
    // , m_handDetectionFPS(0)
    // , m_handDetectionEnabled(false)
    , m_cachedPixmap(640, 480)
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

    loadingCameraLabel = new QLabel("Loading Camera...", this);
    loadingCameraLabel->setAlignment(Qt::AlignCenter);
    QFont loadingFont = loadingCameraLabel->font();
    loadingFont.setPointSize(36);
    loadingFont.setBold(true);
    loadingCameraLabel->setFont(loadingFont);
    loadingCameraLabel->setStyleSheet(
        "color: white; "
        "background-color: rgba(0, 0, 0, 150); "
        "border-radius: 15px; "
        "padding: 10px 20px; "
    );
    loadingCameraLabel->setFixedSize(450, 120);
    loadingCameraLabel->show();

    ui->videoLabel->hide();

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
        connect(cameraWorker, &Camera::frameReady, this, &Capture::updateCameraFeed);
        connect(cameraWorker, &Camera::cameraOpened, this, &Capture::handleCameraOpened);
        connect(cameraWorker, &Camera::error, this, &Capture::handleCameraError);
    } else {
        qCritical() << "Capture: ERROR: cameraWorker is NULL! Camera features will not function.";
        loadingCameraLabel->hide();
        ui->videoLabel->show();
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText(
            "Camera worker not provided or is NULL.\nCannot initialize camera.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }

    // Initialize Enhanced Person Detection and Segmentation
    initializePersonDetection();
    
    // Temporarily disabled Hand Detection initialization
    // initializeHandDetection();
    // m_handDetectionEnabled = false;
    // Initialize MediaPipe-like tracker
    // m_handTracker.initialize(640, 480);

    // Temporarily disabled segmentation thread and connections
    // // Run segmentation in a background thread so UI stays responsive
    // m_segmentationThread = new QThread(this);
    // m_tfliteSegmentation->moveToThread(m_segmentationThread);
    // connect(m_segmentationThread, &QThread::started, m_tfliteSegmentation, &TFLiteDeepLabv3::startRealtimeProcessing);
    // connect(m_segmentationThread, &QThread::finished, m_tfliteSegmentation, &TFLiteDeepLabv3::stopRealtimeProcessing);
    // m_segmentationThread->start();

    // Temporarily disabled TFLite signal connections
    // // Connect TFLite signals
    // connect(m_tfliteSegmentation, &TFLiteDeepLabv3::segmentationResultReady,
    //         this, &Capture::onSegmentationResultReady);
    // connect(m_tfliteSegmentation, &TFLiteDeepLabv3::processingError,
    //         this, &Capture::onSegmentationError);
    // connect(m_tfliteSegmentation, &TFLiteDeepLabv3::modelLoaded,
    //         this, &Capture::onTFLiteModelLoaded);

    // Temporarily disabled segmentation widget signal connections
    // // Connect segmentation widget signals
    // connect(m_segmentationWidget, &TFLiteSegmentationWidget::showSegmentationChanged,
    //         this, &Capture::setShowSegmentation);
    // connect(m_segmentationWidget, &TFLiteSegmentationWidget::confidenceThresholdChanged,
    //         this, &Capture::setSegmentationConfidenceThreshold);
    // connect(m_segmentationWidget, &TFLiteSegmentationWidget::performanceModeChanged,
    //         this, &Capture::setSegmentationPerformanceMode);

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

    // Initialize and start performance timers
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

    qDebug() << "Capture UI initialized. Loading Camera...";
}

Capture::~Capture()
{
    // Stop and delete QTimers owned by Capture
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; countdownTimer = nullptr; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; recordTimer = nullptr; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; recordingFrameTimer = nullptr; }
    
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
    // loadingCameraLabel has 'this' as parent, so it *will* be deleted by Qt's object tree.
    // However, explicitly nulling out the pointer is good practice.
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; }
    if (statusOverlay){ delete statusOverlay; statusOverlay = nullptr; }
    if (loadingCameraLabel){ delete loadingCameraLabel; loadingCameraLabel = nullptr; }
    if (videoLabelFPS){ delete videoLabelFPS; videoLabelFPS = nullptr; } // Only if you actually 'new' this somewhere



    // DO NOT DELETE cameraWorker or cameraThread here.
    // They are passed in as existing objects, implying Capture does not own them.
    // If Capture *were* responsible for stopping and deleting the camera thread,
    // that logic would be handled by the class that *owns* the cameraThread and cameraWorker.
    // For Capture, we just null out the pointers to prevent dangling pointers in case
    // some other part of Capture's code tries to use them after they've been destroyed externally.
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
        // Ensure that the parent widget of loadingCameraLabel (centeringWidget) is hidden
        if (loadingCameraLabel->parentWidget()) {
            loadingCameraLabel->parentWidget()->hide();
        } else {
            loadingCameraLabel->hide(); // Fallback
        }
        ui->videoLabel->show();
    } else {
        qWarning() << "Capture: Camera worker reported open failure.";
        if (loadingCameraLabel->parentWidget()) {
            loadingCameraLabel->parentWidget()->hide();
        } else {
            loadingCameraLabel->hide(); // Fallback
        }
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
    if (loadingCameraLabel->parentWidget()) {
        loadingCameraLabel->parentWidget()->hide();
    } else {
        loadingCameraLabel->hide();
    }
    ui->videoLabel->show();
    ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
    ui->videoLabel->setText(QString("Error: %1").arg(msg));
    ui->videoLabel->setAlignment(Qt::AlignCenter);
}

void Capture::updateCameraFeed(const QImage &image)
{
    // Start loopTimer at the very beginning of the function to measure total time for one update cycle.
    loopTimer.start(); // Measure time for this entire call

    if (image.isNull()) {
        qWarning() << "Capture: Received null QImage from Camera.";
        // Performance stats should still be calculated for every attempt to process a frame
        qint64 currentLoopTime = loopTimer.elapsed();
        totalTime += currentLoopTime;
        frameCount++;
        if (frameCount % 60 == 0) {
            printPerformanceStats();
        }
        return;
    }

    // Hide the centeringWidget containing loadingCameraLabel when frames arrive
    bool cameraStarted = false;
    
    if (loadingCameraLabel->parentWidget() && !loadingCameraLabel->parentWidget()->isHidden()) {
        loadingCameraLabel->parentWidget()->hide();
        ui->videoLabel->show();
        cameraStarted = true;
    } else if (!loadingCameraLabel->isHidden()) {
        loadingCameraLabel->hide();
        ui->videoLabel->show();
        cameraStarted = true;
    }
    
    // Show initial performance stats when camera first starts (only once)
    if (cameraStarted) {
        qDebug() << "----------------------------------------";
        qDebug() << "=== CAMERA STARTED - PERFORMANCE MONITORING ACTIVE ===";
        qDebug() << "Performance stats will be displayed automatically every 60 frames";
        qDebug() << "Debug widget can be toggled with 'D' key";
        qDebug() << "----------------------------------------";
    }

    // Store the original image for potential future use
    m_originalCameraImage = image;

    // Convert QImage to cv::Mat for processing
    cv::Mat frame = qImageToCvMat(image);
    if (frame.empty()) {
        qWarning() << "Failed to convert QImage to cv::Mat";
        return;
    }

    // Temporarily disabled frame storage
    // m_currentFrame = frame.clone();

    cv::Mat displayFrame = frame.clone();
    // Temporarily disabled frame processing flag
    // bool frameProcessed = false;

    // Temporarily disabled hand detection processing
    // if (m_handDetectionEnabled) {
    //     // Update MediaPipe-like tracker first
    //     m_handTracker.update(frame);
    //     if (m_handTracker.shouldTriggerCapture()) {
    //         startCountdown();
    //     }
    //     // Keep existing detector as secondary signal (can be disabled later)
    //     if (m_handDetector && m_handDetector->isInitialized()) {
    //         qDebug() << "🖐️ Processing hand detection - ENABLED";
    //         processFrameWithHandDetection(displayFrame);
    //         frameProcessed = true;
    //     }
    // } else {
    //     qDebug() << "🖐️ Hand detection DISABLED - skipping processing";
    // }

    // Unified Person Detection and Segmentation processing
    if (m_displayMode == RectangleMode || m_displayMode == SegmentationMode) {
        // Process every 3rd frame for maximum GPU efficiency (20 FPS target) - minimal CPU usage
        if (frameCount % 3 == 0) {
            QMutexLocker locker(&m_personDetectionMutex);
            m_currentFrame = frame.clone();
            
            // Process unified detection in background thread
            QFuture<cv::Mat> future = QtConcurrent::run([this]() {
                return processFrameWithUnifiedDetection(m_currentFrame);
            });
            m_personDetectionWatcher->setFuture(future);
        }
        
        // Display based on current mode
        QPixmap pixmap;
        cv::Mat processedFrame;
        
        if (m_displayMode == SegmentationMode) {
            // Display the segmented frame (black background with edge-based silhouettes)
            {
                QMutexLocker segLocker(&m_personDetectionMutex);
                if (!m_lastSegmentedFrame.empty()) {
                    processedFrame = m_lastSegmentedFrame.clone();
                    qDebug() << "✅ Using edge-based segmentation frame, size:" << processedFrame.size().width << "x" << processedFrame.size().height;
                } else {
                    qDebug() << "⚠️ No segmentation frame available, using original";
                    processedFrame = displayFrame.clone();
                }
            }
        } else {
            // Rectangle mode - use original frame (rectangles will be drawn in createSegmentedFrame)
            {
                QMutexLocker segLocker(&m_personDetectionMutex);
                if (!m_lastSegmentedFrame.empty()) {
                    processedFrame = m_lastSegmentedFrame.clone();
                    qDebug() << "✅ Using rectangle detection frame, size:" << processedFrame.size().width << "x" << processedFrame.size().height;
                } else {
                    qDebug() << "⚠️ No detection frame available, using original";
                    processedFrame = displayFrame.clone();
                }
            }
        }
        
        if (!processedFrame.empty()) {
            QImage qImage = cvMatToQImage(processedFrame);
            pixmap = QPixmap::fromImage(qImage);
        } else {
            QImage qImage = cvMatToQImage(displayFrame);
            pixmap = QPixmap::fromImage(qImage);
        }
        
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
    } else {
        // Display original frame when person detection is disabled
        QImage qImage = cvMatToQImage(displayFrame);
        QPixmap pixmap = QPixmap::fromImage(qImage);
        
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
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Calculate current FPS for real-time display
    static QElapsedTimer fpsTimer;
    static int fpsFrameCount = 0;
    if (fpsFrameCount == 0) {
        fpsTimer.start();
    }
    fpsFrameCount++;
    
    if (fpsFrameCount >= 60) { // Update FPS every 60 frames (1 second at 60 FPS)
        double fpsDuration = fpsTimer.elapsed() / 1000.0;
        m_currentFPS = fpsDuration > 0 ? fpsFrameCount / fpsDuration : 0;
        fpsFrameCount = 0;
        fpsTimer.start();
    }

    // Print performance stats every 60 frames (aligned with 60 FPS target)
    if (frameCount % 60 == 0) {
        printPerformanceStats();
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

        // --- FIX for Centering "Loading Camera" ---
        // Create a layout to center the fixed-size loadingCameraLabel
        QVBoxLayout *centeringLayout = new QVBoxLayout();
        centeringLayout->addStretch(); // Top stretch
        QHBoxLayout *hCenteringLayout = new QHBoxLayout();
        hCenteringLayout->addStretch();                  // Left stretch
        hCenteringLayout->addWidget(loadingCameraLabel); // Add the fixed-size label
        hCenteringLayout->addStretch();                  // Right stretch
        centeringLayout->addLayout(hCenteringLayout);    // Add horizontal layout to vertical
        centeringLayout->addStretch();                   // Bottom stretch

        QWidget *centeringWidget = new QWidget(this); // Create a wrapper widget
        centeringWidget->setLayout(centeringLayout);
        centeringWidget->setContentsMargins(0, 0, 0, 0);             // Ensure no extra margins
        centeringWidget->setAttribute(Qt::WA_TranslucentBackground); // Keep background transparent
        // Initially show the centeringWidget, as loadingCameraLabel is shown by default
        centeringWidget->show();

        stackedLayout->addWidget(centeringWidget); // Layer 1: Loading text (now centered)
        // ------------------------------------------

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

    // Now, instead of raising loadingCameraLabel, raise its parent centeringWidget
    if (loadingCameraLabel->parentWidget()) {
        loadingCameraLabel->parentWidget()->raise();
    } else {
        loadingCameraLabel->raise(); // Fallback, though should be within centeringWidget now
    }

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
    
    // Reset timers for next batch
    frameCount = 0;
    totalTime = 0;
    frameTimer.start(); // Restart frameTimer for the next measurement period
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

    ui->capture->setEnabled(false);
    countdownValue = 5;
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();
    
    // Add fade-in animation for the countdown label
    QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
    animation->setDuration(300);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->start();

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
            ui->capture->setEnabled(true);
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

// Temporarily disabled TFLite segmentation initialization
/*
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
}
*/

// Temporarily disabled hand detection initialization
/*
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
*/

// Temporarily disabled hand detection methods
/*
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
*/

// Temporarily disabled segmentation methods
/*
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
*/

// Temporarily disabled segmentation methods
/*
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
*/

// Temporarily disabled hand detection methods
/*
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

/*
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
*/

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
        case Qt::Key_S:
            // Three-way toggle: Normal -> Rectangles -> Segmentation -> Normal
            switch (m_displayMode) {
                case NormalMode:
                    m_displayMode = RectangleMode;
                    qDebug() << "Switched to RECTANGLE MODE (Original frame + Green rectangles)";
                    break;
                case RectangleMode:
                    m_displayMode = SegmentationMode;
                    qDebug() << "Switched to SEGMENTATION MODE (Black background + Edge-based silhouettes)";
                    break;
                case SegmentationMode:
                    m_displayMode = NormalMode;
                    qDebug() << "Switched to NORMAL MODE (Original camera view)";
                    break;
            }
            
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
    qDebug() << "Capture widget shown - algorithms disabled";
    
    // Start camera timer if not already active
    if (cameraTimer && !cameraTimer->isActive()) {
        cameraTimer->start(16); // ~60 FPS - ULTRA-LIGHT processing
    }
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    qDebug() << "Capture widget hidden - algorithms disabled";
    
    // Stop camera timer when page is hidden
    if (cameraTimer && cameraTimer->isActive()) {
        cameraTimer->stop();
    }
}

// Temporarily disabled hand detection processing methods
/*
void Capture::processFrameWithHandDetection(const cv::Mat &frame)
{
    if (!m_handDetector || !m_handDetector->isInitialized()) {
        return;
    }
    
    m_handDetectionTimer.start();
    
    try {
        // Detect hands in the frame
        // QList<AdvancedHandDetection> detections = m_handDetector->detect(frame);
        
        // Store the detections
        // {
        //     QMutexLocker locker(&m_handDetectionMutex);
        //     m_lastHandDetections = detections;
        // }
        
        // Update timing info
        // m_lastHandDetectionTime = m_handDetectionTimer.elapsed() / 1000.0;
        // m_handDetectionFPS = (m_lastHandDetectionTime > 0) ? 1.0 / m_lastHandDetectionTime : 0;
        
        // Apply hand detection visualization to the frame
        // cv::Mat displayFrame = frame.clone();
        // drawHandBoundingBoxes(displayFrame, detections);
        
    } catch (const std::exception& e) {
        qWarning() << "Exception in hand detection:" << e.what();
    }
}
*/

// Temporarily disabled hand detection processing methods
// Function drawHandBoundingBoxes removed to avoid compilation errors

// Temporarily disabled segmentation processing methods
/*
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
    } 
    else {
        qWarning() << "Failed to load TFLite model";
        QMessageBox::warning(this, "Error", "Failed to load TFLite Deeplabv3 model.");
    }
}

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
*/

// Temporarily disabled static function for async processing
/*
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
*/

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
        QString debugInfo = QString("FPS: %1 | %2 | People: %3")
                           .arg(m_currentFPS)
                           .arg(modeText)
                           .arg(peopleDetected);
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
    
    // if (handDetectionLabel) {
    //     QString handStatus = m_showHandDetection ? "ON (Tracking)" : "OFF";
    //     QString handTime = QString::number(m_lastHandDetectionTime * 1000, 'f', 1);
    //     handDetectionLabel->setText(QString("Hand Detection: %1 (%2ms)").arg(handStatus).arg(handTime));
    // }
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
        cv::Mat segmentedFrame = createSegmentedFrame(frame, motionFiltered);
        
        // Update timing info
        m_lastPersonDetectionTime = m_personDetectionTimer.elapsed() / 1000.0;
        m_personDetectionFPS = (m_lastPersonDetectionTime > 0) ? 1.0 / m_lastPersonDetectionTime : 0;
        
        // Log people detection for visibility
        if (motionFiltered.size() > 0) {
            qDebug() << "🎯 PEOPLE DETECTED:" << motionFiltered.size() << "person(s) in frame (motion filtered from" << found.size() << "detections)";
            qDebug() << "🎯 Detection details:";
            for (size_t i = 0; i < motionFiltered.size(); i++) {
                qDebug() << "🎯 Person" << i << "at" << motionFiltered[i].x << motionFiltered[i].y << motionFiltered[i].width << "x" << motionFiltered[i].height;
            }
        } else {
            qDebug() << "⚠️ NO PEOPLE DETECTED in frame (total detections:" << found.size() << ")";
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
        // Create black background for edge-based segmentation
        cv::Mat segmentedFrame = cv::Mat::zeros(frame.size(), frame.type());
        
        for (int i = 0; i < maxDetections; i++) {
            const auto& detection = detections[i];
            
            // Get enhanced edge-based segmentation mask for this person
            cv::Mat personMask = enhancedSilhouetteSegment(frame, detection);
            
            // Apply mask to extract person
            cv::Mat personRegion;
            frame.copyTo(personRegion, personMask);
            
            // Add to segmented frame (black background + edge-based silhouettes)
            cv::add(segmentedFrame, personRegion, segmentedFrame);
        }
        
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
    // Person-focused silhouette segmentation with CUDA-accelerated edge detection
    // Validate detection rectangle
    if (detection.x < 0 || detection.y < 0 || 
        detection.width <= 0 || detection.height <= 0 ||
        detection.x + detection.width > frame.cols ||
        detection.y + detection.height > frame.rows) {
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }
    
    // Create tight rectangle around the person (minimal expansion)
    cv::Rect expandedRect = detection;
    expandedRect.x = std::max(0, expandedRect.x - 10);
    expandedRect.y = std::max(0, expandedRect.y - 10);
    expandedRect.width = std::min(frame.cols - expandedRect.x, expandedRect.width + 20);
    expandedRect.height = std::min(frame.rows - expandedRect.y, expandedRect.height + 20);
    
    // Validate expanded rectangle
    if (expandedRect.width <= 0 || expandedRect.height <= 0) {
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }
    
    // Create ROI for silhouette extraction
    cv::Mat roi = frame(expandedRect);
    cv::Mat roiMask = cv::Mat::zeros(roi.size(), CV_8UC1);
    
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    
    // CUDA-accelerated edge detection if available
    cv::Mat edges;
    if (m_useCUDA) {
        try {
            // Upload to GPU for CUDA-accelerated edge detection
            cv::cuda::GpuMat gpu_blurred;
            gpu_blurred.upload(blurred);
            
            // Create CUDA Canny edge detector
            cv::Ptr<cv::cuda::CannyEdgeDetector> canny_detector = cv::cuda::createCannyEdgeDetector(50, 150);
            
            // Apply Canny edge detection on GPU
            cv::cuda::GpuMat gpu_edges;
            canny_detector->detect(gpu_blurred, gpu_edges);
            
            // Download result back to CPU
            gpu_edges.download(edges);
            
            qDebug() << "🎮 CUDA-accelerated edge detection applied";
        } catch (const cv::Exception& e) {
            qWarning() << "CUDA edge detection failed, falling back to CPU:" << e.what();
            // Fallback to CPU edge detection
            cv::Canny(blurred, edges, 50, 150);
        }
    } else {
        // CPU edge detection
        cv::Canny(blurred, edges, 50, 150);
    }
    
    // Dilate edges to connect broken contours
    cv::Mat kernel_edge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::dilate(edges, edges, kernel_edge);
    
    // Find contours from edges
    std::vector<std::vector<cv::Point>> edgeContours;
    cv::findContours(edges, edgeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter contours based on person-like characteristics
    std::vector<std::vector<cv::Point>> validContours;
    cv::Point detectionCenter(expandedRect.width/2, expandedRect.height/2);
    
    for (const auto& contour : edgeContours) {
        double area = cv::contourArea(contour);
        
        // Only consider contours with reasonable size for a person
        if (area > 100 && area < expandedRect.width * expandedRect.height * 0.8) {
            // Get bounding rectangle
            cv::Rect contourRect = cv::boundingRect(contour);
            
            // Check if contour is centered in the detection area
            cv::Point contourCenter(contourRect.x + contourRect.width/2, contourRect.y + contourRect.height/2);
            double distance = cv::norm(contourCenter - detectionCenter);
            double maxDistance = std::min(expandedRect.width, expandedRect.height) * 0.6;
            
            // Check aspect ratio (person should be taller than wide)
            double aspectRatio = (double)contourRect.height / contourRect.width;
            
            if (distance < maxDistance && aspectRatio > 1.2) {
                validContours.push_back(contour);
            }
        }
    }
    
    // If no valid edge contours found, try alternative approach
    if (validContours.empty()) {
        // Use gradient-based segmentation
        cv::Mat gradX, gradY, gradient;
        cv::Sobel(blurred, gradX, CV_16S, 1, 0, 3);
        cv::Sobel(blurred, gradY, CV_16S, 0, 1, 3);
        
        // Convert to magnitude
        cv::convertScaleAbs(gradX, gradX);
        cv::convertScaleAbs(gradY, gradY);
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, gradient);
        
        // Threshold gradient
        cv::Mat gradientMask;
        cv::threshold(gradient, gradientMask, 30, 255, cv::THRESH_BINARY);
        
        // Find contours from gradient
        cv::findContours(gradientMask, validContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }
    
    // Create mask from valid contours
    if (!validContours.empty()) {
        // Sort contours by area
        std::sort(validContours.begin(), validContours.end(), 
                 [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                     return cv::contourArea(a) > cv::contourArea(b);
                 });
        
        // Use the largest valid contour
        cv::drawContours(roiMask, validContours, 0, cv::Scalar(255), -1);
        
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
        
        // Apply final morphological cleanup
        cv::Mat kernel_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(roiMask, roiMask, cv::MORPH_CLOSE, kernel_clean);
    }
    
    // Create final mask for the entire frame
    cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    roiMask.copyTo(finalMask(expandedRect));
    
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
            
            // Calculate resize dimensions for optimal detection accuracy
            int new_width = cvRound(gpu_gray.cols * 0.6); // Increased for better accuracy
            int new_height = cvRound(gpu_gray.rows * 0.6); // Increased for better accuracy
            
            // Ensure minimum dimensions for HOG detection (HOG needs at least 64x128)
            new_width = std::max(new_width, 160); // Increased minimum for better detection
            new_height = std::max(new_height, 320); // Increased minimum for better detection
            
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
                    
                    // Fallback to CPU HOG detection (working state)
                    cv::Mat resized;
                    cv::resize(frame, resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
                    m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
                    
                    // Scale results back up to original size
                    double scale_factor = 1.0 / 0.6;
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
            double scale_factor = 1.0 / 0.6; // 1/0.6 = 1.67
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
            
            // Fallback to CPU HOG detection (working state)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
            
            // Scale results back up to original size
            double scale_factor = 1.0 / 0.6; // 1/0.6 = 1.67
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
        } catch (...) {
            qWarning() << "🎮 Unknown CUDA error, falling back to CPU";
            m_cudaUtilized = false; // Switch to CPU
            
            // Fallback to CPU HOG detection (working state)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
            
            // Scale results back up to original size
            double scale_factor = 1.0 / 0.6; // 1/0.6 = 1.67
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
            
            // Resize for optimal GPU performance
            cv::UMat gpu_resized;
            cv::resize(gpu_gray, gpu_resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
            
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
            
            // Fallback to CPU processing (working state)
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
            m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
            
            // Scale results back up to original size
            double scale_factor = 1.0 / 0.6; // 1/0.6 = 1.67
            for (auto& rect : found) {
                rect.x = cvRound(rect.x * scale_factor);
                rect.y = cvRound(rect.y * scale_factor);
                rect.width = cvRound(rect.width * scale_factor);
                rect.height = cvRound(rect.height * scale_factor);
            }
        }
        
    } else {
            // CPU fallback (working state)
    m_gpuUtilized = false;
    m_cudaUtilized = false;
    
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), 0.6, 0.6, cv::INTER_LINEAR);
        
        // Run detection with balanced speed/accuracy for 30 FPS
        m_hogDetector.detectMultiScale(resized, found, 0.0, cv::Size(8,8), cv::Size(), 1.05, 2, false);
        
        // Scale results back up to original size
        double scale_factor = 1.0 / 0.6; // 1/0.6 = 1.67
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
            }
        } catch (const std::exception& e) {
            qWarning() << "Exception in person detection finished callback:" << e.what();
        }
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
    // Cycle through modes: Normal -> Rectangle -> Segmentation -> Normal
    switch (m_displayMode) {
        case NormalMode:
            m_displayMode = RectangleMode;
            break;
        case RectangleMode:
            m_displayMode = SegmentationMode;
            break;
        case SegmentationMode:
            m_displayMode = NormalMode;
            break;
    }
    
    // Reset utilization when switching to normal mode
    if (m_displayMode == NormalMode) {
        m_gpuUtilized = false;
        m_cudaUtilized = false;
    }
    
    updatePersonDetectionButton();
    qDebug() << "Person detection toggled to mode:" << m_displayMode;
}

void Capture::updatePersonDetectionButton()
{
    if (personDetectionButton) {
        QString buttonText;
        QString buttonStyle;
        
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






