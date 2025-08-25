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
#include <QMessageBox>
#include <QDateTime>
#include <QStackedLayout>
#include <QThread>
#include <QFileInfo>
#include <opencv2/opencv.hpp>
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
    , m_tfliteSegmentation(nullptr)
    , m_segmentationWidget(nullptr)
    , m_showSegmentation(true)
    , m_segmentationConfidenceThreshold(0.5)
    , m_currentFrame()
    , m_lastSegmentedFrame()
    , m_segmentationMutex()
    , m_segmentationTimer()
    , m_lastSegmentationTime(0.0)
    , m_segmentationFPS(0)
    , m_tfliteModelLoaded(false)
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , fpsLabel(nullptr)
    , segmentationLabel(nullptr)
    , segmentationButton(nullptr)
    , handDetectionLabel(nullptr)
    , handDetectionButton(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_segmentationWatcher(nullptr)
    , m_processingAsync(false)
    , m_asyncMutex()
    , m_lastProcessedFrame()
    , m_handDetector(new HandDetector())
    , m_showHandDetection(true)
    , m_handDetectionMutex()
    , m_handDetectionTimer()
    , m_lastHandDetectionTime(0.0)
    , m_handDetectionFPS(0)
    , m_handDetectionEnabled(false)
    , m_cachedPixmap(640, 480)
{
    ui->setupUi(this);

    setContentsMargins(0, 0, 0, 0);
    
    // Enable keyboard focus for this widget
    setFocusPolicy(Qt::StrongFocus);
    setFocus();

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

    // Initialize TFLite Segmentation
    initializeTFLiteSegmentation();
    
    // Initialize Hand Detection (disabled by default)
    initializeHandDetection();
    m_handDetectionEnabled = false;
    m_captureReady = false;  // Start with capture not ready
    // Initialize MediaPipe-like tracker
    // TODO: Initialize hand tracker when available

    // Run segmentation in a background thread so UI stays responsive
    m_segmentationThread = new QThread(this);
    // TODO: Implement segmentation when TFLite is available
    m_segmentationThread->start();

    // TODO: Connect segmentation widget signals when TFLite is available
    // TODO: Connect segmentation widget signals when TFLite is available
            // TODO: Connect segmentation widget signals when TFLite is available

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
    debugUpdateTimer->start(1000);

    // Initialize async processing
    m_segmentationWatcher = new QFutureWatcher<cv::Mat>(this);
        // TODO: Connect segmentation finished signal when TFLite is available
            
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
    if (m_segmentationWatcher) {
        disconnect(m_segmentationWatcher, nullptr, this, nullptr);
    }
    if (debugUpdateTimer) {
        disconnect(debugUpdateTimer, nullptr, this, nullptr);
    }

    // Stop and delete QTimers owned by Capture
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; countdownTimer = nullptr; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; recordTimer = nullptr; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; recordingFrameTimer = nullptr; }
    if (debugUpdateTimer){ debugUpdateTimer->stop(); delete debugUpdateTimer; debugUpdateTimer = nullptr; }

    // Delete QLabels created on the heap and parented to Capture or ui->overlayWidget
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; }
    if (loadingCameraLabel){ delete loadingCameraLabel; loadingCameraLabel = nullptr; }
    if (videoLabelFPS){ delete videoLabelFPS; videoLabelFPS = nullptr; }

    // Clean up debug widgets
    if (debugWidget){ delete debugWidget; debugWidget = nullptr; }
    if (debugLabel){ delete debugLabel; debugLabel = nullptr; }
    if (fpsLabel){ delete fpsLabel; fpsLabel = nullptr; }
    if (segmentationLabel){ delete segmentationLabel; segmentationLabel = nullptr; }
    if (segmentationButton){ delete segmentationButton; segmentationButton = nullptr; }
    if (handDetectionLabel){ delete handDetectionLabel; handDetectionLabel = nullptr; }
    if (handDetectionButton){ delete handDetectionButton; handDetectionButton = nullptr; }

    // Clean up hand detector
    if (m_handDetector){ delete m_handDetector; m_handDetector = nullptr; }

    // Clean up segmentation watcher
    if (m_segmentationWatcher){ delete m_segmentationWatcher; m_segmentationWatcher = nullptr; }

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

    // Store current frame for TFLite processing
    m_currentFrame = frame.clone();

    cv::Mat displayFrame = frame.clone();
    bool frameProcessed = false;

    // Process frame with hand detection if enabled (optimized performance)
    if (m_handDetectionEnabled) {
        // Only process every 4th frame to reduce lag
        static int handDetectionFrameSkip = 0;
        handDetectionFrameSkip++;
        
        if (handDetectionFrameSkip >= 4) {
            handDetectionFrameSkip = 0;
            
            if (m_handDetector && m_handDetector->isInitialized()) {
                static int processingCounter = 0;
                if (++processingCounter % 60 == 0) { // Every 60 frames
                    qDebug() << "Processing hand detection - Enabled:" << m_handDetectionEnabled;
                }
                processFrameWithHandDetection(displayFrame);
                frameProcessed = true;
            } else {
                qDebug() << "Hand detector not ready - Enabled:" << m_handDetectionEnabled 
                         << "Initialized:" << (m_handDetector ? m_handDetector->isInitialized() : false);
            }
        }
    } else {
        static int disabledCounter = 0;
        if (++disabledCounter % 120 == 0) { // Every 120 frames
            qDebug() << "Hand detection is DISABLED - Counter:" << disabledCounter;
            // REMOVED: Auto-re-enable to prevent unwanted captures
        }
    }

    // Process frame with segmentation if enabled - ASYNC PROCESSING
    if (m_showSegmentation) {
        // Check if we should start a new async processing
                                m_asyncMutex.lock();
            if (!m_processingAsync && m_segmentationWatcher && !m_segmentationWatcher->isRunning()) {
            // Only process every 3rd frame for better performance (adjust as needed)
            static int frameSkipCounter = 0;
            frameSkipCounter++;
            
            if (frameSkipCounter >= 3) {
                frameSkipCounter = 0;
                m_processingAsync = true;
                m_lastProcessedFrame = frame.clone();
                
                // Start async processing
                QFuture<cv::Mat> future = QtConcurrent::run(processFrameAsync, frame, m_tfliteSegmentation);
                m_segmentationWatcher->setFuture(future);
            }
        }
        m_asyncMutex.unlock();
        
        // Display the last available segmented frame or original frame
        QPixmap pixmap;
        cv::Mat segmentedFrame;
        QList<HandDetection> handDetections;
        
        {
            m_segmentationMutex.lock();
            if (!m_lastSegmentedFrame.empty()) {
                segmentedFrame = m_lastSegmentedFrame.clone();
            }
            m_segmentationMutex.unlock();
        }
        
        if (!segmentedFrame.empty()) {
            // Apply hand detection to segmented frame if enabled
            if (m_handDetectionEnabled && m_handDetector && m_handDetector->isInitialized()) {
                m_handDetectionMutex.lock();
                handDetections = m_lastHandDetections;
                m_handDetectionMutex.unlock();
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
        // Display original frame when segmentation is disabled
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
    qDebug() << "Segmentation Enabled:" << (m_showSegmentation ? "YES" : "NO");
    qDebug() << "Hand Detection Enabled:" << (m_showHandDetection ? "YES" : "NO");
    qDebug() << "Segmentation FPS:" << QString::number(m_segmentationFPS, 'f', 1) << "FPS";
    qDebug() << "Hand Detection FPS:" << QString::number(m_handDetectionFPS, 'f', 1) << "FPS";
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
    
    // Segmentation label
    segmentationLabel = new QLabel("Segmentation: OFF", debugWidget);
    segmentationLabel->setStyleSheet("QLabel { color: #ffaa00; font-size: 12px; }");
    debugLayout->addWidget(segmentationLabel);
    
    // Segmentation button
    segmentationButton = new QPushButton("Disable Segmentation", debugWidget);
    segmentationButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    connect(segmentationButton, &QPushButton::clicked, this, &Capture::toggleSegmentation);
    debugLayout->addWidget(segmentationButton);
    
    // Hand detection label
    handDetectionLabel = new QLabel("Hand Detection: OFF", debugWidget);
    handDetectionLabel->setStyleSheet("QLabel { color: #00aaff; font-size: 12px; }");
    debugLayout->addWidget(handDetectionLabel);
    
    // Hand detection button
    handDetectionButton = new QPushButton("Disable Hand Detection", debugWidget);
    handDetectionButton->setStyleSheet("QPushButton { color: white; font-size: 12px; background-color: #d32f2f; border: 1px solid white; padding: 5px; border-radius: 3px; }");
    connect(handDetectionButton, &QPushButton::clicked, this, &Capture::toggleHandDetection);
    debugLayout->addWidget(handDetectionButton);
    
    // Performance tips label
    QLabel *tipsLabel = new QLabel("Press 'P' for stats, 'D' to hide", debugWidget);
    tipsLabel->setStyleSheet("QLabel { color: #cccccc; font-size: 10px; font-style: italic; }");
    debugLayout->addWidget(tipsLabel);
    
    // Add debug widget to the main widget instead of videoLabel's layout
    debugWidget->setParent(this);
    debugWidget->move(10, 10); // Position in top-left corner
    debugWidget->resize(220, 180); // Increased size for more information
    debugWidget->raise(); // Ensure it's on top
    
    debugWidget->show(); // Show debug widget so user can enable segmentation and hand detection
}

void Capture::initializeTFLiteSegmentation()
{
    // Try to load the TFLite model
    QString modelPath = "deeplabv3.tflite";
    if (QFileInfo::exists(modelPath)) {
        qDebug() << "Loading TFLite model from:" << modelPath;
        // TODO: Initialize TFLite model when available
            } else {
        qDebug() << "TFLite model not found, using OpenCV fallback segmentation";
        // Initialize with OpenCV fallback instead of showing error
        // TODO: Initialize OpenCV fallback when available
    }

    // Set initial parameters
            // TODO: Set confidence threshold when TFLite is available
            // TODO: Set performance mode when TFLite is available
}

void Capture::initializeHandDetection()
{
    if (m_handDetector) {
        if (m_handDetector->initialize()) {
            qDebug() << "Hand detector adapter initialized successfully";
            
            // Check if CUDA is available
            if (m_handDetector->isCudaAvailable()) {
                qDebug() << "🚀 CUDA-accelerated hand detection is ACTIVE";
                qDebug() << "GPU: Using CUDA for hand detection processing";
            } else {
                qDebug() << "⚠️ CUDA not available, using CPU fallback";
                qDebug() << "CPU: Using CPU-based hand detection processing";
            }
            
            m_handDetector->setConfidenceThreshold(0.3); // More lenient threshold for poor camera quality
            m_handDetector->setShowBoundingBox(true);
            m_handDetector->setPerformanceMode(1); // Set to Balanced mode
            
            // Connect to performance monitoring signals
            connect(m_handDetector, &HandDetector::processingTimeUpdated,
                    this, [this](double processingTime) {
                m_lastHandDetectionTime = processingTime;
                if (processingTime > 0) {
                    m_handDetectionFPS = 1000.0 / processingTime;
                }
            });
            
            connect(m_handDetector, &HandDetector::detectorTypeChanged,
                    this, [this](const QString& detectorType) {
                qDebug() << "Hand detector switched to:" << detectorType;
            });
            
        } else {
            qWarning() << "Failed to initialize hand detector adapter";
        }
    }
}

void Capture::enableHandDetection(bool enable)
{
    m_handDetectionEnabled = enable;
    if (enable) {
        qDebug() << "🖐️ Hand detection ENABLED";
    } else {
        qDebug() << "🖐️ Hand detection DISABLED";
    }
}

void Capture::setHandDetectionEnabled(bool enabled)
{
    enableHandDetection(enabled);
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
    // TODO: Set confidence threshold when TFLite is available
    qDebug() << "Segmentation confidence threshold set to:" << threshold;
}

double Capture::getSegmentationConfidenceThreshold() const
{
    return m_segmentationConfidenceThreshold;
}

cv::Mat Capture::getLastSegmentedFrame() const
{
    const_cast<QMutex&>(m_segmentationMutex).lock();
    cv::Mat result = m_lastSegmentedFrame.clone();
    const_cast<QMutex&>(m_segmentationMutex).unlock();
    return result;
}

void Capture::saveSegmentedFrame(const QString& filename)
{
    m_segmentationMutex.lock();
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
    m_segmentationMutex.unlock();
}

double Capture::getSegmentationProcessingTime() const
{
    return m_lastSegmentationTime;
}

void Capture::setSegmentationPerformanceMode(int mode)
{
    // TODO: Set performance mode when TFLite is available
    qDebug() << "Segmentation performance mode set to:" << mode;
}



void Capture::toggleSegmentation()
{
    setShowSegmentation(!getShowSegmentation());
}

void Capture::updateSegmentationButton()
{
    if (segmentationButton) {
        segmentationButton->setText(getShowSegmentation() ? "Hide Segmentation" : "Show Segmentation");
    }
}

bool Capture::isTFLiteModelLoaded() const
{
    return m_tfliteModelLoaded;
}



// Segmentation callback implementations
void Capture::onSegmentationResultReady(const cv::Mat& segmentedFrame)
{
    m_segmentationMutex.lock();
    m_lastSegmentedFrame = segmentedFrame.clone();
    m_segmentationCompleted = true;
    m_segmentationMutex.unlock();
}

void Capture::onSegmentationError(const QString& error)
{
    qWarning() << "Segmentation error:" << error;
    m_segmentationCompleted = false;
}

void Capture::onTFLiteModelLoaded(bool success)
{
    m_tfliteModelLoaded = success;
    qDebug() << "TFLite model loaded:" << success;
}

void Capture::onSegmentationFinished()
{
    m_processingAsync = false;
    m_segmentationCompleted = true;
}

void Capture::updateSegmentationDisplay()
{
    // TODO: Update segmentation display when available
}

void Capture::showSegmentationNotification(QWidget* /*parent*/, bool /*show*/)
{
    // TODO: Show segmentation notification when available
}

void Capture::enableHandDetectionForCapture()
{
    enableHandDetection(true);
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
    m_handDetectionEnabled = false;
    m_captureReady = false;
    if (m_handDetector) {
        m_handDetector->resetGestureState();
        qDebug() << "🖐️ Hand detection completely reset";
    }
    
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

QList<HandDetection> Capture::getLastHandDetections() const
{
    const_cast<QMutex&>(m_handDetectionMutex).lock();
    QList<HandDetection> result = m_lastHandDetections;
    const_cast<QMutex&>(m_handDetectionMutex).unlock();
    return result;
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
                    QList<HandDetection> debugDetections = m_handDetector->detect(m_currentFrame);
                    qDebug() << "Manual hand detection test found" << debugDetections.size() << "hands";
                    qDebug() << "Detector type:" << m_handDetector->getDetectorType();
                    qDebug() << "Average processing time:" << m_handDetector->getAverageProcessingTime() << "ms";
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
    qDebug() << "Capture widget shown - ENABLING HAND DETECTION";
    
    // Start camera when page is shown (only if not already open)
    if (cameraWorker && !cameraWorker->isCameraOpen()) {
        qDebug() << "Starting camera...";
        cameraWorker->startCamera();
    } else if (cameraWorker && cameraWorker->isCameraOpen()) {
        qDebug() << "Camera already open - skipping restart";
    }
    
    // ENABLE HAND DETECTION IMMEDIATELY
    m_handDetectionEnabled = true;
    if (m_handDetector) {
        m_handDetector->resetGestureState();
    }
    qDebug() << "🖐️ Hand detection ENABLED - close your hand to trigger capture automatically";
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    qDebug() << "Capture widget hidden - OPTIMIZED camera and hand detection shutdown";
    
    // Disable hand detection when page is hidden (but keep camera running for faster return)
    enableHandDetection(false);
    qDebug() << "🖐️ Hand detection DISABLED";
    
    // Note: Camera is now controlled by the main page change handler in brbooth.cpp
    // This prevents lag when returning to capture page
}

void Capture::processFrameWithHandDetection(const cv::Mat &frame)
{
    if (!m_handDetector || !m_handDetector->isInitialized()) {
        return;
    }
    
    // Double-check that hand detection is enabled
    if (!m_handDetectionEnabled) {
        return;
    }
    
    m_handDetectionTimer.start();
    
    try {
        // Detect hands in the frame using the adapter
        QList<HandDetection> detections = m_handDetector->detect(frame);
        
        // Debug: Check if any detections were found
        static int noDetectionCounter = 0;
        if (detections.isEmpty()) {
            noDetectionCounter++;
            if (noDetectionCounter % 60 == 0) { // Every 60 frames
                qDebug() << "No hand detections found - Counter:" << noDetectionCounter;
            }
        } else {
            noDetectionCounter = 0;
            qDebug() << "Found" << detections.size() << "hand detection(s)";
        }
        
        // Store the detections
        {
            m_handDetectionMutex.lock();
            m_lastHandDetections = detections;
            m_handDetectionMutex.unlock();
        }
        
        // Update timing info
        m_lastHandDetectionTime = m_handDetectionTimer.elapsed() / 1000.0;
        m_handDetectionFPS = (m_lastHandDetectionTime > 0) ? 1.0 / m_lastHandDetectionTime : 0;
        
        // Only process detections if confidence is high enough
        for (const auto& detection : detections) {
            if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
                // Use the detection result directly
                bool isClosed = detection.isClosed;
                
                // Debug: Force test hand closing detection
                static int testCounter = 0;
                if (++testCounter % 60 == 0) { // Every 60 frames
                    qDebug() << "=== HAND DETECTION TEST ===";
                    qDebug() << "Detection confidence:" << static_cast<int>(detection.confidence * 100) << "%";
                    qDebug() << "Detection isClosed:" << isClosed;
                    qDebug() << "Landmarks count:" << detection.landmarks.size();
                    qDebug() << "Bounding box:" << detection.boundingBox.x << detection.boundingBox.y 
                             << detection.boundingBox.width << "x" << detection.boundingBox.height;
                    
                    // Test if we can detect hand closing manually
                    if (!detection.landmarks.empty()) {
                        bool manualClosed = m_handDetector->isHandClosed(detection.landmarks);
                        qDebug() << "Manual isHandClosed check:" << manualClosed;
                    }
                    qDebug() << "==========================";
                }
                
                // Update hand state tracking
                qDebug() << "🖐️ Updating hand state - isClosed:" << isClosed;
                m_handDetector->updateHandState(isClosed);
                
                // Check if capture should be triggered - automatically click capture button when hand closed
                bool shouldTrigger = m_handDetector->shouldTriggerCapture();
                if (shouldTrigger) {
                    qDebug() << "🎯 HAND CLOSED DETECTED! Automatically triggering capture button...";
                    
                    // Only trigger if countdown is not already active
                    if (!countdownTimer || !countdownTimer->isActive()) {
                        // Automatically "click" the capture button
                        on_capture_clicked();
                        qDebug() << "🎯 Capture button automatically clicked by hand detection!";
                        
                        // Disable hand detection after triggering capture
                        m_handDetectionEnabled = false;
                        qDebug() << "🖐️ Hand detection DISABLED after capture trigger - ready for next capture";
                    } else {
                        qDebug() << "🎯 Countdown already active, ignoring hand trigger";
                    }
                    break; // Only trigger once per frame
                } else {
                    // Debug why trigger is false
                    static int triggerDebugCounter = 0;
                    if (++triggerDebugCounter % 120 == 0) { // Every 120 frames
                        qDebug() << "🔍 Trigger check - shouldTrigger:" << shouldTrigger 
                                 << "Hand closed:" << isClosed
                                 << "Timer valid:" << m_handDetector->isHandClosedTimerValid()
                                 << "Countdown active:" << (countdownTimer && countdownTimer->isActive())
                                 << "Hand detection enabled:" << m_handDetectionEnabled;
                    }
                }
                
                // Debug output for hand detection (reduced frequency for performance)
                static int debugCounter = 0;
                if (++debugCounter % 120 == 0) { // Every 120 frames (reduced from 30)
                    qDebug() << "Hand detected - Confidence:" << static_cast<int>(detection.confidence * 100) << "%"
                             << "Is closed:" << isClosed
                             << "Landmarks:" << detection.landmarks.size()
                             << "Should trigger:" << m_handDetector->shouldTriggerCapture()
                             << "Countdown active:" << (countdownTimer && countdownTimer->isActive());
                }
            }
        }
        
        // Log performance info less frequently
        static int frameCounter = 0;
        if (++frameCounter % 200 == 0) {
            qDebug() << "Hand Detection Performance:"
                     << "Type:" << m_handDetector->getDetectorType()
                     << "Time:" << m_handDetector->getAverageProcessingTime() << "ms"
                     << "FPS:" << m_handDetector->getCurrentFPS()
                     << "Detections:" << detections.size();
        }
        
    } catch (const std::exception& e) {
        qWarning() << "Exception in hand detection:" << e.what();
    }
}

void Capture::drawHandBoundingBoxes(cv::Mat &/*frame*/, const QList<HandDetection> &detections)
{
    // REMOVED: No more bounding box drawing to avoid conflicts with segmentation
    // Only show gesture status in console for debugging
    for (const auto& detection : detections) {
        if (detection.confidence >= m_handDetector->getConfidenceThreshold()) {
            // Check if capture should be triggered - automatically click capture button when hand closed
            if (m_handDetector->shouldTriggerCapture()) {
                qDebug() << "🎯 HAND CLOSED DETECTED! Automatically triggering capture button...";
                
                // Only trigger if countdown is not already active
                if (!countdownTimer || !countdownTimer->isActive()) {
                    // Automatically "click" the capture button
                    on_capture_clicked();
                    qDebug() << "🎯 Capture button automatically clicked by hand detection!";
                } else {
                    qDebug() << "🎯 Countdown already active, ignoring hand trigger";
                }
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



void Capture::onHandDetectionFinished()
{
    // This slot can be used for async hand detection if needed in the future
    qDebug() << "Hand detection processing completed";
}

// Static function for async processing
cv::Mat Capture::processFrameAsync(const cv::Mat &frame, void *segmentation)
{
    if (!segmentation) {
        return frame.clone();
    }
    
    // TODO: Implement segmentation when available
    return frame.clone();
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
        QString segTime = QString::number(m_lastSegmentationTime * 1000, 'f', 1);
        segmentationLabel->setText(QString("Segmentation: %1 (%2ms)").arg(segStatus).arg(segTime));
    }
    
    if (handDetectionLabel) {
        QString handStatus = m_showHandDetection ? "ON (Tracking)" : "OFF";
        QString handTime = QString::number(m_lastHandDetectionTime * 1000, 'f', 1);
        QString detectorType = m_handDetector ? m_handDetector->getDetectorType() : "Unknown";
        QString avgTime = m_handDetector ? QString::number(m_handDetector->getAverageProcessingTime(), 'f', 1) : "0.0";
        handDetectionLabel->setText(QString("Hand Detection: %1 (%2ms) [%3] Avg: %4ms").arg(handStatus).arg(handTime).arg(detectorType).arg(avgTime));
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
}




