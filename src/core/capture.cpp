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
#include <QHBoxLayout>
#include <QSizePolicy>
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
    , debugWidget(nullptr)
    , debugLabel(nullptr)
    , fpsLabel(nullptr)
    , handDetectionLabel(nullptr)
    , handDetectionButton(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
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


    
    // Initialize Hand Detection (disabled by default)
    initializeHandDetection();
    m_handDetectionEnabled = false;
    // Hand detection will be enabled only when this page is shown





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

    // Delete QLabels created on the heap and parented to Capture or ui->overlayWidget
    // loadingCameraLabel has 'this' as parent, so it *will* be deleted by Qt's object tree.
    // However, explicitly nulling out the pointer is good practice.
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; }
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


    m_currentFrame = frame;

    cv::Mat displayFrame = frame;
    bool frameProcessed = false;
    
    // Performance-based frame skipping
    static int skipFrameCount = 0;
    static QElapsedTimer performanceTimer;
    static double lastFPS = 30.0;
    
    if (performanceTimer.isValid()) {
        double elapsed = performanceTimer.elapsed() / 1000.0;
        if (elapsed > 0) {
            lastFPS = frameCount / elapsed;
        }
    } else {
        performanceTimer.start();
    }
    
    // Skip frames if FPS drops below 20
    if (lastFPS < 20.0) {
        skipFrameCount++;
        if (skipFrameCount % 2 == 0) { // Skip every other frame when FPS is low
            return;
        }
    } else {
        skipFrameCount = 0;
    }

    // Process frame with hand detection if enabled (optimized performance)
    if (m_handDetectionEnabled) {
        // Use optimized HandDetector for fast hand detection
        if (m_handDetector && m_handDetector->isInitialized()) {
            // Process every 2nd frame for better accuracy (50% of frames)
            if (frameCount % 2 == 0) {
                qDebug() << "🔍 Processing hand detection frame" << frameCount << "| Enabled:" << m_handDetectionEnabled;
                processFrameWithHandDetection(displayFrame);
                frameProcessed = true;
            }
        }
    }

    // Display original frame (optimized conversion)
    QImage qImage = cvMatToQImage(displayFrame);
    QPixmap pixmap = QPixmap::fromImage(qImage, Qt::ColorOnly);
    
    if (ui->videoLabel) {
        QSize labelSize = ui->videoLabel->size();
        QPixmap scaledPixmap = pixmap.scaled(
            labelSize,
            Qt::KeepAspectRatioByExpanding,
            Qt::SmoothTransformation
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

    // Print performance stats every 240 frames (further reduced for performance)
    if (frameCount % 240 == 0) {
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
    qDebug() << "Hand Detection Enabled:" << (m_showHandDetection ? "YES" : "NO");
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
        
        // Re-enable hand detection if countdown was cancelled
        m_handDetectionEnabled = true;
        qDebug() << "Hand detection re-enabled after countdown cancellation";
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

    // Prevent multiple countdowns from running simultaneously
    if (countdownTimer && countdownTimer->isActive()) {
        qDebug() << "Countdown already in progress, ignoring button click";
        return;
    }

    // Disable hand detection to prevent conflicts during countdown
    m_handDetectionEnabled = false;
    qDebug() << "🔍 Hand detection DISABLED for manual capture";

    ui->capture->setEnabled(false);
    
    // Start the countdown immediately
    countdownValue = 5;
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();
    countdownLabel->raise(); // Bring to front
    
    // Add fade-in animation for the countdown label
    QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
    animation->setDuration(300);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->start();
    
    // Start the countdown timer
    if (countdownTimer) {
        countdownTimer->start(1000); // 1 second intervals
        qDebug() << "🎬 Manual countdown started! 5 seconds to capture...";
    } else {
        qWarning() << "Countdown timer is null! Cannot start countdown.";
    }
}

void Capture::startCountdown()
{
    // Only start countdown if not already running
    if (countdownTimer && !countdownTimer->isActive()) {
        // Disable hand detection to prevent conflicts during countdown
        m_handDetectionEnabled = false;
        qDebug() << "🔍 Hand detection DISABLED for hand-triggered capture";
        
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
    qDebug() << "Countdown update: " << countdownValue;
    
    if (countdownValue > 0) {
        countdownLabel->setText(QString::number(countdownValue));
        qDebug() << "Countdown: " << countdownValue;
    } else {
        qDebug() << "Countdown finished! Performing capture...";
        countdownTimer->stop();
        countdownLabel->hide();

        if (m_currentCaptureMode == ImageCaptureMode) {
            performImageCapture();
            ui->capture->setEnabled(true);
            // Re-enable hand detection after capture
            m_handDetectionEnabled = true;
            qDebug() << "Hand detection re-enabled after image capture";
        } else if (m_currentCaptureMode == VideoRecordMode) {
            startRecording();
            // Hand detection will be re-enabled when recording stops
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
    
    debugWidget->show(); // Show debug widget so user can enable hand detection
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

void Capture::enableHandDetectionForCapture()
{
    // Only enable hand detection if no countdown is active
    if (!countdownTimer || !countdownTimer->isActive()) {
        setHandDetectionEnabled(true);
        qDebug() << "Hand detection enabled for capture page";
        qDebug() << "🎯 Hand detection automatically enabled - ready for gesture capture!";
        qDebug() << "✋ Close your hand to trigger image capture automatically!";
    } else {
        qDebug() << "Hand detection not enabled - countdown is active";
    }
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
    qDebug() << "Capture widget shown - ENABLING hand detection with 5-second delay";
    enableHandDetection(true); // Enable hand detection when page is shown
    
    // Start camera timer if not already active
    if (cameraTimer && !cameraTimer->isActive()) {
        cameraTimer->start(16); // ~60 FPS - ULTRA-LIGHT processing
    }
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    qDebug() << "Capture widget hidden - DISABLING hand detection";
    enableHandDetection(false); // Disable hand detection when page is hidden
    
    // Stop camera timer when page is hidden
    if (cameraTimer && cameraTimer->isActive()) {
        cameraTimer->stop();
    }
}

void Capture::processFrameWithHandDetection(const cv::Mat &frame)
{
    if (!m_handDetector || !m_handDetector->isInitialized()) {
        return;
    }
    
    try {
        // Detect hands in the frame (optimized)
        QList<HandDetection> detections = m_handDetector->detect(frame);
        
        // Check if we should trigger capture - this is the only thing we care about
        if (m_handDetector->shouldTriggerCapture()) {
            qDebug() << "🚀 INITIATING CAPTURE COUNTDOWN! Hand gesture confirmed!";
            startCountdown();
        }
        
    } catch (const std::exception& e) {
        qWarning() << "Exception in hand detection:" << e.what();
    }
}

void Capture::drawHandBoundingBoxes(cv::Mat &/*frame*/, const QList<HandDetection> &/*detections*/)
{
    // Visualization disabled - only capture triggering is needed
}





void Capture::onHandDetectionFinished()
{
    // This slot can be used for async hand detection if needed in the future
    qDebug() << "Hand detection processing completed";
}



void Capture::updateDebugDisplay()
{
    if (debugLabel) {
        QString debugInfo = QString("FPS: %1 | Hand: %2 | Hand FPS: %3")
                           .arg(m_currentFPS)
                           .arg(m_showHandDetection ? "ON" : "OFF")
                           .arg(QString::number(m_handDetectionFPS, 'f', 1));
        debugLabel->setText(debugInfo);
    }
    
    if (fpsLabel) {
        fpsLabel->setText(QString("FPS: %1").arg(m_currentFPS));
    }
    
    if (handDetectionLabel) {
        QString handStatus = m_showHandDetection ? "ON (Tracking)" : "OFF";
        QString handTime = QString::number(m_lastHandDetectionTime * 1000, 'f', 1);
        handDetectionLabel->setText(QString("Hand Detection: %1 (%2ms)").arg(handStatus).arg(handTime));
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
    
    // Re-enable hand detection after recording
    m_handDetectionEnabled = true;
    qDebug() << "Hand detection re-enabled after video recording";

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




