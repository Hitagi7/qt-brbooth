#include "capture.h"
#include "ui_capture.h"
#include "foreground.h"
#include "camera.h"

#include <QThread>
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
#include <QMessageBox>
#include <QStackedLayout>
#include <QPainter>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

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
    , m_currentVideoTemplate("Default", 5) // Assuming a constructor for VideoTemplate
    , m_recordedSeconds(0)
    , m_recordedFrames()
    , m_capturedImage()
    , stackedLayout(nullptr)
    , loadingCameraLabel(nullptr)
    , videoLabelFPS(nullptr) // Initialize pointer to nullptr
    , loopTimer()
    , totalTime(0)           // Initialize totalTime
    , frameCount(0)          // Initialize frameCount
    , frameTimer()
    , overlayImageLabel(nullptr)
    , m_personScaleFactor(1.0) // Initialize to 1.0 (normal size) - matches slider at 0
    , m_lastDetectedPersonRect() // Initialize empty rectangle
    , m_personDetected(false) // Initialize person detection flag
    , personDetector(nullptr) // Initialize person detector pointer
    , hogEnabled(true) // Enable HOG by default with optimizations
    , frameSkipCounter(0) // Initialize frame skip counter
{
    ui->setupUi(this);

    setContentsMargins(0, 0, 0, 0);

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

    loadingCameraLabel = new QLabel("Loading Camera...", this); // Set text here
    // --- FIX for "oading Camera." - Increased size and added padding for robustness ---
    loadingCameraLabel->setAlignment(Qt::AlignCenter);
    QFont loadingFont = loadingCameraLabel->font();
    loadingFont.setPointSize(36);
    loadingFont.setBold(true);
    loadingCameraLabel->setFont(loadingFont);
    // Increased horizontal padding in stylesheet to give text more room
    loadingCameraLabel->setStyleSheet(
        "color: white; "
        "background-color: rgba(0, 0, 0, 150); "
        "border-radius: 15px; "
        "padding: 10px 20px; " // Added padding (top/bottom 10px, left/right 20px)
        );
    // Adjusted fixed size to accommodate padding and ensure text fits
    // A rough estimate: "Loading Camera." at 36pt bold might need more than 350px width.
    // Let's try to calculate a safer size or rely on sizeHint if not fixed.
    // For fixed size, let's just make it generously larger.
    loadingCameraLabel->setFixedSize(450, 120); // Increased size (from 350, 100)
    // ---------------------------------------------------------------------------------
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
    ui->overlayWidget->setStyleSheet("background-color: transparent;"); // Ensure overlay is transparent


    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setPageStep(tickStep); // Page step also set to tick interval
    ui->verticalSlider->setValue(0); // Set to 100 for 1.0x scaling (normal size, no scaling) - slider is inverted

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
        ui->videoLabel->setText("Camera worker not provided or is NULL.\nCannot initialize camera.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }
    // Initialize and start performance timers
    loopTimer.start();
    frameTimer.start();

    ui->capture->setEnabled(true);

    // Countdown label overlays on the overlayWidget
    countdownLabel = new QLabel(ui->overlayWidget);
    countdownLabel->setAlignment(Qt::AlignCenter);
    QFont font = countdownLabel->font();
    font.setPointSize(100);
    font.setBold(true);
    countdownLabel->setFont(font);
    countdownLabel->setStyleSheet("color:white; background-color: rgba(0, 0, 0, 150); border-radius: 20px;");
    countdownLabel->setFixedSize(200,200);
    countdownLabel->hide();

    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);

    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);


    // --- INITIALIZE ADVANCED HOG DETECTOR ---
    personDetector = new SimplePersonDetector();
    if (personDetector->initialize()) {
        qDebug() << "Advanced HOG person detector initialized successfully";
        qDebug() << "HOG detection enabled with fallback detection";
    } else {
        qWarning() << "Failed to initialize advanced HOG detector";
        hogEnabled = false;
    }
    // --- END INITIALIZE HOG DETECTOR ---

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

    // Clean up person detector
    if (personDetector) {
        delete personDetector;
        personDetector = nullptr;
    }

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

void Capture::handleCameraOpened(bool success, double actual_width, double actual_height, double actual_fps)
{
    Q_UNUSED(actual_width);
    Q_UNUSED(actual_height);
    Q_UNUSED(actual_fps);

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
    if (loadingCameraLabel->parentWidget() && !loadingCameraLabel->parentWidget()->isHidden()) {
        loadingCameraLabel->parentWidget()->hide();
        ui->videoLabel->show();
    } else if (!loadingCameraLabel->isHidden()) {
        loadingCameraLabel->hide();
        ui->videoLabel->show();
    }

    QPixmap pixmap = QPixmap::fromImage(image);

    QSize labelSize = ui->videoLabel->size();
    // --- HOG PERSON DETECTION (OPTIMIZED) ---
    if (hogEnabled && personDetector && personDetector->isInitialized()) {
        frameSkipCounter++;
        if (frameSkipCounter >= HOG_FRAME_SKIP) {
            detectPersonWithHOG(image);
            frameSkipCounter = 0;
        }
    }
    // --- END HOG PERSON DETECTION ---

    QPixmap scaledPixmap = pixmap.scaled(
        labelSize,
        Qt::KeepAspectRatioByExpanding,
        Qt::FastTransformation
        );

    // --- APPLY FRAME SCALING ---
    if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
        // Scale the entire frame instead of just the person
        QSize originalSize = scaledPixmap.size();
        int newWidth = qRound(originalSize.width() * m_personScaleFactor);
        int newHeight = qRound(originalSize.height() * m_personScaleFactor);
        
        scaledPixmap = scaledPixmap.scaled(
            newWidth, newHeight,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        );
        
        qDebug() << "Applied frame scaling with factor:" << m_personScaleFactor << "Original size:" << originalSize << "New size:" << scaledPixmap.size();
    }
    // --- END FRAME SCALING ---
    
    ui->videoLabel->setPixmap(scaledPixmap);
    ui->videoLabel->setAlignment(Qt::AlignCenter);
    ui->videoLabel->update(); // Request a repaint

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

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

        stackedLayout->addWidget(ui->videoLabel);       // Layer 0: Camera feed (background)

        // --- FIX for Centering "Loading Camera" ---
        // Create a layout to center the fixed-size loadingCameraLabel
        QVBoxLayout* centeringLayout = new QVBoxLayout();
        centeringLayout->addStretch(); // Top stretch
        QHBoxLayout* hCenteringLayout = new QHBoxLayout();
        hCenteringLayout->addStretch(); // Left stretch
        hCenteringLayout->addWidget(loadingCameraLabel); // Add the fixed-size label
        hCenteringLayout->addStretch(); // Right stretch
        centeringLayout->addLayout(hCenteringLayout); // Add horizontal layout to vertical
        centeringLayout->addStretch(); // Bottom stretch

        QWidget* centeringWidget = new QWidget(this); // Create a wrapper widget
        centeringWidget->setLayout(centeringLayout);
        centeringWidget->setContentsMargins(0,0,0,0); // Ensure no extra margins
        centeringWidget->setAttribute(Qt::WA_TranslucentBackground); // Keep background transparent
        // Initially show the centeringWidget, as loadingCameraLabel is shown by default
        centeringWidget->show();


        stackedLayout->addWidget(centeringWidget);      // Layer 1: Loading text (now centered)
        // ------------------------------------------

        stackedLayout->addWidget(ui->overlayWidget);    // Layer 2: UI elements (buttons, slider)
        if (overlayImageLabel) {
            stackedLayout->addWidget(overlayImageLabel); // Layer 3: Foreground image (top)
        }

        if (layout()) {
            delete layout();
        }

        QGridLayout* mainLayout = new QGridLayout(this);
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
    if (ui->back) ui->back->raise();
    if (ui->capture) ui->capture->raise();
    if (ui->verticalSlider) ui->verticalSlider->raise();
    if (countdownLabel) countdownLabel->raise();

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
    ui->back->setStyleSheet(
        "QPushButton {"
        "   background: transparent;"
        "   border: none;"
        "   color: white;"
        "}"
        );

    ui->capture->setStyleSheet(
        "QPushButton {"
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
        "}"
        );

    ui->verticalSlider->setStyleSheet(
        "QSlider::groove:vertical {"
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
        "}"
        );

    ui->overlayWidget->setStyleSheet("background: transparent;");
    qDebug() << "Clean professional overlay styles applied";
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

    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to:" << static_cast<int>(mode);
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
}


void Capture::detectPersonWithHOG(const QImage& image) {
    if (!personDetector || !personDetector->isInitialized()) {
        return;
    }

    try {
        // Convert QImage to cv::Mat
        cv::Mat frame = qImageToCvMat(image);
        if (frame.empty()) {
            return;
        }

        // Use the advanced HOG detector
        QList<SimpleDetection> detections = personDetector->detect(frame);

        // Find the best detection (highest confidence)
        QRect bestDetection = findBestPersonDetection(detections);
        
        if (!bestDetection.isEmpty()) {
            m_personDetected = true;
            m_lastDetectedPersonRect = bestDetection;
            qDebug() << "Advanced HOG detected person at:" << bestDetection << "with" << detections.size() << "total detections";
        } else {
            m_personDetected = false;
            m_lastDetectedPersonRect = QRect();
            qDebug() << "No person detected by advanced HOG";
        }
    } catch (const cv::Exception& e) {
        qWarning() << "Advanced HOG detection error:" << e.what();
        m_personDetected = false;
        m_lastDetectedPersonRect = QRect();
    }
}

QRect Capture::findBestPersonDetection(const QList<SimpleDetection>& detections) {
    if (detections.isEmpty()) {
        return QRect();
    }

    // Find the detection with highest confidence
    SimpleDetection bestDetection = detections.first();
    double maxConfidence = bestDetection.confidence;

    for (const auto& detection : detections) {
        if (detection.confidence > maxConfidence) {
            maxConfidence = detection.confidence;
            bestDetection = detection;
        }
    }

    // Convert cv::Rect to QRect
    return QRect(bestDetection.boundingBox.x, bestDetection.boundingBox.y, 
                bestDetection.boundingBox.width, bestDetection.boundingBox.height);
}

void Capture::printPerformanceStats() {
    if (frameCount == 0) return; // Avoid division by zero

    double avgLoopTime = (double)totalTime / frameCount; // Average time per updateCameraFeed call

    // Calculate FPS based on the total time elapsed for the batch
    double batchDurationSeconds = (double)frameTimer.elapsed() / 1000.0;
    if (batchDurationSeconds == 0) return; // Avoid division by zero

    double measuredFPS = (double)frameCount / batchDurationSeconds;


    qDebug() << "----------------------------------------";
    qDebug() << "Avg loop time per frame (measured over " << frameCount << " frames):" << avgLoopTime << "ms";
    qDebug() << "Camera/Display FPS (measured over " << frameCount << " frames):" << measuredFPS << "FPS";
    qDebug() << "Frame processing efficiency:" << (avgLoopTime < 16.67 ? "GOOD" : "NEEDS OPTIMIZATION");
    qDebug() << "----------------------------------------";
    // Reset timers for next batch
    frameCount = 0;
    totalTime = 0;
    frameTimer.start(); // Restart frameTimer for the next measurement period
}

// YOLO-related methods have been removed and replaced with HOG detection

void Capture::captureRecordingFrame()
{
    if (!m_isRecording) return;

    if (!ui->videoLabel->pixmap().isNull()) {
        m_recordedFrames.append(ui->videoLabel->pixmap());
    } else {
        qWarning() << "No pixmap available on videoLabel for recording frame.";
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
        QMessageBox::warning(this, "Camera Not Ready", "Camera is not open. Please ensure it's connected and drivers are installed.");
        return;
    }

    ui->capture->setEnabled(false);
    countdownValue = 5;
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();

    QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
    animation->setDuration(300);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->start();

    countdownTimer->start(1000);
}

void Capture::updateCountdown()
{
    countdownValue--;
    if (countdownValue > 0){
        countdownLabel->setText(QString::number(countdownValue));
    } else {
        countdownTimer->stop();
        countdownLabel->hide();

        if(m_currentCaptureMode == ImageCaptureMode){
            performImageCapture();
            ui->capture->setEnabled(true);
        }else if(m_currentCaptureMode == VideoRecordMode){
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
    qDebug() << "Recording: " + QString::number(m_recordedSeconds) + " / " + QString::number(m_currentVideoTemplate.durationSeconds) + "s";
}

void Capture::startRecording()
{
    if(!cameraWorker->isCameraOpen()){
        qWarning() << "Cannot start recording: Camera not opened by worker.";
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;

    int frameIntervalMs = 1000 / m_targetRecordingFPS;

    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
    qDebug() << "Recording started at target FPS: " + QString::number(m_targetRecordingFPS) + " frames/sec";
}

void Capture::stopRecording()
{
    if (!m_isRecording) return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;
    qDebug() << "Recording stopped. Captured " + QString::number(m_recordedFrames.size()) + " frames.";

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    emit showFinalOutputPage();
    ui->capture->setEnabled(true);
}

void Capture::performImageCapture()
{
    if (!ui->videoLabel->pixmap().isNull()) {
        QPixmap cameraPixmap = ui->videoLabel->pixmap();
        QPixmap compositedPixmap = cameraPixmap.copy();

        // Get the selected overlay/template path
        QString overlayPath;
        if (foreground) {
            overlayPath = foreground->getSelectedForeground();
        }
        if (!overlayPath.isEmpty()) {
            QPixmap overlayPixmap(overlayPath);
            if (!overlayPixmap.isNull()) {
                // Scale overlay to match camera image size
                QPixmap scaledOverlay = overlayPixmap.scaled(
                    compositedPixmap.size(),
                    Qt::KeepAspectRatioByExpanding,
                    Qt::SmoothTransformation
                    );
                        // Composite overlay onto camera image
        QPainter painter(&compositedPixmap);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
        painter.drawPixmap(0, 0, scaledOverlay);
        painter.end();
        
        // --- APPLY FRAME SCALING TO CAPTURED IMAGE ---
        // Apply frame scaling to the final composited image
        if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
            QSize originalSize = compositedPixmap.size();
            int newWidth = qRound(originalSize.width() * m_personScaleFactor);
            int newHeight = qRound(originalSize.height() * m_personScaleFactor);
            
            compositedPixmap = compositedPixmap.scaled(
                newWidth, newHeight,
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation
            );
            qDebug() << "Applied frame scaling to captured image with factor:" << m_personScaleFactor << "Original size:" << originalSize << "New size:" << compositedPixmap.size();
        }
        // --- END FRAME SCALING ---
            }
        }
        m_capturedImage = compositedPixmap;
        emit imageCaptured(m_capturedImage);
        qDebug() << "Image captured and composited with overlay.";
    } else {
        qWarning() << "Failed to capture image: videoLabel pixmap is empty.";
        QMessageBox::warning(this, "Capture Failed", "No camera feed available to capture an image.");
    }
    emit showFinalOutputPage();
}

// Helper function to convert QImage to cv::Mat
cv::Mat Capture::qImageToCvMat(const QImage &inImage)
{
    switch (inImage.format()) {
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
        // For 32-bit formats, OpenCV typically expects BGRA or BGR.
        // Assuming BGRA or similar layout where alpha is last.
        // OpenCV Mat constructor takes (rows, cols, type, data, step)
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC4,
                       (void*)inImage.constBits(), inImage.bytesPerLine());
    case QImage::Format_RGB888:
        // RGB888 in QImage is typically 3 bytes per pixel, RGB order.
        // OpenCV expects BGR by default, so a clone and conversion might be needed,
        // or ensure `cvtColor` is used later if written as RGB.
        // For direct conversion, it's safer to convert QImage to a known OpenCV format first if needed.
        // Here, we clone because `inImage.constBits()` might not be persistent.
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC3,
                       (void*)inImage.constBits(), inImage.bytesPerLine()).clone();
    case QImage::Format_Indexed8:
    {
        // 8-bit grayscale image
        cv::Mat mat(inImage.height(), inImage.width(), CV_8UC1,
                    (void*)inImage.constBits(), inImage.bytesPerLine());
        return mat.clone(); // Clone to ensure data is owned by Mat
    }
    default:
        qWarning() << "qImageToCvMat - QImage format not handled: " << inImage.format();
        // Convert to a supported format if not directly convertible
        QImage convertedImage = inImage.convertToFormat(QImage::Format_RGB32);
        return cv::Mat(convertedImage.height(), convertedImage.width(), CV_8UC4,
                       (void*)convertedImage.constBits(), convertedImage.bytesPerLine());
    }
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
        return QImage(rgb.data, rgb.cols, mat.rows, mat.step, QImage::Format_RGB888);
    }
    case CV_8UC3: {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    case CV_8UC1: {
        static QVector<QRgb> sColorTable;
        if (sColorTable.isEmpty())
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }
    default:
        return QImage();
    }
}



void Capture::on_verticalSlider_valueChanged(int value)
{
    int tickInterval = ui->verticalSlider->tickInterval();
    if (tickInterval == 0) return;
    int snappedValue = qRound((double)value / tickInterval) * tickInterval;
    snappedValue = qBound(ui->verticalSlider->minimum(), snappedValue, ui->verticalSlider->maximum());
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
        

        
        // Apply frame scaling immediately if we have a valid pixmap
        if (!ui->videoLabel->pixmap().isNull()) {
            QPixmap currentPixmap = ui->videoLabel->pixmap();
            qDebug() << "Applying frame scaling with factor:" << m_personScaleFactor;
            
            QSize originalSize = currentPixmap.size();
            int newWidth = qRound(originalSize.width() * m_personScaleFactor);
            int newHeight = qRound(originalSize.height() * m_personScaleFactor);
            
            QPixmap scaledPixmap = currentPixmap.scaled(
                newWidth, newHeight,
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation
            );
            
            ui->videoLabel->setPixmap(scaledPixmap);
            qDebug() << "Applied frame scaling: Original size:" << originalSize << "New size:" << scaledPixmap.size();
        } else {
            qDebug() << "No pixmap available for frame scaling";
        }
    }
    // --- END SCALING FUNCTIONALITY ---
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
}

QPixmap Capture::applyPersonScaling(const QPixmap& originalPixmap, const QRect& personRect, double scaleFactor)
{
    if (originalPixmap.isNull() || personRect.isEmpty() || qAbs(scaleFactor - 1.0) < 0.01) {
        return originalPixmap; // Return original if no scaling needed
    }
    
    qDebug() << "Applying scaling: factor=" << scaleFactor << "rect=" << personRect;
    
    QPixmap result = originalPixmap.copy();
    QPainter painter(&result);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    
    // Extract the person region from the original image
    QPixmap personRegion = originalPixmap.copy(personRect);
    
    // Scale the person region to the new size
    int scaledWidth = qRound(personRect.width() * scaleFactor);
    int scaledHeight = qRound(personRect.height() * scaleFactor);
    
    QPixmap scaledPersonRegion = personRegion.scaled(
        scaledWidth, scaledHeight,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    
    // Calculate the center point of the original person region
    QPoint center = personRect.center();
    
    // Calculate the new position to keep the person centered
    int newX = center.x() - scaledPersonRegion.width() / 2;
    int newY = center.y() - scaledPersonRegion.height() / 2;
    
    // Ensure the scaled region stays within the image bounds
    newX = qBound(0, newX, originalPixmap.width() - scaledPersonRegion.width());
    newY = qBound(0, newY, originalPixmap.height() - scaledPersonRegion.height());
    
    // Draw the scaled person region back to the result
    painter.drawPixmap(newX, newY, scaledPersonRegion);
    painter.end();
    
    qDebug() << "Scaling complete: new size=" << scaledPersonRegion.size() << "new pos=(" << newX << "," << newY << ")";
    
    return result;
}


