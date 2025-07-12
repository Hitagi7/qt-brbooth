#include "capture.h"
#include "ui_capture.h"
#include "foreground.h" // Needed for Foreground class

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
#include <opencv2/opencv.hpp>

Capture::Capture(QWidget *parent, Foreground *fg)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode) // Using enum directly
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5) // Using struct constructor
    , m_recordedSeconds(0)
    , stackedLayout(nullptr)
    , totalTime(0)            // Only one initialization
    , frameCount(0)           // Only one initialization
    , foreground(fg)
    , overlayImageLabel(nullptr) // Initialize here, then new in body
{
    ui->setupUi(this);

    setContentsMargins(0, 0, 0, 0);

    // QT Foreground Overlay ========================================================
    overlayImageLabel = new QLabel(ui->overlayWidget); // Create the QLabel
    QString selectedOverlay;
    if (foreground) {
        selectedOverlay = foreground->getSelectedForeground();
    } else {
        qWarning() << "Error: foreground is nullptr!";
    }
    qDebug() << "Selected overlay path:" << selectedOverlay;
    overlayImageLabel->setAttribute(Qt::WA_TranslucentBackground);
    overlayImageLabel->setStyleSheet("background: transparent;");
    overlayImageLabel->setScaledContents(true); // Optional: scale the pixmap
    overlayImageLabel->resize(this->size());
    overlayImageLabel->hide();
    connect(foreground, &Foreground::foregroundChanged, this, &Capture::updateForegroundOverlay);
    qDebug() << "Selected overlay path:" << selectedOverlay; // This line is redundant, already printed above
    QPixmap overlayPixmap(selectedOverlay);
    overlayImageLabel->setPixmap(overlayPixmap);

    // IMMEDIATELY setup stacked layout after UI setup
    setupStackedLayoutHybrid(); // This will now add stackedLayout to mainLayout

    // Force initial size match - important for correct rendering at start
    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());
    // Also resize the overlayImageLabel since it's now part of the stacked layout
    if (overlayImageLabel) {
        overlayImageLabel->resize(this->size());
    }

    ui->videoLabel->show();
    ui->overlayWidget->show();
    ui->videoLabel->update();
    ui->overlayWidget->update();

    updateGeometry(); // Ensures layout calculations are up-to-date

    updateOverlayStyles();

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;");
    ui->videoLabel->setScaledContents(false); // We're scaling manually in updateCameraFeed
    ui->videoLabel->setAlignment(Qt::AlignCenter); // Center the image within the label

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
    ui->verticalSlider->setValue(0);

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    ui->capture->setEnabled(false);

    // Camera open logic
    bool cameraOpenedSuccessfully = false;
    if (!cap.open(1)) {
        qWarning() << "Error: Could not open camera with index 1. Trying index 0...";
        if (cap.open(0)) cameraOpenedSuccessfully = true;
        else qWarning() << "Error: Could not open camera with index 0 either.";
    } else {
        cameraOpenedSuccessfully = true;
    }

    if (!cameraOpenedSuccessfully) {
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText("Camera not available.\nCheck connection and drivers.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->capture->setEnabled(false);
        return;
    }

    qDebug() << "Attempting to set camera resolution to 1280x720.";
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    qDebug() << "Attempting to set camera FPS to 60.";
    cap.set(cv::CAP_PROP_FPS, 60.0);

    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    qDebug() << "========================================";
    qDebug() << "Camera settings REQUESTED: 1280x720 @ 60 FPS";
    qDebug() << "Camera settings ACTUAL: " << actual_width << "x" << actual_height << " @ " << actual_fps << " FPS";
    qDebug() << "========================================";
    if (actual_fps < 59) {
        qWarning() << "WARNING: Camera did not accept 60 FPS request. Actual FPS is" << actual_fps;
    }

    // Use actual camera FPS for timer interval
    int timerInterval = qMax(1, static_cast<int>(1000.0 / actual_fps));
    qDebug() << "Setting camera timer interval to" << timerInterval << "ms for" << actual_fps << "FPS";

    cameraTimer = new QTimer(this);
    // FIX: Set cameraTimer to PreciseTimer for smoother updates
    cameraTimer->setTimerType(Qt::PreciseTimer);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(timerInterval);

    // Initialize and start performance timers
    loopTimer.start();
    frameTimer.start();

    ui->capture->setEnabled(true);

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
    qDebug() << "OpenCV Camera display timer started for" << actual_fps << "FPS";

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);

    qDebug() << "OpenCV Camera started successfully with hybrid stacked layout and optimized FPS!";
}

Capture::~Capture()
{
    // Ensure all timers are stopped and deleted
    if (cameraTimer){ cameraTimer->stop(); delete cameraTimer; cameraTimer = nullptr; }
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; countdownTimer = nullptr; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; recordTimer = nullptr; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; recordingFrameTimer = nullptr; }
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; } // Delete overlayImageLabel
    if (cap.isOpened()) { cap.release(); } // Release camera
    delete ui;
}

void Capture::setupStackedLayoutHybrid()
{
    qDebug() << "Setting up hybrid stacked layout...";

    // Ensure videoLabel and overlayWidget are parented to 'this'
    ui->videoLabel->setParent(this);
    ui->overlayWidget->setParent(this);

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

        stackedLayout->addWidget(ui->videoLabel);   // Background
        stackedLayout->addWidget(ui->overlayWidget); // Foreground (for buttons, countdown, etc.)
        stackedLayout->addWidget(overlayImageLabel); // Overlay image on top of everything else

        // Replace existing layout if needed
        if (layout()) {
            delete layout(); // remove any previous layout
        }

        // Create the main QGridLayout and add the stackedLayout to it
        QGridLayout* mainLayout = new QGridLayout(this);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);
        mainLayout->addLayout(stackedLayout, 0, 0);
        mainLayout->setRowStretch(0, 1);
        mainLayout->setColumnStretch(0, 1);

        setLayout(mainLayout); // Set this as the main layout for the Capture widget
    }

    ui->overlayWidget->raise();
    ui->back->raise();
    ui->capture->raise();
    ui->verticalSlider->raise();
    // Ensure overlayImageLabel is on top of the video feed but below controls
    if (overlayImageLabel) {
        overlayImageLabel->raise();
    }
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

    // Ensure labels and overlay widgets match the parent's size
    ui->videoLabel->resize(size());
    ui->overlayWidget->resize(size());

    // Resize the overlay image label too
    if (overlayImageLabel) {
        overlayImageLabel->resize(size()); // Resize to Capture widget's size
        overlayImageLabel->move(0, 0);     // Position at top-left of Capture widget
    }

    // Reposition countdown label to center
    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
}

void Capture::updateCameraFeed()
{
    // FIX: Removed isProcessingFrame flag. Relying on QTimer interval and cap.read()
    //      to manage frame processing flow.

    // Start loopTimer at the very beginning of the function to measure total time for one frame update cycle.
    loopTimer.start();

    if (!ui->videoLabel || !cap.isOpened()) {
        loopTimer.restart(); // Restart loopTimer for the next attempt
        return;
    }

    cv::Mat frame;

    if (!cap.read(frame) || frame.empty()) {
        qWarning() << "Failed to read frame from camera or frame is empty.";
        loopTimer.restart(); // Restart for the next attempt
        return;
    }

    cv::flip(frame, frame, 1); // Mirror the frame

    QImage image = cvMatToQImage(frame);
    if (image.isNull()) {
        qWarning() << "Failed to convert cv::Mat to QImage.";
        loopTimer.restart(); // Restart for the next attempt
        return;
    }

    QPixmap pixmap = QPixmap::fromImage(image);

    // Scale the pixmap to fill the QLabel
    QSize labelSize = ui->videoLabel->size();

    // Use FastTransformation for real-time video (much faster than SmoothTransformation)
    QPixmap scaledPixmap = pixmap.scaled(
        labelSize,
        Qt::KeepAspectRatioByExpanding, // Use KeepAspectRatioByExpanding to fill the label completely
        Qt::FastTransformation
        );

    ui->videoLabel->setPixmap(scaledPixmap);
    ui->videoLabel->setAlignment(Qt::AlignCenter); // Align the scaled pixmap center
    ui->videoLabel->update(); // Explicitly request repaint

    // --- Performance stats ---
    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Print stats every 60 frames (or chosen interval)
    if (frameCount % 60 == 0) {
        double avgLoopTime = (double)totalTime / frameCount;
        // The frameTimer measures the total time over 'frameCount' actual frame updates.
        // This gives a more accurate FPS over that period.
        double measuredFPS = 1000.0 / ((double)frameTimer.elapsed() / frameCount);
        qDebug() << "----------------------------------------";
        qDebug() << "Avg loop time (last 60 frames):" << avgLoopTime << "ms";
        qDebug() << "Current FPS (measured over 60 frames):" << measuredFPS << "FPS";
        qDebug() << "Frame processing efficiency:" << (avgLoopTime < 1000.0 / cap.get(cv::CAP_PROP_FPS) ? "GOOD" : "NEEDS OPTIMIZATION");
        qDebug() << "----------------------------------------";
        // Reset timers for next batch
        frameCount = 0;
        totalTime = 0;
        frameTimer.start(); // Restart frameTimer for the next 60-frame measurement
    }
}

void Capture::captureRecordingFrame()
{
    if (!m_isRecording || !cap.isOpened()) return;

    cv::Mat frame;
    // It's important to read a fresh frame for recording, not reuse the display frame
    if (cap.read(frame)) {
        if (frame.empty()) return;
        cv::flip(frame, frame, 1);
        QImage imageToStore = cvMatToQImage(frame);
        if (!imageToStore.isNull()) {
            m_recordedFrames.append(QPixmap::fromImage(imageToStore));
        } else {
            qWarning() << "Failed to convert frame to QImage for recording.";
        }
    } else {
        qWarning() << "Failed to read frame for recording.";
    }
}

void Capture::on_back_clicked()
{
    if (countdownTimer->isActive()) {
        countdownTimer->stop();
        countdownLabel->hide();
        countdownValue = 0;
    }
    if (m_isRecording) {
        stopRecording();
    } else { // No else if needed, this handles the case where not recording and not in countdown
        ui->capture->setEnabled(true);
    }
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    ui->capture->setEnabled(false); // Disable button to prevent multiple clicks during countdown
    countdownValue = 5; // Start countdown from 5
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();

    QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
    animation->setDuration(300);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->start();

    countdownTimer->start(1000); // Tick every second
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
            ui->capture->setEnabled(true); // Re-enable capture button after image capture
        }else if(m_currentCaptureMode == VideoRecordMode){
            startRecording();
            // Button remains disabled during recording, re-enabled in stopRecording()
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

void Capture::startRecording()
{
    if(!cap.isOpened()){
        qWarning() << "Cannot start recording: Camera not opened.";
        ui->capture->setEnabled(true); // Re-enable if camera isn't ready
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;
    // Use the target recording FPS for frame capture interval
    // This is distinct from the display FPS if you want to record at a specific rate.
    int frameIntervalMs = 1000 / m_targetRecordingFPS;
    recordTimer->start(1000); // Update recording duration every second
    recordingFrameTimer->start(frameIntervalMs); // Capture frames at target FPS
    qDebug() << "Recording started at target FPS:" << m_targetRecordingFPS << "frames/sec";
}

void Capture::stopRecording()
{
    if (!m_isRecording) return; // Only stop if currently recording

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;
    qDebug() << "Recording stopped. Captured" << m_recordedFrames.size() << "frames.";

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    emit showFinalOutputPage();
    ui->capture->setEnabled(true); // Re-enable capture button after recording
}

void Capture::performImageCapture()
{
    cv::Mat frameToCapture;
    if (cap.read(frameToCapture)){
        if(frameToCapture.empty()) {
            qWarning() << "Captured frame is empty.";
            return;
        }
        cv::flip(frameToCapture, frameToCapture, 1);
        QImage capturedImageQ = cvMatToQImage(frameToCapture);
        if(!capturedImageQ.isNull()){
            m_capturedImage = QPixmap::fromImage(capturedImageQ);
            emit imageCaptured(m_capturedImage);
            qDebug() << "Image captured successfully.";
        } else {
            qWarning() << "Failed to convert captured frame to QImage.";
        }
    } else {
        qWarning() << "Failed to read frame for image capture.";
    }
    emit showFinalOutputPage();
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: { // BGRA with alpha channel
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
        // Using QImage constructor that doesn't copy data if source is const and aligned
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    }
    case CV_8UC3: { // BGR (common from OpenCV)
        // Convert BGR to RGB by swapping Red and Blue channels
        // QImage::Format_RGB888 expects RGB byte order. OpenCV's default is BGR.
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped(); // Efficiently swaps R and B channels
    }
    case CV_8UC1: { // Grayscale
        static QVector<QRgb> sColorTable;
        if (sColorTable.isEmpty()) {
            for (int i = 0; i < 256; ++i) {
                sColorTable.push_back(qRgb(i, i, i));
            }
        }
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image; // No copy needed unless you modify the QImage later
    }
    default:
        qWarning() << "Unsupported cv::Mat format: " << mat.type();
        return QImage(); // Return null QImage for unsupported types
    }
}

void Capture::on_verticalSlider_valueChanged(int value)
{
    int tickInterval = ui->verticalSlider->tickInterval();
    if (tickInterval == 0) return; // Avoid division by zero
    int snappedValue = qRound((double)value / tickInterval) * tickInterval;
    snappedValue = qBound(ui->verticalSlider->minimum(), snappedValue, ui->verticalSlider->maximum());
    if (value != snappedValue) {
        ui->verticalSlider->setValue(snappedValue);
    }
    // You might want to do something with the 'snappedValue' here,
    // e.g., adjust a camera setting like zoom or exposure if the slider controls that.
}

void Capture::updateForegroundOverlay(const QString &path)
{
    qDebug() << "Foreground overlay updated to:" << path;

    if (!overlayImageLabel) {
        qWarning() << "overlayImageLabel is null!";
        return;
    }

    QPixmap overlayPixmap(path);
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->resize(this->size()); // Ensure it scales with window
    overlayImageLabel->show();
}
