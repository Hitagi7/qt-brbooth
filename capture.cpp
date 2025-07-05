#include "capture.h"
#include "ui_capture.h"
#include "videotemplate.h"
#include "iconhover.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QThread> // Still needed for QThread::msleep, even if minimal
#include <QElapsedTimer>

#include <opencv2/opencv.hpp>

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , videoLabel(nullptr)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode (ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)

{
    ui->setupUi(this);

    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);

    int tickStep = 10; // Define your desired tick and snap interval

    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides); // Try both sides
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep); // Make single steps snap to ticks

    // Optional: If you want valueChanged signal only on release
    // ui->verticalSlider->setTracking(false);

    ui->verticalSlider->setValue(100); // Initial value

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    // Disable capture button by default (will be enabled if camera opens)
    ui->capture->setEnabled(false);

    // --- Corrected Camera Opening Logic ---
    bool cameraOpenedSuccessfully = false;
    if (!cap.open(1)) {
        qWarning() << "Error: Could not open camera with index 1. Trying index 0...";
        if (cap.open(0)) { // Only try index 0 if index 1 failed
            cameraOpenedSuccessfully = true;
        } else {
            qWarning() << "Error: Could not open camera with index 0 either. Please check camera connection and drivers.";
        }
    } else {
        cameraOpenedSuccessfully = true; // Camera 1 opened successfully
    }

    if (!cameraOpenedSuccessfully) {
        // Handle no camera found scenario
        qWarning() << "No camera found or could not be opened. Disabling capture.";

        // Display error message in the video feed area
        ui->videoFeedWidget->setStyleSheet("background-color: #333; color: white; border-radius: 10px;"); // Darker grey
        QLabel *errorLabel = new QLabel("Camera not available.\nCheck connection and drivers.", ui->videoFeedWidget);
        errorLabel->setAlignment(Qt::AlignCenter);
        QFont errorFont = errorLabel->font();
        errorFont.setPointSize(18);
        errorFont.setBold(true);
        errorLabel->setFont(errorFont);
        errorLabel->setStyleSheet("color: #FF5555;"); // Red text for error

        QVBoxLayout *errorLayout = new QVBoxLayout(ui->videoFeedWidget);
        errorLayout->addWidget(errorLabel);
        errorLayout->setContentsMargins(20, 20, 20, 20); // Add some padding

        return; // Exit constructor if no camera is available
    }

    // --- If camera opened successfully, proceed with setup ---

    // Attempt to set camera resolution and FPS
    qDebug() << "Attempting to set camera resolution to 1280x720.";
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    qDebug() << "Attempting to set camera FPS to 60.";
    cap.set(cv::CAP_PROP_FPS, 60.0);

    // --- DIAGNOSTIC: CHECK ACTUAL CAMERA SETTINGS ---
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

    // Video feed label setup
    videoLabel = new QLabel(ui->videoFeedWidget);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setScaledContents(true);

    // Layout for video feed widget
    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoLabel);
    videoLayout->setContentsMargins(0, 0, 0, 0);

    // Camera timer for continuous feed updates
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(1000 / 60); // Target 60 FPS updates

    // Enable capture button immediately after camera is opened and stream starts
    ui->capture->setEnabled(true);

    // CountdownTimer Setup
    countdownLabel = new QLabel(ui->videoFeedWidget);
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

    // Record Timer Setup
    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    // FIXED: Use precise timer for recording - calculate exact interval
    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    qDebug() << "OpenCV Camera started successfully!";
    qDebug() << "OpenCV Camera display timer started for 60 FPS.";
}

Capture::~Capture()
{
    // Stop and delete the timer
    if (cameraTimer){
        cameraTimer->stop();
        delete cameraTimer;
        cameraTimer = nullptr;
    }

    if (countdownTimer){
        countdownTimer->stop();
        delete countdownTimer;
        countdownTimer = nullptr;
    }

    if(recordTimer){
        recordTimer->stop();
        delete recordTimer;
        recordTimer = nullptr;
    }

    if(recordingFrameTimer){
        recordingFrameTimer->stop();
        delete recordingFrameTimer;
        recordingFrameTimer = nullptr;
    }

    // Release the camera resource
    if (cap.isOpened()) {
        cap.release();
    }

    delete ui; // Deletes the UI components, including videoFeedWidget and its children like videoLabel
}

void Capture::resizeEvent(QResizeEvent *event){
    //Ensures countdownLabel is centered when widget resizes
    if(countdownLabel){
        int x = (ui->videoFeedWidget->width() - countdownLabel->width())/2;
        int y = (ui->videoFeedWidget->height() - countdownLabel->height())/2;
        countdownLabel->move(x,y);
    }
    QWidget::resizeEvent(event);
}

void Capture::setCaptureMode(CaptureMode mode){
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to: " << (mode == ImageCaptureMode ? "Image Capture Mode" : "Video Record Mode");
}

void Capture::setVideoTemplate(const VideoTemplate &templateData)
{
    m_currentVideoTemplate = templateData;
}

void Capture::updateCameraFeed()
{
    static QElapsedTimer frameTimer;
    static qint64 frameCount = 0;
    static qint64 totalTime = 0;

    if (frameCount == 0) {
        frameTimer.start();
    }

    QElapsedTimer loopTimer;
    loopTimer.start();

    if (!videoLabel || !cap.isOpened()) {
        return;
    }

    cv::Mat frame;
    if (cap.read(frame)) {
        if (frame.empty()) {
            qWarning() << "Read empty frame from camera!";
            return;
        }

        cv::flip(frame, frame, 1);

        QImage image = cvMatToQImage(frame);

        if (!image.isNull()) {
            QPixmap pixmap = QPixmap::fromImage(image);
            videoLabel->setPixmap(pixmap.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
        } else {
            qWarning() << "Failed to convert cv::Mat to QImage!";
        }

        // Recording is now handled by separate timer - no frame capture here
    } else {
        qWarning() << "Failed to read frame from camera! Stopping timer.";
        cameraTimer->stop();
        ui->capture->setEnabled(false); // Also disable capture button if stream fails
        if (m_isRecording){
            stopRecording();
        }
    }

    qint64 loopTime = loopTimer.elapsed();
    totalTime += loopTime;
    frameCount++;

    if (frameCount % 60 == 0) {
        qDebug() << "----------------------------------------";
        qDebug() << "Avg loop time (last 60 frames):" << (double)totalTime / frameCount << "ms";
        qDebug() << "Current FPS (measured over 60 frames):" << 1000.0 / ((double)frameTimer.elapsed() / frameCount) << "FPS";
        qDebug() << "----------------------------------------";
        frameCount = 0;
        totalTime = 0;
        frameTimer.start();
    }
}

void Capture::captureRecordingFrame()
{
    if (!m_isRecording || !cap.isOpened()) {
        return;
    }

    // FIXED: Simple approach - just capture every time the timer fires
    cv::Mat frame;
    if (cap.read(frame)) {
        if (frame.empty()) {
            qWarning() << "Read empty frame for recording!";
            return;
        }

        cv::flip(frame, frame, 1);

        QImage imageToStore = cvMatToQImage(frame);
        if (!imageToStore.isNull()) {
            m_recordedFrames.append(QPixmap::fromImage(imageToStore));
            qDebug() << "Captured frame #" << m_recordedFrames.size() << " for recording";
        } else {
            qWarning() << "Failed to convert cv::Mat to QImage for recording!";
        }
    } else {
        qWarning() << "Failed to read frame from camera for recording!";
    }
}

void Capture::on_back_clicked()
{
    // 1. Stop and hide the countdown timer/label if active
    if (countdownTimer->isActive()) {
        countdownTimer->stop();
        countdownLabel->hide();
        // Reset countdown value for the next time
        countdownValue = 0;
        qDebug() << "Countdown stopped by back button.";
    }

    // 2. Stop any ongoing video recording
    if (m_isRecording) {
        stopRecording();
        qDebug() << "Recording stopped by back button.";
    }
    // else if the capture button is disabled (e.g., during photo countdown or just after capture start)
    // ensure it's re-enabled and reset for a potential next photo capture
    else if (!ui->capture->isEnabled()) {
        ui->capture->setEnabled(true);
        qDebug() << "Capture button reset by back button.";
    }

    // 3. Emit the signal to go back to the previous page
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
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
    qDebug() << "Recording: " << m_recordedSeconds << " / " << m_currentVideoTemplate.durationSeconds << " seconds";
    qDebug() << "Total frames captured so far: " << m_recordedFrames.size();
    qDebug() << "Expected frames so far: " << (m_recordedSeconds * m_targetRecordingFPS);
    qDebug() << "Actual FPS so far: " << ((double)m_recordedFrames.size() / m_recordedSeconds);

    //Stop automatically after m_currentVideoTemplate.durationSeconds
    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        stopRecording();
    }
}

void Capture::startRecording()
{
    if(!cap.isOpened()){
        qWarning() << "Cannot start recording: Camera not open!";
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;

    // FIXED: Use precise interval calculation for exactly 60 FPS
    int frameIntervalMs = 1000 / m_targetRecordingFPS; // 16.67ms for 60 FPS

    recordTimer->start(1000); // For tracking seconds
    recordingFrameTimer->start(frameIntervalMs); // Precise timing for 60 FPS

    qDebug() << "Recording started at" << m_targetRecordingFPS << "FPS";
    qDebug() << "Frame interval: " << frameIntervalMs << "ms";
}

void Capture::stopRecording()
{
    if (!m_isRecording)
    {
        return;
    }

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;

    qDebug() << "Recording stopped. Captured " << m_recordedFrames.size() << " frames.";
    qDebug() << "Expected frames for " << m_recordedSeconds << " seconds at " << m_targetRecordingFPS << " FPS: " << (m_recordedSeconds * m_targetRecordingFPS);
    qDebug() << "Actual average FPS: " << ((double)m_recordedFrames.size() / m_recordedSeconds);

    if (!m_recordedFrames.isEmpty()) {
        // MODIFIED: Emit the list of QPixmaps
        emit videoRecorded(m_recordedFrames);
    } else {
        qWarning() << "No frames recorded for video!";
    }

    emit showFinalOutputPage(); // Transition to final page
    ui->capture->setEnabled(true);
}

void Capture::performImageCapture()
{
    cv::Mat frameToCapture; // The actual frame we want to keep

    // --- Actual capture of the final frame ---
    if (cap.read(frameToCapture)){
        if(frameToCapture.empty()){
            qWarning() << "Captured an empty frame!";
            return;
        }

        qDebug() << "Captured frame size: "
                 << frameToCapture.cols << "x" << frameToCapture.rows;  // <--- Add this

        cv::flip(frameToCapture, frameToCapture, 1);

        QImage capturedImageQ = cvMatToQImage(frameToCapture);

        if(!capturedImageQ.isNull()){
            m_capturedImage = QPixmap::fromImage(capturedImageQ);
            qDebug() << "Image captured successfully! Size: " << m_capturedImage.size();
            emit imageCaptured(m_capturedImage);
        }
        else{
            qWarning() << "Failed to convert captured cv::Mat to QImage!";
        }
    } else{
        qWarning() << "Failed to read frame from camera for actual capture!";
    }
    emit showFinalOutputPage();
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4:
    {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    }
    case CV_8UC3:
    {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    case CV_8UC1:
    {
        static QVector<QRgb> sColorTable;
        if (sColorTable.isEmpty())
        {
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        }
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }
    default:
        qWarning() << "cvMatToQImage - Mat type not handled: " << mat.type();
        return QImage();
    }
}

void Capture::on_verticalSlider_valueChanged(int value)
{
    int tickInterval = ui->verticalSlider->tickInterval();
    if (tickInterval == 0) return; // Avoid division by zero

    // Calculate the nearest tick value
    int snappedValue = qRound( (double)value / tickInterval ) * tickInterval;

    // Optional: Ensure it's within range
    snappedValue = qBound(ui->verticalSlider->minimum(), snappedValue, ui->verticalSlider->maximum());

    // If the current value is not already snapped, set it to the snapped value
    if (value != snappedValue) {
        ui->verticalSlider->setValue(snappedValue);
    }
}

