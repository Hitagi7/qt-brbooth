#include "capture.h"
#include "ui_capture.h"
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QElapsedTimer>
#include <QVBoxLayout> // Still needed for countdownLabel positioning, but not for main layout
#include <QGridLayout> // New: For main layout
#include <opencv2/opencv.hpp>

// Removed: #include <QPainter> // No longer needed for debugging borders

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)
    , stackedLayout(nullptr)
    , videoLabelFPS(nullptr) // Initialize pointer to nullptr
    , totalTime(0)           // Initialize totalTime
    , frameCount(0)          // Initialize frameCount
    , isProcessingFrame(false) // Initialize frame processing flag
{
    ui->setupUi(this);

    // Ensure no margins or spacing for the Capture widget itself
    setContentsMargins(0, 0, 0, 0);

    // Setup the main grid layout for the Capture widget
    QGridLayout* mainLayout = new QGridLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // IMMEDIATELY setup stacked layout after UI setup
    setupStackedLayoutHybrid(); // This will now add stackedLayout to mainLayout

    // Force initial size match
    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());

    // Force show and update
    ui->videoLabel->show();
    ui->overlayWidget->show();
    ui->videoLabel->update();
    ui->overlayWidget->update();

    // CRITICAL FIX: Explicitly set the size of videoLabel and overlayWidget
    // to match the Capture widget's initial size immediately after layout setup.
    // This addresses the initial QSize(100, 30) issue.
    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());
    // Force a repaint and layout update
    updateGeometry();

    // Update overlay styles for better visibility
    updateOverlayStyles();

    // Ensure videoLabel fills and paints properly
    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;");
    ui->videoLabel->setScaledContents(false);  // ❗ Disable it – we're scaling manually
    ui->videoLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    // Also ensure overlayWidget expands fully
    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    // Corrected: ui->videoLabel->setStyleSheet was applied twice, this one is for overlayWidget
    ui->overlayWidget->setStyleSheet("background-color: transparent;"); // Ensure overlay is transparent

    // Setup slider
    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setValue(0);

    // Setup icons and button states
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

    // Set camera properties
    qDebug() << "Attempting to set camera resolution to 1280x720.";
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Request 60 FPS from the camera hardware
    qDebug() << "Attempting to set camera FPS to 60.";
    cap.set(cv::CAP_PROP_FPS, 60.0);

    // --- DIAGNOSTIC 1: CHECK ACTUAL CAMERA SETTINGS ---
    // Check what the camera driver actually settled on.
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

    // ✅ FIX 1: Use actual camera FPS for timer instead of fixed 60 FPS
    int timerInterval = qMax(1, static_cast<int>(1000.0 / actual_fps));
    qDebug() << "Setting camera timer interval to" << timerInterval << "ms for" << actual_fps << "FPS";

    // Start camera timer for updates with dynamic interval
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(timerInterval); // ✅ Use actual camera FPS

    // Initialize and start performance timers
    // These should be started here to ensure valid elapsed times from the first frame.
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

    //QT Foreground Overlay ========================================================s
    QLabel* overlayImageLabel = new QLabel(ui->overlayWidget);
    QPixmap overlayPixmap(":/foreground/templates/foreground/2.png"); // Replace with the correct path
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->setScaledContents(true); // Make it scale with the window
    overlayImageLabel->setAttribute(Qt::WA_TransparentForMouseEvents); // Don't block input
    overlayImageLabel->resize(this->size());
    overlayImageLabel->move(0, 0); // Align top-left
    overlayImageLabel->show();

    // Setup timers
    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);
    qDebug() << "OpenCV Camera display timer started for" << actual_fps << "FPS";

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    // Connect signals/slots for buttons and slider
    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);

    qDebug() << "OpenCV Camera started successfully with hybrid stacked layout and optimized FPS!";
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

    // 🟥 Debug border (make sure this shows!)
    ui->videoLabel->setStyleSheet("background-color: black; border: 5px solid red;");
    ui->overlayWidget->setStyleSheet("background-color: transparent; border: 5px dashed green;");

    // Fix stacking
    if (!stackedLayout) {
        stackedLayout = new QStackedLayout;
        stackedLayout->setStackingMode(QStackedLayout::StackAll);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        stackedLayout->setSpacing(0);

        stackedLayout->addWidget(ui->videoLabel);      // Background
        stackedLayout->addWidget(ui->overlayWidget);  // Foreground

        // ⬇️ Replace existing layout if needed
        if (layout()) {
            delete layout(); // remove any previous layout
        }

        QGridLayout* mainLayout = new QGridLayout(this);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);
        mainLayout->addLayout(stackedLayout, 0, 0);
        mainLayout->setRowStretch(0, 1);
        mainLayout->setColumnStretch(0, 1);

        setLayout(mainLayout);
    }

    ui->overlayWidget->raise();
    ui->back->raise();
    ui->capture->raise();
    ui->verticalSlider->raise();
}

void Capture::updateOverlayStyles()
{
    qDebug() << "Updating overlay styles with clean professional appearance...";

    // Update back button - clean transparent style
    ui->back->setStyleSheet(
        "QPushButton {"
        "    background: transparent;"
        "    border: none;"
        "    color: white;"
        "}"
        );

    // Update capture button - clean green style matching your original design
    ui->capture->setStyleSheet(
        "QPushButton {"
        "    border-radius: 9px;"
        "    border-bottom: 3px solid rgba(2, 2, 2, 200);" // Subtle shadow
        "    background: rgba(11, 194, 0, 200);" // Your original green color
        "    color: white;"
        "    font-size: 16px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background: rgba(8, 154, 0, 230);" // Your original hover color
        "}"
        "QPushButton:disabled {"
        "    background: rgba(100, 100, 100, 150);"
        "    color: rgba(200, 200, 200, 150);"
        "    border-bottom: 3px solid rgba(50, 50, 50, 150);"
        "}"
        );

    // Update slider - clean style matching your original design
    ui->verticalSlider->setStyleSheet(
        "QSlider::groove:vertical {"
        "    background: rgba(0, 0, 0, 80);" // Semi-transparent dark groove
        "    width: 30px;"
        "    border-radius: 15px;"
        "    border: none;"
        "}"
        "QSlider::handle:vertical {"
        "    background: rgba(13, 77, 38, 220);" // Your original dark green handle
        "    border: 1px solid rgba(30, 144, 255, 180);" // Subtle blue border
        "    width: 60px;"
        "    height: 13px;"
        "    border-radius: 7px;"
        "    margin: 0 -15px;"
        "}"
        "QSlider::sub-page:vertical {"
        "    background: rgba(0, 0, 0, 60);" // Semi-transparent filled part
        "    border-top-left-radius: 15px;"
        "    border-top-right-radius: 15px;"
        "    border-bottom-left-radius: 0px;"
        "    border-bottom-right-radius: 0px;"
        "}"
        "QSlider::add-page:vertical {"
        "    background: rgba(11, 194, 0, 180);" // Your original green color
        "    border-bottom-left-radius: 15px;"
        "    border-bottom-right-radius: 15px;"
        "    border-top-left-radius: 0px;"
        "    border-top-right-radius: 0px;"
        "}"
        );

    // Ensure overlay widget is transparent
    ui->overlayWidget->setStyleSheet("background: transparent;");

    qDebug() << "Clean professional overlay styles applied";
}

Capture::~Capture()
{
    if (cameraTimer){ cameraTimer->stop(); delete cameraTimer; }
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; }
    if (cap.isOpened()) { cap.release(); }
    delete ui;
}

void Capture::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    ui->videoLabel->resize(size());
    ui->overlayWidget->resize(size());

    // Resize the overlay image label too
    for (QObject* child : ui->overlayWidget->children()) {
        QLabel* label = qobject_cast<QLabel*>(child);
        if (label && label != countdownLabel) {
            label->resize(size());
            label->move(0, 0);
        }
    }


    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }

    update();
    if (ui->videoLabel) ui->videoLabel->resize(this->size());
    if (ui->overlayWidget) ui->overlayWidget->resize(this->size());
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
}

void Capture::updateCameraFeed()
{
    // ✅ FIX 3: Frame skipping logic - Skip if still processing previous frame
    if (isProcessingFrame) {
        qDebug() << "Frame skipped - still processing previous frame";
        return;
    }

    isProcessingFrame = true;

    // Start loopTimer at the very beginning of the function to measure total time for one frame update cycle.
    loopTimer.start();

    if (!ui->videoLabel || !cap.isOpened()) {
        isProcessingFrame = false;
        return;
    }

    cv::Mat frame;

    if (!cap.read(frame) || frame.empty()) {
        // Restart loopTimer even if frame read fails, to prevent it from running indefinitely
        // before the next successful read.
        loopTimer.restart();
        isProcessingFrame = false;
        return;
    }

    cv::flip(frame, frame, 1); // Mirror

    QImage image = cvMatToQImage(frame);
    if (image.isNull()) {
        loopTimer.restart(); // Restart for the next attempt
        isProcessingFrame = false;
        return;
    }

    QPixmap pixmap = QPixmap::fromImage(image);

    // Force fill QLabel with NO aspect ratio
    QSize labelSize = ui->videoLabel->size();

    // ✅ FIX 2: Use FastTransformation for real-time video (much faster than SmoothTransformation)
    QPixmap scaledPixmap = pixmap.scaled(
        labelSize,
        Qt::IgnoreAspectRatio,
        Qt::FastTransformation  // ✅ Much faster for real-time video
        );

    ui->videoLabel->setPixmap(scaledPixmap);
    ui->videoLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    ui->videoLabel->update();

    // --- Performance stats ---
    // loopTime is the duration of the current frame processing
    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    // Print stats every 60 frames
    if (frameCount % 60 == 0) {
        double avgLoopTime = (double)totalTime / frameCount;
        double measuredFPS = 1000.0 / ((double)frameTimer.elapsed() / frameCount);
        qDebug() << "----------------------------------------";
        qDebug() << "Avg loop time (last 60 frames):" << avgLoopTime << "ms";
        qDebug() << "Current FPS (measured over 60 frames):" << measuredFPS << "FPS";
        qDebug() << "Frame processing efficiency:" << (avgLoopTime < 16.67 ? "GOOD" : "NEEDS OPTIMIZATION");
        qDebug() << "----------------------------------------";
        // Reset timers for next batch
        frameCount = 0;
        totalTime = 0;
        frameTimer.start(); // Restart frameTimer for the next 60-frame measurement
    }

    isProcessingFrame = false; // ✅ Mark frame processing as complete
}

void Capture::captureRecordingFrame()
{
    if (!m_isRecording || !cap.isOpened()) return;

    cv::Mat frame;
    if (cap.read(frame)) {
        if (frame.empty()) return;
        cv::flip(frame, frame, 1);
        QImage imageToStore = cvMatToQImage(frame);
        if (!imageToStore.isNull()) {
            m_recordedFrames.append(QPixmap::fromImage(imageToStore));
        }
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
    } else if (!ui->capture->isEnabled()) {
        ui->capture->setEnabled(true);
    }
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
    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        stopRecording();
    }
}

void Capture::startRecording()
{
    if(!cap.isOpened()){
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;
    int frameIntervalMs = 1000 / m_targetRecordingFPS;
    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
}

void Capture::stopRecording()
{
    if (!m_isRecording) return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    emit showFinalOutputPage();
    ui->capture->setEnabled(true);
}

void Capture::performImageCapture()
{
    cv::Mat frameToCapture;
    if (cap.read(frameToCapture)){
        if(frameToCapture.empty()) return;
        cv::flip(frameToCapture, frameToCapture, 1);
        QImage capturedImageQ = cvMatToQImage(frameToCapture);
        if(!capturedImageQ.isNull()){
            m_capturedImage = QPixmap::fromImage(capturedImageQ);
            emit imageCaptured(m_capturedImage);
        }
    }
    emit showFinalOutputPage();
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
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
}
