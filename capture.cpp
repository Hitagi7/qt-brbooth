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
#include <QVBoxLayout>
#include <opencv2/opencv.hpp>
//test
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
{
    ui->setupUi(this);


    // IMMEDIATELY setup stacked layout after UI setup
    setupStackedLayoutHybrid();

    // Update overlay styles for better visibility
    updateOverlayStyles();

    // Setup slider
    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setValue(100);

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
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60.0);

    // Start camera timer for updates
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(1000 / 60);

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

    // Setup timers
    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    // Connect signals/slots for buttons and slider
    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);

    qDebug() << "OpenCV Camera started successfully with hybrid stacked layout!";
}

void Capture::setupStackedLayoutHybrid()
{
    qDebug() << "Setting up hybrid stacked layout...";

    // CRITICAL: Set widget properties BEFORE layout manipulation
    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setScaledContents(true);
    ui->videoLabel->setStyleSheet("background-color: black;");

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setStyleSheet("background: transparent;");

    // FORCE the overlayWidget to be on top with raise() and higher Z-order
    ui->overlayWidget->raise();
    ui->overlayWidget->setAttribute(Qt::WA_TranslucentBackground, false);
    ui->overlayWidget->setAttribute(Qt::WA_NoSystemBackground, false);

    // Make sure buttons are visible and have proper Z-order
    ui->back->raise();
    ui->capture->raise();
    ui->verticalSlider->raise();
    ui->back->show();
    ui->capture->show();
    ui->verticalSlider->show();

    // Get the current layout and remove widgets from it
    QLayout *currentLayout = layout();
    if (currentLayout) {
        qDebug() << "Removing widgets from current layout...";

        // Remove widgets from current layout
        currentLayout->removeWidget(ui->videoLabel);
        currentLayout->removeWidget(ui->overlayWidget);

        // Delete the old layout
        delete currentLayout;
    }

    // Create new stacked layout
    stackedLayout = new QStackedLayout;
    stackedLayout->setStackingMode(QStackedLayout::StackAll); // CRITICAL: Allow all widgets to be visible

    // Add widgets to stacked layout
    stackedLayout->addWidget(ui->videoLabel);    // Background layer (index 0)
    stackedLayout->addWidget(ui->overlayWidget); // Foreground layer (index 1)

    // Ensure the overlay is the current/top widget
    stackedLayout->setCurrentWidget(ui->overlayWidget);

    // Set the stacked layout as the main layout
    setLayout(stackedLayout);

    // FORCE update and show
    ui->overlayWidget->update();
    ui->overlayWidget->show();
    ui->back->update();
    ui->capture->update();
    ui->verticalSlider->update();


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
    if(countdownLabel){
        int x = (ui->overlayWidget->width() - countdownLabel->width())/2;
        int y = (ui->overlayWidget->height() - countdownLabel->height())/2;
        countdownLabel->move(x,y);
    }
    QWidget::resizeEvent(event);
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
}

void Capture::updateCameraFeed()
{
    if (!ui->videoLabel || !cap.isOpened()) return;

    cv::Mat frame;
    if (cap.read(frame)) {
        if (frame.empty()) return;
        cv::flip(frame, frame, 1);
        if (frame.empty()) {
            qWarning() << "Read empty frame from camera!";
            return;
        }

        // cv::flip(frame, frame, 1);

        QImage image = cvMatToQImage(frame);
        if (!image.isNull()) {
            QPixmap pixmap = QPixmap::fromImage(image);

            // FORCE the pixmap to fill the EXACT size of the widget
            QSize widgetSize = this->size(); // Use main widget size, not videoLabel size
            QPixmap scaledPixmap = pixmap.scaled(
                widgetSize,
                Qt::KeepAspectRatioByExpanding,
                Qt::SmoothTransformation
                );

            // If the scaled pixmap is larger than the widget, crop it
            if (scaledPixmap.size() != widgetSize) {
                int x = (scaledPixmap.width() - widgetSize.width()) / 2;
                int y = (scaledPixmap.height() - widgetSize.height()) / 2;
                scaledPixmap = scaledPixmap.copy(x, y, widgetSize.width(), widgetSize.height());
            }

            ui->videoLabel->setPixmap(scaledPixmap);
        }
    }
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
