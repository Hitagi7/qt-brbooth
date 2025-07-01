// capture.cpp
#include "capture.h"
#include "ui_capture.h"
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
{
    ui->setupUi(this);

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

            // REMOVED: m_isCameraReadyForCapture logic here
        } else {
            qWarning() << "Failed to convert cv::Mat to QImage!";
        }
    } else {
        qWarning() << "Failed to read frame from camera! Stopping timer.";
        cameraTimer->stop();
        ui->capture->setEnabled(false); // Also disable capture button if stream fails
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

void Capture::on_back_clicked()
{
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    // REMOVED: m_isCameraReadyForCapture check here

    if (m_currentCaptureMode == ImageCaptureMode) {
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
    } else {
        qDebug() << "Capture button clicked in Video Record Mode. (Video recording logic not yet implemented).";
    }
}

void Capture::updateCountdown()
{
    countdownValue--;
    if (countdownValue > 0){
        countdownLabel->setText(QString::number(countdownValue));
    } else {
        countdownTimer->stop();
        countdownLabel->hide();
        performImageCapture();

        // Always re-enable after countdown
        ui->capture->setEnabled(true);
    }
}

void Capture::performImageCapture()
{
    cv::Mat tempFrame; // Temporary frame for discards
    cv::Mat frameToCapture; // The actual frame we want to keep

    // REMOVED: m_needsInitialRecapture (double-take) logic

    // Standard discard for robustness (you can remove this loop too if you want the absolute minimum)
    int framesToDiscard = 10;
    qDebug() << "Discarding standard " << framesToDiscard << " frames before actual capture...";
    for (int i = 0; i < framesToDiscard; ++i) {
        if (!cap.read(tempFrame)) {
            qWarning() << "Warning: Failed to read a pre-capture dummy frame during discard phase.";
            break;
        }
    }
    // QThread::msleep(50); // Optional small delay

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
