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
#include <QElapsedTimer> // For performance measurement

#include <opencv2/opencv.hpp>

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , videoLabel(nullptr)
{
    ui->setupUi(this);

    // ... (Your icon setup code remains the same)
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    if (!cap.open(1)) {
        qWarning() << "Error: Could not open camera with index 1. Trying index 0...";
        if (!cap.open(0)) {
            // ... (Your error handling remains the same)
            return;
        }
    }

    // --- SOLUTION A (MOST LIKELY FIX): REQUEST A LOWER RESOLUTION ---
    // Change 1920x1080 to 1280x720, which is much more likely to support 60 FPS.
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


    videoLabel = new QLabel(ui->videoFeedWidget);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setAlignment(Qt::AlignCenter);

    // Using Qt::FastTransformation is quicker than SmoothTransformation, which helps performance.
    videoLabel->setScaledContents(false); // We will handle scaling manually for better control

    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoLabel);
    videoLayout->setContentsMargins(0, 0, 0, 0);

    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(1000 / 60); // Target 60 FPS updates

    qDebug() << "OpenCV Camera display timer started for 60 FPS.";
}

Capture::~Capture()
{
    // Stop the timer
    if (cameraTimer && cameraTimer->isActive()) { // Check for nullptr before isActive()
        cameraTimer->stop();
        delete cameraTimer; // Clean up the QTimer
        cameraTimer = nullptr;
    }

    // Release the camera resource
    if (cap.isOpened()) {
        cap.release();
    }

    delete ui; // Deletes the UI components, including videoFeedWidget and its children like videoLabel
}

void Capture::updateCameraFeed()
{
    // --- DIAGNOSTIC 2: MEASURE EXECUTION TIME ---
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
            // --- SOLUTION B: OPTIMIZE SCALING ---
            // Use FastTransformation as it is less CPU-intensive.
            QPixmap pixmap = QPixmap::fromImage(image);
            videoLabel->setPixmap(pixmap.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
        } else {
            qWarning() << "Failed to convert cv::Mat to QImage!";
        }
    } else {
        qWarning() << "Failed to read frame from camera! Stopping timer.";
        cameraTimer->stop();
    }

    // --- DIAGNOSTIC 2 (continued): Print performance stats ---
    qint64 loopTime = loopTimer.elapsed();
    totalTime += loopTime;
    frameCount++;

    // Print stats every 60 frames
    if (frameCount % 60 == 0) {
        qDebug() << "----------------------------------------";
        qDebug() << "Avg loop time (last 60 frames):" << (double)totalTime / frameCount << "ms";
        qDebug() << "Current FPS (measured over 60 frames):" << 1000.0 / ((double)frameTimer.elapsed() / frameCount) << "FPS";
        qDebug() << "----------------------------------------";
        // Reset timers for next batch
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
    emit showFinalOutputPage();
}

// Helper function to convert cv::Mat to QImage
QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: // 8-bit, 4 channel (e.g., BGRA or RGBA)
    {
        // For OpenCV typically giving BGRA, QImage::Format_ARGB32 is ARGB.
        // It's safer to convert to a known format like RGB first for display.
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB); // Convert BGRA to RGB
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    }

    case CV_8UC3: // 8-bit, 3 channel (BGR in OpenCV)
    {
        // OpenCV uses BGR by default for 3-channel images from cameras/files.
        // Qt's QImage::Format_RGB888 expects RGB. So, a swap is necessary.
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped(); // Correctly swaps R and B channels
    }

    case CV_8UC1: // 8-bit, 1 channel (Grayscale)
    {
        static QVector<QRgb> sColorTable;
        // Only create our color table once
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
