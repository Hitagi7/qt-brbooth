// capture.cpp
#include "capture.h"
#include "ui_capture.h"
#include "iconhover.h" // Assuming this is a custom class for icon hover effects
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug> // For qDebug() output
#include <QImage> // For QImage conversion
#include <QPixmap> // For QPixmap conversion
#include <QTimer> // For QTimer

#include <opencv2/opencv.hpp>



Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr) // Initialize member variables
    , videoLabel(nullptr)  // Initialize member variables
{
    ui->setupUi(this);

    // Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    // --- OpenCV Camera integration starts here ---

    // Open the default camera (usually index 0)
    // Check if cap is opened successfully *before* proceeding
    if (!cap.open(0)) {
        qWarning() << "Error: Could not open camera with OpenCV!";
        ui->videoFeedWidget->setStyleSheet("background-color: grey; color: white; border-radius: 10px;");

        // Create error message label
        QLabel *errorLabel = new QLabel("Camera not available", ui->videoFeedWidget);
        errorLabel->setAlignment(Qt::AlignCenter);
        errorLabel->setStyleSheet("color: white; font-size: 16px;");

        QVBoxLayout *errorLayout = new QVBoxLayout(ui->videoFeedWidget);
        errorLayout->addWidget(errorLabel);

        return; // Exit if no camera is available
    }

    // Create a QLabel to display the video feed and store it as member variable
    videoLabel = new QLabel(ui->videoFeedWidget);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setScaledContents(true);
    videoLabel->setAlignment(Qt::AlignCenter);

    // Layout the QLabel within its parent widget
    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoLabel);
    videoLayout->setContentsMargins(0, 0, 0, 0);

    // Set up a timer to grab frames from OpenCV and update the QLabel
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(30); // Update every 30 milliseconds (approx 33 FPS)

    qDebug() << "OpenCV Camera started successfully!";
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

    // videoLabel is a child of ui->videoFeedWidget, so it will be deleted
    // when ui->videoFeedWidget is deleted, or when the parent QWidget is deleted.
    // No need to delete videoLabel explicitly here unless it's not parented correctly.
    // However, if you explicitly create it with `new QLabel(ui->videoFeedWidget);`
    // and `ui->videoFeedWidget` is a QWidget member, it's owned by the QObject hierarchy.
    // If you used `new QLabel();` without a parent, you would need to delete it.
    // Given the `new QLabel(ui->videoFeedWidget);`, it's generally fine.

    delete ui; // Deletes the UI components, including videoFeedWidget and its children like videoLabel
}

void Capture::on_back_clicked()
{
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    emit showFinalOutputPage();
}

void Capture::updateCameraFeed()
{
    if (!videoLabel || !cap.isOpened()) { // Safety check for both label and camera
        qWarning() << "UpdateCameraFeed called without valid videoLabel or camera!";
        return;
    }

    cv::Mat frame;
    if (cap.read(frame)) {
        // Convert the OpenCV Mat to QImage
        // Ensure frame is not empty after reading
        if (frame.empty()) {
            qWarning() << "Read empty frame from camera!";
            return;
        }

        QImage image = cvMatToQImage(frame);

        if (!image.isNull()) {
            // Convert QImage to QPixmap and set it to the QLabel
            QPixmap pixmap = QPixmap::fromImage(image);
            // Scale pixmap to fit the label, keeping aspect ratio
            videoLabel->setPixmap(pixmap.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        } else {
            qWarning() << "Failed to convert cv::Mat to QImage!";
        }
    } else {
        qWarning() << "Failed to read frame from camera!";
        // Optionally, stop the timer or show an error message
        cameraTimer->stop();
        videoLabel->setText("Camera error");
        videoLabel->setStyleSheet("color: red; font-size: 14px;");
    }
}

// Helper function to convert cv::Mat to QImage
QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: // 8-bit, 4 channel (e.g., BGRA or RGBA)
    {
        // Assuming BGRA in OpenCV, Qt's Format_ARGB32 is ARGB. Swapping might be needed.
        // For direct conversion, Format_ARGB32 usually expects ARGB. If OpenCV Mat is BGRA, it needs conversion.
        // Let's assume it's BGRA for typical OpenCV camera output.
        // A direct cast might work if it's already ARGB in the correct byte order.
        // If it's BGRA, you'd typically convert to RGB and then to QImage::Format_RGB888 or similar.
        // For simplicity, if your camera directly gives BGRA, this might be okay.
        // Otherwise, consider cv::cvtColor(mat, mat_rgb, cv::COLOR_BGRA2RGBA)
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
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


