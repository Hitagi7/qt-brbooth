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
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QThread>

#include <opencv2/opencv.hpp>

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr) // Initialize member variables
    , videoLabel(nullptr)  // Initialize member variables
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode (ImageCaptureMode)
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
    if (!cap.open(1)) {
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

    // Set camera resolution for a more controlled aspect ratio if desired
    // This is optional but can help ensure a consistent input frame size
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // Read and discard a few frames to allow auto-exposure/white-balance to settle
    cv::Mat dummyFrame;
    int warmUpDurationSeconds = 10;
    int estimatedFramesToRead = warmUpDurationSeconds * 30; // 150 frames for 5 seconds at 30 FPS
    for (int i = 0; i < estimatedFramesToRead; ++i) {
        if (!cap.read(dummyFrame)) {
            qWarning() << "Warning: Failed to read dummy frame during warm-up at frame" << i;
        }
    }

    // Create a QLabel to display the video feed and store it as member variable
    videoLabel = new QLabel(ui->videoFeedWidget);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setScaledContents(true); // Ensures content scales to fit the label
    videoLabel->setAlignment(Qt::AlignCenter);

    // Layout the QLabel within its parent widget
    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoLabel);
    videoLayout->setContentsMargins(0, 0, 0, 0);

    // To help with "tight" aspect ratio, you might want to adjust
    // the size policy or minimum size of `ui->videoFeedWidget` in your UI file (.ui)
    // or programmatically here, to match the desired 4:3 or 16:9 aspect ratio
    // (e.g., set a fixed minimum size or preferred size).
    // For example, if your camera is 640x480 (4:3):
    // ui->videoFeedWidget->setMinimumSize(640, 480);
    // ui->videoFeedWidget->setMaximumSize(640, 480); // if you want fixed size
    // Or set a preferred size:
    // ui->videoFeedWidget->setPreferredSize(QSize(640, 480));


    // Set up a timer to grab frames from OpenCV and update the QLabel
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(30); // Update every 30 milliseconds (approx 33 FPS)

    //CountdownTimer Setup
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
}

Capture::~Capture()
{
    // Stop andd delete the timer
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

void Capture::on_back_clicked()
{
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    if (m_currentCaptureMode == ImageCaptureMode) {
        ui->capture->setEnabled(false);

        // Initialize countdown
        countdownValue = 5;
        countdownLabel->setText(QString::number(countdownValue));
        countdownLabel->show();

        // Animate the countdown label (optional, for visual effect)
        QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
        animation->setDuration(300); // Fade in
        animation->setStartValue(0.0);
        animation->setEndValue(1.0);
        animation->start();

        // Start the countdown timer
        countdownTimer->start(1000); // 1000 milliseconds = 1 second
    } else {
        // This block will be executed if m_currentCaptureMode is VideoRecordMode.
        // For now, it just prints a message. This is where video recording logic would go later.
        qDebug() << "Capture button clicked in Video Record Mode. (Video recording logic not yet implemented).";
        // Optionally, you might want to show a message to the user that video mode is not active.
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

        ui->capture->setEnabled(true);
    }
}

void Capture::performImageCapture()
{
    cv::Mat frameToCapture;
    if (cap.read(frameToCapture)){
        if(frameToCapture.empty()){
            qWarning() << "Captured an empty frame!";
            return;
        }

        //Mirror the captured frame
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
        qWarning() << "Failed to read frame from camera";
    }
    emit showFinalOutputPage();;
}
void Capture::updateCameraFeed()
{
    if(!videoLabel || !cap.isOpened()){
        return;
    }

    cv::Mat frame;
    if (cap.read(frame)){
        if(frame.empty()){
            qWarning() << "Read empty frame from camera!";
            return;
        }

        cv::flip(frame, frame, 1);
        QImage image = cvMatToQImage(frame);

        if(!image.isNull()){
            QPixmap pixmap = QPixmap::fromImage(image);
            videoLabel->setPixmap(pixmap.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }else{
            qWarning() << "Failed to convert cv::Mat to QImage for live feed";
        }
    }else {
        qWarning() << "Failed to read frame from camera to live feed";
        if (cameraTimer->isActive()){
            cameraTimer->stop();
            videoLabel->setText("Camera stream interrupted");
            videoLabel->setStyleSheet("color: red; font-size: 14px;");
        }
    }
}

// Helper function to convert cv::Mat to QImage
QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: // 8-bit, 4 channel (e.g., BGRA or RGBA)
    {
        // For OpenCV typically giving BGRA, QImage::Format_ARGB32 is ARGB.
        // A direct cast might lead to incorrect color representation (e.g., blue instead of red).
        // It's safer to convert to a known format like RGB first.
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
