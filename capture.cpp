#include "capture.h"
#include "ui_capture.h"
#include "iconhover.h"
#include <QVBoxLayout>
#include <QLabel>

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
{
    ui->setupUi(this);

    //Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);

    // --- OpenCV Camera integration starts here ---

    // Open the default camera (usually index 0)
    // You can iterate through cameras if needed, similar to QMediaDevices::videoInputs()
    if (!cap.open(0)) { // 0 for the default camera
        qWarning() << "Error: Could not open camera with OpenCV!";
        ui->videoFeedWidget->setStyleSheet("background-color: grey; color: white; border-radius: 10px;"); // Visual feedback
        return; // Exit if no camera is available
    }

    // Create a QLabel to display the video feed
    // Assuming ui->videoFeedWidget is a QWidget in your .ui file where you want the video to appear.
    // We'll add a QLabel to it.
    QLabel *videoLabel = new QLabel(ui->videoFeedWidget);
    videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    videoLabel->setScaledContents(true); // Scale the pixmap to fit the label

    // Layout the QLabel within its parent widget (ui->videoFeedWidget)
    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoLabel);
    videoLayout->setContentsMargins(0, 0, 0, 0); // Remove any margins around the video feed

    // Set up a timer to grab frames from OpenCV and update the QLabel
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(30); // Update every 30 milliseconds (approx 33 FPS)

    qDebug() << "OpenCV Camera started successfully!";

    // --- OpenCV Camera Integration Ends Here ---

}

Capture::~Capture()
{
    // Stop the timer
    if (cameraTimer->isActive()) {
        cameraTimer->stop();
    }
    delete cameraTimer;

    // Release the camera resource
    if (cap.isOpened()) {
        cap.release();
    }

    delete ui;
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
    cv::Mat frame;
    if (cap.read(frame)) { // Read a new frame from the camera
        // Convert the OpenCV Mat to QImage
        QImage image = cvMatToQImage(frame);

        // Convert QImage to QPixmap and set it to the QLabel
        // Assuming you created a QLabel named 'videoLabel' and added it to ui->videoFeedWidget.
        // You'll need to store 'videoLabel' as a member variable or find it.
        // For simplicity, let's cast ui->videoFeedWidget's layout item to QLabel.
        // A better approach is to make `videoLabel` a member of `Capture` class.
        QLabel *videoLabel = qobject_cast<QLabel*>(ui->videoFeedWidget->layout()->itemAt(0)->widget());
        if (videoLabel) {
            videoLabel->setPixmap(QPixmap::fromImage(image).scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    } else {
        qWarning() << "Failed to read frame from camera!";
        // Optionally, stop the timer or show an error message
        cameraTimer->stop();
    }
}

// Helper function to convert cv::Mat to QImage
QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped(); // OpenCV uses BGR, Qt uses RGB
    }

    // 8-bit, 1 channel
    case CV_8UC1:
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


