#include "final.h"
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include <QVBoxLayout>
#include "iconhover.h"
#include "ui_final.h"
#include <QFileDialog> // Required for QFileDialog
#include <QMessageBox> // Required for QMessageBox
#include <QImage>
#include <QDateTime> // For generating unique filenames
#include <opencv2/opencv.hpp> // For video writing

Final::Final(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Final)
    , videoPlaybackTimer(nullptr)
    , m_currentFrameIndex(0)
{
    ui->setupUi(this);

    //Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);

    //Setup for displaying captured image
    QVBoxLayout *finalWidgetLayout = qobject_cast<QVBoxLayout*>(ui->finalWidget->layout());
    if(!finalWidgetLayout){
        finalWidgetLayout = new QVBoxLayout(ui->finalWidget);
        finalWidgetLayout->setContentsMargins(0,0,0,0);
    }

    imageDisplayLabel = new QLabel(ui->finalWidget);
    imageDisplayLabel->setAlignment(Qt::AlignCenter);
    imageDisplayLabel->setScaledContents(true);

    finalWidgetLayout->addWidget(imageDisplayLabel);
    finalWidgetLayout->setStretchFactor(imageDisplayLabel,1);

    videoPlaybackTimer = new QTimer(this);
    connect(videoPlaybackTimer, &QTimer::timeout, this, &Final::playNextFrame);

}

Final::~Final()
{
    if(videoPlaybackTimer){
        videoPlaybackTimer->stop();
        delete videoPlaybackTimer;
        videoPlaybackTimer = nullptr;
    }
    delete ui;
}

void Final::on_back_clicked()
{
    if (videoPlaybackTimer->isActive()){
        videoPlaybackTimer->stop();
    }
    emit backToCapturePage();
}

void Final::setImage(const QPixmap &image)
{
    if (videoPlaybackTimer->isActive()){
        videoPlaybackTimer->stop();
    }

    m_videoFrames.clear();
    m_currentFrameIndex = 0;

    if (!image.isNull()) {
        imageDisplayLabel->setScaledContents(true); // Prevent stretching
        imageDisplayLabel->setPixmap(image);
        imageDisplayLabel->setText("");
    } else {
        imageDisplayLabel->clear();
    }
}

void Final::on_save_clicked()
{
    if (!m_videoFrames.isEmpty()) {
        // We have video frames, so save as a video
        saveVideoToFile();
    } else {
        // No video frames, proceed with saving an image
        QPixmap imageToSave = imageDisplayLabel->pixmap();

        if (imageToSave.isNull()) {
            QMessageBox::warning(this, "Save Image", "No image to save.");
            return;
        }

        // Get a file name from the user
        QString fileName = QFileDialog::getSaveFileName(this, "Save Image",
                                                        QDir::homePath() + "/untitled.png", // Default path and filename
                                                        "Images (*.png *.jpg *.bmp *.gif)"); // Supported image formats

        if (!fileName.isEmpty()) {
            // Save the image
            if (imageToSave.save(fileName)) {
                QMessageBox::information(this, "Save Image", "Image saved successfully!");
            } else {
                QMessageBox::critical(this, "Save Image", "Failed to save image.");
            }
        }
    }
    emit backToLandingPage();
}

void Final::setVideo(const QList<QPixmap> &frames)
{
    // Stop any existing playback
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames = frames; // Store the received frames
    m_currentFrameIndex = 0; // Start from the first frame

    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Playing back video with " << m_videoFrames.size() << " frames.";
        // Determine playback speed. Assuming Capture's cameraTimer started at 1000/60 for 60 FPS
        videoPlaybackTimer->start(1000 / 60); // Aim for 60 FPS playback
        playNextFrame(); // Display the first frame immediately
    } else {
        qWarning() << "No video frames provided for playback!";
        imageDisplayLabel->clear(); // Clear the display if no frames
    }
}

// ADDITION: New slot to advance to the next frame for video playback
void Final::playNextFrame()
{
    if (m_videoFrames.isEmpty()) {
        videoPlaybackTimer->stop();
        imageDisplayLabel->clear();
        qDebug() << "No frames left to play.";
        return;
    }

    if (m_currentFrameIndex < m_videoFrames.size()) {
        imageDisplayLabel->setPixmap(m_videoFrames.at(m_currentFrameIndex).scaled(imageDisplayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)); // Scale for display
        m_currentFrameIndex++;
    } else {
        m_currentFrameIndex = 0; // Loop back to the beginning
    }
}

// Helper function to convert QImage to cv::Mat
cv::Mat QImageToCvMat(const QImage &inImage)
{
    switch (inImage.format()) {
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC4, (void*)inImage.constBits(), inImage.bytesPerLine());
    case QImage::Format_RGB888:
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC3, (void*)inImage.constBits(), inImage.bytesPerLine()).clone(); // .clone() for RGB888 to ensure data is contiguous
    case QImage::Format_Indexed8:
    {
        cv::Mat mat(inImage.height(), inImage.width(), CV_8UC1, (void*)inImage.constBits(), inImage.bytesPerLine());
        return mat.clone(); // Clone to ensure a deep copy
    }
    default:
        qWarning() << "QImageToCvMat - QImage format not handled: " << inImage.format();
        return cv::Mat();
    }
}


// ADDITION: New function to save video frames to a file
void Final::saveVideoToFile()
{
    if (m_videoFrames.isEmpty()) {
        QMessageBox::warning(this, "Save Video", "No video frames to save.");
        return;
    }

    // Get a file name from the user (CAN BE CHANGED)
    QString defaultFileName = QDir::homePath() + "/video_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".avi";
    QString fileName = QFileDialog::getSaveFileName(this, "Save Video",
                                                    defaultFileName,
                                                    "Videos (*.avi *.mp4)"); // Offer AVI and MP4

    if (fileName.isEmpty()) {
        return; // User cancelled
    }

    // Determine the codec based on the file extension
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Default to MJPG
    if (fileName.endsWith(".mp4", Qt::CaseInsensitive)) {
        fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V'); // H.264 is preferred for MP4, but MP4V is widely supported.
        // You might need to check if your OpenCV build supports H.264 (e.g., with FFMPEG backend).
        // 'XVID' can also be an option for AVI.
    }


    // Get frame size from the first frame
    QSize frameSize = m_videoFrames.first().size();
    int width = frameSize.width();
    int height = frameSize.height();

    // Determine FPS (assuming 60 FPS capture)
    double fps = 60.0; // This should ideally come from the Capture class if it's dynamic

    cv::VideoWriter videoWriter;

    // Open the video writer
    if (!videoWriter.open(fileName.toStdString(), fourcc, fps, cv::Size(width, height), true)) {
        QMessageBox::critical(this, "Save Video", "Failed to open video writer. Check codecs and file path.");
        qWarning() << "Failed to open video writer for file: " << fileName << " with FOURCC: " << fourcc;
        return;
    }

    // Write frames to the video file
    for (const QPixmap &pixmap : m_videoFrames) {
        QImage image = pixmap.toImage();
        if (image.isNull()) {
            qWarning() << "Failed to convert QPixmap to QImage during video saving.";
            continue;
        }

        cv::Mat frame = QImageToCvMat(image);
        if (frame.empty()) {
            qWarning() << "Failed to convert QImage to cv::Mat during video saving.";
            continue;
        }
        // Ensure the frame is in 3 channels (BGR for OpenCV) for video writer
        if (frame.channels() == 4) { // If QImage was ARGB/RGB32, it's 4 channels
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) { // If it was grayscale
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        videoWriter.write(frame);
    }

    videoWriter.release(); // Release the video writer
    QMessageBox::information(this, "Save Video", "Video saved successfully!");
    qDebug() << "Video saved to: " << fileName;
}
