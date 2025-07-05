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
#include <QStandardPaths>
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

        // Get Downloads folder path
        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            // Fallback to C:/Downloads if QStandardPaths doesn't work
            downloadsPath = "C:/Downloads"; // Placeholder download location
        }

        // Generate unique filename with timestamp
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString fileName = downloadsPath + "/image_" + timestamp + ".png";

        // Create directory if it doesn't exist
        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath);
        }

        // Save the image
        if (imageToSave.save(fileName)) {
            QMessageBox::information(this, "Save Image",
                                     QString("Image saved successfully to:\n%1").arg(fileName));
        } else {
            QMessageBox::critical(this, "Save Image", "Failed to save image.");
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
        // FIXED: Use 60 FPS for playback to match recording FPS
        videoPlaybackTimer->start(1000 / 60); // Aim for 60 FPS playback to match recording
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

    // Get Downloads folder path
    QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    if (downloadsPath.isEmpty()) {
        // Fallback to C:/Downloads if QStandardPaths doesn't work
        downloadsPath = "C:/Downloads";
    }

    // Generate unique filename with timestamp
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString fileName = downloadsPath + "/video_" + timestamp + ".avi";

    // Create directory if it doesn't exist
    QDir dir;
    if (!dir.exists(downloadsPath)) {
        dir.mkpath(downloadsPath);
    }

    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    QSize frameSize = m_videoFrames.first().size();
    int width = frameSize.width();
    int height = frameSize.height();

    // FIXED: Calculate actual FPS based on captured frames
    // If we have 285 frames for 10 seconds, that's 28.5 FPS
    // Use this as the save FPS to get normal playback speed
    double actualFPS = (double)m_videoFrames.size() / 10.0; // Assuming 10 seconds duration
    qDebug() << "Saving video with actual FPS: " << actualFPS;

    cv::VideoWriter videoWriter;

    if (!videoWriter.open(fileName.toStdString(), fourcc, actualFPS, cv::Size(width, height), true)) {
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

        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        videoWriter.write(frame);
    }

    videoWriter.release();
    QMessageBox::information(this, "Save Video",
                             QString("Video saved successfully at %1 FPS to:\n%2").arg(actualFPS, 0, 'f', 1).arg(fileName));
    qDebug() << "Video saved to: " << fileName;
}
