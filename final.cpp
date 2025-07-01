#include "final.h"
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include <QVBoxLayout>
#include "iconhover.h"
#include "ui_final.h"
#include <QFileDialog> // Required for QFileDialog
#include <QMessageBox> // Required for QMessageBox

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
        imageDisplayLabel->setPixmap(m_videoFrames.at(m_currentFrameIndex));
        m_currentFrameIndex++;
    } else {
        m_currentFrameIndex = 0; // Loop back to the beginning
    }
}
