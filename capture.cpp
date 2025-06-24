#include "capture.h"
#include "ui_capture.h"
#include "iconhover.h"

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

    // -- Camera integration starts here --
    // Check for available camera devices
    QList<QCameraDevice> devices = QMediaDevices::videoInputs();
    if (devices.isEmpty()) {
        qWarning() << "No camera found!";
        ui->videoFeedWidget->setStyleSheet("background-color: grey; color: white; border-radius: 10px;"); // Visual feedback
        return; // Exit if no camera is available
    }

    // Instantiate QCamera with the first available device
    camera = new QCamera(devices.first(), this);

    // Create the QVideoWidget for displaying the video
    videoOutput = new QVideoWidget(ui->videoFeedWidget);

    // Create a QMediaCaptureSession and set its camera and video output
    captureSession = new QMediaCaptureSession(this); // 'this' sets parent
    captureSession->setCamera(camera);
    captureSession->setVideoOutput(videoOutput);

    // Layout the QVideoWidget within its parent widget (ui->videoFeedWidget)
    QVBoxLayout *videoLayout = new QVBoxLayout(ui->videoFeedWidget);
    videoLayout->addWidget(videoOutput);
    videoLayout->setContentsMargins(0, 0, 0, 0); // Remove any margins around the video feed

    // Ensure the videoOutput resizes with its parent
    videoOutput->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Start the camera
    camera->start();

    qDebug() << "Camera started successfully!";

    // --- Webcam Integration Ends Here ---

}

Capture::~Capture()
{
    // Stop the camera before deleting
    if (camera && camera->isActive()) {
        camera->stop();
    }
    delete captureSession;
    delete camera;
    delete videoOutput;
    delete ui;
}

void Capture::on_back_clicked()
{
    emit backtoBackgroundPage();
}

