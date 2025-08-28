#include "core/camera.h"
#include <QBuffer>

QImage cvMatToQImage(const cv::Mat &mat)
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
        if (sColorTable.isEmpty()) {
            for (int i = 0; i < 256; ++i) {
                sColorTable.push_back(qRgb(i, i, i));
            }
        }
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }
    default:
        qWarning() << "Unsupported cv::Mat format in Camera: " << mat.type();
        return QImage();
    }
}

Camera::Camera(QObject *parent)
    : QObject(parent)
    , cameraReadTimer(new QTimer(this))
    , m_desiredWidth(1280)
    , m_desiredHeight(720)
    , m_desiredFps(60.0)
    , m_firstFrameEmitted(false)
{
    cameraReadTimer->setTimerType(Qt::PreciseTimer);
    connect(cameraReadTimer, &QTimer::timeout, this, &Camera::processFrame);
    loopTimer.start();
    frameTimer.start();
    qDebug() << "Camera worker created.";
}

Camera::~Camera()
{
    stopCamera();
    qDebug() << "Camera worker destroyed.";
}

void Camera::setDesiredCameraProperties(int width, int height, double fps)
{
    m_desiredWidth = width;
    m_desiredHeight = height;
    m_desiredFps = fps;
    qDebug() << "Camera: Desired properties set to W:" << width << "H:" << height << "FPS:" << fps;
}

bool Camera::isCameraOpen() const
{
    return cap.isOpened();
}

void Camera::startCamera()
{
    qDebug() << "Camera: Attempting to open camera and set properties...";

    // Reset first frame flag
    m_firstFrameEmitted = false;

    if (cap.isOpened()) {
        cap.release();
        qDebug() << "Camera: Releasing previous camera instance.";
    }

    // Start camera opening process asynchronously
    QTimer::singleShot(0, [this]() {
        if (!cap.open(0, cv::CAP_ANY)) {
            qWarning() << "Camera: Error: Could not open camera with index 0.";
            emit error("Camera not available. Check connection and drivers.");
            emit cameraOpened(false, 0, 0, 0);
            return;
        }

        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, m_desiredWidth);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_desiredHeight);
        cap.set(cv::CAP_PROP_FPS, m_desiredFps);

        // Get actual camera properties
        double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap.get(cv::CAP_PROP_FPS);

        qDebug() << "========================================";
        qDebug() << "Camera: Camera settings REQUESTED:" << m_desiredWidth << "x" << m_desiredHeight
                 << "@" << m_desiredFps << "FPS";
        qDebug() << "Camera: Camera settings ACTUAL: " << actual_width << "x" << actual_height << "@"
                 << actual_fps << "FPS";
        qDebug() << "========================================";

        if (qAbs(actual_fps - m_desiredFps) > 1.0) {
            qWarning() << "Camera: WARNING: Camera did not accept desired FPS request. Actual FPS is"
                       << actual_fps;
        }

        if (qAbs(actual_width - m_desiredWidth) > 1.0 || qAbs(actual_height - m_desiredHeight) > 1.0) {
            qWarning()
                << "Camera: WARNING: Camera did not accept desired resolution. Actual resolution is"
                << actual_width << "x" << actual_height;
        }

        // Start timer for continuous frame grabbing (optimized for 30 FPS minimum)
        int timerInterval = 33; // 33ms = ~30 FPS (1000ms / 30fps = 33.33ms)
        if (actual_fps > 0 && actual_fps < 30) {
            timerInterval = qMax(1, static_cast<int>(1000.0 / actual_fps));
        }

        // Ensure timer is started in the correct thread context
        QMetaObject::invokeMethod(cameraReadTimer, "start", Qt::QueuedConnection, Q_ARG(int, timerInterval));
        emit cameraOpened(true, actual_width, actual_height, actual_fps);
        qDebug() << "📹 Camera: Camera started successfully in worker thread. Timer interval:"
                 << timerInterval << "ms";
        qDebug() << "📹 Camera: Ready for capture!";
    });
}

void Camera::stopCamera()
{
    if (cameraReadTimer->isActive()) {
        cameraReadTimer->stop();
        qDebug() << "Camera: Timer stopped.";
    }
    if (cap.isOpened()) {
        cap.release();
        qDebug() << "Camera: Camera released.";
    }
    qDebug() << "Camera: Camera stopped.";
}

void Camera::processFrame()
{
    loopTimer.start();

    cv::Mat frame;
    if (!cap.isOpened()) {
        qWarning() << "Camera: Camera not opened, skipping frame.";
        loopTimer.restart();
        return;
    }
    
    if (!cap.read(frame) || frame.empty()) {
        qWarning() << "Camera: Failed to read frame from camera or frame is empty.";
        loopTimer.restart();
        return;
    }

    cv::flip(frame, frame, 1);
    QImage image = cvMatToQImage(frame);
    if (image.isNull()) {
        qWarning() << "Camera: Failed to convert cv::Mat to QImage.";
        loopTimer.restart();
        return;
    }

    // Emit first frame signal if this is the first frame
    if (!m_firstFrameEmitted) {
        m_firstFrameEmitted = true;
        emit firstFrameEmitted();
        qDebug() << "Camera: First frame emitted immediately.";
    }

    // Add error handling for frame emission
    try {
        emit frameReady(image);
    } catch (const std::exception& e) {
        qWarning() << "Camera: Exception during frame emission:" << e.what();
        loopTimer.restart();
        return;
    }

    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    if (frameCount % 240 == 0) { // Further reduced frequency for performance
        double avgLoopTime = (double) totalTime / frameCount;
        double measuredFPS = 1000.0 / ((double) frameTimer.elapsed() / frameCount);
        qDebug() << "----------------- Worker Thread Stats ------------------";
        qDebug() << "Camera: Avg loop time (last 240 frames):" << avgLoopTime << "ms";
        qDebug() << "Camera: Current FPS (measured over 240 frames):" << measuredFPS << "FPS";
        qDebug() << "Camera: Frame processing efficiency:"
                 << (avgLoopTime < (1000.0 / cap.get(cv::CAP_PROP_FPS)) ? "GOOD"
                                                                        : "NEEDS OPTIMIZATION");
        qDebug() << "--------------------------------------------------------";
        frameCount = 0;
        totalTime = 0;
        frameTimer.restart();
    }
}
