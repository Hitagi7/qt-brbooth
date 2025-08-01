#include "camera.h"
#include <QBuffer>
#include <chrono>

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

    if (cap.isOpened()) {
        cap.release();
        qDebug() << "Camera: Releasing previous camera instance.";
    }

    if (!cap.open(0, cv::CAP_ANY)) {
        qWarning() << "Camera: Error: Could not open camera with index 0.";
        emit error("Camera not available. Check connection and drivers.");
        emit cameraOpened(false, 0, 0, 0);
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, m_desiredWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_desiredHeight);
    cap.set(cv::CAP_PROP_FPS, m_desiredFps);

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

    // ✅ Emit first frame immediately for faster feedback
    cv::Mat firstFrame;
    if (cap.read(firstFrame) && !firstFrame.empty()) {
        cv::flip(firstFrame, firstFrame, 1);
        QImage firstImage = cvMatToQImage(firstFrame);
        if (!firstImage.isNull()) {
            emit frameReady(firstImage.copy());
            qDebug() << "Camera: First frame emitted immediately.";
        }
    } else {
        qWarning() << "Camera: Failed to read first frame immediately.";
    }

    // 🔁 Start timer for continuous frame grabbing
    int timerInterval = 0;
    if (actual_fps > 0) {
        timerInterval = qMax(1, static_cast<int>(1000.0 / actual_fps));
    } else {
        timerInterval = 33;
        qWarning() << "Camera: WARNING: Actual FPS is 0, defaulting timer interval to 33ms.";
    }

    cameraReadTimer->start(timerInterval);
    emit cameraOpened(true, actual_width, actual_height, actual_fps);
    qDebug() << "Camera: Camera started successfully in worker thread. Timer interval:"
             << timerInterval << "ms";
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
    if (!cap.isOpened() || !cap.read(frame) || frame.empty()) {
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

    emit frameReady(image.copy());

    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    if (frameCount % 60 == 0) {
        double avgLoopTime = (double) totalTime / frameCount;
        double measuredFPS = 1000.0 / ((double) frameTimer.elapsed() / frameCount);
        qDebug() << "----------------- Worker Thread Stats ------------------";
        qDebug() << "Camera: Avg loop time (last 60 frames):" << avgLoopTime << "ms";
        qDebug() << "Camera: Current FPS (measured over 60 frames):" << measuredFPS << "FPS";
        qDebug() << "Camera: Frame processing efficiency:"
                 << (avgLoopTime < (1000.0 / cap.get(cv::CAP_PROP_FPS)) ? "GOOD"
                                                                        : "NEEDS OPTIMIZATION");
        qDebug() << "--------------------------------------------------------";
        frameCount = 0;
        totalTime = 0;
        frameTimer.restart();
    }
}
