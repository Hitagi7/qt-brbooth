#ifndef CAMERA_H
#define CAMERA_H

#include <QDebug>
#include <QElapsedTimer>
#include <QImage>
#include <QObject>
#include <QTimer>
#include <opencv2/opencv.hpp>

// Forward declaration for cvMatToQImage utility
QImage cvMatToQImage(const cv::Mat &mat);

class Camera : public QObject
{
    Q_OBJECT

public:
    explicit Camera(QObject *parent = nullptr);
    ~Camera();

    void setDesiredCameraProperties(int width, int height, double fps);
    bool isCameraOpen() const;

signals:
    void frameReady(const QImage &frame);
    void cameraOpened(bool success, double actual_width, double actual_height, double actual_fps);
    void error(const QString &msg);
    // REMOVED: void firstFrameAvailable(); // No longer needed for immediate display

public slots:
    void startCamera();
    void stopCamera();

private slots:
    void processFrame();

private:
    cv::VideoCapture cap;
    QTimer *cameraReadTimer;
    QElapsedTimer loopTimer;
    QElapsedTimer frameTimer;
    long long totalTime = 0;
    int frameCount = 0;

    int m_desiredWidth;
    int m_desiredHeight;
    double m_desiredFps;

    // REMOVED: bool m_firstFrameReadySent; // No longer needed for immediate display
};

#endif // CAMERA_H
