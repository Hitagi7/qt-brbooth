#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>
#include <QCamera> // Still needed if you keep any QCamera related logic, but for OpenCV preview, you won't need it for the direct feed.
#include <QVideoWidget> // Not directly used for OpenCV preview, but if you have other video elements, keep it.
#include <QMediaCaptureSession> // Not directly used for OpenCV preview
#include <QMediaDevices>
#include <QDebug>
#include <QTimer> // For updating OpenCV frames

// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

namespace Ui {
class Capture;
}

class Capture : public QWidget
{
    Q_OBJECT

public:
    explicit Capture(QWidget *parent = nullptr);
    ~Capture();


signals:
    void backtoPreviousPage();
    void showFinalOutputPage();

private slots:
    void on_back_clicked();
    void on_capture_clicked();
    void updateCameraFeed(); // New slot for updating the feed

private:
    Ui::Capture *ui;

    // Declare Qcamera-related objects
    // QCamera *camera;
    // QVideoWidget *videoOutput; // Displays the video
    // QMediaCaptureSession *captureSession; // Connects camera to videoOutput

    // OpenCV related members
    cv::VideoCapture cap; // OpenCV video capture object
    QTimer *cameraTimer;  // Timer to grab frames

    // Helper function to convert cv::Mat to QImage
    QImage cvMatToQImage(const cv::Mat &mat);
};

#endif // CAPTURE_H
