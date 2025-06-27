// capture.h
#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget> // Base class for Capture
#include <QTimer>  // Required for QTimer member
#include <QLabel>  // Required for QLabel member

// Required for OpenCV types (cv::VideoCapture and cv::Mat) used as members
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
QT_END_NAMESPACE

class Capture : public QWidget
{
    Q_OBJECT // Required for signals and slots

public:
    explicit Capture(QWidget *parent = nullptr);
    ~Capture();

signals:
    void backtoPreviousPage();
    void showFinalOutputPage();

private slots:
    void on_back_clicked();
    void on_capture_clicked();
    void updateCameraFeed(); // Declaration for the slot used to update camera feed

private:
    Ui::Capture *ui;

    // Member variables for OpenCV camera and video display
    QTimer *cameraTimer;      // QTimer object to trigger frame updates
    QLabel *videoLabel;       // QLabel to display the video feed
    cv::VideoCapture cap;     // OpenCV VideoCapture object for camera access

    // Helper function declaration
    QImage cvMatToQImage(const cv::Mat &mat); // Helper to convert OpenCV Mat to QImage
};

#endif // CAPTURE_H
