// capture.h
#ifndef CAPTURE_H
#define CAPTURE_H

#include <QLabel>  // Required for QLabel member
#include <QTimer>  // Required for QTimer member
#include <QWidget> // Base class for Capture
#include <QImage>
#include <QPixmap>
#include <QPushButton>
#include <QThread>

// Required for OpenCV types (cv::VideoCapture and cv::Mat) used as members
#include <opencv2/opencv.hpp>

QT_BEGIN_NAMESPACE
namespace Ui {
class Capture;
}
QT_END_NAMESPACE

class Capture : public QWidget
{
    Q_OBJECT // Required for signals and slots

public:
    enum CaptureMode{
        ImageCaptureMode, //for taking a single capture
        VideoRecordMode   //for recording a video
    };
    explicit Capture(QWidget *parent = nullptr);
    ~Capture();

    void setCaptureMode(CaptureMode mode);

signals:
    void backtoPreviousPage();
    void showFinalOutputPage();
    void imageCaptured(const QPixmap &image);

protected:
    void resizeEvent(QResizeEvent *event) override;
private slots:
    void on_back_clicked();
    void on_capture_clicked();
    void updateCameraFeed(); // Declaration for the slot used to update camera feed
    void updateCountdown();
    void performImageCapture();

private:
    Ui::Capture *ui;

    // Member variables for OpenCV camera and video display
    QTimer *cameraTimer;  // QTimer object to trigger frame updates
    QLabel *videoLabel;   // QLabel to display the video feed
    cv::VideoCapture cap; // OpenCV VideoCapture object for camera access

    // Helper function declaration
    QImage cvMatToQImage(const cv::Mat &mat); // Helper to convert OpenCV Mat to QImage
    QPixmap m_capturedImage; //stores the last captured image

    //Countdown Timers;
    QTimer *countdownTimer; //Timer for the 5-second countdown
    QLabel *countdownLabel; //Label to display the countdown
    int countdownValue; //current value of the countdown

    CaptureMode m_currentCaptureMode; //stores current mode of operation for Capture page
};

#endif // CAPTURE_H
