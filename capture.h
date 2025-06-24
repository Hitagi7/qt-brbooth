#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>
#include <QMouseEvent> //Included for Icon Hover for Back Button

#include <QCamera>
#include <QCameraDevice>
#include <QMediaDevices>
#include <QVideoWidget>           // For displaying the video feed
#include <QMediaCaptureSession>   // To connect camera to video output

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
    void backtoBackgroundPage();

private slots:
    void on_back_clicked();

private:
    Ui::Capture *ui;

    // Declare camera-related objects
    QCamera *camera;
    QVideoWidget *videoOutput; // Displays the video
    QMediaCaptureSession *captureSession; // Connects camera to videoOutput

};

#endif // CAPTURE_H
