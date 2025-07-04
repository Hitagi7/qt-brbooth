#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QPixmap>
#include <QList>
#include <opencv2/opencv.hpp>
#include "videotemplate.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
QT_END_NAMESPACE

class Capture : public QWidget
{
    Q_OBJECT

public:
    enum CaptureMode {
        ImageCaptureMode,
        VideoRecordMode
    };

    Capture(QWidget *parent = nullptr);
    ~Capture();

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate &templateData);

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateCameraFeed();
    void captureRecordingFrame();
    void on_back_clicked();
    void on_capture_clicked();
    void updateCountdown();
    void updateRecordTimer();

private:
    Ui::Capture *ui;
    cv::VideoCapture cap;
    QTimer *cameraTimer;
    QLabel *videoLabel;
    QTimer *countdownTimer;
    QLabel *countdownLabel;
    int countdownValue;
    CaptureMode m_currentCaptureMode;
    bool m_isRecording;
    QTimer *recordTimer;
    QTimer *recordingFrameTimer;
    int m_targetRecordingFPS;
    VideoTemplate m_currentVideoTemplate;
    int m_recordedSeconds;
    QList<QPixmap> m_recordedFrames;
    QPixmap m_capturedImage;

    void startRecording();
    void stopRecording();
    void performImageCapture();
    QImage cvMatToQImage(const cv::Mat &mat);

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);
    void showFinalOutputPage();
};

#endif // CAPTURE_H
