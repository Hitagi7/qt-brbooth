#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>
#include <QPixmap>
#include <QVector>
#include <QTimer>
#include <QLabel>
#include <QStackedLayout>
#include <QGridLayout>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include "videotemplate.h"
#include "foreground.h" // Include Foreground header as it's used in the constructor and as a member


// Forward declarations for UI namespace
namespace Ui {
class Capture;
}

class Capture : public QWidget
{
    Q_OBJECT

public:
    // Moved enum inside the class to be a member type, making it accessible via Capture::
    enum CaptureMode {
        ImageCaptureMode,
        VideoRecordMode
    };

    explicit Capture(QWidget *parent = nullptr, Foreground *fg = nullptr); //pass foreground class
    ~Capture();

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate &templateData);

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QVector<QPixmap> &frames);
    void showFinalOutputPage(); // Signal to show the final output page

protected:
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateCameraFeed();
    void updateCountdown();
    void updateRecordTimer();
    void captureRecordingFrame();
    void on_back_clicked();
    void on_capture_clicked();
    void on_verticalSlider_valueChanged(int value);
    void updateForegroundOverlay(const QString &path); // New slot for overlay updates

private:
    Ui::Capture *ui;

    cv::VideoCapture cap;

    QTimer *cameraTimer;
    QTimer *countdownTimer;
    QLabel *countdownLabel;
    int countdownValue;

    CaptureMode m_currentCaptureMode; // Use the enum
    VideoTemplate m_currentVideoTemplate; // Use the struct

    bool m_isRecording;
    QTimer *recordTimer;
    QTimer *recordingFrameTimer;
    int m_targetRecordingFPS;
    int m_recordedSeconds;
    QVector<QPixmap> m_recordedFrames;
    QPixmap m_capturedImage;

    QStackedLayout *stackedLayout;

    QElapsedTimer loopTimer;
    QElapsedTimer frameTimer;
    qint64 totalTime;
    int frameCount;
    bool isProcessingFrame;

    void setupStackedLayoutHybrid();
    void updateOverlayStyles();
    void startRecording();
    void stopRecording();
    void performImageCapture();

    QImage cvMatToQImage(const cv::Mat &mat);


    // pass foreground
    Foreground *foreground;
    QLabel* overlayImageLabel; // Initialized to nullptr in constructor, no need for = nullptr here.
};

// FIX: Correctly scope the enum for Q_DECLARE_METATYPE
// This must be outside the class definition, but after the enum's full definition.
Q_DECLARE_METATYPE(Capture::CaptureMode)


#endif // CAPTURE_H
