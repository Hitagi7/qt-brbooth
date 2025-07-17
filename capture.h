#ifndef CAPTURE_H
#define CAPTURE_H

#include <QLabel>  // Required for QLabel member
#include <QTimer>  // Required for QTimer member
#include <QWidget> // Base class for Capture
#include <QImage>
#include <QPixmap>
#include <QPushButton>
#include <QThread>
#include <QList>
#include <QElapsedTimer>     // Required for QElapsedTimer
#include <QMessageBox>       // Required for QMessageBox
#include <QPropertyAnimation> // Required for QPropertyAnimation
#include <QStackedLayout>    // Required for QStackedLayout
#include <QGridLayout>       // Required for QGridLayout, used in setupStackedLayoutHybrid
#include <QSlider>           // Required for QSlider, used for ui->verticalSlider
#include "videotemplate.h"
#include "persondetector.h"
#include "camera.h"          // Your custom Camera class

// Forward declarations
namespace Ui { class Capture; }
class Foreground; // Forward declaration for Foreground

class Capture : public QWidget
{
    Q_OBJECT

public:
    explicit Capture(QWidget *parent = nullptr, Foreground *fg = nullptr,
                     Camera *existingCameraWorker = nullptr, QThread *existingCameraThread = nullptr);
    ~Capture();

    // Define CaptureMode enum here, inside the class (already correct)
    enum CaptureMode {
        ImageCaptureMode,
        VideoRecordMode
    };

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate& templateData);

protected:
    void resizeEvent(QResizeEvent *event) override;

signals:
    void backtoPreviousPage();
    void showFinalOutputPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);

private slots:
    void updateCameraFeed(const QImage &frame);
    void handleCameraOpened(bool success, double actual_width, double actual_height, double actual_fps);
    void handleCameraError(const QString &msg);

    void updateCountdown();
    void updateRecordTimer();
    void captureRecordingFrame();

    void on_back_clicked();
    void on_capture_clicked();
    void on_verticalSlider_valueChanged(int value);

    void onPeopleDetected(int count);
    void processCurrentFrame();

    void updateForegroundOverlay(const QString &path);
    void setupStackedLayoutHybrid();
    void updateOverlayStyles();


private:
    void performImageCapture();
    void startRecording();
    void stopRecording();

    Ui::Capture *ui;

    // IMPORTANT: Reorder these to match constructor initializer list for 'initialized after' warning
    Foreground *foreground;    // Declared first as it's initialized before cameraThread in Ctor
    QThread *cameraThread;
    Camera *cameraWorker;

    //Countdown Timers;
    QTimer *countdownTimer; //Timer for the 5-second countdown
    QLabel *countdownLabel; //Label to display the countdown
    int countdownValue; //current value of the countdown

    CaptureMode m_currentCaptureMode;
    bool m_isRecording;
    QTimer *recordTimer;
    QTimer *recordingFrameTimer;
    int m_targetRecordingFPS;
    VideoTemplate m_currentVideoTemplate;
    int m_recordedSeconds;
    QList<QPixmap> m_recordedFrames;
    QPixmap m_capturedImage;

    QStackedLayout *stackedLayout;
    QLabel *overlayImageLabel;
    QLabel *loadingCameraLabel;

    // Member variables for OpenCV camera and video display
    QTimer *cameraTimer;  // QTimer object to trigger frame updates
    QLabel *videoLabel;   // QLabel to display the video feed
    QLabel *yoloLabel;    // QLabel to display YOLO detection results
    cv::VideoCapture cap; // OpenCV VideoCapture object for camera access

    bool m_cameraFullyReady;

    //Yolov5
    PersonDetector* personDetector;
    QTimer* detectionTimer;
    int lastPersonCount;
};

#endif // CAPTURE_H
