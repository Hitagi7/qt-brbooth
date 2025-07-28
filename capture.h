#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>           // Required for QWidget base class
#include <QTimer>            // Required for QTimer
#include <QImage>            // Required for QImage
#include <QPixmap>           // Required for QPixmap
#include <QElapsedTimer>     // Required for QElapsedTimer
#include <QThread>           // CRUCIAL: For QThread definition and usage
#include <QMessageBox>       // Required for QMessageBox
#include <QPropertyAnimation> // Required for QPropertyAnimation
#include <QLabel>            // Required for QLabel
#include <QStackedLayout>    // Required for QStackedLayout
#include <QGridLayout>       // Required for QGridLayout, used in setupStackedLayoutHybrid
#include <QPushButton>       // Required for QPushButton, used for ui->back, ui->capture
#include <QSlider>           // Required for QSlider, used for ui->verticalSlider
#include "videotemplate.h"   // Your custom VideoTemplate class
#include "camera.h"          // Your custom Camera class

// --- NEW INCLUDES FOR QPROCESS AND JSON ---
#include <QProcess>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDir>
#include <QDateTime>
#include <QCoreApplication> // For applicationDirPath()
#include <opencv2/opencv.hpp>
// --- END NEW INCLUDES ---

QT_BEGIN_NAMESPACE
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
    void setVideoTemplate(const VideoTemplate &templateData);

protected:
    void resizeEvent(QResizeEvent *event) override;

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);
    void showFinalOutputPage();
    void personDetectedInFrame();

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

    void updateForegroundOverlay(const QString &path);
    void setupStackedLayoutHybrid();
    void updateOverlayStyles();

    // --- NEW SLOTS FOR ASYNCHRONOUS QPROCESS ---
    void handleYoloOutput();
    void handleYoloError();
    void handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleYoloErrorOccurred(QProcess::ProcessError error);
    void printPerformanceStats(); // <-- ADDED THIS DECLARATION
    // --- END NEW SLOTS ---

private:
    // Declare these private functions here (already correct)
    void performImageCapture();
    void startRecording();
    void stopRecording();

    Ui::Capture *ui;

    // IMPORTANT: Reorder these to match constructor initializer list for 'initialized after' warning
    Foreground *foreground;    // Declared first as it's initialized before cameraThread in Ctor
    QThread *cameraThread;
    Camera *cameraWorker;

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

    QStackedLayout *stackedLayout;
    QLabel *loadingCameraLabel;

    // Performance tracking members
    QLabel *videoLabelFPS;
    QElapsedTimer loopTimer;
    qint64 totalTime;
    int frameCount;
    QElapsedTimer frameTimer;

    // --- MODIFIED/NEW MEMBERS FOR ASYNCHRONOUS YOLO ---
    QProcess *yoloProcess; // QProcess member
    bool isProcessingFrame; // Flag to manage concurrent detection calls
    QString currentTempImagePath; // To keep track of the temp image being processed
    // --- END MODIFIED/NEW MEMBERS ---

    // pass foreground
    QLabel* overlayImageLabel = nullptr;
    // --- MODIFIED: detectPersonInImage now returns void, processing done in slot ---
    void detectPersonInImage(const QString& imagePath);

    // Helper functions for OpenCV conversion
    cv::Mat qImageToCvMat(const QImage &inImage);
    QImage cvMatToQImage(const cv::Mat &mat);

};

#endif // CAPTURE_H
