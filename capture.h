#ifndef CAPTURE_H
#define CAPTURE_H

#include <QElapsedTimer>      // Required for QElapsedTimer
#include <QImage>             // Required for QImage
#include <QMessageBox>        // Required for QMessageBox
#include <QPixmap>            // Required for QPixmap
#include <QPropertyAnimation> // Required for QPropertyAnimation
#include <QLabel>            // Required for QLabel
#include <QStackedLayout>    // Required for QStackedLayout
#include <QGridLayout>       // Required for QGridLayout, used in setupStackedLayoutHybrid
#include <QPushButton>       // Required for QPushButton, used for ui->back, ui->capture
#include <QSlider>           // Required for QSlider, used for ui->verticalSlider
#include <QThread>            // CRUCIAL: For QThread definition and usage
#include <QTimer>             // Required for QTimer
#include <QWidget>            // Required for QWidget base class
#include "videotemplate.h"   // Your custom VideoTemplate class
#include "camera.h"          // Your custom Camera class



QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
class Foreground; // Forward declaration for Foreground

class Capture : public QWidget
{
    Q_OBJECT

public:
    explicit Capture(QWidget *parent = nullptr,
                     Foreground *fg = nullptr,
                     Camera *existingCameraWorker = nullptr,
                     QThread *existingCameraThread = nullptr);
    ~Capture();

    // Define CaptureMode enum here, inside the class (already correct)
    enum CaptureMode { ImageCaptureMode, VideoRecordMode };

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate &templateData);
    


protected:
    void resizeEvent(QResizeEvent *event) override;

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames, double fps);
    void showFinalOutputPage();
    void personDetectedInFrame();
    void foregroundPathChanged(const QString &foregroundPath);

private slots:
    void updateCameraFeed(const QImage &frame);
    void handleCameraOpened(bool success,
                            double actual_width,
                            double actual_height,
                            double actual_fps);
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

    // --- PERFORMANCE MONITORING ---
    void printPerformanceStats();
    // --- END PERFORMANCE MONITORING ---

private:
    // Declare these private functions here (already correct)
    void performImageCapture();
    void startRecording();
    void stopRecording();

    Ui::Capture *ui;

    // IMPORTANT: Reorder these to match constructor initializer list for 'initialized after' warning
    Foreground *foreground; // Declared first as it's initialized before cameraThread in Ctor
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
    double m_actualCameraFPS;  // Store the actual camera FPS for correct playback
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



    // pass foreground
    QLabel* overlayImageLabel = nullptr;
    
    // --- FRAME SCALING MEMBERS ---
    double m_personScaleFactor;  // Current scaling factor for entire frame (1.0 to 0.5)
    QImage m_originalCameraImage;  // Store original camera image for capture (without display scaling)
    QSize m_cachedLabelSize;  // Cached label size for better recording performance
    double m_adjustedRecordingFPS;  // Store the adjusted FPS used during recording
    // --- END FRAME SCALING MEMBERS ---
    

    




};

#endif // CAPTURE_H
