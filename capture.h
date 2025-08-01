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

// --- OPENCV INCLUDES FOR HOG DETECTION ---
#include <opencv2/opencv.hpp>
#include "simplepersondetector.h"
// --- END OPENCV INCLUDES ---

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

    // --- HOG DETECTION MEMBERS ---
    SimplePersonDetector* personDetector; // Advanced HOG person detector
    bool hogEnabled; // Flag to enable/disable HOG detection
    int frameSkipCounter; // Counter to skip frames for performance
    static const int HOG_FRAME_SKIP = 15; // Process HOG every 15 frames (less lag)
    // --- END HOG DETECTION MEMBERS ---

    // pass foreground
    QLabel* overlayImageLabel = nullptr;
    
    // --- FRAME SCALING MEMBERS ---
    double m_personScaleFactor;  // Current scaling factor for entire frame (1.0 to 0.5)
    QRect m_lastDetectedPersonRect;  // Last detected person bounding box (for HOG detection)
    bool m_personDetected;  // Flag to track if person was detected in current frame (for HOG detection)
    // --- END FRAME SCALING MEMBERS ---
    
    // --- HOG PERSON DETECTION ---
    void detectPersonWithHOG(const QImage& image);
    QRect findBestPersonDetection(const QList<SimpleDetection>& detections);
    // --- END HOG PERSON DETECTION ---
    
    // --- NEW METHOD FOR PERSON SCALING ---
    QPixmap applyPersonScaling(const QPixmap& originalPixmap, const QRect& personRect, double scaleFactor);
    // --- END NEW METHOD ---

    // Helper functions for OpenCV conversion
    cv::Mat qImageToCvMat(const QImage &inImage);
    QImage cvMatToQImage(const cv::Mat &mat);

};

#endif // CAPTURE_H
