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
#include <QList>              // Required for QList<HandDetection> and QList<QPixmap>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QMutex>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QShowEvent>
#include <QHideEvent>
#include <opencv2/opencv.hpp>
#include "core/videotemplate.h"   // Your custom VideoTemplate class
#include "core/camera.h"          // Your custom Camera class
#include "ui/foreground.h"        // Foreground class
#include "core/common_types.h"    // Common data structures
#include "algorithms/hand_detection/hand_detector.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
class Foreground; // Forward declaration for Foreground

// Forward declarations
class HandDetector;

QT_END_NAMESPACE

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
    


    // Hand Detection Control Methods
    void setShowHandDetection(bool show);
    bool getShowHandDetection() const;
    void setHandDetectionConfidenceThreshold(double threshold);
    double getHandDetectionConfidenceThreshold() const;
    QList<HandDetection> getLastHandDetections() const;
    void toggleHandDetection();
    void updateHandDetectionButton();
    double getHandDetectionProcessingTime() const;
    void enableHandDetectionForCapture(); // Enable hand detection when capture page is shown
    void setHandDetectionEnabled(bool enabled);

protected:
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

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
    void startCountdown();
    void updateRecordTimer();
    void captureRecordingFrame();

    void on_back_clicked();
    void on_capture_clicked();
    void on_verticalSlider_valueChanged(int value);

    void updateForegroundOverlay(const QString &path);
    void setupStackedLayoutHybrid();
    void updateOverlayStyles();



    // Hand Detection Slots
    void onHandDetectionFinished();

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
    QTimer *cameraTimer;
    int m_targetRecordingFPS;
    double m_actualCameraFPS;  // Store the actual camera FPS for correct playback
    VideoTemplate m_currentVideoTemplate;
    int m_recordedSeconds;
    QList<QPixmap> m_recordedFrames;
    QPixmap m_capturedImage;

    // Hybrid stacked layout components
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
    
    // Current frame storage
    cv::Mat m_currentFrame;
    


    // Hand Detection Methods
    void processFrameWithHandDetection(const cv::Mat &frame);
    void applyHandDetectionToFrame(cv::Mat &frame);
    void drawHandBoundingBoxes(cv::Mat &frame, const QList<HandDetection> &detections);
    void initializeHandDetection();
    void enableHandDetection(bool enable);
    
    // Hand detection initialization
    
    // Hand detection state
    bool m_handDetectionEnabled;

    // Debug Display Members
    
    // --- FRAME SCALING MEMBERS ---
    double m_personScaleFactor;  // Current scaling factor for entire frame (1.0 to 0.5)
    QImage m_originalCameraImage;  // Store original camera image for capture (without display scaling)
    QSize m_cachedLabelSize;  // Cached label size for better recording performance
    double m_adjustedRecordingFPS;  // Store the adjusted FPS used during recording
    // --- END FRAME SCALING MEMBERS ---



    // Debug and UI Members
    QWidget *debugWidget;
    QLabel *debugLabel;
    QLabel *fpsLabel;
    QLabel *handDetectionLabel;
    QPushButton *handDetectionButton;
    QTimer *debugUpdateTimer;
    int m_currentFPS;



    // Hand Detection Members
    HandDetector *m_handDetector;
    bool m_showHandDetection;
    mutable QMutex m_handDetectionMutex;
    QElapsedTimer m_handDetectionTimer;
    double m_lastHandDetectionTime;
    double m_handDetectionFPS;
    // HandTrackerMP removed - now using consolidated HandDetector
    QList<HandDetection> m_lastHandDetections;

    // Performance optimization
    QPixmap m_cachedPixmap;



    // Helper methods (implemented in .cpp)
    void updateDebugDisplay();
    void setupDebugDisplay();

    // --- PERFORMANCE MONITORING ---
    void printPerformanceStats();
    // --- END PERFORMANCE MONITORING ---

    // Utility functions
    QImage cvMatToQImage(const cv::Mat &mat);
    cv::Mat qImageToCvMat(const QImage &image);
};

#endif // CAPTURE_H
