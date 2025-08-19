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
#include <QList>              // Required for QList<AdvancedHandDetection> and QList<QPixmap>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <QMutex>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QShowEvent>
#include <QHideEvent>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "core/videotemplate.h"   // Your custom VideoTemplate class
#include "core/camera.h"          // Your custom Camera class
#include "ui/foreground.h"        // Foreground class
#include "core/common_types.h"    // Common data structures
// Temporarily disabled algorithms
// #include "algorithms/tflite_segmentation_widget.h"
// #include "algorithms/tflite_deeplabv3.h"
// #include "algorithms/advanced_hand_detector.h"
// #include "algorithms/mediapipe_like_hand_tracker.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
class Foreground; // Forward declaration for Foreground

// Forward declarations
// Temporarily disabled algorithms
// class TFLiteDeepLabv3;
// class TFLiteSegmentationWidget;
// class AdvancedHandDetector;

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
    
    // Unified Person Detection and Segmentation Control Methods
    void setShowPersonDetection(bool show);
    bool getShowPersonDetection() const;
    void setPersonDetectionConfidenceThreshold(double threshold);
    double getPersonDetectionConfidenceThreshold() const;
    void togglePersonDetection();
    void updatePersonDetectionButton();
    double getPersonDetectionProcessingTime() const;
    bool isGPUAvailable() const;
    bool isCUDAAvailable() const;

    // Temporarily disabled Hand Detection Control Methods
    // void setShowHandDetection(bool show);
    // bool getShowHandDetection() const;
    // void setHandDetectionConfidenceThreshold(double threshold);
    // double getHandDetectionConfidenceThreshold() const;
    // QList<AdvancedHandDetection> getLastHandDetections() const;
    // void toggleHandDetection();
    // void updateHandDetectionButton();
    // double getHandDetectionProcessingTime() const;

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
    void segmentationCompleted();

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

    // Unified Person Detection and Segmentation Slots
    void onPersonDetectionFinished();

    // Temporarily disabled Hand Detection Slots
    // void onHandDetectionFinished();

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
    
    // Status overlay for key presses
    QLabel* statusOverlay = nullptr;
    
    // Temporarily disabled TFLite Segmentation Methods
    // void processFrameWithTFLite(const cv::Mat &frame);
    // void applySegmentationToFrame(cv::Mat &frame);

    // Temporarily disabled Hand Detection Methods
    // void processFrameWithHandDetection(const cv::Mat &frame);
    // void applyHandDetectionToFrame(cv::Mat &frame);
    // void drawHandBoundingBoxes(cv::Mat &frame, const QList<AdvancedHandDetection> &detections);
    // void initializeHandDetection();
    // void enableHandDetection(bool enable);
    
    // Temporarily disabled Public method to control hand detection
    // void setHandDetectionEnabled(bool enabled);
    
    // Temporarily disabled Hand detection state
    // bool m_handDetectionEnabled;

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
    QLabel *gpuStatusLabel;
    QLabel *cudaStatusLabel;
    QLabel *personDetectionLabel;
    QPushButton *personDetectionButton;
    QLabel *personSegmentationLabel;
    QPushButton *personSegmentationButton;
    QLabel *handDetectionLabel;
    QPushButton *handDetectionButton;
    QTimer *debugUpdateTimer;
    int m_currentFPS;

    // Unified Person Detection and Segmentation Members
    enum DisplayMode { NormalMode, RectangleMode, SegmentationMode };
    DisplayMode m_displayMode;  // Three-way toggle: Normal -> Rectangles -> Segmentation -> Normal
    double m_personDetectionFPS;
    double m_lastPersonDetectionTime;
    cv::Mat m_currentFrame;
    cv::Mat m_lastSegmentedFrame;
    mutable QMutex m_personDetectionMutex;
    QElapsedTimer m_personDetectionTimer;
    cv::HOGDescriptor m_hogDetector;  // CPU fallback
    cv::HOGDescriptor m_hogDetectorDaimler;  // CPU fallback
    cv::Ptr<cv::cuda::HOG> m_cudaHogDetector;  // CUDA-accelerated HOG detection
    cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgSubtractor;
    bool m_useGPU;
    bool m_useCUDA;
    bool m_gpuUtilized;
    bool m_cudaUtilized;
    QFutureWatcher<cv::Mat> *m_personDetectionWatcher;
    std::vector<cv::Rect> m_lastDetections;

    // Temporarily disabled Hand Detection Members
    // AdvancedHandDetector *m_handDetector;
    // bool m_showHandDetection;
    // mutable QMutex m_handDetectionMutex;
    // QElapsedTimer m_handDetectionTimer;
    // double m_lastHandDetectionTime;
    // double m_handDetectionFPS;
    // HandTrackerMP m_handTracker;
    // QList<AdvancedHandDetection> m_lastHandDetections;

    // Performance optimization
    QPixmap m_cachedPixmap;

    // Temporarily disabled Static helper functions
    // static cv::Mat processFrameAsync(const cv::Mat &frame, TFLiteDeepLabv3 *segmentation);

    // Unified Person Detection and Segmentation methods
    void initializePersonDetection();
    cv::Mat processFrameWithUnifiedDetection(const cv::Mat &frame);
    cv::Mat createSegmentedFrame(const cv::Mat &frame, const std::vector<cv::Rect> &detections);
    cv::Mat enhancedSilhouetteSegment(const cv::Mat &frame, const cv::Rect &detection);
    std::vector<cv::Rect> detectPeople(const cv::Mat &frame);
    std::vector<cv::Rect> filterByMotion(const std::vector<cv::Rect> &detections, const cv::Mat &motionMask);
    cv::Mat getMotionMask(const cv::Mat &frame);
    void adjustRect(cv::Rect &r) const;
    
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
