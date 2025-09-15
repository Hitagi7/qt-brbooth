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
#include <QQueue>              // Required for QQueue<cv::Mat>
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
#include <opencv2/cudacodec.hpp>
#include "core/videotemplate.h"   // Your custom VideoTemplate class
#include "core/camera.h"          // Your custom Camera class
#include "ui/foreground.h"        // Foreground class
#include "core/common_types.h"    // Common data structures
#include "algorithms/hand_detection/hand_detector.h"
#include "algorithms/lighting_correction/lighting_corrector.h"

// 🚀 GPU Memory Pool for optimized CUDA operations
class GPUMemoryPool {
private:
    // Pre-allocated GPU buffers for triple buffering
    cv::cuda::GpuMat gpuFrameBuffers[3];        // Frame buffers
    cv::cuda::GpuMat gpuSegmentationBuffers[2]; // Segmentation buffers
    cv::cuda::GpuMat gpuDetectionBuffers[2];    // Detection buffers
    cv::cuda::GpuMat gpuTempBuffers[2];         // Temporary processing buffers
    
    // 🚀 Guided Filtering GPU buffers for edge-blending
    cv::cuda::GpuMat gpuGuidedFilterBuffers[4]; // Guided filter processing buffers
    cv::cuda::GpuMat gpuBoxFilterBuffers[2];    // Box filter intermediate buffers
    
    // 🚀 Edge Blurring GPU buffers for enhanced edge processing
    cv::cuda::GpuMat gpuEdgeBlurBuffers[3];     // Edge blurring processing buffers
    cv::cuda::GpuMat gpuEdgeDetectionBuffers[2]; // Edge detection intermediate buffers
    
    // Reusable CUDA filters (create once, use many times)
    cv::Ptr<cv::cuda::Filter> morphCloseFilter;
    cv::Ptr<cv::cuda::Filter> morphOpenFilter;
    cv::Ptr<cv::cuda::Filter> morphDilateFilter;
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector;
    
    // CUDA streams for parallel processing
    cv::cuda::Stream detectionStream;
    cv::cuda::Stream segmentationStream;
    cv::cuda::Stream compositionStream;
    
    // Buffer rotation indices
    int currentFrameBuffer = 0;
    int currentSegBuffer = 0;
    int currentDetBuffer = 0;
    int currentTempBuffer = 0;
    
    // 🚀 Guided Filtering buffer rotation indices
    int currentGuidedFilterBuffer = 0;
    int currentBoxFilterBuffer = 0;
    
    // 🚀 Edge Blurring buffer rotation indices
    int currentEdgeBlurBuffer = 0;
    int currentEdgeDetectionBuffer = 0;
    
    // Pool state
    bool initialized = false;
    int poolWidth = 0;
    int poolHeight = 0;
    
public:
    GPUMemoryPool();
    ~GPUMemoryPool();
    
    // Initialization
    void initialize(int width, int height);
    bool isInitialized() const { return initialized; }
    
    // Buffer management
    cv::cuda::GpuMat& getNextFrameBuffer();
    cv::cuda::GpuMat& getNextSegmentationBuffer();
    cv::cuda::GpuMat& getNextDetectionBuffer();
    cv::cuda::GpuMat& getNextTempBuffer();
    
    // 🚀 Guided Filtering buffer management
    cv::cuda::GpuMat& getNextGuidedFilterBuffer();
    cv::cuda::GpuMat& getNextBoxFilterBuffer();
    
    // 🚀 Edge Blurring buffer management
    cv::cuda::GpuMat& getNextEdgeBlurBuffer();
    cv::cuda::GpuMat& getNextEdgeDetectionBuffer();
    
    // Filter access
    cv::Ptr<cv::cuda::Filter>& getMorphCloseFilter() { return morphCloseFilter; }
    cv::Ptr<cv::cuda::Filter>& getMorphOpenFilter() { return morphOpenFilter; }
    cv::Ptr<cv::cuda::Filter>& getMorphDilateFilter() { return morphDilateFilter; }
    cv::Ptr<cv::cuda::CannyEdgeDetector>& getCannyDetector() { return cannyDetector; }
    
    // Stream access
    cv::cuda::Stream& getDetectionStream() { return detectionStream; }
    cv::cuda::Stream& getSegmentationStream() { return segmentationStream; }
    cv::cuda::Stream& getCompositionStream() { return compositionStream; }
    
    // Memory management
    void release();
    void resetBuffers();
};

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
    // Dynamic video background control
    void enableDynamicVideoBackground(const QString &videoPath);
    void disableDynamicVideoBackground();
    bool isDynamicVideoBackgroundEnabled() const;
    void resetDynamicVideoToStart(); // Reset video to beginning for re-recording
    void clearDynamicVideoPath(); // Clear dynamic video path for mode switching
    
    // Background Template Control Methods
    void setSelectedBackgroundTemplate(const QString &path);
    QString getSelectedBackgroundTemplate() const;
    
    // Video Template Duration Control Methods
    void setVideoTemplateDuration(int durationSeconds);
    int getVideoTemplateDuration() const;

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
    void enableProcessingModes(); // Safely enable processing modes after camera is stable
    void disableProcessingModes(); // Disable heavy processing modes for non-capture pages
    
    // Segmentation Control Methods for Capture Interface
    void enableSegmentationInCapture(); // Enable segmentation when capture page is active
    void disableSegmentationOutsideCapture(); // Disable segmentation when leaving capture page
    void restoreSegmentationState(); // Restore last segmentation state when returning to capture
    bool isSegmentationEnabledInCapture() const; // Check if segmentation is enabled for capture
    void setSegmentationMode(int mode); // Set specific segmentation mode (0=Normal, 1=Rectangle, 2=Segmentation)
    
    // Resource Management Methods
    void cleanupResources(); // Clean up all resources when leaving capture page
    void initializeResources(); // Initialize resources when entering capture page
    
    // Loading camera label management
    void showLoadingCameraLabel();
    void hideLoadingCameraLabel();
    
    void handleFirstFrame();
    
    // Capture Mode Control
    void setCaptureReady(bool ready);
    bool isCaptureReady() const;
    
    // Page Reset
    void resetCapturePage();
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

protected:
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void imageCapturedWithComparison(const QPixmap &correctedImage, const QPixmap &originalImage);
    void imageCapturedForLoading(const QPixmap &originalImage); // Original image for loading page preview
    void videoRecorded(const QList<QPixmap> &frames, double fps);
    void videoRecordedWithComparison(const QList<QPixmap> &correctedFrames, const QList<QPixmap> &originalFrames, double fps);
    void videoRecordedForLoading(const QList<QPixmap> &originalFrames, double fps); // Original frames for loading page preview
    void videoProcessingProgress(int percent);
    void showLoadingPage(); // Show loading UI during post-processing
    void showFinalOutputPage();
    void personDetectedInFrame();
    void foregroundPathChanged(const QString &foregroundPath);
    void handTriggeredCapture(); // Signal for hand detection to trigger capture

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
    
    // Video Playback Timer Slots
    void onVideoPlaybackTimer(); // Handle video frame advancement at native frame rate

    // Hand Detection Slots
    void startHandTriggeredCountdown();
    void onHandTriggeredCapture();
    
    // 🚀 Asynchronous Recording Slots
    void processRecordingFrame();

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
    QList<QPixmap> m_originalRecordedFrames; // Frames before lighting correction for comparison
    QPixmap m_capturedImage;

    // Hybrid stacked layout components
    QStackedLayout *stackedLayout;

    // Performance tracking members
    QLabel *videoLabelFPS;
    QElapsedTimer loopTimer;
    qint64 totalTime;
    int frameCount;
    QElapsedTimer frameTimer;
    int fpsFrameCount;

    // pass foreground
    QLabel* overlayImageLabel = nullptr;
    


    // Hand Detection Methods (using hand_detector.h/.cpp)
    void processFrameWithHandDetection(const cv::Mat &frame);
    void applyHandDetectionToFrame(cv::Mat &frame);
    void drawHandBoundingBoxes(cv::Mat &frame, const QList<HandDetection> &detections);
    void initializeHandDetection();
    void enableHandDetection(bool enable);
    
    // Status overlay for key presses
    QLabel* statusOverlay = nullptr;
    
    // Loading camera label
    QLabel* loadingCameraLabel = nullptr;
    
    // Camera initialization tracking
    bool m_cameraFirstInitialized = false;
    


    
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



    // Hand Detection Members (using hand_detector.h/.cpp)
    HandDetector *m_handDetector;
    bool m_showHandDetection;
    bool m_handDetectionEnabled;  // Add missing member
    mutable QMutex m_handDetectionMutex;
    QElapsedTimer m_handDetectionTimer;
    double m_lastHandDetectionTime;
    double m_handDetectionFPS;
    QList<HandDetection> m_lastHandDetections;
    QFuture<QList<HandDetection>> m_handDetectionFuture;
    
    // Capture Mode State
    bool m_captureReady;  // Only allow hand detection to trigger capture when true
    
    // Unified Person Detection and Segmentation Members
    enum DisplayMode { NormalMode, RectangleMode, SegmentationMode };
    DisplayMode m_displayMode;  // Three-way toggle: Normal -> Rectangles -> Segmentation -> Normal
    DisplayMode m_lastSegmentationMode;  // Store the last segmentation mode when leaving capture page
    bool m_segmentationEnabledInCapture;  // Track if segmentation should be enabled in capture interface
    
    // Background Template Members
    QString m_selectedBackgroundTemplate;  // Store the selected background template path
    bool m_useBackgroundTemplate;  // Track if background template should be used in segmentation

    // Dynamic Video Background Members
    bool m_useDynamicVideoBackground; // If true, use video frames as segmentation background
    QString m_dynamicVideoPath; // Absolute path to selected video
    cv::VideoCapture m_dynamicVideoCap; // Reader for dynamic background
    cv::Mat m_dynamicVideoFrame; // Last fetched frame for reuse if needed
    cv::Ptr<cv::cudacodec::VideoReader> m_dynamicGpuReader; // GPU video reader if available
    cv::cuda::GpuMat m_dynamicGpuFrame; // GPU frame buffer
    
    // Video Playback Timer for Phase 1: Frame Rate Synchronization
    QTimer *m_videoPlaybackTimer; // Separate timer for video frame rate synchronization
    double m_videoFrameRate; // Native video frame rate (FPS)
    int m_videoFrameInterval; // Timer interval in milliseconds
    bool m_videoPlaybackActive; // Track if video playback is active
    
    // Phase 2A: GPU-Only Video Processing Members
    cv::cuda::GpuMat m_gpuVideoFrame; // GPU video frame buffer
    cv::cuda::GpuMat m_gpuSegmentedFrame; // GPU segmented frame buffer
    cv::cuda::GpuMat m_gpuPersonMask; // GPU person mask buffer
    cv::cuda::GpuMat m_gpuBackgroundFrame; // GPU background frame buffer
    bool m_gpuOnlyProcessingEnabled; // Enable GPU-only processing pipeline
    bool m_gpuProcessingAvailable; // Check if GPU processing is available
    
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

    cv::Mat m_selectedTemplate;

    // 🚀 GPU Memory Pool for optimized CUDA operations
    GPUMemoryPool m_gpuMemoryPool;
    bool m_gpuMemoryPoolInitialized;
    
    // 🚀 Asynchronous Recording System
    QThread *m_recordingThread;
    QTimer *m_recordingFrameTimer;
    QMutex m_recordingMutex;
    QQueue<cv::Mat> m_recordingFrameQueue;
    bool m_recordingThreadActive;
    cv::cuda::Stream m_recordingStream;
    cv::cuda::GpuMat m_recordingGpuBuffer;

    // Performance optimization
    QPixmap m_cachedPixmap;

    // Unified Person Detection and Segmentation methods
    void initializePersonDetection();
    cv::Mat processFrameWithUnifiedDetection(const cv::Mat &frame);
    cv::Mat createSegmentedFrame(const cv::Mat &frame, const std::vector<cv::Rect> &detections);
    cv::Mat enhancedSilhouetteSegment(const cv::Mat &frame, const cv::Rect &detection);
    
    // 🚀 Lightweight Processing for Recording Performance
    cv::Mat createLightweightSegmentedFrame(const cv::Mat &frame);
    
    // Phase 2A: GPU-Only Processing Methods
    void initializeGPUOnlyProcessing();
    bool isGPUOnlyProcessingAvailable() const;
    cv::Mat processFrameWithGPUOnlyPipeline(const cv::Mat &frame);
    cv::Mat createSegmentedFrameGPUOnly(const cv::Mat &frame, const std::vector<cv::Rect> &detections);
    cv::Mat enhancedSilhouetteSegmentGPUOnly(const cv::cuda::GpuMat &gpuFrame, const cv::Rect &detection);
    void validateGPUResults(const cv::Mat &gpuResult, const cv::Mat &cpuResult);
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

    // 🚀 Asynchronous Recording Methods
    void initializeRecordingSystem();
    void cleanupRecordingSystem();
    void queueFrameForRecording(const cv::Mat &frame);
    QPixmap processFrameForRecordingGPU(const cv::Mat &frame);
    
    // Lighting Correction Methods
    void initializeLightingCorrection();
    void setLightingCorrectionEnabled(bool enabled);
    bool isLightingCorrectionEnabled() const;
    bool isGPULightingAvailable() const;
    void setReferenceTemplate(const QString &templatePath);
    cv::Mat applyPersonLightingCorrection(const cv::Mat &inputImage, const cv::Mat &personMask);
    cv::Mat createPersonMaskFromSegmentedFrame(const cv::Mat &segmentedFrame);
    cv::Mat applyPersonColorMatching(const cv::Mat &segmentedFrame);
    cv::Mat applyLightingToRawPersonRegion(const cv::Mat &personRegion, const cv::Mat &personMask);
    cv::Mat applyPostProcessingLighting();
    QList<QPixmap> processRecordedVideoWithLighting(const QList<QPixmap> &inputFrames, double fps);
    cv::Mat applyDynamicFrameEdgeBlending(const cv::Mat &composedFrame, 
                                          const cv::Mat &rawPersonRegion, 
                                          const cv::Mat &rawPersonMask, 
                                          const cv::Mat &backgroundFrame = cv::Mat());
    
    // 🚀 Optimized Async Lighting Processing (POST-PROCESSING ONLY)
    void initializeAsyncLightingSystem();
    void cleanupAsyncLightingSystem();
    // REMOVED: Real-time lighting methods - lighting only applied in post-processing
    
    // Lighting Correction Member
    LightingCorrector *m_lightingCorrector;
    
    // 🚀 Simplified Lighting Processing (POST-PROCESSING ONLY)
    QThread *m_lightingProcessingThread;      // Keep for future use if needed
    QFutureWatcher<QList<QPixmap>> *m_lightingWatcher; // Keep for future use if needed
    QMutex m_lightingMutex;                   // Thread safety for lighting operations
    
    // Lighting Comparison Storage
    cv::Mat m_originalCapturedImage;      // Original image without lighting correction
    cv::Mat m_lightingCorrectedImage;     // Image with lighting correction applied
    bool m_hasLightingComparison;         // Whether we have both versions for comparison
    bool m_hasVideoLightingComparison;    // Whether we have both video versions for comparison
    
    // Raw Person Data for Post-Processing
    cv::Mat m_lastRawPersonRegion;
    cv::Mat m_lastRawPersonMask;
    cv::Mat m_lastTemplateBackground;
    // Per-frame raw data for dynamic recording post-process
    QList<cv::Mat> m_recordedRawPersonRegions;
    QList<cv::Mat> m_recordedRawPersonMasks;
    QList<cv::Mat> m_recordedBackgroundFrames;
    
    // Utility functions
    QImage cvMatToQImage(const cv::Mat &mat);
    cv::Mat qImageToCvMat(const QImage &image);
    QString resolveTemplatePath(const QString &templatePath);
};

#endif // CAPTURE_H
