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
#include <QList>              // Required for QList<QPixmap>
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
#include <opencv2/cudafilters.hpp>  // Required for cv::cuda::Filter
#include "core/videotemplate.h"   // Your custom VideoTemplate class
#include "core/camera.h"          // Your custom Camera class
#include "ui/foreground.h"        // Foreground class
#include "core/common_types.h"    // Common data structures
#include "algorithms/lighting_correction/lighting_corrector.h"
#include <array>

//  GPU Memory Pool for optimized CUDA operations
class GPUMemoryPool {
private:
    // Pre-allocated GPU buffers for triple buffering
    cv::cuda::GpuMat gpuFrameBuffers[3];        // Frame buffers
    cv::cuda::GpuMat gpuSegmentationBuffers[2]; // Segmentation buffers
    cv::cuda::GpuMat gpuDetectionBuffers[2];    // Detection buffers
    cv::cuda::GpuMat gpuTempBuffers[2];         // Temporary processing buffers
    
    //  Guided Filtering GPU buffers for edge-blending
    cv::cuda::GpuMat gpuGuidedFilterBuffers[4]; // Guided filter processing buffers
    cv::cuda::GpuMat gpuBoxFilterBuffers[2];    // Box filter intermediate buffers
    
    //  Edge Blurring GPU buffers for enhanced edge processing
    cv::cuda::GpuMat gpuEdgeBlurBuffers[3];     // Edge blurring processing buffers
    cv::cuda::GpuMat gpuEdgeDetectionBuffers[2]; // Edge detection intermediate buffers
    
    // Reusable CUDA filters (create once, use many times)
    // Note: cv::Ptr cannot be default-constructed, so these are initialized in initialize() method
    cv::Ptr<cv::cuda::Filter> morphCloseFilter = nullptr;
    cv::Ptr<cv::cuda::Filter> morphOpenFilter = nullptr;
    cv::Ptr<cv::cuda::Filter> morphDilateFilter = nullptr;
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = nullptr;
    
    // CUDA streams for parallel processing
    cv::cuda::Stream detectionStream;
    cv::cuda::Stream segmentationStream;
    cv::cuda::Stream compositionStream;
    
    // Buffer rotation indices
    int currentFrameBuffer = 0;
    int currentSegBuffer = 0;
    int currentDetBuffer = 0;
    int currentTempBuffer = 0;
    
    //  Guided Filtering buffer rotation indices
    int currentGuidedFilterBuffer = 0;
    int currentBoxFilterBuffer = 0;
    
    //  Edge Blurring buffer rotation indices
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
    
    //  Guided Filtering buffer management
    cv::cuda::GpuMat& getNextGuidedFilterBuffer();
    cv::cuda::GpuMat& getNextBoxFilterBuffer();
    
    //  Edge Blurring buffer management
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
    
    // Post-Processing Control
    void startPostProcessing(); // Trigger post-processing after user confirmation
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
    
    // System monitor integration
    void setSystemMonitor(class SystemMonitor* monitor);
    void updatePersonDetectionButton();
    double getPersonDetectionProcessingTime() const;
    bool isGPUAvailable() const;
    bool isCUDAAvailable() const;
    
    // Green-screen segmentation controls
    void setGreenScreenEnabled(bool enabled);
    bool isGreenScreenEnabled() const;
    void setGreenHueRange(int hueMin, int hueMax);
    void setGreenSaturationMin(int sMin);
    void setGreenValueMin(int vMin);

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
    void videoRecordedForConfirm(const QList<QPixmap> &frames, double fps); // Send video to confirm page
    void videoProcessingProgress(int percent);
    void showLoadingPage(); // Show loading UI during post-processing
    void showConfirmPage(); // Show confirm page after dynamic recording
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

    // Unified Person Detection and Segmentation Slots
    void onPersonDetectionFinished();
    
    // Video Playback Timer Slots
    void onVideoPlaybackTimer(); // Handle video frame advancement at native frame rate
    
    //  Asynchronous Video Processing Slot
    void onVideoProcessingFinished();
    
    //  Asynchronous Recording Slots
    void processRecordingFrame();

private:
    // Declare these private functions here (already correct)
    void performImageCapture();
    void startRecording();
    void stopRecording();
    void showCaptureFlash();  // Show flash animation when taking picture

    Ui::Capture *ui;

    // IMPORTANT: Reorder these to match constructor initializer list for 'initialized after' warning
    Foreground *foreground; // Declared first as it's initialized before cameraThread in Ctor
    QThread *cameraThread;
    Camera *cameraWorker;

    QTimer *countdownTimer;
    QLabel *countdownLabel;
    QLabel *flashOverlayLabel;  // Flash overlay for capture animation
    QPropertyAnimation *flashAnimation;  // Animation for flash effect
    QLabel *recordingTimerLabel;  // Timer label showing remaining recording time
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
    
    // System monitor for FPS tracking
    class SystemMonitor* m_systemMonitor;
    
    // Status overlay for key presses
    QLabel* statusOverlay = nullptr;
    
    // Loading camera label
    QLabel* loadingCameraLabel = nullptr;
    
    // Camera initialization tracking
    bool m_cameraFirstInitialized = false;

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
    QTimer *debugUpdateTimer;
    int m_currentFPS;
    bool m_fpsTrackingEnabled;  // Track if FPS monitoring is active (only when in capture interface)

    // Capture Mode State
    bool m_captureReady;
    
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
    mutable QMutex m_dynamicVideoMutex; // Thread-safe access to dynamic video frames
    
    // Video Playback Timer for Phase 1: Frame Rate Synchronization
    QTimer *m_videoPlaybackTimer; // Separate timer for video frame rate synchronization
    double m_videoFrameRate; // Native video frame rate (FPS)
    int m_videoFrameInterval; // Timer interval in milliseconds
    bool m_videoPlaybackActive; // Track if video playback is active
    int m_videoTotalFrames; // Total frame count of template video for exact sync
    
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
    int m_segmentedFrameCounter;  // Track unique frames for accurate FPS measurement
    mutable QMutex m_personDetectionMutex;
    QElapsedTimer m_personDetectionTimer;
    cv::Ptr<cv::BackgroundSubtractorMOG2> m_bgSubtractor;
    cv::Mat m_subtractionReferenceImage;  // Static reference image for background subtraction
    cv::Mat m_subtractionReferenceImage2;  // Second static reference image for background subtraction
    double m_subtractionBlendWeight;  // Weight for blending two reference images (0.0-1.0)
    bool m_useGPU;
    bool m_useCUDA;
    bool m_gpuUtilized;
    bool m_cudaUtilized;
    QFutureWatcher<cv::Mat> *m_personDetectionWatcher;
    std::vector<cv::Rect> m_lastDetections;

    cv::Mat m_selectedTemplate;

    //  GPU Memory Pool for optimized CUDA operations
    GPUMemoryPool m_gpuMemoryPool;
    bool m_gpuMemoryPoolInitialized;
    bool m_firstFrameHandled;  // Track if first frame has been processed
    
    //  Asynchronous Recording System
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
    
    // Phase 2A: GPU-Only Processing Methods
    void initializeGPUOnlyProcessing();
    bool isGPUOnlyProcessingAvailable() const;
    cv::Mat processFrameWithGPUOnlyPipeline(const cv::Mat &frame);
    cv::Mat createSegmentedFrameGPUOnly(const cv::Mat &frame, const std::vector<cv::Rect> &detections);
    cv::Mat enhancedSilhouetteSegmentGPUOnly(const cv::cuda::GpuMat &gpuFrame, const cv::Rect &detection);
    
    // Green-screen helpers
    cv::Mat createGreenScreenPersonMask(const cv::Mat &frame) const;
    cv::cuda::GpuMat createGreenScreenPersonMaskGPU(const cv::cuda::GpuMat &gpuFrame) const;
    cv::cuda::GpuMat removeGreenSpillGPU(const cv::cuda::GpuMat &gpuFrame, const cv::cuda::GpuMat &gpuMask) const;
    std::vector<cv::Rect> deriveDetectionsFromMask(const cv::Mat &mask) const;
    
    // Helper methods (implemented in .cpp)
    void updateDebugDisplay();
    void setupDebugDisplay();

    // --- PERFORMANCE MONITORING ---
    void printPerformanceStats();
    // --- END PERFORMANCE MONITORING ---

    //  Asynchronous Recording Methods
    void initializeRecordingSystem();
    void cleanupRecordingSystem();
    void queueFrameForRecording(const cv::Mat &frame);
    QPixmap processFrameForRecordingGPU(const cv::Mat &frame);
    
    // Lighting Correction Methods
    void initializeLightingCorrection();
    bool isGPULightingAvailable() const;
    void setReferenceTemplate(const QString &templatePath);
    
    // Background Subtraction Reference Image
    void setSubtractionReferenceImage(const QString &imagePath);
    void setSubtractionReferenceImage2(const QString &imagePath);
    void setSubtractionReferenceBlendWeight(double weight);
    cv::Mat createPersonMaskFromSegmentedFrame(const cv::Mat &segmentedFrame);
    cv::Mat applyPersonColorMatching(const cv::Mat &segmentedFrame);
    cv::Mat applyLightingToRawPersonRegion(const cv::Mat &personRegion, const cv::Mat &personMask);
    cv::Mat applyVideoOptimizedLighting(const cv::Mat &personRegion, const cv::Mat &personMask, LightingCorrector* lightingCorrector);
    cv::Mat applyPostProcessingLighting();
    QList<QPixmap> processRecordedVideoWithLighting(const QList<QPixmap> &inputFrames, double fps);
    // Thread-safe wrapper functions for parallel processing
    cv::Mat applyDynamicFrameEdgeBlendingSafe(const cv::Mat &composedFrame,
                                              const cv::Mat &rawPersonRegion,
                                              const cv::Mat &rawPersonMask,
                                              const cv::Mat &backgroundFrame,
                                              LightingCorrector* lightingCorrector,
                                              double personScaleFactor,
                                              const cv::Mat &lastTemplateBackground,
                                              bool useCUDA,
                                              GPUMemoryPool* gpuMemoryPool);
    cv::Mat applySimpleDynamicCompositingSafe(const cv::Mat &composedFrame,
                                              const cv::Mat &rawPersonRegion,
                                              const cv::Mat &rawPersonMask,
                                              const cv::Mat &backgroundFrame,
                                              LightingCorrector* lightingCorrector,
                                              double personScaleFactor,
                                              bool useCUDA);
    
    //  Optimized Async Lighting Processing (POST-PROCESSING ONLY)
    void initializeAsyncLightingSystem();
    void cleanupAsyncLightingSystem();
    // REMOVED: Real-time lighting methods - lighting only applied in post-processing
    
    // Lighting Correction Member
    LightingCorrector *m_lightingCorrector;
    
    //  Simplified Lighting Processing (POST-PROCESSING ONLY)
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
    double m_recordedPersonScaleFactor; // Store scaling factor used during recording
    
    // Utility functions
    cv::Mat qImageToCvMat(const QImage &image);
    QString resolveTemplatePath(const QString &templatePath);
    
    // Green-screen configuration
    bool m_greenScreenEnabled;
    int m_greenHueMin;   // HSV hue min for green
    int m_greenHueMax;   // HSV hue max for green
    int m_greenSatMin;   // HSV min saturation to be considered green
    int m_greenValMin;   // HSV min value to be considered green
    int m_greenMaskOpen; // morph open kernel size
    int m_greenMaskClose;// morph close kernel size
    
    //  GPU Green Screen Filter Cache (prevent memory allocation on every frame)
    cv::Ptr<cv::cuda::CannyEdgeDetector> m_greenScreenCannyDetector;
    cv::Ptr<cv::cuda::Filter> m_greenScreenMorphOpen;
    cv::Ptr<cv::cuda::Filter> m_greenScreenMorphClose;
    cv::Ptr<cv::cuda::Filter> m_greenScreenGaussianBlur;
    
    // Temporal green screen mask smoothing
    mutable cv::Mat m_lastGreenScreenMask;
    mutable int m_greenScreenMaskStableCount;

    struct AdaptiveGreenThresholds {
        int hueMin = 30;
        int hueMax = 95;
        int strictSatMin = 30;
        int relaxedSatMin = 10;
        int strictValMin = 30;
        int relaxedValMin = 10;
        int darkSatMin = 5;
        int darkValMax = 80;
        double cbMin = 60.0;
        double cbMax = 140.0;
        double crMax = 150.0;
        double greenDelta = 8.0;
        double greenRatioMin = 0.42;
        double lumaMin = 30.0;
        double probabilityThreshold = 0.55;
        int guardValueMax = 140;
        int guardSatMax = 80;
        double edgeGuardMin = 45.0;
        int hueGuardPadding = 6;
        double invVarB = 1.0 / 400.0;
        double invVarG = 1.0 / 400.0;
        double invVarR = 1.0 / 400.0;
        double colorDistanceThreshold = 3.2;
        double colorGuardThreshold = 4.8;
    };

    void updateGreenBackgroundModel(const cv::Mat &frame) const;
    AdaptiveGreenThresholds computeAdaptiveGreenThresholds() const;

    mutable bool m_bgModelInitialized;
    mutable double m_bgHueMean;
    mutable double m_bgHueStd;
    mutable double m_bgSatMean;
    mutable double m_bgSatStd;
    mutable double m_bgValMean;
    mutable double m_bgValStd;
    mutable double m_bgCbMean;
    mutable double m_bgCbStd;
    mutable double m_bgCrMean;
    mutable double m_bgCrStd;
    mutable double m_bgRedMean;
    mutable double m_bgGreenMean;
    mutable double m_bgBlueMean;
    mutable double m_bgRedStd;
    mutable double m_bgGreenStd;
    mutable double m_bgBlueStd;
    mutable cv::Matx33d m_bgColorInvCov;
    mutable bool m_bgColorInvCovReady;
};

#endif // CAPTURE_H
