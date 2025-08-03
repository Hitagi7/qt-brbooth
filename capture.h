#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>
#include <QPushButton>
#include <QStackedLayout>
#include <QSlider>
#include <QLabel>
#include <QTimer>
#include <QPixmap>
#include <QList>
#include <QCheckBox>
#include <QApplication>
#include <opencv2/opencv.hpp>
#include "videotemplate.h"
#include "foreground.h"
#include "simplepersondetector.h"
#include "personsegmentation.h"
#include "optimized_detector.h"
#include "fast_segmentation.h"
#include "common_types.h"
// #include "segmentation_manager.h"
// #include "detection_manager.h"
// Temporarily commented out to avoid circular dependencies

// --- NEW INCLUDES FOR QPROCESS AND JSON ---
#include <QProcess>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDir>
#include <QDateTime>
#include <QCoreApplication> // For applicationDirPath()
#include <QElapsedTimer> // Include for QElapsedTimer
#include <QMutex> // Include for thread-safe detection access
// --- END NEW INCLUDES ---

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
QT_END_NAMESPACE

// Forward declarations
class QCheckBox;
class QLabel;
class QTimer;

class Capture : public QWidget
{
    Q_OBJECT

public:
    enum CaptureMode {
        ImageCaptureMode,
        VideoRecordMode
    };

    explicit Capture(QWidget *parent = nullptr, Foreground *fg = nullptr); //pass foreground class
    ~Capture();

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate &templateData);
    
    // --- NEW: Bounding Box Control Methods ---
    void setShowBoundingBoxes(bool show);
    bool getShowBoundingBoxes() const;
    int getDetectionCount() const;
    double getAverageConfidence() const;
    void onBoundingBoxCheckBoxToggled(bool checked);
    void testYoloDetection(); // Test method for YOLO detection
    // --- END NEW ---
    
    // --- NEW: Person Segmentation Methods ---
    void setShowPersonSegmentation(bool show);
    bool getShowPersonSegmentation() const;
    void onSegmentationCheckBoxToggled(bool checked);
    void setSegmentationConfidenceThreshold(double threshold);
    double getSegmentationConfidenceThreshold() const;
    cv::Mat getLastSegmentedFrame() const;
    void saveSegmentedFrame(const QString& filename = "");
    double getSegmentationProcessingTime() const;
    // --- END NEW ---

protected:
    void resizeEvent(QResizeEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private slots:
    void updateCameraFeed();
    void captureRecordingFrame();
    void on_back_clicked();
    void on_capture_clicked();
    void updateCountdown();
    void updateRecordTimer();
    void on_verticalSlider_valueChanged(int value);
    void updateForegroundOverlay(const QString &path);

    // --- NEW SLOTS FOR ASYNCHRONOUS QPROCESS ---
    void handleYoloOutput();
    void handleYoloError();
    void handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleYoloErrorOccurred(QProcess::ProcessError error);
    void printPerformanceStats(); // <-- ADDED THIS DECLARATION
    // --- END NEW SLOTS ---
    
    // --- NEW: Debug Display Slots ---
    void updateDebugDisplay();
    // --- END NEW ---
    
private:
    Ui::Capture *ui;
    cv::VideoCapture cap;
    QTimer *cameraTimer;
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
    void setupStackedLayoutHybrid(); // New method for hybrid approach
    void updateOverlayStyles();      // New method to update overlay styles

    // Hybrid stacked layout components
    QStackedLayout *stackedLayout;

    // Performance tracking members
    QLabel *videoLabelFPS;
    QElapsedTimer loopTimer;
    qint64 totalTime;
    int frameCount;
    QElapsedTimer frameTimer;

    // --- MODIFIED/NEW MEMBERS FOR ASYNCHRONOUS YOLO ---
    QProcess *yoloProcess; // QProcess member (keeping for fallback)
    bool isProcessingFrame; // Flag to manage concurrent detection calls
    QString currentTempImagePath; // To keep track of the temp image being processed
    // --- END MODIFIED/NEW MEMBERS ---
    
    // --- NEW: C++ Person Detector ---
    SimplePersonDetector *m_personDetector; // Simple person detector
    bool m_useCppDetector; // Flag to use C++ detector instead of Python
    // --- END NEW ---

    // --- NEW: Optimized Detector ---
    OptimizedPersonDetector *m_optimizedDetector; // High-performance ONNX detector
    bool m_useOptimizedDetector; // Flag to use optimized detector
    // --- END NEW ---

    // --- NEW: Bounding Box Members ---
    QList<BoundingBox> m_currentDetections; // Store current frame detections
    mutable QMutex m_detectionMutex; // Thread-safe access to detections
    bool m_showBoundingBoxes; // Toggle for showing/hiding boxes
    // --- END NEW ---

    // --- NEW: Person Segmentation Members ---
    PersonSegmentationProcessor *m_segmentationProcessor; // Segmentation processor
    bool m_showPersonSegmentation; // Toggle for showing/hiding segmentation
    double m_segmentationConfidenceThreshold; // Minimum confidence for segmentation
    cv::Mat m_lastSegmentedFrame; // Store last segmented frame
    QList<SegmentationResult> m_currentSegmentations; // Store current segmentation results
    mutable QMutex m_segmentationMutex; // Thread-safe access to segmentation results
    // --- END NEW ---

    // --- NEW: Fast Segmentation Members ---
    FastSegmentationProcessor *m_fastSegmentationProcessor; // Fast segmentation processor
    QList<FastSegmentationResult> m_currentFastSegmentations; // Store current fast segmentation results
    mutable QMutex m_fastSegmentationMutex; // Thread-safe access to fast segmentation results
    // --- END NEW ---

    // pass foreground
    Foreground *foreground;
    QLabel* overlayImageLabel;
    // --- MODIFIED: detectPersonInImage now returns void, processing done in slot ---
    void detectPersonInImage(const QString& imagePath);

    // --- NEW: Bounding Box Drawing Methods ---
    void drawBoundingBoxes(QPixmap& pixmap, const QList<BoundingBox>& detections);
    void updateDetectionResults(const QList<BoundingBox>& detections);
    void showBoundingBoxNotification();
    // --- END NEW ---

    // --- NEW: Person Segmentation Methods ---
    void processPersonSegmentation(const cv::Mat& frame, const QList<BoundingBox>& detections);
    void updateSegmentationResults(const QList<SegmentationResult>& results);
    void applySegmentationToFrame(cv::Mat& frame, const QList<SegmentationResult>& results);
    void showSegmentationNotification();
    // --- END NEW ---

    // --- NEW: Debug Display Members ---
    QWidget *debugWidget;
    QLabel *debugLabel;
    QLabel *fpsLabel;
    QLabel *detectionLabel;
    QCheckBox *boundingBoxCheckBox;
    QCheckBox *segmentationCheckBox;
    QTimer *debugUpdateTimer;
    int m_currentFPS;
    bool m_personDetected;
    int m_detectionCount;
    double m_averageConfidence;
    
    // --- NEW: Debug Display Methods ---
    void setupDebugDisplay();
    // --- END NEW ---

    // --- NEW: Optimized Detector Slots ---
    void onOptimizedDetectionsReady(const QList<OptimizedDetection>& detections);
    void onOptimizedProcessingFinished();
    void processOptimizedSegmentation(const QList<OptimizedDetection>& detections);
    // --- END NEW ---

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);
    void showFinalOutputPage();
    // --- NEW SIGNAL (optional) to notify UI of person detection ---
    void personDetectedInFrame();
};

#endif // CAPTURE_H
