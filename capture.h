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
#include "common_types.h"
#include "tflite_deeplabv3.h"
#include "tflite_segmentation_widget.h"
#include "advanced_hand_detector.h"
#include "mediapipe_like_hand_tracker.h"

// Qt includes for threading and processing
#include <QProcess>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDir>
#include <QDateTime>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QMutex>
#include <QThread>
#include <QMessageBox>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>

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

    explicit Capture(QWidget *parent = nullptr, Foreground *fg = nullptr);
    ~Capture();

    void setCaptureMode(CaptureMode mode);
    void setVideoTemplate(const VideoTemplate &templateData);
    
    // TFLite Deeplabv3 Segmentation Control Methods
    void setShowSegmentation(bool show);
    bool getShowSegmentation() const;
    void setSegmentationConfidenceThreshold(double threshold);
    double getSegmentationConfidenceThreshold() const;
    cv::Mat getLastSegmentedFrame() const;
    void saveSegmentedFrame(const QString& filename = "");
    double getSegmentationProcessingTime() const;
    void setSegmentationPerformanceMode(TFLiteDeepLabv3::PerformanceMode mode);
    bool isTFLiteModelLoaded() const;
    void initializeTFLiteSegmentation();
    void toggleSegmentation();
    void updateSegmentationButton();
    cv::Mat qImageToCvMat(const QImage &image);
    
    // Hand Detection Control Methods
    void setShowHandDetection(bool show);
    bool getShowHandDetection() const;
    void setHandDetectionConfidenceThreshold(double threshold);
    double getHandDetectionConfidenceThreshold() const;
    QList<AdvancedHandDetection> getLastHandDetections() const;
    void toggleHandDetection();
    void updateHandDetectionButton();
    double getHandDetectionProcessingTime() const;

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
    void startCountdown();
    void updateRecordTimer();
    void on_verticalSlider_valueChanged(int value);
    void updateForegroundOverlay(const QString &path);
    
    // TFLite Segmentation Slots
    void onSegmentationResultReady(const QImage &segmentedImage);
    void onSegmentationError(const QString &error);
    void onTFLiteModelLoaded(bool success);
    void updateDebugDisplay();
    void onSegmentationFinished();
    
    // Hand Detection Slots
    void onHandDetectionFinished();

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
    void setupStackedLayoutHybrid();
    void updateOverlayStyles();

    // Hybrid stacked layout components
    QStackedLayout *stackedLayout;

    // Performance tracking members
    QLabel *videoLabelFPS;
    QElapsedTimer loopTimer;
    qint64 totalTime;
    int frameCount;
    QElapsedTimer frameTimer;

    // TFLite Deeplabv3 Segmentation Members
    TFLiteDeepLabv3 *m_tfliteSegmentation;
    TFLiteSegmentationWidget *m_segmentationWidget;
    bool m_showSegmentation;
    double m_segmentationConfidenceThreshold;
    cv::Mat m_currentFrame;
    cv::Mat m_lastSegmentedFrame;
    mutable QMutex m_segmentationMutex;
    QElapsedTimer m_segmentationTimer;
    double m_lastSegmentationTime;
    int m_segmentationFPS;
    bool m_tfliteModelLoaded;
    QThread *m_segmentationThread;

    // pass foreground
    Foreground *foreground;
    QLabel* overlayImageLabel;

    // TFLite Segmentation Methods
    void processFrameWithTFLite(const cv::Mat &frame);
    void applySegmentationToFrame(cv::Mat &frame);
    void updateSegmentationDisplay(const QImage &segmentedImage);
    void showSegmentationNotification();

    // Hand Detection Methods
    void processFrameWithHandDetection(const cv::Mat &frame);
    void applyHandDetectionToFrame(cv::Mat &frame);
    void drawHandBoundingBoxes(cv::Mat &frame, const QList<AdvancedHandDetection> &detections);
    void initializeHandDetection();
    void enableHandDetection(bool enable);
    
    // Public method to control hand detection
    void setHandDetectionEnabled(bool enabled);
    
    // Hand detection state
    bool m_handDetectionEnabled;

    // Debug Display Members
    QWidget *debugWidget;
    QLabel *debugLabel;
    QLabel *fpsLabel;
    QLabel *segmentationLabel;
    QPushButton *segmentationButton;
    QLabel *handDetectionLabel;
    QPushButton *handDetectionButton;
    QTimer *debugUpdateTimer;
    int m_currentFPS;
    
    // Debug Display Methods
    void setupDebugDisplay();

    // Async processing members
    QFutureWatcher<cv::Mat> *m_segmentationWatcher;
    QMutex m_asyncMutex;
    bool m_processingAsync;
    cv::Mat m_lastProcessedFrame;
    QPixmap m_cachedPixmap;
    static cv::Mat processFrameAsync(const cv::Mat &frame, TFLiteDeepLabv3 *segmentation);

    // Hand Detection Members
    AdvancedHandDetector *m_handDetector;
    bool m_showHandDetection;
    QList<AdvancedHandDetection> m_lastHandDetections;
    mutable QMutex m_handDetectionMutex;
    QElapsedTimer m_handDetectionTimer;
    double m_lastHandDetectionTime;
    int m_handDetectionFPS;
    HandTrackerMP m_handTracker; // MediaPipe-like motion+ROI tracker

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);
    void showFinalOutputPage();
    void segmentationCompleted();
};

#endif // CAPTURE_H
