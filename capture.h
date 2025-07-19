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
#include <opencv2/opencv.hpp>
#include "videotemplate.h"
#include "foreground.h"

// --- NEW INCLUDES FOR QPROCESS AND JSON ---
#include <QProcess>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDir>
#include <QDateTime>
#include <QCoreApplication> // For applicationDirPath()
#include <QElapsedTimer> // Include for QElapsedTimer
// --- END NEW INCLUDES ---

QT_BEGIN_NAMESPACE
namespace Ui { class Capture; }
QT_END_NAMESPACE

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

protected:
    void resizeEvent(QResizeEvent *event) override;

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
    QProcess *yoloProcess; // QProcess member
    bool isProcessingFrame; // Flag to manage concurrent detection calls
    QString currentTempImagePath; // To keep track of the temp image being processed
    // --- END MODIFIED/NEW MEMBERS ---

    // pass foreground
    Foreground *foreground;
    QLabel* overlayImageLabel = nullptr;
    // --- MODIFIED: detectPersonInImage now returns void, processing done in slot ---
    void detectPersonInImage(const QString& imagePath);

signals:
    void backtoPreviousPage();
    void imageCaptured(const QPixmap &image);
    void videoRecorded(const QList<QPixmap> &frames);
    void showFinalOutputPage();
    // --- NEW SIGNAL (optional) to notify UI of person detection ---
    void personDetectedInFrame();
};

#endif // CAPTURE_H
