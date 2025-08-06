#ifndef TFLITE_DEEPLABV3_H
#define TFLITE_DEEPLABV3_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QQueue>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <memory>

class TFLiteDeepLabv3 : public QObject
{
    Q_OBJECT

public:
    enum PerformanceMode {
        HighQuality,    // Highest quality, slower processing
        Balanced,       // Balanced quality and speed
        HighSpeed,      // Fast processing, lower quality
        Adaptive        // Automatically adjust based on performance
    };

    explicit TFLiteDeepLabv3(QObject *parent = nullptr);
    ~TFLiteDeepLabv3();

    bool initializeModel(const QString &modelPath);
    bool isModelLoaded() const { return m_modelLoaded; }
    
    // Process a single frame
    cv::Mat segmentFrame(const cv::Mat &inputFrame);
    
    // Start/stop real-time processing
    void startRealtimeProcessing();
    void stopRealtimeProcessing();
    
    // Set processing parameters
    void setInputSize(int width, int height);
    void setConfidenceThreshold(float threshold);
    void setProcessingInterval(int msec);
    void setPerformanceMode(PerformanceMode mode);

signals:
    void segmentationResultReady(const QImage &segmentedImage);
    void processingError(const QString &error);
    void modelLoaded(bool success);

public slots:
    void processFrame(const QImage &frame);
    void processFrame(const cv::Mat &frame);

private slots:
    void processQueuedFrames();

private:
    // Model state
    bool m_modelLoaded;
    
    // Processing parameters
    int m_inputWidth;
    int m_inputHeight;
    float m_confidenceThreshold;
    int m_processingInterval;
    PerformanceMode m_performanceMode;
    
    // Real-time processing
    QTimer *m_processingTimer;
    QThread *m_processingThread;
    QMutex m_frameMutex;
    QQueue<cv::Mat> m_frameQueue;
    QWaitCondition m_frameCondition;
    bool m_processingActive;
    
    // Helper methods
    bool loadModel(const QString &modelPath);
    cv::Mat preprocessFrame(const cv::Mat &inputFrame);
    cv::Mat postprocessSegmentation(const cv::Mat &inputFrame, const std::vector<float> &output);
    cv::Mat performOpenCVSegmentation(const cv::Mat &inputFrame);
    QImage cvMatToQImage(const cv::Mat &mat);
    cv::Mat qImageToCvMat(const QImage &image);
    
    // Color palette for segmentation visualization
    std::vector<cv::Vec3b> m_colorPalette;
    void initializeColorPalette();
};

// Processing thread class
class SegmentationThread : public QThread
{
    Q_OBJECT

public:
    explicit SegmentationThread(TFLiteDeepLabv3 *processor, QObject *parent = nullptr);
    void run() override;
    void stop();

signals:
    void resultReady(const QImage &segmentedImage);

private:
    TFLiteDeepLabv3 *m_processor;
    bool m_running;
};

#endif // TFLITE_DEEPLABV3_H 