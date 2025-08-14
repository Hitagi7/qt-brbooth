#ifndef TFLITE_SEGMENTATION_WIDGET_H
#define TFLITE_SEGMENTATION_WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QComboBox>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include "algorithms/tflite_deeplabv3.h"

class TFLiteSegmentationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TFLiteSegmentationWidget(QWidget *parent = nullptr);
    ~TFLiteSegmentationWidget();

    // Control methods
    void setShowSegmentation(bool show);
    bool getShowSegmentation() const;
    void setConfidenceThreshold(double threshold);
    double getConfidenceThreshold() const;
    void setPerformanceMode(TFLiteDeepLabv3::PerformanceMode mode);
    TFLiteDeepLabv3::PerformanceMode getPerformanceMode() const;

signals:
    void showSegmentationChanged(bool show);
    void confidenceThresholdChanged(double threshold);
    void performanceModeChanged(TFLiteDeepLabv3::PerformanceMode mode);
    void segmentationStarted();
    void segmentationStopped();
    void segmentationError(const QString &error);

private slots:
    void onShowSegmentationToggled(bool checked);
    void onConfidenceThresholdChanged(int value);
    void onPerformanceModeChanged(int index);

private:
    // UI Components
    QGroupBox *m_mainGroupBox;
    QCheckBox *m_showSegmentationCheckBox;
    QLabel *m_confidenceLabel;
    QSlider *m_confidenceSlider;
    QLabel *m_confidenceValueLabel;
    QLabel *m_performanceLabel;
    QComboBox *m_performanceModeComboBox;
    
    // State
    bool m_showSegmentation;
    double m_confidenceThreshold;
    TFLiteDeepLabv3::PerformanceMode m_performanceMode;
    
    void setupUI();
    void updateConfidenceLabel();
};

#endif // TFLITE_SEGMENTATION_WIDGET_H 