#include "tflite_segmentation_widget.h"
#include <QDebug>

TFLiteSegmentationWidget::TFLiteSegmentationWidget(QWidget *parent)
    : QWidget(parent)
    , m_showSegmentation(false)
    , m_confidenceThreshold(0.5)
    , m_performanceMode(TFLiteDeepLabv3::Balanced)
{
    setupUI();
}

TFLiteSegmentationWidget::~TFLiteSegmentationWidget()
{
}

void TFLiteSegmentationWidget::setShowSegmentation(bool show)
{
    m_showSegmentation = show;
    m_showSegmentationCheckBox->setChecked(show);
}

bool TFLiteSegmentationWidget::getShowSegmentation() const
{
    return m_showSegmentation;
}

void TFLiteSegmentationWidget::setConfidenceThreshold(double threshold)
{
    m_confidenceThreshold = threshold;
    m_confidenceSlider->setValue(static_cast<int>(threshold * 100));
    updateConfidenceLabel();
}

double TFLiteSegmentationWidget::getConfidenceThreshold() const
{
    return m_confidenceThreshold;
}

void TFLiteSegmentationWidget::setPerformanceMode(TFLiteDeepLabv3::PerformanceMode mode)
{
    m_performanceMode = mode;
    m_performanceModeComboBox->setCurrentIndex(static_cast<int>(mode));
}

TFLiteDeepLabv3::PerformanceMode TFLiteSegmentationWidget::getPerformanceMode() const
{
    return m_performanceMode;
}

void TFLiteSegmentationWidget::onShowSegmentationToggled(bool checked)
{
    m_showSegmentation = checked;
    emit showSegmentationChanged(checked);
    
    // Emit appropriate signals based on state
    if (checked) {
        emit segmentationStarted();
    } else {
        emit segmentationStopped();
    }
}

void TFLiteSegmentationWidget::onConfidenceThresholdChanged(int value)
{
    m_confidenceThreshold = value / 100.0; // Convert from percentage to decimal
    updateConfidenceLabel();
    emit confidenceThresholdChanged(m_confidenceThreshold);
}

void TFLiteSegmentationWidget::onPerformanceModeChanged(int index)
{
    if (index >= 0 && index < 4) {
        m_performanceMode = static_cast<TFLiteDeepLabv3::PerformanceMode>(index);
        emit performanceModeChanged(m_performanceMode);
    }
}

void TFLiteSegmentationWidget::setupUI()
{
    // Main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(10);

    // Main group box
    m_mainGroupBox = new QGroupBox("TFLite Deeplabv3 Segmentation", this);
    QVBoxLayout *groupLayout = new QVBoxLayout(m_mainGroupBox);

    // Show segmentation checkbox
    m_showSegmentationCheckBox = new QCheckBox("Enable Segmentation", this);
    m_showSegmentationCheckBox->setChecked(m_showSegmentation);
    connect(m_showSegmentationCheckBox, &QCheckBox::toggled, 
            this, &TFLiteSegmentationWidget::onShowSegmentationToggled);
    groupLayout->addWidget(m_showSegmentationCheckBox);

    // Confidence threshold slider
    QHBoxLayout *confidenceLayout = new QHBoxLayout();
    m_confidenceLabel = new QLabel("Confidence Threshold:", this);
    m_confidenceSlider = new QSlider(Qt::Horizontal, this);
    m_confidenceSlider->setRange(10, 100); // 0.1 to 1.0
    m_confidenceSlider->setValue(static_cast<int>(m_confidenceThreshold * 100));
    m_confidenceValueLabel = new QLabel("50%", this);
    m_confidenceValueLabel->setMinimumWidth(40);
    
    connect(m_confidenceSlider, &QSlider::valueChanged,
            this, &TFLiteSegmentationWidget::onConfidenceThresholdChanged);
    
    confidenceLayout->addWidget(m_confidenceLabel);
    confidenceLayout->addWidget(m_confidenceSlider);
    confidenceLayout->addWidget(m_confidenceValueLabel);
    groupLayout->addLayout(confidenceLayout);

    // Performance mode combo box
    QHBoxLayout *performanceLayout = new QHBoxLayout();
    m_performanceLabel = new QLabel("Performance Mode:", this);
    m_performanceModeComboBox = new QComboBox(this);
    m_performanceModeComboBox->addItem("High Quality");
    m_performanceModeComboBox->addItem("Balanced");
    m_performanceModeComboBox->addItem("High Speed");
    m_performanceModeComboBox->addItem("Adaptive");
    m_performanceModeComboBox->setCurrentIndex(static_cast<int>(m_performanceMode));
    
    connect(m_performanceModeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TFLiteSegmentationWidget::onPerformanceModeChanged);
    
    performanceLayout->addWidget(m_performanceLabel);
    performanceLayout->addWidget(m_performanceModeComboBox);
    groupLayout->addLayout(performanceLayout);

    // Add group box to main layout
    mainLayout->addWidget(m_mainGroupBox);
    
    // Add stretch to push everything to the top
    mainLayout->addStretch();
    
    setLayout(mainLayout);
}

void TFLiteSegmentationWidget::updateConfidenceLabel()
{
    m_confidenceValueLabel->setText(QString("%1%").arg(static_cast<int>(m_confidenceThreshold * 100)));
} 