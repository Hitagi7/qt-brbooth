#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QProgressBar>
#include <QTextEdit>
#include <QSplitter>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>
#include <QFormLayout>
#include <QDebug>
#include <QTime>
#include <QFileInfo>

#include "../src/yolo/yolov5detector.h"

class YOLOExampleWidget : public QWidget
{
    Q_OBJECT

public:
    explicit YOLOExampleWidget(QWidget *parent = nullptr)
        : QWidget(parent)
        , m_detector(new YOLOv5Detector(this))
        , m_imageLabel(new QLabel)
        , m_loadImageButton(new QPushButton("Load Image"))
        , m_loadModelButton(new QPushButton("Load YOLO Model"))
        , m_detectButton(new QPushButton("Detect Objects"))
        , m_progressBar(new QProgressBar)
        , m_logTextEdit(new QTextEdit)
        , m_confidenceSlider(new QSlider(Qt::Horizontal))
        , m_nmsSlider(new QSlider(Qt::Horizontal))
        , m_confidenceSpinBox(new QSpinBox)
        , m_nmsSpinBox(new QSpinBox)
    {
        setupUI();
        connectSignals();
        
        // Initialize detector settings
        m_detector->setConfidenceThreshold(0.5f);
        m_detector->setNmsThreshold(0.4f);
        
        logMessage("YOLOv5 Example Application started");
        logMessage("Please load a YOLO model first, then load an image to detect objects");
    }

private slots:
    void loadImage()
    {
        QString fileName = QFileDialog::getOpenFileName(this,
            "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)");
        
        if (!fileName.isEmpty()) {
            m_currentImage.load(fileName);
            if (!m_currentImage.isNull()) {
                displayImage(m_currentImage);
                logMessage(QString("Image loaded: %1 (%2x%3)")
                          .arg(QFileInfo(fileName).fileName())
                          .arg(m_currentImage.width())
                          .arg(m_currentImage.height()));
                
                m_detectButton->setEnabled(m_detector->isInitialized());
            } else {
                QMessageBox::warning(this, "Error", "Failed to load image");
                logMessage("Error: Failed to load image");
            }
        }
    }
    
    void loadModel()
    {
        QString fileName = QFileDialog::getOpenFileName(this,
            "Load YOLO Model", "", "ONNX Model Files (*.onnx)");
        
        if (!fileName.isEmpty()) {
            logMessage("Loading YOLO model...");
            m_progressBar->setRange(0, 0); // Indeterminate progress
            
            bool success = m_detector->initialize(fileName);
            m_progressBar->setRange(0, 1);
            m_progressBar->setValue(1);
            
            if (success) {
                logMessage(QString("Model loaded successfully: %1")
                          .arg(QFileInfo(fileName).fileName()));
                logMessage(QString("Model input size: %1x%2")
                          .arg(m_detector->getModelInputSize().width())
                          .arg(m_detector->getModelInputSize().height()));
                
                m_detectButton->setEnabled(!m_currentImage.isNull());
            } else {
                logMessage("Error: Failed to load YOLO model");
            }
        }
    }
    
    void detectObjects()
    {
        if (m_currentImage.isNull() || !m_detector->isInitialized()) {
            return;
        }
        
        logMessage("Starting object detection...");
        m_progressBar->setRange(0, 0); // Indeterminate progress
        m_detectButton->setEnabled(false);
        
        // Run detection in a separate thread (via Qt's event system)
        QTimer::singleShot(0, [this]() {
            QVector<Detection> detections = m_detector->detectObjects(m_currentImage);
            
            m_progressBar->setRange(0, 1);
            m_progressBar->setValue(1);
            m_detectButton->setEnabled(true);
            
            if (!detections.isEmpty()) {
                // Draw bounding boxes on image
                QImage resultImage = m_detector->drawBoundingBoxes(m_currentImage, detections);
                displayImage(resultImage);
                
                // Log detection results
                logMessage(QString("Detection completed: %1 objects found").arg(detections.size()));
                for (const Detection& detection : detections) {
                    logMessage(QString("  - %1: %.3f (%.1f, %.1f, %.1f, %.1f)")
                              .arg(detection.className)
                              .arg(detection.confidence)
                              .arg(detection.boundingBox.x())
                              .arg(detection.boundingBox.y())
                              .arg(detection.boundingBox.width())
                              .arg(detection.boundingBox.height()));
                }
            } else {
                logMessage("No objects detected");
            }
        });
    }
    
    void onConfidenceChanged(int value)
    {
        float threshold = value / 100.0f;
        m_detector->setConfidenceThreshold(threshold);
        m_confidenceSpinBox->setValue(value);
        logMessage(QString("Confidence threshold: %.2f").arg(threshold));
    }
    
    void onNmsChanged(int value)
    {
        float threshold = value / 100.0f;
        m_detector->setNmsThreshold(threshold);
        m_nmsSpinBox->setValue(value);
        logMessage(QString("NMS threshold: %.2f").arg(threshold));
    }
    
    void onDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs)
    {
        Q_UNUSED(detections)
        logMessage(QString("Processing time: %1ms").arg(processingTimeMs));
    }
    
    void onDetectorError(const QString& error)
    {
        logMessage(QString("Detector error: %1").arg(error));
        QMessageBox::warning(this, "Detector Error", error);
    }

private:
    void setupUI()
    {
        // Main layout
        auto* mainLayout = new QHBoxLayout(this);
        auto* splitter = new QSplitter(Qt::Horizontal);
        
        // Left panel - image display
        auto* imageWidget = new QWidget;
        auto* imageLayout = new QVBoxLayout(imageWidget);
        
        m_imageLabel->setMinimumSize(640, 480);
        m_imageLabel->setStyleSheet("border: 1px solid black;");
        m_imageLabel->setAlignment(Qt::AlignCenter);
        m_imageLabel->setText("No image loaded");
        m_imageLabel->setScaledContents(true);
        
        imageLayout->addWidget(m_imageLabel);
        
        // Right panel - controls and log
        auto* controlWidget = new QWidget;
        auto* controlLayout = new QVBoxLayout(controlWidget);
        
        // File operations group
        auto* fileGroup = new QGroupBox("File Operations");
        auto* fileLayout = new QVBoxLayout(fileGroup);
        fileLayout->addWidget(m_loadModelButton);
        fileLayout->addWidget(m_loadImageButton);
        fileLayout->addWidget(m_detectButton);
        m_detectButton->setEnabled(false);
        
        // Detection parameters group
        auto* paramGroup = new QGroupBox("Detection Parameters");
        auto* paramLayout = new QFormLayout(paramGroup);
        
        // Confidence threshold
        m_confidenceSlider->setRange(10, 90);
        m_confidenceSlider->setValue(50);
        m_confidenceSpinBox->setRange(10, 90);
        m_confidenceSpinBox->setValue(50);
        m_confidenceSpinBox->setSuffix("%");
        
        auto* confLayout = new QHBoxLayout;
        confLayout->addWidget(m_confidenceSlider);
        confLayout->addWidget(m_confidenceSpinBox);
        paramLayout->addRow("Confidence:", confLayout);
        
        // NMS threshold
        m_nmsSlider->setRange(10, 80);
        m_nmsSlider->setValue(40);
        m_nmsSpinBox->setRange(10, 80);
        m_nmsSpinBox->setValue(40);
        m_nmsSpinBox->setSuffix("%");
        
        auto* nmsLayout = new QHBoxLayout;
        nmsLayout->addWidget(m_nmsSlider);
        nmsLayout->addWidget(m_nmsSpinBox);
        paramLayout->addRow("NMS:", nmsLayout);
        
        // Progress bar
        m_progressBar->setVisible(false);
        
        // Log area
        auto* logGroup = new QGroupBox("Log");
        auto* logLayout = new QVBoxLayout(logGroup);
        m_logTextEdit->setMaximumHeight(200);
        m_logTextEdit->setReadOnly(true);
        logLayout->addWidget(m_logTextEdit);
        
        // Assemble control panel
        controlLayout->addWidget(fileGroup);
        controlLayout->addWidget(paramGroup);
        controlLayout->addWidget(m_progressBar);
        controlLayout->addWidget(logGroup);
        controlLayout->addStretch();
        
        // Set minimum widths
        imageWidget->setMinimumWidth(650);
        controlWidget->setMinimumWidth(300);
        controlWidget->setMaximumWidth(400);
        
        // Add to splitter
        splitter->addWidget(imageWidget);
        splitter->addWidget(controlWidget);
        splitter->setStretchFactor(0, 1);
        splitter->setStretchFactor(1, 0);
        
        mainLayout->addWidget(splitter);
        
        setWindowTitle("YOLOv5 Object Detection Example");
        resize(1000, 700);
    }
    
    void connectSignals()
    {
        connect(m_loadImageButton, &QPushButton::clicked, this, &YOLOExampleWidget::loadImage);
        connect(m_loadModelButton, &QPushButton::clicked, this, &YOLOExampleWidget::loadModel);
        connect(m_detectButton, &QPushButton::clicked, this, &YOLOExampleWidget::detectObjects);
        
        connect(m_confidenceSlider, &QSlider::valueChanged, this, &YOLOExampleWidget::onConfidenceChanged);
        connect(m_nmsSlider, &QSlider::valueChanged, this, &YOLOExampleWidget::onNmsChanged);
        
        connect(m_confidenceSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
                m_confidenceSlider, &QSlider::setValue);
        connect(m_nmsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
                m_nmsSlider, &QSlider::setValue);
        
        connect(m_detector, &YOLOv5Detector::detectionCompleted,
                this, &YOLOExampleWidget::onDetectionCompleted);
        connect(m_detector, &YOLOv5Detector::errorOccurred,
                this, &YOLOExampleWidget::onDetectorError);
    }
    
    void displayImage(const QImage& image)
    {
        if (!image.isNull()) {
            QPixmap pixmap = QPixmap::fromImage(image);
            QPixmap scaledPixmap = pixmap.scaled(m_imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            m_imageLabel->setPixmap(scaledPixmap);
        }
    }
    
    void logMessage(const QString& message)
    {
        QString timestamp = QTime::currentTime().toString("hh:mm:ss");
        m_logTextEdit->append(QString("[%1] %2").arg(timestamp, message));
    }

private:
    YOLOv5Detector* m_detector;
    QImage m_currentImage;
    
    // UI components
    QLabel* m_imageLabel;
    QPushButton* m_loadImageButton;
    QPushButton* m_loadModelButton;
    QPushButton* m_detectButton;
    QProgressBar* m_progressBar;
    QTextEdit* m_logTextEdit;
    QSlider* m_confidenceSlider;
    QSlider* m_nmsSlider;
    QSpinBox* m_confidenceSpinBox;
    QSpinBox* m_nmsSpinBox;
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    YOLOExampleWidget window;
    window.show();
    
    return app.exec();
}

#include "yolo_example.moc"