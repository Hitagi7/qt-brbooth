#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QCamera>
#include <QCameraDevice>
#include <QMediaDevices>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include "persondetector.h"

/**
 * @brief Example Qt application demonstrating YOLOv5n person detection
 * 
 * This application shows how to integrate the PersonDetector class
 * into a Qt booth application for real-time person detection.
 */
class PersonDetectionExample : public QWidget
{
    Q_OBJECT

public:
    PersonDetectionExample(QWidget *parent = nullptr)
        : QWidget(parent)
        , m_detector(nullptr)
        , m_camera(nullptr)
        , m_timer(new QTimer(this))
        , m_imageLabel(new QLabel)
        , m_statusLabel(new QLabel("Ready"))
        , m_personCountLabel(new QLabel("People detected: 0"))
        , m_loadModelButton(new QPushButton("Load YOLO Model"))
        , m_startCameraButton(new QPushButton("Start Camera"))
        , m_stopCameraButton(new QPushButton("Stop Camera"))
        , m_loadImageButton(new QPushButton("Load Test Image"))
    {
        setupUI();
        setupConnections();
        initializeCamera();
    }

    ~PersonDetectionExample()
    {
        if (m_camera) {
            m_camera->stop();
        }
        delete m_detector;
    }

private slots:
    void loadModel()
    {
        QString modelPath = QFileDialog::getOpenFileName(
            this, "Select YOLOv5n ONNX Model", "", "ONNX Models (*.onnx)");
        
        if (!modelPath.isEmpty()) {
            delete m_detector;
            m_detector = new PersonDetector(modelPath, 0.5f, 0.4f);
            
            if (m_detector->isInitialized()) {
                m_statusLabel->setText("Model loaded: " + QFileInfo(modelPath).baseName());
                m_startCameraButton->setEnabled(true);
                m_loadImageButton->setEnabled(true);
                
                QMessageBox::information(this, "Success", 
                    "YOLOv5n model loaded successfully!\n"
                    "Input size: 640x640\n"
                    "Person detection ready.");
            } else {
                m_statusLabel->setText("Failed to load model");
                QMessageBox::critical(this, "Error", 
                    "Failed to load YOLO model. Please check:\n"
                    "1. ONNX Runtime is installed\n"
                    "2. Model file is valid YOLOv5n ONNX format\n"
                    "3. File permissions");
            }
        }
    }

    void startCamera()
    {
        if (!m_detector || !m_detector->isInitialized()) {
            QMessageBox::warning(this, "Warning", "Please load a YOLO model first.");
            return;
        }

        if (m_cap.open(0)) {
            m_timer->start(30); // ~33 FPS
            m_startCameraButton->setEnabled(false);
            m_stopCameraButton->setEnabled(true);
            m_statusLabel->setText("Camera started - Real-time person detection active");
        } else {
            QMessageBox::critical(this, "Error", "Failed to open camera.");
        }
    }

    void stopCamera()
    {
        m_timer->stop();
        m_cap.release();
        m_startCameraButton->setEnabled(true);
        m_stopCameraButton->setEnabled(false);
        m_statusLabel->setText("Camera stopped");
    }

    void processFrame()
    {
        if (!m_cap.isOpened() || !m_detector) {
            return;
        }

        cv::Mat frame;
        if (m_cap.read(frame)) {
            // Flip for mirror effect (typical for booth applications)
            cv::flip(frame, frame, 1);

            // Detect persons
            QVector<PersonDetection> detections = m_detector->detectPersons(frame);
            
            // Draw detections
            m_detector->drawDetections(frame, detections);
            
            // Convert to QImage and display
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
            
            // Scale to fit display
            QPixmap pixmap = QPixmap::fromImage(qimg).scaled(640, 480, Qt::KeepAspectRatio);
            m_imageLabel->setPixmap(pixmap);
            
            // Update person count
            m_personCountLabel->setText(QString("People detected: %1").arg(detections.size()));
            
            // Log detections
            for (const auto& detection : detections) {
                qDebug() << "Person detected at" << detection.boundingBox 
                         << "with confidence" << detection.confidence;
            }
        }
    }

    void loadTestImage()
    {
        if (!m_detector || !m_detector->isInitialized()) {
            QMessageBox::warning(this, "Warning", "Please load a YOLO model first.");
            return;
        }

        QString imagePath = QFileDialog::getOpenFileName(
            this, "Select Test Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        
        if (!imagePath.isEmpty()) {
            QImage image(imagePath);
            if (!image.isNull()) {
                // Detect persons
                QVector<PersonDetection> detections = m_detector->detectPersons(image);
                
                // Draw detections
                QImage result = m_detector->drawDetections(image, detections);
                
                // Scale and display
                QPixmap pixmap = QPixmap::fromImage(result).scaled(640, 480, Qt::KeepAspectRatio);
                m_imageLabel->setPixmap(pixmap);
                
                // Update count
                m_personCountLabel->setText(QString("People detected: %1").arg(detections.size()));
                
                // Show detection details
                QString details = QString("Image: %1\nPeople found: %2\n\nDetections:\n")
                                 .arg(QFileInfo(imagePath).baseName())
                                 .arg(detections.size());
                
                for (int i = 0; i < detections.size(); ++i) {
                    const auto& det = detections[i];
                    details += QString("Person %1: Confidence %2%, Box (%3,%4) %5x%6\n")
                              .arg(i + 1)
                              .arg(det.confidence * 100, 0, 'f', 1)
                              .arg(det.boundingBox.x())
                              .arg(det.boundingBox.y())
                              .arg(det.boundingBox.width())
                              .arg(det.boundingBox.height());
                }
                
                QMessageBox::information(this, "Detection Results", details);
            }
        }
    }

private:
    void setupUI()
    {
        setWindowTitle("YOLOv5n Person Detection Example - Qt Booth");
        setMinimumSize(800, 600);

        // Main layout
        QVBoxLayout *mainLayout = new QVBoxLayout(this);

        // Control buttons
        QHBoxLayout *buttonLayout = new QHBoxLayout;
        buttonLayout->addWidget(m_loadModelButton);
        buttonLayout->addWidget(m_startCameraButton);
        buttonLayout->addWidget(m_stopCameraButton);
        buttonLayout->addWidget(m_loadImageButton);
        buttonLayout->addStretch();

        // Image display
        m_imageLabel->setMinimumSize(640, 480);
        m_imageLabel->setStyleSheet("border: 2px solid gray;");
        m_imageLabel->setAlignment(Qt::AlignCenter);
        m_imageLabel->setText("Load a model and start camera or load test image");

        // Status labels
        QHBoxLayout *statusLayout = new QHBoxLayout;
        statusLayout->addWidget(new QLabel("Status:"));
        statusLayout->addWidget(m_statusLabel);
        statusLayout->addStretch();
        statusLayout->addWidget(m_personCountLabel);

        // Add to main layout
        mainLayout->addLayout(buttonLayout);
        mainLayout->addWidget(m_imageLabel);
        mainLayout->addLayout(statusLayout);

        // Initial button states
        m_startCameraButton->setEnabled(false);
        m_stopCameraButton->setEnabled(false);
        m_loadImageButton->setEnabled(false);
    }

    void setupConnections()
    {
        connect(m_loadModelButton, &QPushButton::clicked, this, &PersonDetectionExample::loadModel);
        connect(m_startCameraButton, &QPushButton::clicked, this, &PersonDetectionExample::startCamera);
        connect(m_stopCameraButton, &QPushButton::clicked, this, &PersonDetectionExample::stopCamera);
        connect(m_loadImageButton, &QPushButton::clicked, this, &PersonDetectionExample::loadTestImage);
        connect(m_timer, &QTimer::timeout, this, &PersonDetectionExample::processFrame);
    }

    void initializeCamera()
    {
        // Check available cameras
        auto cameras = QMediaDevices::videoInputs();
        if (cameras.isEmpty()) {
            m_statusLabel->setText("No cameras available");
            qWarning() << "No cameras found";
        } else {
            qDebug() << "Available cameras:" << cameras.size();
            for (const auto& camera : cameras) {
                qDebug() << "Camera:" << camera.description();
            }
        }
    }

private:
    PersonDetector *m_detector;
    QCamera *m_camera;
    cv::VideoCapture m_cap;
    QTimer *m_timer;
    
    // UI components
    QLabel *m_imageLabel;
    QLabel *m_statusLabel;
    QLabel *m_personCountLabel;
    QPushButton *m_loadModelButton;
    QPushButton *m_startCameraButton;
    QPushButton *m_stopCameraButton;
    QPushButton *m_loadImageButton;
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    qDebug() << "YOLOv5n Person Detection Example";
    qDebug() << "OpenCV Version:" << CV_VERSION;
    
#ifdef HAVE_ONNXRUNTIME
    qDebug() << "ONNX Runtime: Available";
#else
    qDebug() << "ONNX Runtime: NOT AVAILABLE - Person detection will not work";
    QMessageBox::critical(nullptr, "Missing Dependency", 
        "ONNX Runtime is not available.\n"
        "Please install ONNX Runtime to use person detection.\n\n"
        "Ubuntu/Debian: apt install libonnxruntime-dev\n"
        "Or download from: https://github.com/microsoft/onnxruntime/releases");
#endif

    PersonDetectionExample window;
    window.show();

    return app.exec();
}

#include "person_detection_example.moc"