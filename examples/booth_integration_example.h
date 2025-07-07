/**
 * @file booth_integration_example.h
 * @brief Example showing minimal integration of PersonDetector into existing booth capture system
 * 
 * This file demonstrates how to integrate YOLOv5n person detection into the existing
 * Qt booth application with minimal code changes.
 */

#ifndef BOOTH_INTEGRATION_EXAMPLE_H
#define BOOTH_INTEGRATION_EXAMPLE_H

// Add this include to capture.h
#include "persondetector.h"

/**
 * STEP 1: Add to capture.h - Add these members to the Capture class
 */
class CaptureWithPersonDetection : public QWidget
{
    Q_OBJECT

public:
    // ... existing members ...

private:
    // Add these new members for person detection
    PersonDetector *m_personDetector;
    int m_currentPersonCount;
    bool m_personDetectionEnabled;
    QTimer *m_personDetectionTimer;
    
    // Person detection methods
    void initializePersonDetection();
    void updatePersonDetection();
    void enablePersonDetection(bool enabled);

public slots:
    // ... existing slots ...
    void setPersonDetectionEnabled(bool enabled);
    void setPersonDetectionModel(const QString& modelPath);

signals:
    // ... existing signals ...
    
    // New person detection signals
    void personCountChanged(int count);
    void personEntered();
    void personLeft();
    void personDetectionStatusChanged(bool enabled);
};

/**
 * STEP 2: Add to capture.cpp - Implementation
 */

// Add to constructor
void initializeCaptureWithPersonDetection()
{
    // ... existing initialization code ...
    
    // Initialize person detection
    m_personDetector = nullptr;
    m_currentPersonCount = 0;
    m_personDetectionEnabled = false;
    
    // Create timer for person detection (separate from camera timer for performance)
    m_personDetectionTimer = new QTimer(this);
    m_personDetectionTimer->setInterval(200); // 5 FPS for person detection
    connect(m_personDetectionTimer, &QTimer::timeout, this, &Capture::updatePersonDetection);
    
    // Initialize with default model if available
    initializePersonDetection();
}

void initializePersonDetection()
{
    // Try to load default model
    QString defaultModelPath = "models/yolov5n.onnx";
    if (QFile::exists(defaultModelPath)) {
        setPersonDetectionModel(defaultModelPath);
    } else {
        qDebug() << "Person detection: Default model not found at" << defaultModelPath;
        qDebug() << "Person detection disabled. Load model manually to enable.";
    }
}

void setPersonDetectionModel(const QString& modelPath)
{
    // Clean up existing detector
    delete m_personDetector;
    m_personDetector = nullptr;
    
    if (!modelPath.isEmpty() && QFile::exists(modelPath)) {
        m_personDetector = new PersonDetector(modelPath, 0.5f, 0.4f);
        
        if (m_personDetector->isInitialized()) {
            qDebug() << "Person detection model loaded:" << modelPath;
            setPersonDetectionEnabled(true);
            emit personDetectionStatusChanged(true);
        } else {
            qDebug() << "Failed to initialize person detector with model:" << modelPath;
            delete m_personDetector;
            m_personDetector = nullptr;
            emit personDetectionStatusChanged(false);
        }
    }
}

void setPersonDetectionEnabled(bool enabled)
{
    if (!m_personDetector) {
        enabled = false;
    }
    
    m_personDetectionEnabled = enabled;
    
    if (enabled && cap.isOpened()) {
        m_personDetectionTimer->start();
    } else {
        m_personDetectionTimer->stop();
    }
    
    qDebug() << "Person detection" << (enabled ? "enabled" : "disabled");
}

void updatePersonDetection()
{
    if (!m_personDetectionEnabled || !m_personDetector || !cap.isOpened()) {
        return;
    }
    
    cv::Mat frame;
    if (cap.read(frame)) {
        cv::flip(frame, frame, 1); // Mirror effect
        
        // Count people in current frame
        int newPersonCount = m_personDetector->countPersons(frame);
        
        // Check for changes in person count
        if (newPersonCount != m_currentPersonCount) {
            if (newPersonCount > m_currentPersonCount) {
                qDebug() << "Person entered. Count:" << newPersonCount;
                emit personEntered();
            } else if (newPersonCount < m_currentPersonCount) {
                qDebug() << "Person left. Count:" << newPersonCount;
                emit personLeft();
            }
            
            m_currentPersonCount = newPersonCount;
            emit personCountChanged(newPersonCount);
        }
    }
}

/**
 * STEP 3: Update existing camera feed method to optionally show person detection
 */
void updateCameraFeedWithPersonDetection()
{
    // ... existing camera feed code ...
    
    if (cap.read(frame)) {
        cv::flip(frame, frame, 1);
        
        // Optional: Draw person detection overlay
        if (m_personDetectionEnabled && m_personDetector) {
            QVector<PersonDetection> detections = m_personDetector->detectPersons(frame);
            m_personDetector->drawDetections(frame, detections);
        }
        
        // Convert and display frame
        QImage image = cvMatToQImage(frame);
        if (!image.isNull()) {
            QPixmap pixmap = QPixmap::fromImage(image);
            videoLabel->setPixmap(pixmap.scaled(videoLabel->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
        }
    }
    
    // ... rest of existing code ...
}

/**
 * STEP 4: Add to brbooth.cpp - Connect person detection signals
 */
void connectPersonDetectionSignals()
{
    // Connect person detection signals to booth logic
    if (capturePage) {
        connect(capturePage, &Capture::personEntered, this, [this]() {
            qDebug() << "Booth: Person detected, ready for interaction";
            // Optional: Auto-start capture sequence, show welcome message, etc.
        });
        
        connect(capturePage, &Capture::personLeft, this, [this]() {
            qDebug() << "Booth: Person left";
            // Optional: Reset booth state, save analytics, etc.
        });
        
        connect(capturePage, &Capture::personCountChanged, this, [this](int count) {
            qDebug() << "Booth: Person count changed to" << count;
            // Optional: Update UI, analytics, or booth behavior based on count
        });
    }
}

/**
 * STEP 5: Optional UI Integration - Add person detection controls
 */

// Add to capture.ui or create person detection control widget:

/*
 * Person Detection Controls:
 * - Enable/Disable checkbox
 * - Model path selection button
 * - Person count display
 * - Confidence threshold slider
 * - Detection visualization toggle
 */

class PersonDetectionWidget : public QWidget
{
    Q_OBJECT
    
public:
    PersonDetectionWidget(QWidget *parent = nullptr);
    
    void setCapture(Capture *capture);

private slots:
    void onEnableToggled(bool enabled);
    void onSelectModel();
    void onConfidenceChanged(int value);
    void onPersonCountChanged(int count);

private:
    Capture *m_capture;
    QCheckBox *m_enableCheckBox;
    QPushButton *m_selectModelButton;
    QLabel *m_personCountLabel;
    QSlider *m_confidenceSlider;
    QCheckBox *m_showDetectionsCheckBox;
};

/**
 * STEP 6: Analytics Integration - Track booth usage
 */
class BoothAnalytics : public QObject
{
    Q_OBJECT
    
public:
    struct BoothSession {
        QDateTime startTime;
        QDateTime endTime;
        int maxPersonCount;
        int totalDetections;
        bool captureCompleted;
    };
    
public slots:
    void onPersonEntered();
    void onPersonLeft();
    void onPersonCountChanged(int count);
    void onCaptureCompleted();
    
    void saveAnalytics();
    QList<BoothSession> getAnalytics() const;

private:
    QList<BoothSession> m_sessions;
    BoothSession m_currentSession;
    bool m_sessionActive;
};

#endif // BOOTH_INTEGRATION_EXAMPLE_H

/**
 * USAGE SUMMARY:
 * 
 * 1. Add PersonDetector member to Capture class
 * 2. Initialize detector in Capture constructor
 * 3. Connect person detection signals in brbooth.cpp
 * 4. Optionally add UI controls for person detection
 * 5. Use person events to enhance booth behavior
 * 
 * Key benefits:
 * - Automatic person detection without user interaction
 * - Count people for booth capacity management
 * - Trigger booth actions based on person presence
 * - Collect analytics on booth usage patterns
 * - Enhance user experience with person-aware features
 */