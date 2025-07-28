#include "capture.h"
#include "ui_capture.h"
#include "foreground.h"
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QElapsedTimer>
#include <QVBoxLayout> // Still needed for countdownLabel positioning, but not for main layout
#include <QGridLayout> // New: For main layout
#include <QPainter> // For drawing bounding boxes
#include <QKeyEvent> // For keyboard event handling
#include <opencv2/opencv.hpp>

Capture::Capture(QWidget *parent, Foreground *fg)
    : QWidget(parent)
    , ui(new Ui::Capture)
    , cameraTimer(nullptr)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5)
    , m_recordedSeconds(0)
    , stackedLayout(nullptr)
    , videoLabelFPS(nullptr) // Initialize pointer to nullptr
    , totalTime(0)           // Initialize totalTime
    , frameCount(0)          // Initialize frameCount
    , isProcessingFrame(false) // Initialize frame processing flag
    , foreground(fg)
    , yoloProcess(new QProcess(this)) // Initialize QProcess here!
    , m_showBoundingBoxes(true) // Initialize bounding box display to true
{
    ui->setupUi(this);

    // Ensure no margins or spacing for the Capture widget itself
    setContentsMargins(0, 0, 0, 0);
    
    // Enable keyboard focus for this widget
    setFocusPolicy(Qt::StrongFocus);
    setFocus(); // Set initial focus

    // Setup the main grid layout for the Capture widget
    QGridLayout* mainLayout = new QGridLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    //QT Foreground Overlay ========================================================s
    overlayImageLabel = new QLabel(ui->overlayWidget);
    QString selectedOverlay;
    if (foreground) {
        selectedOverlay = foreground->getSelectedForeground();
    } else {
        qWarning() << "Error: foreground is nullptr!";
    }
    qDebug() << "Selected overlay path:" << selectedOverlay;
    overlayImageLabel->setAttribute(Qt::WA_TranslucentBackground);
    overlayImageLabel->setStyleSheet("background: transparent;");
    overlayImageLabel->setScaledContents(true); // Optional: scale the pixmap
    overlayImageLabel->resize(this->size());
    overlayImageLabel->hide();
    connect(foreground, &Foreground::foregroundChanged, this, &Capture::updateForegroundOverlay);
    qDebug() << "Selected overlay path:" << selectedOverlay;
    QPixmap overlayPixmap(selectedOverlay);
    overlayImageLabel->setPixmap(overlayPixmap);

    // IMMEDIATELY setup stacked layout after UI setup
    setupStackedLayoutHybrid(); // This will now add stackedLayout to mainLayout

    // Force initial size match
    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());

    // Force show and update
    ui->videoLabel->show();
    ui->overlayWidget->show();
    ui->videoLabel->update();
    ui->overlayWidget->update();

    // CRITICAL FIX: Explicitly set the size of videoLabel and overlayWidget
    // to match the Capture widget's initial size immediately after layout setup.
    // This addresses the initial QSize(100, 30) issue.
    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());
    // Force a repaint and layout update
    updateGeometry();

    // Update overlay styles for better visibility
    updateOverlayStyles();

    // Ensure videoLabel fills and paints properly
    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;");
    ui->videoLabel->setScaledContents(false);  // ❗ Disable it – we're scaling manually
    ui->videoLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    // Also ensure overlayWidget expands fully
    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->overlayWidget->setStyleSheet("background-color: transparent;"); // Ensure overlay is transparent

    // Setup slider
    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setValue(0);

    // Setup icons and button states
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    ui->capture->setEnabled(false);

    // Camera open logic
    bool cameraOpenedSuccessfully = false;
    if (!cap.open(1)) {
        qWarning() << "Error: Could not open camera with index 1. Trying index 0...";
        if (cap.open(0)) cameraOpenedSuccessfully = true;
        else qWarning() << "Error: Could not open camera with index 0 either.";
    } else {
        cameraOpenedSuccessfully = true;
    }

    if (!cameraOpenedSuccessfully) {
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText("Camera not available.\nCheck connection and drivers.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->capture->setEnabled(false);
        return;
    }

    // Set camera properties
    qDebug() << "Attempting to set camera resolution to 1280x720.";
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Request 60 FPS from the camera hardware
    qDebug() << "Attempting to set camera FPS to 60.";
    cap.set(cv::CAP_PROP_FPS, 60.0);

    // --- DIAGNOSTIC 1: CHECK ACTUAL CAMERA SETTINGS ---
    // Check what the camera driver actually settled on.
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    qDebug() << "========================================";
    qDebug() << "Camera settings REQUESTED: 1280x720 @ 60 FPS";
    qDebug() << "Camera settings ACTUAL: " << actual_width << "x" << actual_height << " @ " << actual_fps << " FPS";
    qDebug() << "========================================";
    if (actual_fps < 59) {
        qWarning() << "WARNING: Camera did not accept 60 FPS request. Actual FPS is" << actual_fps;
    }

    // ✅ FIX 1: Use actual camera FPS for timer instead of fixed 60 FPS
    int timerInterval = qMax(1, static_cast<int>(1000.0 / actual_fps));
    qDebug() << "Setting camera timer interval to" << timerInterval << "ms for" << actual_fps << "FPS";

    // Start camera timer for updates with dynamic interval
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    cameraTimer->start(timerInterval); // ✅ Use actual camera FPS

    // Initialize and start performance timers
    loopTimer.start();
    frameTimer.start();

    ui->capture->setEnabled(true);

    // Countdown label overlays on the overlayWidget
    countdownLabel = new QLabel(ui->overlayWidget);
    countdownLabel->setAlignment(Qt::AlignCenter);
    QFont font = countdownLabel->font();
    font.setPointSize(100);
    font.setBold(true);
    countdownLabel->setFont(font);
    countdownLabel->setStyleSheet("color:white; background-color: rgba(0, 0, 0, 150); border-radius: 20px;");
    countdownLabel->setFixedSize(200,200);
    countdownLabel->hide();

    // Setup timers
    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);
    qDebug() << "OpenCV Camera display timer started for" << actual_fps << "FPS";

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    // Connect signals/slots for buttons and slider
    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);
    


    // --- CONNECT QPROCESS SIGNALS HERE ---
    connect(yoloProcess, &QProcess::readyReadStandardOutput, this, &Capture::handleYoloOutput);
    connect(yoloProcess, &QProcess::readyReadStandardError, this, &Capture::handleYoloError);
    connect(yoloProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Capture::handleYoloFinished);
    connect(yoloProcess, &QProcess::errorOccurred, this, &Capture::handleYoloErrorOccurred);
    // --- END CONNECT QPROCESS SIGNALS ---

    qDebug() << "OpenCV Camera started successfully with hybrid stacked layout and optimized FPS!";
}

void Capture::setupStackedLayoutHybrid()
{
    qDebug() << "Setting up hybrid stacked layout...";

    // Ensure videoLabel and overlayWidget are parented to 'this'
    ui->videoLabel->setParent(this);
    ui->overlayWidget->setParent(this);

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

    // 🟥 Debug border (make sure this shows!)
    ui->videoLabel->setStyleSheet("background-color: black; border: 5px solid red;");
    ui->overlayWidget->setStyleSheet("background-color: transparent; border: 5px dashed green;");

    // Fix stacking
    if (!stackedLayout) {
        stackedLayout = new QStackedLayout;
        stackedLayout->setStackingMode(QStackedLayout::StackAll);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        stackedLayout->setSpacing(0);

        stackedLayout->addWidget(ui->videoLabel);      // Background
        stackedLayout->addWidget(ui->overlayWidget);  // Foreground
        stackedLayout->addWidget(overlayImageLabel);

        // ⬇️ Replace existing layout if needed
        if (layout()) {
            delete layout(); // remove any previous layout
        }

        QGridLayout* mainLayout = new QGridLayout(this);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);
        mainLayout->addLayout(stackedLayout, 0, 0);
        mainLayout->setRowStretch(0, 1);
        mainLayout->setColumnStretch(0, 1);

        setLayout(mainLayout);
    }

    overlayImageLabel->raise();
    ui->overlayWidget->raise();
    ui->back->raise();
    ui->capture->raise();
    ui->verticalSlider->raise();
}

void Capture::updateOverlayStyles()
{
    qDebug() << "Updating overlay styles with clean professional appearance...";

    // Update back button - clean transparent style
    ui->back->setStyleSheet(
        "QPushButton {"
        "    background: transparent;"
        "    border: none;"
        "    color: white;"
        "}"
        );

    // Update capture button - clean green style matching your original design
    ui->capture->setStyleSheet(
        "QPushButton {"
        "    border-radius: 9px;"
        "    border-bottom: 3px solid rgba(2, 2, 2, 200);" // Subtle shadow
        "    background: rgba(11, 194, 0, 200);" // Your original green color
        "    color: white;"
        "    font-size: 16px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background: rgba(8, 154, 0, 230);" // Your original hover color
        "}"
        "QPushButton:disabled {"
        "    background: rgba(100, 100, 100, 150);"
        "    color: rgba(200, 200, 200, 150);"
        "    border-bottom: 3px solid rgba(50, 50, 50, 150);"
        "}"
        );

    // Update slider - clean style matching your original design
    ui->verticalSlider->setStyleSheet(
        "QSlider::groove:vertical {"
        "    background: rgba(0, 0, 0, 80);" // Semi-transparent dark groove
        "    width: 30px;"
        "    border-radius: 15px;"
        "    border: none;"
        "}"
        "QSlider::handle:vertical {"
        "    background: rgba(13, 77, 38, 220);" // Your original dark green handle
        "    border: 1px solid rgba(30, 144, 255, 180);" // Subtle blue border
        "    width: 60px;"
        "    height: 13px;"
        "    border-radius: 7px;"
        "    margin: 0 -15px;"
        "}"
        "QSlider::sub-page:vertical {"
        "    background: rgba(0, 0, 0, 60);" // Semi-transparent filled part
        "    border-top-left-radius: 15px;"
        "    border-top-right-radius: 15px;"
        "    border-bottom-left-radius: 0px;"
        "    border-bottom-right-radius: 0px;"
        "}"
        "QSlider::add-page:vertical {"
        "    background: rgba(11, 194, 0, 180);" // Your original green color
        "    border-bottom-left-radius: 15px;"
        "    border-bottom-right-radius: 15px;"
        "    border-top-left-radius: 0px;"
        "    border-top-right-radius: 0px;"
        "}"
        );

    // Ensure overlay widget is transparent
    ui->overlayWidget->setStyleSheet("background: transparent;");

    qDebug() << "Clean professional overlay styles applied";
}

Capture::~Capture()
{
    // === FIX FOR "Destroyed while process running" ASSERTION ===
    // Disconnect signals from yoloProcess FIRST to prevent calls to a dying Capture object
    disconnect(yoloProcess, &QProcess::readyReadStandardOutput, this, &Capture::handleYoloOutput);
    disconnect(yoloProcess, &QProcess::readyReadStandardError, this, &Capture::handleYoloError);
    disconnect(yoloProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Capture::handleYoloFinished);
    disconnect(yoloProcess, &QProcess::errorOccurred, this, &Capture::handleYoloErrorOccurred);
    // ==========================================================

    if (cameraTimer){ cameraTimer->stop(); delete cameraTimer; }
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; }
    if (cap.isOpened()) { cap.release(); }

    // --- Terminate YOLO process on exit ---
    if (yoloProcess->state() == QProcess::Running) {
        yoloProcess->terminate();
        yoloProcess->waitForFinished(1000); // Give it a moment to finish
        if (yoloProcess->state() == QProcess::Running) {
            yoloProcess->kill();
        }
    }
    // yoloProcess will be deleted by QObject parent mechanism because 'this' is its parent.
    // --- END Terminate YOLO process ---

    delete ui;
}

void Capture::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    ui->videoLabel->resize(size());
    ui->overlayWidget->resize(size());

    // Resize the overlay image label too
    for (QObject* child : ui->overlayWidget->children()) {
        QLabel* label = qobject_cast<QLabel*>(child);
        if (label && label != countdownLabel) {
            label->resize(size());
            label->move(0, 0);
        }
    }

    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }

    update();
    if (ui->videoLabel) ui->videoLabel->resize(this->size());
    if (ui->overlayWidget) ui->overlayWidget->resize(this->size());
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
}

// --- NEW: Bounding Box Control Methods Implementation ---

void Capture::setShowBoundingBoxes(bool show)
{
    m_showBoundingBoxes = show;
    qDebug() << "Bounding boxes display" << (show ? "enabled" : "disabled");
}

bool Capture::getShowBoundingBoxes() const
{
    return m_showBoundingBoxes;
}

int Capture::getDetectionCount() const
{
    QMutexLocker locker(&m_detectionMutex);
    int count = m_currentDetections.size();
    locker.unlock();
    return count;
}

double Capture::getAverageConfidence() const
{
    QMutexLocker locker(&m_detectionMutex);
    QList<BoundingBox> detections = m_currentDetections; // Copy for thread safety
    locker.unlock();
    
    if (detections.isEmpty()) {
        return 0.0;
    }
    
    double totalConfidence = 0.0;
    for (const BoundingBox& box : detections) {
        totalConfidence += box.confidence;
    }
    return totalConfidence / detections.size();
}

void Capture::detectPersonInImage(const QString& imagePath) {
    // This check is now mostly handled by the caller (updateCameraFeed),
    // but remains as a safeguard.
    if (yoloProcess->state() == QProcess::Running) {
        qDebug() << "YOLO process already running (from detectPersonInImage), skipping start.";
        return;
    }

    currentTempImagePath = imagePath; // Store the path for deletion later

    QString program = "python";
    QStringList arguments;
    arguments << "yolov5/detect.py"
              << "--weights" << "yolov5/yolov5n.pt"
              << "--source" << imagePath
              << "--classes" << "0" // Class ID for 'person'
              << "--nosave"; // CRUCIAL: ensures output is to stdout, not saved to file

    yoloProcess->setWorkingDirectory(QCoreApplication::applicationDirPath() + "/../../../"); // Corrected path

    qDebug() << "Starting YOLOv5 process for frame detection...";
    qDebug() << "Program:" << program;
    qDebug() << "Arguments:" << arguments.join(" ");
    qDebug() << "Working Directory:" << yoloProcess->workingDirectory();
    qDebug() << "Source Image Path:" << imagePath;

    yoloProcess->start(program, arguments);
}

void Capture::updateCameraFeed()
{
    // Start loopTimer at the very beginning of the function to measure total time for one update cycle.
    loopTimer.start(); // Measure time for this entire call

    if (!ui->videoLabel || !cap.isOpened()) {
        isProcessingFrame = false; // Ensure flag is reset if camera fails
        return;
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        isProcessingFrame = false; // Ensure flag is reset if frame read fails
        return;
    }

    cv::flip(frame, frame, 1); // Mirror

    // Display the frame immediately
    QImage image = cvMatToQImage(frame);
    if (!image.isNull()) {
        QPixmap pixmap = QPixmap::fromImage(image);
        
        // Draw bounding boxes if enabled and detections exist
        if (m_showBoundingBoxes) {
            QMutexLocker locker(&m_detectionMutex);
            QList<BoundingBox> detections = m_currentDetections; // Copy for thread safety
            locker.unlock();
            
            if (!detections.isEmpty()) {
                drawBoundingBoxes(pixmap, detections);
            }
        }
        
        QSize labelSize = ui->videoLabel->size();
        QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::IgnoreAspectRatio, Qt::FastTransformation);
        ui->videoLabel->setPixmap(scaledPixmap);
        ui->videoLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        ui->videoLabel->update();
    } else {
        // If image conversion failed, no display, so skip YOLO for this frame.
        isProcessingFrame = false; // Ensure it's false as we can't proceed with YOLO
        qWarning() << "Failed to convert OpenCV frame to QImage.";
        // Fall through to performance stats to account for this frame
    }

    // Now, handle YOLO processing only if not already busy
    if (!isProcessingFrame) { // Proceed with YOLO if free
        // --- YOLOv5n Person Detection Trigger ---
        QString tempImagePath = QDir::temp().filePath(
            "yolo_temp_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz") + ".jpg"
            );
        if (!cv::imwrite(tempImagePath.toStdString(), frame)) { // Save the original (flipped) frame
            qWarning() << "Failed to save temporary image:" << tempImagePath;
            // No YOLO for this frame, but still count the frame for display FPS
            // isProcessingFrame remains false.
            // Fall through to performance stats.
        } else {
            isProcessingFrame = true; // Set flag when starting YOLO
            detectPersonInImage(tempImagePath); // This starts the async process
        }
    } else {
        // qDebug() << "YOLO processing skipped for this frame due to busy state."; // Uncomment for verbose skipping
        // Fall through to performance stats.
    }
    
    // Clear old detections if YOLO is not processing (to avoid stale boxes)
    static int frameCounter = 0;
    {
        QMutexLocker locker(&m_detectionMutex);
        if (!isProcessingFrame && !m_currentDetections.isEmpty()) {
            frameCounter++;
            // Clear detections after 30 frames (about 0.5 seconds at 60 FPS) if no new detection
            if (frameCounter > 30) {
                m_currentDetections.clear();
                frameCounter = 0;
            }
        } else if (isProcessingFrame) {
            // Reset counter when YOLO is processing
            frameCounter = 0;
        }
        locker.unlock();
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = loopTimer.elapsed(); // Time taken for this entire updateCameraFeed call
    totalTime += currentLoopTime;
    frameCount++;

    // Print stats every 60 frames
    if (frameCount % 60 == 0) {
        printPerformanceStats();
    }
}

void Capture::printPerformanceStats() {
    if (frameCount == 0) return; // Avoid division by zero

    double avgLoopTime = (double)totalTime / frameCount; // Average time per updateCameraFeed call

    // Calculate FPS based on the total time elapsed for the batch
    double batchDurationSeconds = (double)frameTimer.elapsed() / 1000.0;
    if (batchDurationSeconds == 0) return; // Avoid division by zero

    double measuredFPS = (double)frameCount / batchDurationSeconds;


    qDebug() << "----------------------------------------";
    qDebug() << "Avg loop time per frame (measured over " << frameCount << " frames):" << avgLoopTime << "ms";
    qDebug() << "Camera/Display FPS (measured over " << frameCount << " frames):" << measuredFPS << "FPS";
    qDebug() << "Frame processing efficiency:" << (avgLoopTime < 16.67 ? "GOOD" : "NEEDS OPTIMIZATION");
    qDebug() << "----------------------------------------";
    // Reset timers for next batch
    frameCount = 0;
    totalTime = 0;
    frameTimer.start(); // Restart frameTimer for the next measurement period
}

void Capture::handleYoloOutput() {
    QByteArray output = yoloProcess->readAllStandardOutput();
    // qDebug() << "YOLOv5 StdOut (Raw):" << output; // Uncomment for verbose output

    QJsonDocument doc = QJsonDocument::fromJson(output);

    if (!doc.isNull() && doc.isArray()) {
        QJsonArray results = doc.array();
        bool personDetected = false;
        QList<BoundingBox> detections;
        
        for (const QJsonValue& imageResult : results) {
            QJsonObject obj = imageResult.toObject();
            QJsonArray detectionArray = obj["detections"].toArray();
            if (!detectionArray.isEmpty()) {
                personDetected = true;
                qDebug() << "Person(s) detected! Number of detections:" << detectionArray.size();
                for (const QJsonValue& detectionValue : detectionArray) {
                    QJsonObject detection = detectionValue.toObject();
                    QJsonArray bbox = detection["bbox"].toArray();
                    double confidence = detection["confidence"].toDouble();
                    
                    // Create bounding box object
                    BoundingBox box(
                        bbox.at(0).toInt(), // x1
                        bbox.at(1).toInt(), // y1
                        bbox.at(2).toInt(), // x2
                        bbox.at(3).toInt(), // y2
                        confidence
                    );
                    detections.append(box);
                    
                    qDebug() << "  BBox:" << box.x1 << box.y1 << box.x2 << box.y2
                             << "Confidence:" << box.confidence;
                }
            }
        }
        
        // Update detection results for drawing
        updateDetectionResults(detections);
        
        if (personDetected) {
            emit personDetectedInFrame(); // Emit signal if any person was found
        } else {
            qDebug() << "No person detected in frame.";
        }
    } else {
        qDebug() << "Detection output is not a valid JSON array or is null. Raw output was:" << output;
        // Clear detections if no valid output
        updateDetectionResults(QList<BoundingBox>());
    }
}

void Capture::handleYoloError() {
    QByteArray errorOutput = yoloProcess->readAllStandardError();
    if (!errorOutput.isEmpty()) {
        qWarning() << "YOLOv5 Python Script Errors/Warnings (stderr):" << errorOutput;
    }
}

void Capture::handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    qDebug() << "YOLOv5 process finished with exit code:" << exitCode << "and status:" << exitStatus;
    if (exitCode != 0) {
        qWarning() << "YOLOv5 script exited with an error. Check stderr for details.";
        qWarning() << "Final Stderr (if any):" << yoloProcess->readAllStandardError();
        // Clear detections on error
        updateDetectionResults(QList<BoundingBox>());
    }
    isProcessingFrame = false; // Always reset the flag when the process finishes

    // Ensure any remaining output is read (edge case)
    yoloProcess->readAllStandardOutput();
    yoloProcess->readAllStandardError();

    // Delete the temporary image file
    if (!currentTempImagePath.isEmpty()) {
        QFile::remove(currentTempImagePath);
        currentTempImagePath.clear();
    }
}

void Capture::handleYoloErrorOccurred(QProcess::ProcessError error) {
    qWarning() << "QProcess experienced an internal error:" << error << yoloProcess->errorString();
    isProcessingFrame = false; // Always reset the flag when an internal QProcess error occurs

    // Clear detections on error
    updateDetectionResults(QList<BoundingBox>());

    // Delete the temporary image file
    if (!currentTempImagePath.isEmpty()) {
        QFile::remove(currentTempImagePath);
        currentTempImagePath.clear();
    }
}

void Capture::captureRecordingFrame()
{
    if (!m_isRecording || !cap.isOpened()) return;

    cv::Mat frame;
    if (cap.read(frame)) {
        if (frame.empty()) return;
        cv::flip(frame, frame, 1);
        QImage imageToStore = cvMatToQImage(frame);
        if (!imageToStore.isNull()) {
            m_recordedFrames.append(QPixmap::fromImage(imageToStore));
        }
    }
}

void Capture::on_back_clicked()
{
    if (countdownTimer->isActive()) {
        countdownTimer->stop();
        countdownLabel->hide();
        countdownValue = 0;
    }
    if (m_isRecording) {
        stopRecording();
    } else if (!ui->capture->isEnabled()) {
        ui->capture->setEnabled(true);
    }
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    ui->capture->setEnabled(false);
    countdownValue = 5;
    countdownLabel->setText(QString::number(countdownValue));
    countdownLabel->show();

    QPropertyAnimation *animation = new QPropertyAnimation(countdownLabel, "windowOpacity", this);
    animation->setDuration(300);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->start();

    countdownTimer->start(1000);
}

void Capture::updateCountdown()
{
    countdownValue--;
    if (countdownValue > 0){
        countdownLabel->setText(QString::number(countdownValue));
    } else {
        countdownTimer->stop();
        countdownLabel->hide();

        if(m_currentCaptureMode == ImageCaptureMode){
            performImageCapture();
            ui->capture->setEnabled(true);
        }else if(m_currentCaptureMode == VideoRecordMode){
            startRecording();
        }
    }
}

void Capture::updateRecordTimer()
{
    m_recordedSeconds++;
    if (m_recordedSeconds >= m_currentVideoTemplate.durationSeconds) {
        stopRecording();
    }
}

void Capture::startRecording()
{
    if(!cap.isOpened()){
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;
    int frameIntervalMs = 1000 / m_targetRecordingFPS;
    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
}

void Capture::stopRecording()
{
    if (!m_isRecording) return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    emit showFinalOutputPage();
    ui->capture->setEnabled(true);
}

void Capture::performImageCapture()
{
    cv::Mat frameToCapture;
    if (cap.read(frameToCapture)){
        if(frameToCapture.empty()) return;
        cv::flip(frameToCapture, frameToCapture, 1);
        QImage capturedImageQ = cvMatToQImage(frameToCapture);
        if(!capturedImageQ.isNull()){
            m_capturedImage = QPixmap::fromImage(capturedImageQ);
            emit imageCaptured(m_capturedImage);
        }
    }
    emit showFinalOutputPage();
}

QImage Capture::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
        return QImage(rgb.data, rgb.cols, mat.rows, mat.step, QImage::Format_RGB888);
    }
    case CV_8UC3: {
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    case CV_8UC1: {
        static QVector<QRgb> sColorTable;
        if (sColorTable.isEmpty())
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }
    default:
        return QImage();
    }
}

void Capture::on_verticalSlider_valueChanged(int value)
{
    int tickInterval = ui->verticalSlider->tickInterval();
    if (tickInterval == 0) return;
    int snappedValue = qRound((double)value / tickInterval) * tickInterval;
    snappedValue = qBound(ui->verticalSlider->minimum(), snappedValue, ui->verticalSlider->maximum());
    if (value != snappedValue) {
        ui->verticalSlider->setValue(snappedValue);
    }
}

void Capture::updateForegroundOverlay(const QString &path)
{
    qDebug() << "Foreground overlay updated to:" << path;

    if (!overlayImageLabel) {
        qWarning() << "overlayImageLabel is null!";
        return;
    }

    QPixmap overlayPixmap(path);
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->resize(this->size()); // Ensure it scales with window
    overlayImageLabel->show();
}



void Capture::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_B:
        // Toggle bounding boxes
        m_showBoundingBoxes = !m_showBoundingBoxes;
        
        qDebug() << "Bounding boxes toggled via keyboard to:" << m_showBoundingBoxes;
        
        // Show a brief on-screen notification
        showBoundingBoxNotification();
        break;
        
    default:
        // Call parent class for other keys
        QWidget::keyPressEvent(event);
        break;
    }
}

void Capture::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    // Ensure this widget has focus when shown
    setFocus();
}

// --- NEW: Bounding Box Drawing Methods Implementation ---

void Capture::updateDetectionResults(const QList<BoundingBox>& detections)
{
    QMutexLocker locker(&m_detectionMutex);
    m_currentDetections = detections;
    locker.unlock();
    
    // Reset frame counter when new detections arrive
    static int frameCounter = 0;
    frameCounter = 0;
}

void Capture::drawBoundingBoxes(QPixmap& pixmap, const QList<BoundingBox>& detections)
{
    QPainter painter(&pixmap);
    
    // Set up the painter for drawing
    painter.setRenderHint(QPainter::Antialiasing);
    
    for (int i = 0; i < detections.size(); ++i) {
        const BoundingBox& box = detections[i];
        
        // Calculate box dimensions
        int width = box.x2 - box.x1;
        int height = box.y2 - box.y1;
        
        // Create rectangle
        QRect rect(box.x1, box.y1, width, height);
        
        // Set up colors based on confidence
        QColor boxColor;
        if (box.confidence > 0.8) {
            boxColor = QColor(0, 255, 0); // Green for high confidence
        } else if (box.confidence > 0.6) {
            boxColor = QColor(255, 255, 0); // Yellow for medium confidence
        } else {
            boxColor = QColor(255, 0, 0); // Red for low confidence
        }
        
        // Draw the bounding box with rounded corners
        QPen pen(boxColor, 3); // 3 pixel thick line
        painter.setPen(pen);
        painter.drawRoundedRect(rect, 5, 5); // Rounded corners
        
        // Draw corner indicators
        int cornerSize = 10;
        painter.setPen(QPen(boxColor, 2));
        
        // Top-left corner
        painter.drawLine(box.x1, box.y1, box.x1 + cornerSize, box.y1);
        painter.drawLine(box.x1, box.y1, box.x1, box.y1 + cornerSize);
        
        // Top-right corner
        painter.drawLine(box.x2 - cornerSize, box.y1, box.x2, box.y1);
        painter.drawLine(box.x2, box.y1, box.x2, box.y1 + cornerSize);
        
        // Bottom-left corner
        painter.drawLine(box.x1, box.y2 - cornerSize, box.x1, box.y2);
        painter.drawLine(box.x1, box.y2, box.x1 + cornerSize, box.y2);
        
        // Bottom-right corner
        painter.drawLine(box.x2 - cornerSize, box.y2, box.x2, box.y2);
        painter.drawLine(box.x2, box.y2 - cornerSize, box.x2, box.y2);
        
        // Draw confidence text with person number
        QString confidenceText = QString("Person %1: %2%").arg(i + 1).arg(static_cast<int>(box.confidence * 100));
        QFont font = painter.font();
        font.setPointSize(12);
        font.setBold(true);
        painter.setFont(font);
        
        // Set up text background
        QRect textRect = painter.fontMetrics().boundingRect(confidenceText);
        textRect.moveTopLeft(QPoint(box.x1, box.y1 - textRect.height() - 5));
        textRect.adjust(-5, -2, 5, 2);
        
        // Draw text background with rounded corners
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(0, 0, 0, 200)); // More opaque black
        painter.drawRoundedRect(textRect, 3, 3);
        
        // Draw text
        painter.setPen(Qt::white);
        painter.drawText(QPoint(box.x1, box.y1 - 5), confidenceText);
        
        // Draw box dimensions (optional)
        QString sizeText = QString("%1x%2").arg(width).arg(height);
        QRect sizeRect = painter.fontMetrics().boundingRect(sizeText);
        sizeRect.moveTopLeft(QPoint(box.x2 - sizeRect.width() - 5, box.y2 + 5));
        sizeRect.adjust(-3, -1, 3, 1);
        
        painter.setBrush(QColor(0, 0, 0, 150));
        painter.drawRoundedRect(sizeRect, 2, 2);
        painter.setPen(Qt::white);
        painter.drawText(QPoint(box.x2 - sizeRect.width() - 2, box.y2 + 15), sizeText);
    }
    
    painter.end();
}

void Capture::showBoundingBoxNotification()
{
    // Create a temporary notification label
    QLabel* notificationLabel = new QLabel(this);
    notificationLabel->setAlignment(Qt::AlignCenter);
    notificationLabel->setStyleSheet(
        "QLabel {"
        "   background-color: rgba(0, 0, 0, 200);"
        "   color: white;"
        "   border-radius: 10px;"
        "   padding: 10px;"
        "   font-size: 16px;"
        "   font-weight: bold;"
        "}"
    );
    
    QString message = m_showBoundingBoxes ? 
        "Bounding Boxes: ON" : 
        "Bounding Boxes: OFF";
    notificationLabel->setText(message);
    
    // Position the notification in the center of the widget
    notificationLabel->adjustSize();
    int x = (width() - notificationLabel->width()) / 2;
    int y = (height() - notificationLabel->height()) / 2;
    notificationLabel->move(x, y);
    
    // Show the notification
    notificationLabel->show();
    notificationLabel->raise(); // Bring to front
    
    // Set up a timer to hide the notification after 1.5 seconds
    QTimer::singleShot(1500, [notificationLabel]() {
        if (notificationLabel) {
            notificationLabel->deleteLater();
        }
    });
}
