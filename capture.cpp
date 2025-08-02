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
#include <QCheckBox> // For bounding box toggle checkbox
#include <QApplication> // For Qt::WindowStaysOnTopHint
#include <QPushButton> // For debug panel buttons
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
    , debugLabel(nullptr)
    , fpsLabel(nullptr)
    , detectionLabel(nullptr)
    , boundingBoxCheckBox(nullptr)
    , debugUpdateTimer(nullptr)
    , m_currentFPS(0)
    , m_personDetected(false)
    , m_detectionCount(0)
    , m_averageConfidence(0.0)
    , overlayImageLabel(nullptr)
    , m_personDetector(new SimplePersonDetector())
    , m_useCppDetector(true) // Use C++ detector by default
    , m_segmentationProcessor(new PersonSegmentationProcessor())
    , m_showPersonSegmentation(false) // Start with segmentation disabled
    , m_segmentationConfidenceThreshold(0.7) // Default high confidence threshold
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

    // --- NEW: Setup Debug Display ---
    setupDebugDisplay();
    // --- END NEW ---

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

    // Create camera timer but DON'T start it yet - only start when page is shown
    cameraTimer = new QTimer(this);
    connect(cameraTimer, &QTimer::timeout, this, &Capture::updateCameraFeed);
    // cameraTimer->start(timerInterval); // ❌ REMOVED: Don't start immediately

    // Initialize performance timers but don't start them yet
    // loopTimer.start(); // ❌ REMOVED: Don't start immediately
    // frameTimer.start(); // ❌ REMOVED: Don't start immediately

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

    // Check if YOLO model file exists
    QString modelPath = QCoreApplication::applicationDirPath() + "/../../../yolov5/yolov5n.pt";
    QFileInfo modelFile(modelPath);
    if (!modelFile.exists()) {
        modelPath = QCoreApplication::applicationDirPath() + "/yolov5/yolov5n.pt";
        modelFile.setFile(modelPath);
    }
    
    if (modelFile.exists()) {
        qDebug() << "YOLOv5 model found at:" << modelPath;
        qDebug() << "Model file size:" << modelFile.size() << "bytes";
        
        // Initialize the C++ person detector
        if (m_useCppDetector) {
            qDebug() << "Initializing C++ person detector...";
            if (m_personDetector->initialize()) {
                qDebug() << "✅ C++ person detector initialized successfully!";
            } else {
                qWarning() << "❌ Failed to initialize C++ person detector, falling back to Python";
                m_useCppDetector = false;
            }
        }
    } else {
        qWarning() << "YOLOv5 model file not found! Expected at:" << modelPath;
        qWarning() << "Detection will not work without the model file.";
        m_useCppDetector = false;
    }

    // Test Python environment
    QProcess testProcess;
    testProcess.start("python3", QStringList() << "-c" << "import torch; print('PyTorch version:', torch.__version__)");
    if (testProcess.waitForStarted(1000)) {
        if (testProcess.waitForFinished(5000)) {
            QByteArray output = testProcess.readAllStandardOutput();
            qDebug() << "Python3 test output:" << output.trimmed();
        } else {
            qWarning() << "Python3 test timed out";
        }
    } else {
        qDebug() << "Python3 not found, trying python...";
        testProcess.start("python", QStringList() << "-c" << "import torch; print('PyTorch version:', torch.__version__)");
        if (testProcess.waitForStarted(1000)) {
            if (testProcess.waitForFinished(5000)) {
                QByteArray output = testProcess.readAllStandardOutput();
                qDebug() << "Python test output:" << output.trimmed();
            } else {
                qWarning() << "Python test timed out";
            }
        } else {
            qWarning() << "Neither python3 nor python found! YOLO detection will not work.";
        }
    }

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
    
    // Reposition debug panel
    QWidget* debugPanel = findChild<QWidget*>("debugPanel");
    if (debugPanel) {
        debugPanel->move(width() - debugPanel->width() - 20, 20);
    }
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
    if (yoloProcess->state() == QProcess::Running) {
        qDebug() << "YOLO process is already running, skipping this frame";
        return;
    }

    // Use python3 explicitly and provide full path to the script
    QString program = "python3";
    QStringList arguments;
    
    // Get the absolute path to the detect.py script
    QString scriptPath = QCoreApplication::applicationDirPath() + "/../../../yolov5/detect.py";
    QFileInfo scriptFile(scriptPath);
    
    if (!scriptFile.exists()) {
        qWarning() << "YOLOv5 detect.py script not found at:" << scriptPath;
        // Try alternative paths
        scriptPath = QCoreApplication::applicationDirPath() + "/yolov5/detect.py";
        scriptFile.setFile(scriptPath);
        if (!scriptFile.exists()) {
            qWarning() << "YOLOv5 detect.py script not found at alternative path:" << scriptPath;
            return;
        }
    }
    
    arguments << scriptPath
              << "--weights" << "yolov5/yolov5n.pt"
              << "--source" << imagePath
              << "--classes" << "0" // Class ID for 'person'
              << "--conf-thres" << "0.25" // Lower confidence threshold for better detection
              << "--iou-thres" << "0.45" // Standard IoU threshold
              << "--nosave"; // CRUCIAL: ensures output is to stdout, not saved to file

    // Set working directory to the project root
    QString workingDir = QCoreApplication::applicationDirPath() + "/../../../";
    QDir dir(workingDir);
    if (!dir.exists()) {
        workingDir = QCoreApplication::applicationDirPath();
    }
    yoloProcess->setWorkingDirectory(workingDir);

    qDebug() << "=== YOLO DIAGNOSTIC INFO ===";
    qDebug() << "Program:" << program;
    qDebug() << "Arguments:" << arguments.join(" ");
    qDebug() << "Working Directory:" << yoloProcess->workingDirectory();
    qDebug() << "Image path:" << imagePath;
    qDebug() << "Script path:" << scriptPath;
    qDebug() << "Current temp image path:" << currentTempImagePath;
    
    // Check if the image file exists
    QFileInfo fileInfo(imagePath);
    if (!fileInfo.exists()) {
        qWarning() << "Image file does not exist:" << imagePath;
        return;
    }
    qDebug() << "Image file exists, size:" << fileInfo.size() << "bytes";

    // Store the current temp image path for cleanup
    currentTempImagePath = imagePath;

    // Try python3 first, fallback to python
    yoloProcess->start(program, arguments);
    
    // If python3 fails, try python
    if (!yoloProcess->waitForStarted(1000)) {
        qDebug() << "python3 failed, trying python...";
        program = "python";
        yoloProcess->start(program, arguments);
        
        if (!yoloProcess->waitForStarted(1000)) {
            qWarning() << "Both python3 and python failed to start YOLO process!";
            qWarning() << "This indicates a Python environment issue.";
            isProcessingFrame = false;
            return;
        }
    }
    
    qDebug() << "YOLO process started successfully with PID:" << yoloProcess->processId();
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

    // --- NEW: Apply person segmentation if enabled ---
    if (m_showPersonSegmentation && m_segmentationProcessor) {
        QMutexLocker locker(&m_segmentationMutex);
        QList<SegmentationResult> segmentations = m_currentSegmentations; // Copy for thread safety
        locker.unlock();
        
        if (!segmentations.isEmpty()) {
            // Apply segmentation immediately for seamless tracking
            applySegmentationToFrame(frame, segmentations);
        }
    }
    // --- END NEW ---

    // Display the frame immediately
    QImage image = cvMatToQImage(frame);
    if (!image.isNull()) {
        QPixmap pixmap = QPixmap::fromImage(image);
        
        // Draw bounding boxes if enabled and detections exist
        if (m_showBoundingBoxes) {
            QMutexLocker locker(&m_detectionMutex);
            QList<BoundingBox> detections = m_currentDetections; // Copy for thread safety
            locker.unlock();
            
            qDebug() << "🔍 Bounding boxes enabled, checking" << detections.size() << "detections";
            
            if (!detections.isEmpty()) {
                qDebug() << "🎯 Drawing" << detections.size() << "bounding boxes on pixmap size:" << pixmap.size();
                
                // Draw bounding boxes directly on the original pixmap (no scaling needed)
                // The detections are already in the correct coordinate system for the original frame
                drawBoundingBoxes(pixmap, detections);
            } else {
                qDebug() << "⚠️ No detections to draw";
            }
        } else {
            qDebug() << "🚫 Bounding boxes disabled";
        }
        
        QSize labelSize = ui->videoLabel->size();
        QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::IgnoreAspectRatio, Qt::FastTransformation);
        ui->videoLabel->setPixmap(scaledPixmap);
        ui->videoLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        ui->videoLabel->update();
    } else {
        qWarning() << "Failed to convert OpenCV frame to QImage";
        isProcessingFrame = false;
        return;
    }

    // --- OPTIMIZED DETECTION PROCESSING: Only process every 3rd frame to maintain good FPS ---
    static int frameSkipCounter = 0;
    frameSkipCounter++;
    
    // Run detection EVERY FRAME for seamless real-time tracking without traces
    if (frameSkipCounter % 1 == 0 && !isProcessingFrame) {
        if (m_useCppDetector && m_personDetector && m_personDetector->isInitialized()) {
            // Use C++ person detector
            isProcessingFrame = true;
            
            try {
                // Run detection directly on the frame
                QList<SimpleDetection> detections = m_personDetector->detect(frame);
                
                // Convert SimpleDetection objects to BoundingBox objects
                QList<BoundingBox> boundingBoxes;
                for (const SimpleDetection& det : detections) {
                    // Validate detection before converting - accept any confidence value
                    if (det.boundingBox.width > 0 && det.boundingBox.height > 0) {
                        BoundingBox box(
                            det.boundingBox.x,
                            det.boundingBox.y,
                            det.boundingBox.x + det.boundingBox.width,
                            det.boundingBox.y + det.boundingBox.height,
                            det.confidence
                        );
                        boundingBoxes.append(box);
                        qDebug() << "✅ Added detection:" << det.boundingBox.x << det.boundingBox.y 
                                << det.boundingBox.width << "x" << det.boundingBox.height 
                                << "conf:" << det.confidence;
                    }
                }
                
                // Update detection results
                updateDetectionResults(boundingBoxes);
                
                // --- NEW: Process person segmentation ---
                if (m_showPersonSegmentation && !boundingBoxes.isEmpty()) {
                    processPersonSegmentation(frame, boundingBoxes);
                }
                // --- END NEW ---
                
                if (boundingBoxes.size() > 1) {
                    qDebug() << "👥 MULTIPLE PEOPLE DETECTED! C++ detector found" << boundingBoxes.size() << "persons";
                } else {
                    qDebug() << "C++ person detector found" << boundingBoxes.size() << "persons";
                }
                
            } catch (const std::exception& e) {
                qWarning() << "Exception in C++ person detection:" << e.what();
                updateDetectionResults(QList<BoundingBox>());
            }
            
            isProcessingFrame = false;
            
        } else {
            // Fallback to Python YOLO
            QString tempImagePath = QString("%1/temp_frame_%2.jpg")
                .arg(QCoreApplication::applicationDirPath())
                .arg(QDateTime::currentMSecsSinceEpoch());
                
            if (!cv::imwrite(tempImagePath.toStdString(), frame)) { // Save the original (flipped) frame
                qWarning() << "Failed to save temporary image:" << tempImagePath;
            } else {
                isProcessingFrame = true; // Set flag when starting YOLO
                detectPersonInImage(tempImagePath); // This starts the async process
            }
        }
    }
    
    // Clear old detections if YOLO is not processing (to avoid stale boxes)
    static int frameCounter = 0;
    {
        QMutexLocker locker(&m_detectionMutex);
        if (!isProcessingFrame && !m_currentDetections.isEmpty()) {
            frameCounter++;
            // Clear detections after 60 frames (about 1 second at 60 FPS) if no new detection
            if (frameCounter > 60) {
                m_currentDetections.clear();
                frameCounter = 0;
                qDebug() << "Cleared stale detections after timeout";
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

    // Calculate current FPS for debug display
    if (frameCount % 10 == 0) { // Update FPS every 10 frames for smoother display
        double currentFPS = 1000.0 / (currentLoopTime > 0 ? currentLoopTime : 1);
        m_currentFPS = static_cast<int>(currentFPS);
    }

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
    qDebug() << "=== YOLO OUTPUT RECEIVED ===";
    qDebug() << "Raw output length:" << output.length() << "bytes";
    qDebug() << "Raw output:" << output;

    // Try to parse as JSON
    QJsonDocument doc = QJsonDocument::fromJson(output);
    
    if (!doc.isNull() && doc.isArray()) {
        QJsonArray results = doc.array();
        bool personDetected = false;
        QList<BoundingBox> detections;
        
        qDebug() << "✅ Successfully parsed JSON with" << results.size() << "results";
        
        for (const QJsonValue& imageResult : results) {
            QJsonObject obj = imageResult.toObject();
            QJsonArray detectionArray = obj["detections"].toArray();
            
            qDebug() << "Processing image result with" << detectionArray.size() << "detections";
            
            if (!detectionArray.isEmpty()) {
                personDetected = true;
                qDebug() << "🎯 Person(s) detected! Number of detections:" << detectionArray.size();
                for (const QJsonValue& detectionValue : detectionArray) {
                    QJsonObject detection = detectionValue.toObject();
                    QJsonArray bbox = detection["bbox"].toArray();
                    double confidence = detection["confidence"].toDouble();
                    
                    if (bbox.size() == 4) {
                        // Create bounding box object
                        BoundingBox box(
                            bbox.at(0).toInt(), // x1
                            bbox.at(1).toInt(), // y1
                            bbox.at(2).toInt(), // x2
                            bbox.at(3).toInt(), // y2
                            confidence
                        );
                        detections.append(box);
                        
                        qDebug() << "  📦 BBox:" << box.x1 << box.y1 << box.x2 << box.y2
                                 << "Confidence:" << box.confidence;
                    } else {
                        qWarning() << "❌ Invalid bbox array size:" << bbox.size();
                    }
                }
            } else {
                qDebug() << "❌ No detections found in this image result";
            }
        }
        
        // Update detection results for drawing
        updateDetectionResults(detections);
        
        // --- NEW: Process person segmentation for YOLO detections ---
        if (m_showPersonSegmentation && !detections.isEmpty()) {
            // We need to reconstruct the original frame for segmentation
            // Since this is called from async YOLO processing, we may need to handle this differently
            // For now, segmentation will primarily work with C++ detector
            qDebug() << "🎭 YOLO segmentation integration - skipping for now (async processing limitation)";
        }
        // --- END NEW ---
        
        // Update debug information
        m_personDetected = personDetected;
        m_detectionCount = detections.size();
        
        // Calculate average confidence
        if (!detections.isEmpty()) {
            double totalConfidence = 0.0;
            for (const BoundingBox& box : detections) {
                totalConfidence += box.confidence;
            }
            m_averageConfidence = totalConfidence / detections.size();
            qDebug() << "📊 Average confidence:" << m_averageConfidence;
        } else {
            m_averageConfidence = 0.0;
        }
        
        if (personDetected) {
            emit personDetectedInFrame(); // Emit signal if any person was found
            qDebug() << "🚀 Emitted personDetectedInFrame signal";
        } else {
            qDebug() << "❌ No person detected in frame.";
        }
    } else {
        qDebug() << "❌ Detection output is not a valid JSON array or is null.";
        qDebug() << "Raw output was:" << output;
        
        // Try to extract any useful information from the output
        QString outputStr = QString::fromUtf8(output);
        if (outputStr.contains("person", Qt::CaseInsensitive)) {
            qDebug() << "ℹ️ Output contains 'person' keyword - might indicate detection";
        }
        if (outputStr.contains("error", Qt::CaseInsensitive)) {
            qWarning() << "⚠️ Output contains 'error' keyword - check for Python errors";
        }
        if (outputStr.contains("torch", Qt::CaseInsensitive)) {
            qDebug() << "ℹ️ Output contains 'torch' keyword - PyTorch related message";
        }
        
        // Clear detections if no valid output
        updateDetectionResults(QList<BoundingBox>());
        
        // Clear debug information
        m_personDetected = false;
        m_detectionCount = 0;
        m_averageConfidence = 0.0;
    }
    
    qDebug() << "=== END YOLO OUTPUT PROCESSING ===";
}

void Capture::handleYoloError() {
    QByteArray errorOutput = yoloProcess->readAllStandardError();
    if (!errorOutput.isEmpty()) {
        qWarning() << "=== YOLO ERROR OUTPUT ===";
        qWarning() << "Error output length:" << errorOutput.length() << "bytes";
        qWarning() << "Error output:" << errorOutput;
        
        // Check for common errors
        QString errorStr = QString::fromUtf8(errorOutput);
        if (errorStr.contains("No module named", Qt::CaseInsensitive)) {
            qWarning() << "❌ Missing Python module - check if ultralytics is installed";
            qWarning() << "   Try: pip install ultralytics";
        } else if (errorStr.contains("FileNotFoundError", Qt::CaseInsensitive)) {
            qWarning() << "❌ File not found error - check script and model paths";
        } else if (errorStr.contains("CUDA", Qt::CaseInsensitive)) {
            qWarning() << "⚠️ CUDA/GPU related error - falling back to CPU";
        } else if (errorStr.contains("torch", Qt::CaseInsensitive)) {
            qWarning() << "❌ PyTorch related error - check PyTorch installation";
            qWarning() << "   Try: pip install torch torchvision";
        } else if (errorStr.contains("ImportError", Qt::CaseInsensitive)) {
            qWarning() << "❌ Import error - missing dependencies";
        } else if (errorStr.contains("PermissionError", Qt::CaseInsensitive)) {
            qWarning() << "❌ Permission error - check file access rights";
        } else if (errorStr.contains("OSError", Qt::CaseInsensitive)) {
            qWarning() << "❌ OS error - check system compatibility";
        } else {
            qWarning() << "⚠️ Unknown error type - check Python environment";
        }
        
        qWarning() << "=== END YOLO ERROR OUTPUT ===";
    }
}

void Capture::handleYoloFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    qDebug() << "YOLOv5 process finished with exit code:" << exitCode << "and status:" << exitStatus;
    
    if (exitCode != 0) {
        qWarning() << "YOLOv5 script exited with an error. Check stderr for details.";
        qWarning() << "Final Stderr (if any):" << yoloProcess->readAllStandardError();
        
        // Clear detections on error
        updateDetectionResults(QList<BoundingBox>());
        
        // Clear debug information
        m_personDetected = false;
        m_detectionCount = 0;
        m_averageConfidence = 0.0;
    } else {
        qDebug() << "YOLOv5 process completed successfully";
    }
    
    isProcessingFrame = false; // Always reset the flag when the process finishes

    // Ensure any remaining output is read (edge case)
    QByteArray remainingOutput = yoloProcess->readAllStandardOutput();
    if (!remainingOutput.isEmpty()) {
        qDebug() << "Processing remaining output after process finished:" << remainingOutput;
        // Process any remaining output
        QByteArray fullOutput = remainingOutput;
        QJsonDocument doc = QJsonDocument::fromJson(fullOutput);
        if (!doc.isNull() && doc.isArray()) {
            // Process the remaining output as if it came through handleYoloOutput
            QJsonArray results = doc.array();
            bool personDetected = false;
            QList<BoundingBox> detections;
            
            for (const QJsonValue& imageResult : results) {
                QJsonObject obj = imageResult.toObject();
                QJsonArray detectionArray = obj["detections"].toArray();
                
                if (!detectionArray.isEmpty()) {
                    personDetected = true;
                    for (const QJsonValue& detectionValue : detectionArray) {
                        QJsonObject detection = detectionValue.toObject();
                        QJsonArray bbox = detection["bbox"].toArray();
                        double confidence = detection["confidence"].toDouble();
                        
                        if (bbox.size() == 4) {
                            BoundingBox box(
                                bbox.at(0).toInt(),
                                bbox.at(1).toInt(),
                                bbox.at(2).toInt(),
                                bbox.at(3).toInt(),
                                confidence
                            );
                            detections.append(box);
                        }
                    }
                }
            }
            
            updateDetectionResults(detections);
            m_personDetected = personDetected;
            m_detectionCount = detections.size();
        }
    }

    // Clean up temporary image file
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
    
    // Clear debug information
    m_personDetected = false;
    m_detectionCount = 0;
    m_averageConfidence = 0.0;

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
        // Toggle bounding boxes with 'B' key
        setShowBoundingBoxes(!getShowBoundingBoxes());
        showBoundingBoxNotification();
        qDebug() << "🎮 Bounding boxes toggled via keyboard to:" << m_showBoundingBoxes;
        break;
        
    case Qt::Key_S:
        // Toggle instant segmentation with 'S' key
        setShowPersonSegmentation(!getShowPersonSegmentation());
        qDebug() << "🚀 INSTANT person segmentation toggled via keyboard to:" << m_showPersonSegmentation;
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
    
    // Start camera timer and performance timers when page is shown
    if (cameraTimer && !cameraTimer->isActive()) {
        qDebug() << "🎬 Starting camera timer - Capture page is now visible";
        cameraTimer->start();
        loopTimer.start();
        frameTimer.start();
    }
}

void Capture::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    
    // Stop camera timer and performance timers when page is hidden
    if (cameraTimer && cameraTimer->isActive()) {
        qDebug() << "⏸️ Stopping camera timer - Capture page is now hidden";
        cameraTimer->stop();
        loopTimer.invalidate();
        frameTimer.invalidate();
    }
}

// --- NEW: Bounding Box Drawing Methods Implementation ---

void Capture::updateDetectionResults(const QList<BoundingBox>& detections)
{
    try {
        QMutexLocker locker(&m_detectionMutex);
        m_currentDetections = detections;
        locker.unlock();
        
        // Update debug display variables
        m_detectionCount = detections.size();
        m_personDetected = (detections.size() > 0);
        
        // Calculate average confidence
        if (!detections.isEmpty()) {
            double totalConfidence = 0.0;
            int validDetections = 0;
            for (const BoundingBox& box : detections) {
                if (box.confidence >= 0.0 && box.confidence <= 1.0) {
                    totalConfidence += box.confidence;
                    validDetections++;
                }
            }
            m_averageConfidence = (validDetections > 0) ? (totalConfidence / validDetections) : 0.0;
        } else {
            m_averageConfidence = 0.0;
        }
    
    qDebug() << "Detection results updated:" << detections.size() << "detections";
    for (int i = 0; i < detections.size(); ++i) {
        const BoundingBox& box = detections[i];
        qDebug() << "  Detection" << i << ":" << box.x1 << box.y1 << box.x2 << box.y2 << "conf:" << box.confidence;
    }
    
    // Reset frame counter when new detections arrive
    static int frameCounter = 0;
    frameCounter = 0;
    
    } catch (const std::exception& e) {
        qWarning() << "Exception in updateDetectionResults:" << e.what();
        m_detectionCount = 0;
        m_personDetected = false;
        m_averageConfidence = 0.0;
    }
}

void Capture::drawBoundingBoxes(QPixmap& pixmap, const QList<BoundingBox>& detections)
{
    QPainter painter(&pixmap);
    
    // Set up the painter for drawing
    painter.setRenderHint(QPainter::Antialiasing);
    
    qDebug() << "🎨 Drawing" << detections.size() << "bounding boxes on pixmap size:" << pixmap.size();
    
    for (int i = 0; i < detections.size(); ++i) {
        const BoundingBox& box = detections[i];
        
        // Validate bounding box coordinates
        if (box.x1 < 0 || box.y1 < 0 || box.x2 <= box.x1 || box.y2 <= box.y1 ||
            box.x2 > pixmap.width() || box.y2 > pixmap.height()) {
            qWarning() << "❌ Invalid bounding box coordinates:" << box.x1 << box.y1 << box.x2 << box.y2
                       << "pixmap size:" << pixmap.size();
            continue;
        }
        
        // Calculate box dimensions
        int width = box.x2 - box.x1;
        int height = box.y2 - box.y1;
        
        // Create rectangle
        QRect rect(box.x1, box.y1, width, height);
        
        qDebug() << "✅ Drawing box" << i << "at" << rect << "with confidence" << box.confidence;
        
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
        QPen pen(boxColor, 4); // Increased thickness for better visibility
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
    qDebug() << "✅ Finished drawing" << detections.size() << "bounding boxes";
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

// --- NEW: Debug Display Methods ---

void Capture::setupDebugDisplay() {
    // Create debug overlay widget (compact for ultra-fast mode)
    debugWidget = new QWidget(this);
    debugWidget->setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; border-radius: 10px; padding: 10px;");
    debugWidget->setFixedSize(400, 180); // Compact size for essentials only
    debugWidget->move(10, 10);
    debugWidget->setWindowFlags(Qt::WindowStaysOnTopHint);
    debugWidget->show();

    // Create layout for debug widgets
    QVBoxLayout* debugLayout = new QVBoxLayout(debugWidget);
    debugLayout->setSpacing(5);

    // FPS Label
    fpsLabel = new QLabel("Camera FPS: --", debugWidget);
    fpsLabel->setStyleSheet("color: white; font-weight: bold;");
    debugLayout->addWidget(fpsLabel);

    // Detection Label
    detectionLabel = new QLabel("Detection Status: --", debugWidget);
    detectionLabel->setStyleSheet("color: white; font-weight: bold;");
    debugLayout->addWidget(detectionLabel);

    // Main Debug Label
    debugLabel = new QLabel("Debug Info: --", debugWidget);
    debugLabel->setStyleSheet("color: white; font-weight: bold;");
    debugLayout->addWidget(debugLabel);

    // Bounding Box Checkbox
    boundingBoxCheckBox = new QCheckBox("Show Bounding Boxes", debugWidget);
    boundingBoxCheckBox->setStyleSheet("color: white; font-weight: bold;");
    boundingBoxCheckBox->setChecked(m_showBoundingBoxes);
    connect(boundingBoxCheckBox, &QCheckBox::toggled, this, &Capture::onBoundingBoxCheckBoxToggled);
    debugLayout->addWidget(boundingBoxCheckBox);
    
    // Person Segmentation Checkbox  
    segmentationCheckBox = new QCheckBox("🎯 REAL Person Shape", debugWidget);
    segmentationCheckBox->setStyleSheet("color: lime; font-weight: bold; font-size: 14px;");
    segmentationCheckBox->setChecked(m_showPersonSegmentation);
    connect(segmentationCheckBox, &QCheckBox::toggled, this, &Capture::onSegmentationCheckBoxToggled);
    debugLayout->addWidget(segmentationCheckBox);

    // Simplified keyboard shortcuts info
    QLabel* shortcutsLabel = new QLabel("🎮 Controls:", debugWidget);
    shortcutsLabel->setStyleSheet("color: cyan; font-weight: bold; font-size: 12px;");
    debugLayout->addWidget(shortcutsLabel);
    
    QLabel* shortcutsInfo = new QLabel("B: Toggle BBox | S: Toggle REAL Person Shape", debugWidget);
    shortcutsInfo->setStyleSheet("color: lightgray; font-size: 11px; line-height: 1.2;");
    debugLayout->addWidget(shortcutsInfo);

    // Update timer
    debugUpdateTimer = new QTimer(this);
    connect(debugUpdateTimer, &QTimer::timeout, this, &Capture::updateDebugDisplay);
    debugUpdateTimer->start(100); // Update every 100ms for real-time display
}

void Capture::updateDebugDisplay() {
    if (!debugLabel || !fpsLabel || !detectionLabel) {
        return;
    }
    
    // Update FPS display
    fpsLabel->setText(QString("Camera FPS: %1").arg(m_currentFPS));
    
    // Update detection status
    QString detectionText;
    if (m_personDetected) {
        detectionText = QString("DETECTED: %1 person(s) | Avg Conf: %2%")
            .arg(m_detectionCount)
            .arg(static_cast<int>(m_averageConfidence * 100));
        detectionLabel->setStyleSheet("color: green; font-weight: bold;");
    } else {
        detectionText = "No person detected";
        detectionLabel->setStyleSheet("color: red; font-weight: bold;");
    }
    detectionLabel->setText(detectionText);
    
    // Update main debug info
    QString debugText = QString("🎯 REAL SHAPE | BBox: %1 | Shape: %2 (%3ms) | Proc: %4")
        .arg(m_showBoundingBoxes ? "ON" : "OFF")
        .arg(m_showPersonSegmentation ? "ON" : "OFF")
        .arg(m_segmentationProcessor ? QString::number(m_segmentationProcessor->getAverageProcessingTime(), 'f', 1) : "0.0")
        .arg(isProcessingFrame ? "Yes" : "No");
    
    debugLabel->setText(debugText);
    
    // Update checkbox states
    if (boundingBoxCheckBox) {
        boundingBoxCheckBox->setChecked(m_showBoundingBoxes);
    }
    if (segmentationCheckBox) {
        segmentationCheckBox->setChecked(m_showPersonSegmentation);
    }
    
    // Show detection details if available
    QMutexLocker locker(&m_detectionMutex);
    if (!m_currentDetections.isEmpty()) {
        QString detailsText = "Detection Details:\n";
        for (int i = 0; i < m_currentDetections.size(); ++i) {
            const BoundingBox& box = m_currentDetections[i];
            detailsText += QString("  Person %1: (%2,%3)-(%4,%5) Conf: %6%\n")
                .arg(i + 1)
                .arg(box.x1).arg(box.y1).arg(box.x2).arg(box.y2)
                .arg(static_cast<int>(box.confidence * 100));
        }
        qDebug() << detailsText.trimmed();
    }
    locker.unlock();
}

void Capture::onBoundingBoxCheckBoxToggled(bool checked)
{
    m_showBoundingBoxes = checked;
    qDebug() << "Bounding boxes toggled via checkbox to:" << m_showBoundingBoxes;
    showBoundingBoxNotification();
}

// --- NEW: Test YOLO Detection Method ---
void Capture::testYoloDetection() {
    qDebug() << "=== MANUAL YOLO DETECTION TEST ===";
    
    // Create a test image with a simple pattern that should be detectable
    cv::Mat testImage(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    // Draw a simple rectangle to simulate a person (should be detectable)
    cv::rectangle(testImage, cv::Point(200, 100), cv::Point(400, 400), cv::Scalar(255, 255, 255), -1);
    
    // Add some noise to make it more realistic
    cv::randn(testImage, cv::Scalar(0, 0, 0), cv::Scalar(30, 30, 30));
    
    QString testImagePath = QString("%1/test_detection_%2.jpg")
        .arg(QCoreApplication::applicationDirPath())
        .arg(QDateTime::currentMSecsSinceEpoch());
    
    if (cv::imwrite(testImagePath.toStdString(), testImage)) {
        qDebug() << "✅ Created test image:" << testImagePath;
        qDebug() << "Image size:" << testImage.cols << "x" << testImage.rows;
        
        // Check if YOLO process is available
        if (yoloProcess->state() == QProcess::Running) {
            qDebug() << "⚠️ YOLO process is currently running, will wait for it to finish...";
            yoloProcess->waitForFinished(5000); // Wait up to 5 seconds
        }
        
        // Reset detection state for clean test
        m_personDetected = false;
        m_detectionCount = 0;
        m_averageConfidence = 0.0;
        updateDetectionResults(QList<BoundingBox>());
        
        // Start detection
        detectPersonInImage(testImagePath);
        
        // Set up a timer to check results after a delay
        QTimer::singleShot(5000, [this, testImagePath]() {
            qDebug() << "=== TEST RESULTS ===";
            qDebug() << "Detection count:" << m_detectionCount;
            qDebug() << "Person detected:" << m_personDetected;
            qDebug() << "Average confidence:" << m_averageConfidence;
            
            QMutexLocker locker(&m_detectionMutex);
            qDebug() << "Current detections:" << m_currentDetections.size();
            for (int i = 0; i < m_currentDetections.size(); ++i) {
                const BoundingBox& box = m_currentDetections[i];
                qDebug() << "  Detection" << i << ":" << box.x1 << box.y1 << box.x2 << box.y2 << "conf:" << box.confidence;
            }
            locker.unlock();
            
            // Clean up test image
            QFile::remove(testImagePath);
            
            if (m_detectionCount > 0) {
                qDebug() << "✅ TEST PASSED: YOLO detection is working!";
            } else {
                qDebug() << "❌ TEST FAILED: No detections found. Check Python environment and model file.";
            }
        });
        
    } else {
        qWarning() << "❌ Failed to create test image";
    }
    
    qDebug() << "=== END TEST ===";
}

// --- NEW: Person Segmentation Implementation ---

void Capture::setShowPersonSegmentation(bool show) {
    m_showPersonSegmentation = show;
    qDebug() << "Person segmentation display" << (show ? "enabled" : "disabled");
    
    if (segmentationCheckBox) {
        segmentationCheckBox->setChecked(show);
    }
}

bool Capture::getShowPersonSegmentation() const {
    return m_showPersonSegmentation;
}

void Capture::onSegmentationCheckBoxToggled(bool checked) {
    setShowPersonSegmentation(checked);
    if (checked) {
        showSegmentationNotification();
    }
}

void Capture::setSegmentationConfidenceThreshold(double threshold) {
    m_segmentationConfidenceThreshold = qBound(0.1, threshold, 1.0);
    qDebug() << "Segmentation confidence threshold set to:" << m_segmentationConfidenceThreshold;
}

double Capture::getSegmentationConfidenceThreshold() const {
    return m_segmentationConfidenceThreshold;
}

cv::Mat Capture::getLastSegmentedFrame() const {
    QMutexLocker locker(&m_segmentationMutex);
    return m_lastSegmentedFrame.clone();
}

void Capture::saveSegmentedFrame(const QString& filename) {
    cv::Mat segmentedFrame = getLastSegmentedFrame();
    if (segmentedFrame.empty()) {
        qWarning() << "No segmented frame available to save";
        return;
    }
    
    QString saveFilename = filename;
    if (saveFilename.isEmpty()) {
        saveFilename = QString("segmented_frame_%1.png")
            .arg(QDateTime::currentMSecsSinceEpoch());
    }
    
    try {
        if (cv::imwrite(saveFilename.toStdString(), segmentedFrame)) {
            qDebug() << "✅ Segmented frame saved as:" << saveFilename;
        } else {
            qWarning() << "❌ Failed to save segmented frame:" << saveFilename;
        }
    } catch (const cv::Exception& e) {
        qWarning() << "❌ OpenCV error saving segmented frame:" << e.what();
    }
}

void Capture::processPersonSegmentation(const cv::Mat& frame, const QList<BoundingBox>& detections) {
    if (!m_segmentationProcessor || !m_showPersonSegmentation || frame.empty()) {
        return;
    }
    
    try {
        qDebug() << "🎭 Processing person segmentation with" << detections.size() << "detections";
        
        // Filter detections by confidence threshold
        QList<BoundingBox> highConfidenceDetections;
        for (const BoundingBox& detection : detections) {
            if (detection.confidence >= m_segmentationConfidenceThreshold) {
                highConfidenceDetections.append(detection);
            }
        }
        
        if (highConfidenceDetections.isEmpty()) {
            qDebug() << "⏭️  No high-confidence detections for segmentation (threshold:" << m_segmentationConfidenceThreshold << ")";
            updateSegmentationResults(QList<SegmentationResult>());
            return;
        }
        
        qDebug() << "🎯 Processing" << highConfidenceDetections.size() << "high-confidence detections";
        
        // Use fast segmentation for real-time performance
        QList<SegmentationResult> results = m_segmentationProcessor->segmentPersonsFast(
            frame, highConfidenceDetections, m_segmentationConfidenceThreshold);
        
        // Update segmentation results
        updateSegmentationResults(results);
        
        qDebug() << "✅ Segmentation processing complete, generated" << results.size() << "valid segments";
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Exception in person segmentation:" << e.what();
        updateSegmentationResults(QList<SegmentationResult>());
    }
}

void Capture::updateSegmentationResults(const QList<SegmentationResult>& results) {
    try {
        QMutexLocker locker(&m_segmentationMutex);
        m_currentSegmentations = results;
        locker.unlock();
        
        qDebug() << "🔄 Segmentation results updated:" << results.size() << "segments";
        
        // Store combined segmented frame if we have valid results
        if (!results.isEmpty()) {
            qDebug() << "📊 Segmentation confidence scores:";
            for (int i = 0; i < results.size(); ++i) {
                const SegmentationResult& result = results[i];
                if (result.isValid) {
                    qDebug() << "  Segment" << i << "confidence:" << result.confidence 
                             << "mask size:" << result.mask.cols << "x" << result.mask.rows;
                }
            }
        }
        
    } catch (const std::exception& e) {
        qWarning() << "❌ Exception updating segmentation results:" << e.what();
    }
}

void Capture::applySegmentationToFrame(cv::Mat& frame, const QList<SegmentationResult>& results) {
    if (!m_showPersonSegmentation || results.isEmpty() || frame.empty()) {
        return;
    }
    
    try {
        // Create combined transparent background image
        cv::Mat transparentFrame = m_segmentationProcessor->combineSegmentations(frame, results);
        
        if (!transparentFrame.empty()) {
            // Store the segmented frame
            QMutexLocker locker(&m_segmentationMutex);
            m_lastSegmentedFrame = transparentFrame.clone();
            locker.unlock();
            
            // For real-time display: show person on TRULY TRANSPARENT background
            cv::Mat displayFrame;
            if (transparentFrame.channels() == 4) {
                // Create completely transparent background (black = transparent in display)
                cv::Mat backgroundFrame = cv::Mat::zeros(frame.size(), CV_8UC3);
                // Keep background completely black for transparency effect
                
                // Split channels
                std::vector<cv::Mat> channels;
                cv::split(transparentFrame, channels);
                
                if (channels.size() >= 4) {
                    // Create RGB image of just the person
                    cv::Mat rgbFrame;
                    std::vector<cv::Mat> rgbChannels = {channels[0], channels[1], channels[2]};
                    cv::merge(rgbChannels, rgbFrame);
                    
                    // Apply mask directly - only show person pixels
                    cv::Mat alpha = channels[3];
                    cv::Mat result = cv::Mat::zeros(frame.size(), CV_8UC3);
                    
                    // Copy person pixels where mask is white, leave rest transparent (black)
                    rgbFrame.copyTo(result, alpha);
                    
                    displayFrame = result;
                } else {
                    displayFrame = transparentFrame;
                }
            } else {
                displayFrame = transparentFrame;
            }
            
            // Replace the original frame with segmented version (person only, transparent background)
            displayFrame.copyTo(frame);
            
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ OpenCV error applying segmentation:" << e.what();
    } catch (const std::exception& e) {
        qWarning() << "❌ Standard error applying segmentation:" << e.what();
    }
}

void Capture::showSegmentationNotification() {
    // Create instant segmentation notification
    QLabel* notificationLabel = new QLabel(this);
    notificationLabel->setAlignment(Qt::AlignCenter);
    notificationLabel->setStyleSheet(
        "QLabel {"
        "   background-color: rgba(0, 255, 0, 240);" // Bright lime green for instant speed
        "   color: black;"
        "   border-radius: 15px;"
        "   padding: 25px;"
        "   font-size: 22px;"
        "   font-weight: bold;"
        "   border: 3px solid rgba(0, 255, 0, 255);"
        "}"
    );
    
    QString message = m_showPersonSegmentation ? 
        "🎯 REAL Person Shape: ON\nAccurate Contour Tracking" : 
        "🎯 REAL Person Shape: OFF";
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
    
    qDebug() << "🚀 INSTANT person segmentation" << (m_showPersonSegmentation ? "enabled" : "disabled") << "- transparent background extraction";
}

double Capture::getSegmentationProcessingTime() const {
    if (m_segmentationProcessor) {
        return m_segmentationProcessor->getAverageProcessingTime();
    }
    return 0.0;
}

// --- END: Person Segmentation Implementation ---
