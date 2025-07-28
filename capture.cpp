#include "capture.h"
#include "ui_capture.h"
#include "foreground.h"
#include "camera.h"

#include <QThread>
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QPropertyAnimation>
#include <QFont>
#include <QResizeEvent>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QMessageBox>
#include <QStackedLayout>
#include <QPainter>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

Capture::Capture(QWidget *parent, Foreground *fg, Camera *existingCameraWorker, QThread *existingCameraThread)

    : QWidget(parent)
    , ui(new Ui::Capture)
    , foreground(fg)
    , cameraThread(existingCameraThread)
    , cameraWorker(existingCameraWorker)
    , countdownTimer(nullptr)
    , countdownLabel(nullptr)
    , countdownValue(0)
    , m_currentCaptureMode(ImageCaptureMode)
    , m_isRecording(false)
    , recordTimer(nullptr)
    , recordingFrameTimer(nullptr)
    , m_targetRecordingFPS(60)
    , m_currentVideoTemplate("Default", 5) // Assuming a constructor for VideoTemplate
    , m_recordedSeconds(0)
    , m_recordedFrames()
    , m_capturedImage()
    , stackedLayout(nullptr)
    , loadingCameraLabel(nullptr)
    , videoLabelFPS(nullptr) // Initialize pointer to nullptr
    , loopTimer()
    , totalTime(0)           // Initialize totalTime
    , frameCount(0)          // Initialize frameCount
    , frameTimer()
    , yoloProcess(new QProcess(this)) // Initialize QProcess here!
    , isProcessingFrame(false) // Initialize frame processing flag
    , currentTempImagePath()
    , overlayImageLabel(nullptr)
{
    ui->setupUi(this);

    setContentsMargins(0, 0, 0, 0);

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
    overlayImageLabel->setScaledContents(true);
    overlayImageLabel->resize(this->size());
    overlayImageLabel->hide();

    loadingCameraLabel = new QLabel("Loading Camera...", this); // Set text here
    // --- FIX for "oading Camera." - Increased size and added padding for robustness ---
    loadingCameraLabel->setAlignment(Qt::AlignCenter);
    QFont loadingFont = loadingCameraLabel->font();
    loadingFont.setPointSize(36);
    loadingFont.setBold(true);
    loadingCameraLabel->setFont(loadingFont);
    // Increased horizontal padding in stylesheet to give text more room
    loadingCameraLabel->setStyleSheet(
        "color: white; "
        "background-color: rgba(0, 0, 0, 150); "
        "border-radius: 15px; "
        "padding: 10px 20px; " // Added padding (top/bottom 10px, left/right 20px)
        );
    // Adjusted fixed size to accommodate padding and ensure text fits
    // A rough estimate: "Loading Camera." at 36pt bold might need more than 350px width.
    // Let's try to calculate a safer size or rely on sizeHint if not fixed.
    // For fixed size, let's just make it generously larger.
    loadingCameraLabel->setFixedSize(450, 120); // Increased size (from 350, 100)
    // ---------------------------------------------------------------------------------
    loadingCameraLabel->show();

    ui->videoLabel->hide();

    connect(foreground, &Foreground::foregroundChanged, this, &Capture::updateForegroundOverlay);
    QPixmap overlayPixmap(selectedOverlay);
    overlayImageLabel->setPixmap(overlayPixmap);

    setupStackedLayoutHybrid();
    updateOverlayStyles();

    ui->videoLabel->resize(this->size());
    ui->overlayWidget->resize(this->size());
    if (overlayImageLabel) {
        overlayImageLabel->resize(this->size());
    }

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;");
    ui->videoLabel->setScaledContents(false);
    ui->videoLabel->setAlignment(Qt::AlignCenter);

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->overlayWidget->setStyleSheet("background-color: transparent;"); // Ensure overlay is transparent


    ui->verticalSlider->setMinimum(0);
    ui->verticalSlider->setMaximum(100);
    int tickStep = 10;
    ui->verticalSlider->setTickPosition(QSlider::TicksBothSides);
    ui->verticalSlider->setTickInterval(tickStep);
    ui->verticalSlider->setSingleStep(tickStep);
    ui->verticalSlider->setValue(0);

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));
    ui->capture->setEnabled(false);

    if (cameraWorker) {
        connect(cameraWorker, &Camera::frameReady, this, &Capture::updateCameraFeed);
        connect(cameraWorker, &Camera::cameraOpened, this, &Capture::handleCameraOpened);
        connect(cameraWorker, &Camera::error, this, &Capture::handleCameraError);
    } else {
        qCritical() << "Capture: ERROR: cameraWorker is NULL! Camera features will not function.";
        loadingCameraLabel->hide();
        ui->videoLabel->show();
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText("Camera worker not provided or is NULL.\nCannot initialize camera.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }
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

    countdownTimer = new QTimer(this);
    connect(countdownTimer, &QTimer::timeout, this, &Capture::updateCountdown);

    recordTimer = new QTimer(this);
    connect(recordTimer, &QTimer::timeout, this, &Capture::updateRecordTimer);

    recordingFrameTimer = new QTimer(this);
    recordingFrameTimer->setTimerType(Qt::PreciseTimer);
    connect(recordingFrameTimer, &QTimer::timeout, this, &Capture::captureRecordingFrame);

    connect(ui->back, &QPushButton::clicked, this, &Capture::on_back_clicked);
    connect(ui->capture, &QPushButton::clicked, this, &Capture::on_capture_clicked);
    connect(ui->verticalSlider, &QSlider::valueChanged, this, &Capture::on_verticalSlider_valueChanged);

    // --- CONNECT QPROCESS SIGNALS ONCE IN CONSTRUCTOR ---
    connect(yoloProcess, &QProcess::readyReadStandardOutput, this, &Capture::handleYoloOutput);
    connect(yoloProcess, &QProcess::readyReadStandardError, this, &Capture::handleYoloError);
    connect(yoloProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Capture::handleYoloFinished);
    connect(yoloProcess, &QProcess::errorOccurred, this, &Capture::handleYoloErrorOccurred);
    // --- END CONNECT QPROCESS SIGNALS ---

    qDebug() << "Capture UI initialized. Loading Camera...";
}

Capture::~Capture()
{
    // Stop and delete QTimers owned by Capture
    if (countdownTimer){ countdownTimer->stop(); delete countdownTimer; countdownTimer = nullptr; }
    if (recordTimer){ recordTimer->stop(); delete recordTimer; recordTimer = nullptr; }
    if (recordingFrameTimer){ recordingFrameTimer->stop(); delete recordingFrameTimer; recordingFrameTimer = nullptr; }

    // Delete QLabels created on the heap and parented to Capture or ui->overlayWidget
    // loadingCameraLabel has 'this' as parent, so it *will* be deleted by Qt's object tree.
    // However, explicitly nulling out the pointer is good practice.
    if (overlayImageLabel){ delete overlayImageLabel; overlayImageLabel = nullptr; }
    if (loadingCameraLabel){ delete loadingCameraLabel; loadingCameraLabel = nullptr; }
    if (videoLabelFPS){ delete videoLabelFPS; videoLabelFPS = nullptr; } // Only if you actually 'new' this somewhere

    // === Corrected FIX for YOLO QProcess cleanup ===
    // Disconnect signals from yoloProcess FIRST to prevent calls to a dying Capture object
    // It's good practice to disconnect if the slots might try to access 'this' after partial destruction.
    disconnect(yoloProcess, &QProcess::readyReadStandardOutput, this, &Capture::handleYoloOutput);
    disconnect(yoloProcess, &QProcess::readyReadStandardError, this, &Capture::handleYoloError);
    disconnect(yoloProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &Capture::handleYoloFinished);
    disconnect(yoloProcess, &QProcess::errorOccurred, this, &Capture::handleYoloErrorOccurred);

    // Terminate YOLO process on exit if it's still running
    if (yoloProcess->state() == QProcess::Running) {
        qDebug() << "Capture: Terminating YOLO process...";
        yoloProcess->terminate();
        if (!yoloProcess->waitForFinished(1000)) { // Give it a moment to finish gracefully
            qWarning() << "Capture: YOLO process did not terminate gracefully, killing...";
            yoloProcess->kill(); // Force kill if it doesn't terminate
            yoloProcess->waitForFinished(500); // Wait briefly after kill
        }
    }
    // yoloProcess will be deleted by QObject parent mechanism because 'this' is its parent.
    // No explicit 'delete yoloProcess;' is needed here.
    // === END Corrected FIX for YOLO QProcess cleanup ===

    // DO NOT DELETE cameraWorker or cameraThread here.
    // They are passed in as existing objects, implying Capture does not own them.
    // If Capture *were* responsible for stopping and deleting the camera thread,
    // that logic would be handled by the class that *owns* the cameraThread and cameraWorker.
    // For Capture, we just null out the pointers to prevent dangling pointers in case
    // some other part of Capture's code tries to use them after they've been destroyed externally.
    cameraWorker = nullptr;
    cameraThread = nullptr;

    delete ui; // Deletes the UI object and all its child widgets.
}

void Capture::handleCameraOpened(bool success, double actual_width, double actual_height, double actual_fps)
{
    Q_UNUSED(actual_width);
    Q_UNUSED(actual_height);
    Q_UNUSED(actual_fps);

    if (success) {
        qDebug() << "Capture: Camera worker reported open success. Enabling capture button.";
        ui->capture->setEnabled(true);
        // Ensure that the parent widget of loadingCameraLabel (centeringWidget) is hidden
        if (loadingCameraLabel->parentWidget()) {
            loadingCameraLabel->parentWidget()->hide();
        } else {
            loadingCameraLabel->hide(); // Fallback
        }
        ui->videoLabel->show();
    } else {
        qWarning() << "Capture: Camera worker reported open failure.";
        if (loadingCameraLabel->parentWidget()) {
            loadingCameraLabel->parentWidget()->hide();
        } else {
            loadingCameraLabel->hide(); // Fallback
        }
        ui->videoLabel->show();
        ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
        ui->videoLabel->setText("Camera failed to open.\nCheck connection and drivers.");
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        ui->capture->setEnabled(false);
    }
}

void Capture::handleCameraError(const QString &msg)
{
    QMessageBox::critical(this, "Camera Error", msg);
    ui->capture->setEnabled(false);
    qWarning() << "Capture: Camera error received:" << msg;
    if (loadingCameraLabel->parentWidget()) {
        loadingCameraLabel->parentWidget()->hide();
    } else {
        loadingCameraLabel->hide();
    }
    ui->videoLabel->show();
    ui->videoLabel->setStyleSheet("background-color: #333; color: white; border-radius: 10px;");
    ui->videoLabel->setText(QString("Error: %1").arg(msg));
    ui->videoLabel->setAlignment(Qt::AlignCenter);
}

void Capture::updateCameraFeed(const QImage &image)
{
    // Start loopTimer at the very beginning of the function to measure total time for one update cycle.
    loopTimer.start(); // Measure time for this entire call

    if (image.isNull()) {
        qWarning() << "Capture: Received null QImage from Camera.";
        isProcessingFrame = false; // Ensure flag is reset if image is null
        // Performance stats should still be calculated for every attempt to process a frame
        qint64 currentLoopTime = loopTimer.elapsed();
        totalTime += currentLoopTime;
        frameCount++;
        if (frameCount % 60 == 0) {
            printPerformanceStats();
        }
        return;
    }

    // Hide the centeringWidget containing loadingCameraLabel when frames arrive
    if (loadingCameraLabel->parentWidget() && !loadingCameraLabel->parentWidget()->isHidden()) {
        loadingCameraLabel->parentWidget()->hide();
        ui->videoLabel->show();
    } else if (!loadingCameraLabel->isHidden()) {
        loadingCameraLabel->hide();
        ui->videoLabel->show();
    }

    QPixmap pixmap = QPixmap::fromImage(image);

    QSize labelSize = ui->videoLabel->size();
    QPixmap scaledPixmap = pixmap.scaled(
        labelSize,
        Qt::KeepAspectRatioByExpanding,
        Qt::FastTransformation
        );

    ui->videoLabel->setPixmap(scaledPixmap);
    ui->videoLabel->setAlignment(Qt::AlignCenter);
    ui->videoLabel->update(); // Request a repaint

    // Now, handle YOLO processing only if not already busy
    if (!isProcessingFrame) {
        QString tempImagePath = QDir::temp().filePath(
            "yolo_temp_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz") + ".jpg"
            );

        // Convert QImage to cv::Mat for saving to file for YOLO
        cv::Mat frameForYolo = qImageToCvMat(image);

        if (!frameForYolo.empty() && cv::imwrite(tempImagePath.toStdString(), frameForYolo)) {
            isProcessingFrame = true; // Set flag when starting YOLO
            detectPersonInImage(tempImagePath); // This starts the async process
        } else {
            qWarning() << "Failed to save temporary image or convert QImage to cv::Mat for YOLO:" << tempImagePath;
            isProcessingFrame = false; // Ensure it's false if saving failed
        }
    } else {
        // YOLO processing skipped for this frame due to busy state.
    }

    // --- Performance stats (always run for every valid frame received) ---
    qint64 currentLoopTime = loopTimer.elapsed();
    totalTime += currentLoopTime;
    frameCount++;

    if (frameCount % 60 == 0) {
        printPerformanceStats();
    }
}


void Capture::setupStackedLayoutHybrid()
{
    qDebug() << "Setting up hybrid stacked layout...";

    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);

    ui->overlayWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayWidget->setMinimumSize(1, 1);
    ui->overlayWidget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);

    if (!stackedLayout) {
        stackedLayout = new QStackedLayout;
        stackedLayout->setStackingMode(QStackedLayout::StackAll);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        stackedLayout->setSpacing(0);

        stackedLayout->addWidget(ui->videoLabel);       // Layer 0: Camera feed (background)

        // --- FIX for Centering "Loading Camera" ---
        // Create a layout to center the fixed-size loadingCameraLabel
        QVBoxLayout* centeringLayout = new QVBoxLayout();
        centeringLayout->addStretch(); // Top stretch
        QHBoxLayout* hCenteringLayout = new QHBoxLayout();
        hCenteringLayout->addStretch(); // Left stretch
        hCenteringLayout->addWidget(loadingCameraLabel); // Add the fixed-size label
        hCenteringLayout->addStretch(); // Right stretch
        centeringLayout->addLayout(hCenteringLayout); // Add horizontal layout to vertical
        centeringLayout->addStretch(); // Bottom stretch

        QWidget* centeringWidget = new QWidget(this); // Create a wrapper widget
        centeringWidget->setLayout(centeringLayout);
        centeringWidget->setContentsMargins(0,0,0,0); // Ensure no extra margins
        centeringWidget->setAttribute(Qt::WA_TranslucentBackground); // Keep background transparent
        // Initially show the centeringWidget, as loadingCameraLabel is shown by default
        centeringWidget->show();


        stackedLayout->addWidget(centeringWidget);      // Layer 1: Loading text (now centered)
        // ------------------------------------------

        stackedLayout->addWidget(ui->overlayWidget);    // Layer 2: UI elements (buttons, slider)
        if (overlayImageLabel) {
            stackedLayout->addWidget(overlayImageLabel); // Layer 3: Foreground image (top)
        }

        if (layout()) {
            delete layout();
        }

        QGridLayout* mainLayout = new QGridLayout(this);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);
        mainLayout->addLayout(stackedLayout, 0, 0);
        mainLayout->setRowStretch(0, 1);
        mainLayout->setColumnStretch(0, 1);

        setLayout(mainLayout);
    }

    if (overlayImageLabel) {
        overlayImageLabel->raise();
    }
    ui->overlayWidget->raise();
    if (ui->back) ui->back->raise();
    if (ui->capture) ui->capture->raise();
    if (ui->verticalSlider) ui->verticalSlider->raise();
    if (countdownLabel) countdownLabel->raise();

    // Now, instead of raising loadingCameraLabel, raise its parent centeringWidget
    if (loadingCameraLabel->parentWidget()) {
        loadingCameraLabel->parentWidget()->raise();
    } else {
        loadingCameraLabel->raise(); // Fallback, though should be within centeringWidget now
    }

    qDebug() << "Hybrid stacked layout setup complete.";
}

void Capture::updateOverlayStyles()
{
    qDebug() << "Updating overlay styles with clean professional appearance...";
    ui->back->setStyleSheet(
        "QPushButton {"
        "   background: transparent;"
        "   border: none;"
        "   color: white;"
        "}"
        );

    ui->capture->setStyleSheet(
        "QPushButton {"
        "   border-radius: 9px;"
        "   border-bottom: 3px solid rgba(2, 2, 2, 200);"
        "   background: rgba(11, 194, 0, 200);"
        "   color: white;"
        "   font-size: 16px;"
        "   font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "   background: rgba(8, 154, 0, 230);"
        "}"
        "QPushButton:disabled {"
        "   background: rgba(100, 100, 100, 150);"
        "   color: rgba(200, 200, 200, 150);"
        "   border-bottom: 3px solid rgba(50, 50, 50, 150);"
        "}"
        );

    ui->verticalSlider->setStyleSheet(
        "QSlider::groove:vertical {"
        "   background: rgba(0, 0, 0, 80);"
        "   width: 30px;"
        "   border-radius: 15px;"
        "   border: none;"
        "}"
        "QSlider::handle:vertical {"
        "   background: rgba(13, 77, 38, 220);"
        "   border: 1px solid rgba(30, 144, 255, 180);"
        "   width: 60px;"
        "   height: 13px;"
        "   border-radius: 7px;"
        "   margin: 0 -15px;"
        "}"
        "QSlider::sub-page:vertical {"
        "   background: rgba(0, 0, 0, 60);"
        "   border-top-left-radius: 15px;"
        "   border-top-right-radius: 15px;"
        "   border-bottom-left-radius: 0px;"
        "   border-bottom-right-radius: 0px;"
        "}"
        "QSlider::add-page:vertical {"
        "   background: rgba(11, 194, 0, 180);"
        "   border-bottom-left-radius: 15px;"
        "   border-bottom-right-radius: 15px;"
        "   border-top-left-radius: 0px;"
        "   border-top-right-radius: 0px;"
        "}"
        );

    ui->overlayWidget->setStyleSheet("background: transparent;");
    qDebug() << "Clean professional overlay styles applied";
}



void Capture::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    ui->videoLabel->resize(size());
    ui->overlayWidget->resize(size());
    if (overlayImageLabel) {
        overlayImageLabel->resize(size());
        overlayImageLabel->move(0, 0);
    }

    if (countdownLabel) {
        int x = (width() - countdownLabel->width()) / 2;
        int y = (height() - countdownLabel->height()) / 2;
        countdownLabel->move(x, y);
    }
}

void Capture::setCaptureMode(CaptureMode mode) {
    m_currentCaptureMode = mode;
    qDebug() << "Capture mode set to:" << static_cast<int>(mode);
}

void Capture::setVideoTemplate(const VideoTemplate &templateData) {
    m_currentVideoTemplate = templateData;
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
        for (const QJsonValue& imageResult : results) {
            QJsonObject obj = imageResult.toObject();
            QJsonArray detections = obj["detections"].toArray();
            if (!detections.isEmpty()) {
                personDetected = true;
                qDebug() << "Person(s) detected! Number of detections:" << detections.size();
                for (const QJsonValue& detectionValue : detections) {
                    QJsonObject detection = detectionValue.toObject();
                    QJsonArray bbox = detection["bbox"].toArray();
                    double confidence = detection["confidence"].toDouble();
                    qDebug() << "  BBox:" << bbox.at(0).toInt() << bbox.at(1).toInt()
                             << bbox.at(2).toInt() << bbox.at(3).toInt()
                             << "Confidence:" << confidence;
                    // TODO: Emit a signal here with the bbox data to update your UI (e.g., draw rects)
                }
            }
        }
        if (personDetected) {
            emit personDetectedInFrame(); // Emit signal if any person was found
        } else {
            qDebug() << "No person detected in frame.";
        }
    } else {
        qDebug() << "Detection output is not a valid JSON array or is null. Raw output was:" << output;
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

    // Delete the temporary image file
    if (!currentTempImagePath.isEmpty()) {
        QFile::remove(currentTempImagePath);
        currentTempImagePath.clear();
    }
}

void Capture::captureRecordingFrame()
{
    if (!m_isRecording) return;

    if (!ui->videoLabel->pixmap().isNull()) {
        m_recordedFrames.append(ui->videoLabel->pixmap());
    } else {
        qWarning() << "No pixmap available on videoLabel for recording frame.";
    }
}

void Capture::on_back_clicked()
{
    qDebug() << "DEBUG: Back button clicked in Capture! Emitting backtoPreviousPage.";
    if (countdownTimer->isActive()) {
        countdownTimer->stop();
        countdownLabel->hide();
        countdownValue = 0;
    }
    if (m_isRecording) {
        stopRecording();
    }
    ui->capture->setEnabled(true);
    emit backtoPreviousPage();
}

void Capture::on_capture_clicked()
{
    if (!cameraWorker || !cameraWorker->isCameraOpen()) {
        QMessageBox::warning(this, "Camera Not Ready", "Camera is not open. Please ensure it's connected and drivers are installed.");
        return;
    }

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
    qDebug() << "Recording: " + QString::number(m_recordedSeconds) + " / " + QString::number(m_currentVideoTemplate.durationSeconds) + "s";
}

void Capture::startRecording()
{
    if(!cameraWorker->isCameraOpen()){
        qWarning() << "Cannot start recording: Camera not opened by worker.";
        ui->capture->setEnabled(true);
        return;
    }

    m_recordedFrames.clear();
    m_isRecording = true;
    m_recordedSeconds = 0;

    int frameIntervalMs = 1000 / m_targetRecordingFPS;

    recordTimer->start(1000);
    recordingFrameTimer->start(frameIntervalMs);
    qDebug() << "Recording started at target FPS: " + QString::number(m_targetRecordingFPS) + " frames/sec";
}

void Capture::stopRecording()
{
    if (!m_isRecording) return;

    recordTimer->stop();
    recordingFrameTimer->stop();
    m_isRecording = false;
    qDebug() << "Recording stopped. Captured " + QString::number(m_recordedFrames.size()) + " frames.";

    if (!m_recordedFrames.isEmpty()) {
        emit videoRecorded(m_recordedFrames);
    }
    emit showFinalOutputPage();
    ui->capture->setEnabled(true);
}

void Capture::performImageCapture()
{
    if (!ui->videoLabel->pixmap().isNull()) {
        QPixmap cameraPixmap = ui->videoLabel->pixmap();
        QPixmap compositedPixmap = cameraPixmap.copy();

        // Get the selected overlay/template path
        QString overlayPath;
        if (foreground) {
            overlayPath = foreground->getSelectedForeground();
        }
        if (!overlayPath.isEmpty()) {
            QPixmap overlayPixmap(overlayPath);
            if (!overlayPixmap.isNull()) {
                // Scale overlay to match camera image size
                QPixmap scaledOverlay = overlayPixmap.scaled(
                    compositedPixmap.size(),
                    Qt::KeepAspectRatioByExpanding,
                    Qt::SmoothTransformation
                    );
                // Composite overlay onto camera image
                QPainter painter(&compositedPixmap);
                painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
                painter.drawPixmap(0, 0, scaledOverlay);
                painter.end();
            }
        }
        m_capturedImage = compositedPixmap;
        emit imageCaptured(m_capturedImage);
        qDebug() << "Image captured and composited with overlay.";
    } else {
        qWarning() << "Failed to capture image: videoLabel pixmap is empty.";
        QMessageBox::warning(this, "Capture Failed", "No camera feed available to capture an image.");
    }
    emit showFinalOutputPage();
}

// Helper function to convert QImage to cv::Mat
cv::Mat Capture::qImageToCvMat(const QImage &inImage)
{
    switch (inImage.format()) {
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
        // For 32-bit formats, OpenCV typically expects BGRA or BGR.
        // Assuming BGRA or similar layout where alpha is last.
        // OpenCV Mat constructor takes (rows, cols, type, data, step)
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC4,
                       (void*)inImage.constBits(), inImage.bytesPerLine());
    case QImage::Format_RGB888:
        // RGB888 in QImage is typically 3 bytes per pixel, RGB order.
        // OpenCV expects BGR by default, so a clone and conversion might be needed,
        // or ensure `cvtColor` is used later if written as RGB.
        // For direct conversion, it's safer to convert QImage to a known OpenCV format first if needed.
        // Here, we clone because `inImage.constBits()` might not be persistent.
        return cv::Mat(inImage.height(), inImage.width(), CV_8UC3,
                       (void*)inImage.constBits(), inImage.bytesPerLine()).clone();
    case QImage::Format_Indexed8:
    {
        // 8-bit grayscale image
        cv::Mat mat(inImage.height(), inImage.width(), CV_8UC1,
                    (void*)inImage.constBits(), inImage.bytesPerLine());
        return mat.clone(); // Clone to ensure data is owned by Mat
    }
    default:
        qWarning() << "qImageToCvMat - QImage format not handled: " << inImage.format();
        // Convert to a supported format if not directly convertible
        QImage convertedImage = inImage.convertToFormat(QImage::Format_RGB32);
        return cv::Mat(convertedImage.height(), convertedImage.width(), CV_8UC4,
                       (void*)convertedImage.constBits(), convertedImage.bytesPerLine());
    }
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
        qWarning() << "overlayImageLabel is null! Cannot update overlay.";
        return;
    }

    QPixmap overlayPixmap(path);
    if (overlayPixmap.isNull()) {
        qWarning() << "Failed to load overlay image from path:" << path;
        overlayImageLabel->hide();
        return;
    }
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->show();
}
