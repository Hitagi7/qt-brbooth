#include "ui/final.h"
#include <QDebug>
#include <QDir>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include "ui/iconhover.h" // Assuming this is your custom class for button hover effects
#include "ui_final.h"  // Generated UI header
#include <QMessageBox>
#include <QFileDialog>
#include <QStandardPaths>
#include <QImage>
#include <QDateTime>
#include <opencv2/opencv.hpp>

// Explicitly include layouts, although they are in final.h, it's good practice
#include <QGridLayout>
#include <QStackedLayout>

Final::Final(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Final)
    , videoPlaybackTimer(nullptr)
    , m_currentFrameIndex(0)
    , m_videoFPS(30.0)
    , m_stackedLayout(nullptr)
    , m_lastLoadedImage() // Initialize the QPixmap member
    , overlayImageLabel(nullptr)
{
    ui->setupUi(this); // Initialize UI elements from final.ui

    // Ensure the main Final widget itself has no margins or spacing
    setContentsMargins(0, 0, 0, 0);

    // Create overlay image label for pixel art game UI elements (same as capture interface)
    overlayImageLabel = new QLabel(this); // Create as child of Final widget, not overlayFinal
    overlayImageLabel->setAttribute(Qt::WA_TranslucentBackground);
    overlayImageLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true); // Allow mouse events to pass through
    overlayImageLabel->setStyleSheet("background: transparent;");
    overlayImageLabel->setScaledContents(true); // Same as capture interface
    overlayImageLabel->resize(this->size());
    overlayImageLabel->move(0, 0);
    
    // Initially hide the overlay - it will be set when foreground is provided
    overlayImageLabel->hide();

    // --- LAYOUT SETUP using same logic as capture interface ---

    // 1. Create the QGridLayout as the main layout for the Final widget
    QGridLayout *mainLayout = new QGridLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0); // No padding for the grid itself
    mainLayout->setSpacing(0);                  // No spacing between grid cells

    // 2. Create the QStackedLayout to stack widgets (same as capture interface)
    m_stackedLayout = new QStackedLayout; // Initialize the member
    m_stackedLayout->setStackingMode(
        QStackedLayout::StackAll); // Makes all contained widgets visible and layered
    m_stackedLayout->setContentsMargins(0, 0, 0, 0); // No padding inside the stacked layout
    m_stackedLayout->setSpacing(0);                  // No spacing between stacked items

    // 3. Add widgets to stacked layout in same order as capture interface
    m_stackedLayout->addWidget(ui->videoLabel); // Layer 0: Video/image display (background)
    m_stackedLayout->addWidget(ui->overlayFinal); // Layer 1: UI elements (buttons)
    if (overlayImageLabel) {
        m_stackedLayout->addWidget(overlayImageLabel); // Layer 2: Foreground template (top)
    }

    // 4. Add the stacked layout to the main grid layout, making it stretch
    mainLayout->addLayout(m_stackedLayout, 0, 0); // Place the stacked layout in the first cell
    mainLayout->setRowStretch(0, 1); // Make this row stretchable
    mainLayout->setColumnStretch(0, 1); // Make this column stretchable

    // 5. Set the main grid layout for the Final widget
    setLayout(mainLayout); // Apply the mainLayout to the Final widget

    // --- Configure properties of the UI elements ---

    // Configure videoLabel (background display)
    ui->videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->videoLabel->setMinimumSize(1, 1);
    ui->videoLabel->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->videoLabel->setStyleSheet("background-color: black;"); // Set background to black if no image
    ui->videoLabel->setScaledContents(false);                  // We handle scaling manually
    ui->videoLabel->setAlignment(Qt::AlignCenter); // Center the scaled image within the label

    // Configure overlayFinal (foreground for buttons)
    ui->overlayFinal->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->overlayFinal->setMinimumSize(1, 1);
    ui->overlayFinal->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    ui->overlayFinal->setStyleSheet("background-color: transparent;"); // Crucial for transparency

    // Ensure proper layering (same as capture interface)
    if (overlayImageLabel) {
        overlayImageLabel->raise(); // Foreground template on top
    }
    ui->overlayFinal->raise(); // UI elements (buttons) on top
    ui->back->raise();         // Ensure back button is clickable
    ui->save->raise();         // Ensure save button is clickable

    // --- Existing setup for buttons and timers ---

    // Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg")); // Adjust path as necessary
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);

    // Set button styles
    ui->back->setStyleSheet("QPushButton {"
                            "   background: transparent;"
                            "   border: none;"
                            "   color: white;"
                            "}");

    ui->save->setStyleSheet("QPushButton {"
                            "   border-radius: 9px;"
                            "   border-bottom: 3px solid rgba(2, 2, 2, 200);" // Subtle shadow
                            "   background: rgba(11, 194, 0, 200);" // Your original green color
                            "   color: white;"
                            "   font-size: 16px;"
                            "   font-weight: bold;"
                            "}"
                            "QPushButton:hover {"
                            "   background: rgba(8, 154, 0, 230);" // Your original hover color
                            "}");

    videoPlaybackTimer = new QTimer(this);
    videoPlaybackTimer->setTimerType(Qt::PreciseTimer); // Use precise timer for better timing accuracy
    connect(videoPlaybackTimer, &QTimer::timeout, this, &Final::playNextFrame);

    // <== CRITICAL: Defer the initial display refresh to ensure layout has settled
    // This addresses the "small image on first capture" problem.
    QTimer::singleShot(0, this, &Final::refreshDisplay);
}

Final::~Final()
{
    if (videoPlaybackTimer) {
        videoPlaybackTimer->stop();
        delete videoPlaybackTimer;
        videoPlaybackTimer = nullptr;
    }
    
    // Clean up overlay image label
    if (overlayImageLabel) {
        delete overlayImageLabel;
        overlayImageLabel = nullptr;
    }
    
    delete ui;
}

// --- NEW/MODIFIED FUNCTIONS ---

void Final::refreshDisplay()
{
    // If video frames are loaded, display the current frame
    if (!m_videoFrames.isEmpty()) {
        // playNextFrame() will handle getting the current frame and scaling it.
        // We ensure the timer is running for continuous playback.
        if (!videoPlaybackTimer->isActive()) {
            // Use the stored video FPS for correct playback speed
            int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / m_videoFPS));
            videoPlaybackTimer->start(playbackIntervalMs);
        }
        playNextFrame(); // Display the current frame (and advance for next call)
    }
    // If no video frames, but a single image is loaded
    else if (!m_lastLoadedImage.isNull()) {
        // The image is already scaled from the capture interface, so display it as-is
        // Only apply minimal scaling if the image is too large for the label
        QSize labelSize = ui->videoLabel->size();
        QSize imageSize = m_lastLoadedImage.size();
        
        qDebug() << "Final interface displaying image. Image size:" << imageSize << "Label size:" << labelSize;
        
        // Only scale down if the image is larger than the label
        if (imageSize.width() > labelSize.width() || imageSize.height() > labelSize.height()) {
            QPixmap scaledImage = m_lastLoadedImage.scaled(
                labelSize,
                Qt::KeepAspectRatio,
                Qt::FastTransformation
            );
            ui->videoLabel->setPixmap(scaledImage);
            qDebug() << "Image was re-scaled down to:" << scaledImage.size();
        } else {
            // Display the image as-is since it's already properly scaled
            ui->videoLabel->setPixmap(m_lastLoadedImage);
            qDebug() << "Image displayed as-is (no re-scaling needed)";
        }
        
        ui->videoLabel->setAlignment(Qt::AlignCenter);
    }
    // If nothing is loaded, clear the display
    else {
        ui->videoLabel->clear();
    }
}

void Final::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event); // Call the base class implementation

    // Ensure ui->videoLabel and ui->overlayFinal fill the entire widget's current size
    if (ui->videoLabel)
        ui->videoLabel->resize(this->size());
    if (ui->overlayFinal)
        ui->overlayFinal->resize(this->size());
    if (overlayImageLabel) {
        overlayImageLabel->resize(this->size());
        overlayImageLabel->move(0, 0);
        
        // Use the same resize logic as capture interface
        overlayImageLabel->resize(this->size());
        overlayImageLabel->move(0, 0);
    }

    // Ensure buttons remain on top after resize (same as capture interface)
    ui->overlayFinal->raise();
    ui->back->raise();
    ui->save->raise();

    // <== CRITICAL: Re-scale and display the content when the widget resizes
    refreshDisplay();
}

void Final::setImage(const QPixmap &image)
{
    // Stop video playback if active
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames.clear(); // Clear any existing video frames
    m_currentFrameIndex = 0;
    m_lastLoadedImage = image; // Store the scaled image from capture interface

    qDebug() << "Final interface received image with size:" << image.size();

    ui->videoLabel->setText(""); // Clear any previous text

    refreshDisplay(); // Display the image immediately with proper scaling
}

void Final::setVideo(const QList<QPixmap> &frames, double fps)
{
    // Stop any current playback
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames = frames; // Store the list of video frames
    m_currentFrameIndex = 0;
    m_videoFPS = fps; // Store the FPS for debugging and potential future use
    m_lastLoadedImage = QPixmap(); // Clear last image if switching to video

    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Playing back video with " << m_videoFrames.size() << " frames at " << fps << " FPS.";
        
        // Calculate the correct playback interval based on the actual camera FPS
        int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / fps));
        videoPlaybackTimer->start(playbackIntervalMs);
        
        refreshDisplay(); // Display the first frame immediately with proper scaling
    } else {
        qWarning() << "No video frames provided for playback!";
        ui->videoLabel->clear(); // Clear display if no frames
    }
}

void Final::setForegroundOverlay(const QString &foregroundPath)
{
    if (!overlayImageLabel) {
        qWarning() << "overlayImageLabel is null! Cannot set foreground overlay.";
        return;
    }

    if (foregroundPath.isEmpty()) {
        qDebug() << "No foreground path provided, hiding overlay.";
        overlayImageLabel->hide();
        return;
    }

    QPixmap overlayPixmap(foregroundPath);
    if (overlayPixmap.isNull()) {
        qWarning() << "Failed to load foreground overlay from path:" << foregroundPath;
        overlayImageLabel->hide();
        return;
    }

    // Use the same logic as capture interface: set scaled contents and show
    overlayImageLabel->setPixmap(overlayPixmap);
    overlayImageLabel->setScaledContents(true); // Same as capture interface
    overlayImageLabel->show();
    
    // Ensure buttons remain on top after setting overlay (same as capture interface)
    ui->overlayFinal->raise();
    ui->back->raise();
    ui->save->raise();
    
    qDebug() << "Foreground overlay set successfully from:" << foregroundPath;
}

void Final::playNextFrame()
{
    if (m_videoFrames.isEmpty()) {
        videoPlaybackTimer->stop();
        ui->videoLabel->clear();
        qDebug() << "No frames left to play or video playback stopped.";
        return;
    }

    // Loop video playback
    if (m_currentFrameIndex >= m_videoFrames.size()) {
        m_currentFrameIndex = 0;
    }

    // Get the current frame - it's already scaled from the capture interface
    QPixmap currentFrame = m_videoFrames.at(m_currentFrameIndex);
    QSize labelSize = ui->videoLabel->size();
    QSize frameSize = currentFrame.size();
    
    // Only scale down if the frame is larger than the label
    if (frameSize.width() > labelSize.width() || frameSize.height() > labelSize.height()) {
        QPixmap scaledFrame = currentFrame.scaled(
            labelSize,
            Qt::KeepAspectRatio,
            Qt::FastTransformation
        );
        ui->videoLabel->setPixmap(scaledFrame);
    } else {
        // Display the frame as-is since it's already properly scaled
        ui->videoLabel->setPixmap(currentFrame);
    }
    ui->videoLabel->setAlignment(Qt::AlignCenter);

    m_currentFrameIndex++; // Advance to the next frame for the next timer timeout
}

// --- STANDARD SLOTS AND HELPER FUNCTIONS (unchanged from your original) ---

void Final::on_back_clicked()
{
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }
    emit backToCapturePage();
}

void Final::on_save_clicked()
{
    if (!m_videoFrames.isEmpty()) {
        // We have video frames, so save as a video
        saveVideoToFile();
    } else {
        // No video frames, proceed with saving an image
        // Use the currently displayed pixmap from ui->videoLabel
        QPixmap imageToSave = ui->videoLabel->pixmap();

        if (imageToSave.isNull()) {
            QMessageBox::warning(this, "Save Image", "No image to save.");
            return;
        }

        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            // Fallback for systems without standard download path (e.g., some Linux setups or custom env)
            downloadsPath
                = "C:/Downloads"; // Or a more cross-platform default like QDir::currentPath()
        }

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QString fileName = downloadsPath + "/image_" + timestamp + ".png";

        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath); // Create the directory if it doesn't exist
        }

        if (imageToSave.save(fileName)) {
            QMessageBox::information(this,
                                     "Save Image",
                                     QString("Image saved successfully to:\n%1").arg(fileName));
        } else {
            QMessageBox::critical(this, "Save Image", "Failed to save image.");
        }
    }
    emit backToLandingPage();
}

// Helper function to convert QImage to cv::Mat
// This function needs to handle various QImage formats.
cv::Mat QImageToCvMat(const QImage &inImage)
{
    switch (inImage.format()) {
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
        // For 32-bit formats, OpenCV typically expects BGRA or BGR.
        // Assuming BGRA or similar layout where alpha is last.
        // OpenCV Mat constructor takes (rows, cols, type, data, step)
        return cv::Mat(inImage.height(),
                       inImage.width(),
                       CV_8UC4,
                       (void *) inImage.constBits(),
                       inImage.bytesPerLine());
    case QImage::Format_RGB888:
        // RGB888 in QImage is typically 3 bytes per pixel, RGB order.
        // OpenCV expects BGR by default, so a clone and conversion might be needed,
        // or ensure `cvtColor` is used later if written as RGB.
        // For direct conversion, it's safer to convert QImage to a known OpenCV format first if needed.
        // Here, we clone because `inImage.constBits()` might not be persistent.
        return cv::Mat(inImage.height(),
                       inImage.width(),
                       CV_8UC3,
                       (void *) inImage.constBits(),
                       inImage.bytesPerLine())
            .clone();
    case QImage::Format_Indexed8: {
        // 8-bit grayscale image
        cv::Mat mat(inImage.height(),
                    inImage.width(),
                    CV_8UC1,
                    (void *) inImage.constBits(),
                    inImage.bytesPerLine());
        return mat.clone(); // Clone to ensure data is owned by Mat
    }
    default:
        qWarning() << "QImageToCvMat - QImage format not handled: " << inImage.format();
        // Convert to a supported format if not directly convertible
        QImage convertedImage = inImage.convertToFormat(QImage::Format_RGB32);
        return cv::Mat(convertedImage.height(),
                       convertedImage.width(),
                       CV_8UC4,
                       (void *) convertedImage.constBits(),
                       convertedImage.bytesPerLine());
    }
}

// Helper function to save video frames to an AVI file
void Final::saveVideoToFile()
{
    if (m_videoFrames.isEmpty()) {
        QMessageBox::warning(this, "Save Video", "No video frames to save.");
        return;
    }

    QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    if (downloadsPath.isEmpty()) {
        downloadsPath = "C:/Downloads"; // Fallback
    }

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString fileName = downloadsPath + "/video_" + timestamp + ".avi";

    QDir dir;
    if (!dir.exists(downloadsPath)) {
        dir.mkpath(downloadsPath);
    }

    // Get frame size from the first pixmap
    QSize frameSize = m_videoFrames.first().size();
    int width = frameSize.width();
    int height = frameSize.height();

    // Smart frame rate selection: use original rate for standard rates, round for non-standard
    double frameRate = m_videoFPS;
    
    // If it's already a standard rate, keep it
    if (frameRate == 24.0 || frameRate == 25.0 || frameRate == 30.0 || 
        frameRate == 50.0 || frameRate == 60.0) {
        // Keep the original rate
    }
    // Otherwise, round to nearest standard rate to maintain speed while improving compatibility
    else if (frameRate <= 26.0) {
        frameRate = 25.0;  // PAL standard
    } else if (frameRate <= 35.0) {
        frameRate = 30.0;  // NTSC standard
    } else if (frameRate <= 55.0) {
        frameRate = 50.0;  // PAL HD standard
    } else {
        frameRate = 60.0;  // NTSC HD standard
    }
    
    qDebug() << "Saving video with " << m_videoFrames.size() << " frames at " << frameRate << " FPS (original: " << m_videoFPS << " FPS)";

    cv::VideoWriter videoWriter;

    // Use MJPG codec for better compatibility and performance
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MJPG codec for better compatibility
    
    // Open the video writer with MJPG
    if (!videoWriter.open(fileName.toStdString(), fourcc, frameRate, cv::Size(width, height), true)) {
        QMessageBox::critical(this,
                              "Save Video",
                              "Failed to open video writer. Check codecs and file path.");
        qWarning() << "Failed to open video writer for file: " << fileName.toStdString().c_str();
        return;
    }

    // Write each frame with optimized conversion
    for (const QPixmap &pixmap : m_videoFrames) {
        // Convert QPixmap to QImage with optimized format
        QImage image = pixmap.toImage().convertToFormat(QImage::Format_RGB888);
        if (image.isNull()) {
            qWarning() << "Failed to convert QPixmap to QImage during video saving.";
            continue;
        }

        // Convert QImage to OpenCV Mat more efficiently
        cv::Mat frame(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::Mat frameCopy = frame.clone(); // Create a copy to ensure data ownership

        if (frameCopy.empty()) {
            qWarning() << "Failed to convert QImage to cv::Mat during video saving.";
            continue;
        }

        // Convert RGB to BGR (OpenCV expects BGR)
        cv::cvtColor(frameCopy, frameCopy, cv::COLOR_RGB2BGR);

        videoWriter.write(frameCopy);
    }

    videoWriter.release(); // Release the video writer
    QMessageBox::information(this,
                             "Save Video",
                             QString("Video saved successfully at %1 FPS to:\n%2")
                                 .arg(frameRate, 0, 'f', 1)
                                 .arg(fileName));
    qDebug() << "Video saved to: " << fileName;
}
