#include "ui/final.h"
#include <QDebug>
#include <QDir>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include <QPainter>
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
    , m_hasComparisonVideos(false) // Initialize comparison flag
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
    // If video frames are loaded, only ensure timer is running (first frame handled in setVideo)
    if (!m_videoFrames.isEmpty()) {
        // We don't call playNextFrame() here anymore since setVideo() handles the first frame
        // and the timer handles subsequent frames
        if (!videoPlaybackTimer->isActive()) {
            // Use the stored video FPS for correct playback speed
            int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / m_videoFPS));
            videoPlaybackTimer->start(playbackIntervalMs);
            qDebug() << "Video timer restarted in refreshDisplay()";
        }
    }
    // If no video frames, but a single image is loaded
    else if (!m_lastLoadedImage.isNull()) {
        // FIX: Display the image as-is from capture interface (preserve user scaling)
        // The image already has the user's desired scaling applied from the capture interface
        QSize imageSize = m_lastLoadedImage.size();
        
        qDebug() << "Final interface displaying image with preserved scaling. Image size:" << imageSize;
        
        // Display the image as-is - DO NOT re-scale to preserve capture interface scaling
        ui->videoLabel->setPixmap(m_lastLoadedImage);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        
        qDebug() << "Image displayed with preserved scaling from capture interface";
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

void Final::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event); // Call the base class implementation
    
    // Reset video playback to the beginning when the final page is shown
    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Final page shown - resetting video to beginning";
        
        // Stop current playback
        if (videoPlaybackTimer->isActive()) {
            videoPlaybackTimer->stop();
        }
        
        // Reset to frame 0
        m_currentFrameIndex = 0;
        
        // FIX: Display the first frame as-is (preserve capture interface scaling)
        QPixmap firstFrame = m_videoFrames.at(0);
        QSize frameSize = firstFrame.size();
        
        qDebug() << "Displaying first frame with preserved scaling. Frame size:" << frameSize;
        
        // Display frame as-is - DO NOT re-scale to preserve capture interface scaling
        ui->videoLabel->setPixmap(firstFrame);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        
        // Restart the timer for continuous playback from the beginning
        int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / m_videoFPS));
        videoPlaybackTimer->start(playbackIntervalMs);
        
        qDebug() << "Video reset to frame 0 and playback restarted with preserved scaling";
    }
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

void Final::setImageWithComparison(const QPixmap &image, const QPixmap &originalImage)
{
    // Stop video playback if active
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames.clear(); // Clear any existing video frames
    m_currentFrameIndex = 0;
    m_lastLoadedImage = image; // Store the lighting-corrected image for display
    m_originalImage = originalImage; // Store the original image without lighting correction
    m_hasComparisonImages = true; // We have both versions

    qDebug() << "Final interface received comparison images - corrected:" << image.size() << "original:" << originalImage.size();

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
    m_currentFrameIndex = 0; // Start from the beginning
    m_videoFPS = fps; // Store the FPS for debugging and potential future use
    m_lastLoadedImage = QPixmap(); // Clear last image if switching to video

    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Playing back video with " << m_videoFrames.size() << " frames at " << fps << " FPS.";
        
        // FIX: Display the first frame as-is (preserve capture interface scaling)
        QPixmap firstFrame = m_videoFrames.at(0);
        QSize frameSize = firstFrame.size();
        
        qDebug() << "Displaying first frame with preserved scaling. Frame size:" << frameSize;
        
        // Display frame as-is - DO NOT re-scale to preserve capture interface scaling
        ui->videoLabel->setPixmap(firstFrame);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        
        // Calculate the correct playback interval based on the actual camera FPS
        int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / fps));
        videoPlaybackTimer->start(playbackIntervalMs);
        
        qDebug() << "Video starts from frame 0 with preserved scaling, timer interval:" << playbackIntervalMs << "ms";
    } else {
        qWarning() << "No video frames provided for playback!";
        ui->videoLabel->clear(); // Clear display if no frames
    }
}

void Final::setVideoWithComparison(const QList<QPixmap> &frames, const QList<QPixmap> &originalFrames, double fps)
{
    // Stop any current playback
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames = frames; // Store the processed video frames
    m_originalVideoFrames = originalFrames; // Store the original video frames
    m_hasComparisonVideos = true; // Enable comparison mode
    m_currentFrameIndex = 0; // Start from the beginning
    m_videoFPS = fps; // Store the FPS for debugging and potential future use
    m_lastLoadedImage = QPixmap(); // Clear last image if switching to video

    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Playing back video with comparison - Processed:" << m_videoFrames.size() 
                 << "Original:" << m_originalVideoFrames.size() << "frames at" << fps << "FPS.";
        
        // FIX: Display the first frame as-is (preserve capture interface scaling)
        QPixmap firstFrame = m_videoFrames.at(0);
        QSize frameSize = firstFrame.size();
        
        qDebug() << "Displaying first comparison frame with preserved scaling. Frame size:" << frameSize;
        
        // Display frame as-is - DO NOT re-scale to preserve capture interface scaling
        ui->videoLabel->setPixmap(firstFrame);
        ui->videoLabel->setAlignment(Qt::AlignCenter);
        
        // Calculate the correct playback interval based on the actual camera FPS
        int playbackIntervalMs = qMax(1, static_cast<int>(1000.0 / fps));
        videoPlaybackTimer->start(playbackIntervalMs);
        
        qDebug() << "Video with comparison starts from frame 0 with preserved scaling, timer interval:" << playbackIntervalMs << "ms";
    } else {
        qWarning() << "No video frames provided for comparison playback!";
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

    // Advance to the next frame first (this is called by timer for subsequent frames)
    m_currentFrameIndex++;
    
    // Loop video playback
    if (m_currentFrameIndex >= m_videoFrames.size()) {
        m_currentFrameIndex = 0;
        qDebug() << "Video looped back to frame 0";
    }

    // FIX: Get and display the current frame as-is (preserve capture interface scaling)
    // The frames are already scaled from the capture interface, so display them without modification
    QPixmap currentFrame = m_videoFrames.at(m_currentFrameIndex);
    
    // Display frame as-is - DO NOT re-scale to preserve capture interface scaling
    ui->videoLabel->setPixmap(currentFrame);
    ui->videoLabel->setAlignment(Qt::AlignCenter);
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
        // Create composite image with foreground overlay if present
        QPixmap imageToSave = ui->videoLabel->pixmap();

        if (imageToSave.isNull()) {
            QMessageBox::warning(this, "Save Image", "No image to save.");
            return;
        }
        
        // Composite the foreground overlay on top if present
        if (overlayImageLabel && !overlayImageLabel->isHidden() && !overlayImageLabel->pixmap().isNull()) {
            qDebug() << "Compositing foreground overlay into saved image";
            
            // Create a new pixmap for the composite
            QPixmap composite = imageToSave.copy();
            
            // Draw the foreground overlay on top
            QPainter painter(&composite);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.setRenderHint(QPainter::SmoothPixmapTransform);
            
            // Get the overlay pixmap and scale it to match the base image size
            QPixmap overlayPixmap = overlayImageLabel->pixmap();
            QPixmap scaledOverlay = overlayPixmap.scaled(composite.size(), 
                                                         Qt::IgnoreAspectRatio, 
                                                         Qt::SmoothTransformation);
            
            // Draw the overlay on top of the base image
            painter.drawPixmap(0, 0, scaledOverlay);
            painter.end();
            
            imageToSave = composite;
            qDebug() << "Foreground overlay composited successfully";
        } else {
            qDebug() << "No foreground overlay to composite";
        }

        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            // Fallback for systems without standard download path (e.g., some Linux setups or custom env)
            downloadsPath
                = "C:/Downloads"; // Or a more cross-platform default like QDir::currentPath()
        }

        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath); // Create the directory if it doesn't exist
        }

        if (m_hasComparisonImages) {
            // Save both versions for comparison
            QString correctedFileName = downloadsPath + "/image_lighting_corrected_" + timestamp + ".png";
            QString originalFileName = downloadsPath + "/image_original_" + timestamp + ".png";
            
            bool correctedSaved = imageToSave.save(correctedFileName);
            bool originalSaved = m_originalImage.save(originalFileName);
            
            if (correctedSaved && originalSaved) {
                QMessageBox::information(this,
                                         "Save Images",
                                         QString("Both images saved successfully:\n\n"
                                                "Lighting Corrected: %1\n"
                                                "Original: %2").arg(correctedFileName, originalFileName));
            } else {
                QString errorMsg = "Failed to save: ";
                if (!correctedSaved) errorMsg += "lighting corrected image ";
                if (!originalSaved) errorMsg += "original image ";
                QMessageBox::critical(this, "Save Images", errorMsg);
            }
        } else {
            // Save single image
            QString fileName = downloadsPath + "/image_" + timestamp + ".png";
            
            if (imageToSave.save(fileName)) {
                QMessageBox::information(this,
                                         "Save Image",
                                         QString("Image saved successfully to:\n%1").arg(fileName));
            } else {
                QMessageBox::critical(this, "Save Image", "Failed to save image.");
            }
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
    
    QDir dir;
    if (!dir.exists(downloadsPath)) {
        dir.mkpath(downloadsPath);
    }

    // Handle comparison videos (both enhanced and original)
    if (m_hasComparisonVideos && !m_originalVideoFrames.isEmpty()) {
        QString enhancedFileName = downloadsPath + "/video_lighting_enhanced_" + timestamp + ".avi";
        QString originalFileName = downloadsPath + "/video_original_" + timestamp + ".avi";
        
        qDebug() << "Saving both video versions - Enhanced:" << m_videoFrames.size() 
                 << "Original:" << m_originalVideoFrames.size() << "frames";
        
        bool enhancedSaved = saveVideoFramesToFile(m_videoFrames, enhancedFileName);
        bool originalSaved = saveVideoFramesToFile(m_originalVideoFrames, originalFileName);
        
        if (enhancedSaved && originalSaved) {
            QMessageBox::information(this,
                                     "Save Videos",
                                     QString("Both videos saved successfully:\n\n"
                                             "Enhanced: %1\n"
                                             "Original: %2")
                                         .arg(enhancedFileName)
                                         .arg(originalFileName));
        } else if (enhancedSaved) {
            QMessageBox::warning(this,
                                 "Save Videos", 
                                 QString("Enhanced video saved successfully:\n%1\n\n"
                                         "Failed to save original video.")
                                     .arg(enhancedFileName));
        } else if (originalSaved) {
            QMessageBox::warning(this,
                                 "Save Videos", 
                                 QString("Original video saved successfully:\n%1\n\n"
                                         "Failed to save enhanced video.")
                                     .arg(originalFileName));
        } else {
            QMessageBox::critical(this, "Save Videos", "Failed to save both videos.");
        }
        return;
    }

    // Handle single video (no comparison)
    QString fileName = downloadsPath + "/video_" + timestamp + ".avi";
    bool saved = saveVideoFramesToFile(m_videoFrames, fileName);
    
    if (saved) {
        QMessageBox::information(this, "Save Video", QString("Video saved successfully:\n%1").arg(fileName));
    } else {
        QMessageBox::critical(this, "Save Video", "Failed to save video.");
    }
}

bool Final::saveVideoFramesToFile(const QList<QPixmap> &frames, const QString &fileName)
{
    if (frames.isEmpty()) {
        qWarning() << "No frames to save for file:" << fileName;
        return false;
    }

    // Check if we need to composite foreground overlay
    bool hasOverlay = overlayImageLabel && !overlayImageLabel->isHidden() && !overlayImageLabel->pixmap().isNull();
    QPixmap scaledOverlay;
    
    if (hasOverlay) {
        qDebug() << "Compositing foreground overlay into video frames";
        // Pre-scale the overlay to match frame size for efficiency
        QPixmap overlayPixmap = overlayImageLabel->pixmap();
        QSize frameSize = frames.first().size();
        scaledOverlay = overlayPixmap.scaled(frameSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    } else {
        qDebug() << "No foreground overlay to composite for video";
    }

    // Get frame size from the first pixmap
    QSize frameSize = frames.first().size();
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
    
    qDebug() << "Saving video with " << frames.size() << " frames at " << frameRate << " FPS (original: " << m_videoFPS << " FPS) to:" << fileName;

    cv::VideoWriter videoWriter;

    // Use MJPG codec for better compatibility and performance
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // MJPG codec for better compatibility
    
    // Open the video writer with MJPG
    if (!videoWriter.open(fileName.toStdString(), fourcc, frameRate, cv::Size(width, height), true)) {
        qWarning() << "Failed to open video writer for file: " << fileName;
        return false;
    }

    // Write each frame with optimized conversion
    for (const QPixmap &pixmap : frames) {
        // Composite foreground overlay if present
        QPixmap frameToSave = pixmap;
        if (hasOverlay) {
            frameToSave = pixmap.copy();
            QPainter painter(&frameToSave);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.setRenderHint(QPainter::SmoothPixmapTransform);
            painter.drawPixmap(0, 0, scaledOverlay);
            painter.end();
        }
        
        // Convert QPixmap to QImage with optimized format
        QImage image = frameToSave.toImage().convertToFormat(QImage::Format_RGB888);
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
    qDebug() << "Video saved successfully to:" << fileName;
    return true;
}
