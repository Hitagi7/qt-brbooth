#include "final.h"
#include <QDateTime>
#include <QFileDialog>
#include <QImage>
#include <QMessageBox>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QStyle>
#include <QVBoxLayout> // Might be implicitly used by UI file for buttons
#include "iconhover.h" // Assuming this is your custom class for button hover effects
#include "ui_final.h"  // Generated UI header

// OpenCV includes (ensure your .pro file links to OpenCV libraries)
#include <opencv2/opencv.hpp>

// Explicitly include layouts, although they are in final.h, it's good practice
#include <QGridLayout>
#include <QStackedLayout>

Final::Final(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Final)
    , videoPlaybackTimer(nullptr)
    , m_currentFrameIndex(0)
    , m_stackedLayout(nullptr)
    , m_lastLoadedImage() // Initialize the QPixmap member
{
    ui->setupUi(this); // Initialize UI elements from final.ui

    // Ensure the main Final widget itself has no margins or spacing
    setContentsMargins(0, 0, 0, 0);

    // --- NEW LAYOUT SETUP leveraging final.ui components ---

    // 1. Create the QGridLayout as the main layout for the Final widget
    QGridLayout *mainLayout = new QGridLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0); // No padding for the grid itself
    mainLayout->setSpacing(0);                  // No spacing between grid cells

    // 2. Create the QStackedLayout to stack videoLabel and overlayFinal
    m_stackedLayout = new QStackedLayout; // Initialize the member
    m_stackedLayout->setStackingMode(
        QStackedLayout::StackAll); // Makes all contained widgets visible and layered
    m_stackedLayout->setContentsMargins(0, 0, 0, 0); // No padding inside the stacked layout
    m_stackedLayout->setSpacing(0);                  // No spacing between stacked items

    // 3. Add existing UI widgets to the stacked layout
    m_stackedLayout->addWidget(
        ui->videoLabel); // This will be the background layer (for image/video display)
    m_stackedLayout->addWidget(
        ui->overlayFinal); // This will be the foreground layer (it already contains your buttons)

    // 4. Add the stacked layout to the main grid layout, making it stretch
    mainLayout->addLayout(m_stackedLayout,
                          0,
                          0);        // Place the stacked layout in the first cell (row 0, col 0)
    mainLayout->setRowStretch(0, 1); // Make this row stretchable, so it takes all vertical space
    mainLayout->setColumnStretch(0,
                                 1); // Make this column stretchable, so it takes all horizontal space

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

    // Ensure overlayFinal is on top
    ui->overlayFinal->raise(); // Brings overlayFinal to the front of the stacked layout.
    ui->back->raise();         // Ensure buttons are on top of overlay
    ui->save->raise();         // Ensure buttons are on top of overlay

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
            videoPlaybackTimer->start(1000 / 60); // Start if not already playing
        }
        playNextFrame(); // Display the current frame (and advance for next call)
    }
    // If no video frames, but a single image is loaded
    else if (!m_lastLoadedImage.isNull()) {
        // Scale the stored original image to the current size of ui->videoLabel
        QPixmap scaledImage = m_lastLoadedImage
                                  .scaled(ui->videoLabel->size(),
                                          Qt::IgnoreAspectRatio, // <== IMPORTANT: Stretches to fill
                                          Qt::SmoothTransformation);
        ui->videoLabel->setPixmap(scaledImage);
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
    m_lastLoadedImage = image; // Store the original image

    ui->videoLabel->setText(""); // Clear any previous text

    refreshDisplay(); // Display the image immediately with proper scaling
}

void Final::setVideo(const QList<QPixmap> &frames)
{
    // Stop any current playback
    if (videoPlaybackTimer->isActive()) {
        videoPlaybackTimer->stop();
    }

    m_videoFrames = frames; // Store the list of video frames
    m_currentFrameIndex = 0;
    m_lastLoadedImage = QPixmap(); // Clear last image if switching to video

    if (!m_videoFrames.isEmpty()) {
        qDebug() << "Playing back video with " << m_videoFrames.size() << " frames.";
        videoPlaybackTimer->start(1000 / 60); // Start timer for continuous playback
        refreshDisplay(); // Display the first frame immediately with proper scaling
    } else {
        qWarning() << "No video frames provided for playback!";
        ui->videoLabel->clear(); // Clear display if no frames
    }
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

    // Get the current frame and scale it to the size of ui->videoLabel
    QPixmap currentFrame = m_videoFrames.at(m_currentFrameIndex);
    QPixmap scaledFrame = currentFrame
                              .scaled(ui->videoLabel->size(),
                                      Qt::IgnoreAspectRatio, // <== IMPORTANT: Stretches to fill
                                      Qt::SmoothTransformation);
    ui->videoLabel->setPixmap(scaledFrame);

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

    // Define the FourCC code for the video codec (MJPG is widely supported)
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    // Get frame size from the first pixmap
    QSize frameSize = m_videoFrames.first().size();
    int width = frameSize.width();
    int height = frameSize.height();

    // Assuming video was captured at 60 FPS for 10 seconds, adjust if different
    double frameRate = 60.0;         // The target playback frame rate for the saved video
    if (m_videoFrames.size() > 10) { // If you have enough frames for 10 seconds at 60fps
        // Adjust frameRate based on how many frames collected over 10s or similar,
        // or hardcode based on your capture rate.
        // This 'actualFPS' calculation was previously `(double)m_videoFrames.size() / 10.0;`
        // which implies frames were captured over 10 seconds.
        // It's crucial to match the capture rate or desired playback rate.
    }

    cv::VideoWriter videoWriter;

    // Open the video writer
    // Parameters: filename, fourcc, fps, frame size, isColor
    if (!videoWriter.open(fileName.toStdString(), fourcc, frameRate, cv::Size(width, height), true)) {
        QMessageBox::critical(this,
                              "Save Video",
                              "Failed to open video writer. Check codecs and file path.");
        qWarning() << "Failed to open video writer for file: " << fileName.toStdString().c_str()
                   << " with FOURCC: " << fourcc;
        return;
    }

    // Write each frame
    for (const QPixmap &pixmap : m_videoFrames) {
        QImage image = pixmap.toImage();
        if (image.isNull()) {
            qWarning() << "Failed to convert QPixmap to QImage during video saving.";
            continue;
        }

        cv::Mat frame = QImageToCvMat(image); // Convert QImage to OpenCV Mat
        if (frame.empty()) {
            qWarning() << "Failed to convert QImage to cv::Mat during video saving.";
            continue;
        }

        // Ensure the frame is 3-channel BGR for the video writer (common requirement)
        if (frame.channels() == 4) { // e.g., ARGB32
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else if (frame.channels() == 1) { // e.g., grayscale
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }

        videoWriter.write(frame);
    }

    videoWriter.release(); // Release the video writer
    QMessageBox::information(this,
                             "Save Video",
                             QString("Video saved successfully at %1 FPS to:\n%2")
                                 .arg(frameRate, 0, 'f', 1)
                                 .arg(fileName));
    qDebug() << "Video saved to: " << fileName;
}
