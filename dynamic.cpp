#include "dynamic.h"
#include "ui_dynamic.h" // This MUST be included for Ui::Dynamic to be defined
#include "iconhover.h"  // Assuming this is a local class for hover effects
#include <QStyle>       // For QStyle::polish
#include <QDebug>
#include <QMouseEvent>
#include <QFile>
#include <QDir>
#include <QImageReader>
#include <QTimer>
#include <QResource>
#include <QCoreApplication>
#include <QScreen>
#include <QApplication>
#include <QVBoxLayout>
#include <QEvent>
#include <QMovie>
#include <QLabel>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QPushButton>
#include <QResizeEvent>
#include <QFrame>       // For ui->fglabel if it's a QFrame

// Assume CV_VERSION is defined from an OpenCV include in brbooth.cpp or a global define
// #ifdef CV_VERSION
// #include <opencv2/opencv.hpp>
// #endif

Dynamic::Dynamic(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic) // Initialize the Ui::Dynamic pointer
    , fullscreenVideoWidget(nullptr)
    , fullscreenPlayer(nullptr)
{
    qDebug() << "Dynamic::Dynamic - Constructor started.";
    ui->setupUi(this); // This call populates the 'ui' object with widgets from dynamic.ui

    // Back button setup
    if (!ui->back) {
        qWarning() << "CRITICAL ERROR: 'back' QPushButton not found in Dynamic UI (from ui->setupUi)!";
    } else {
        QIcon backIcon(":/icons/Icons/normal.svg");
        if (backIcon.isNull()) {
            qWarning() << "WARNING: Back button icon ':/icons/Icons/normal.svg' not found! Setting text fallback.";
        }
        ui->back->setIcon(backIcon);
        ui->back->setIconSize(QSize(100, 100));
        ui->back->setFlat(true);
        qDebug() << "Dynamic::Dynamic - Back button setup complete.";

        // Assuming Iconhover is defined and works as expected
        Iconhover* backButtonHover = new Iconhover(this); // Parent Iconhover to Dynamic widget
        ui->back->installEventFilter(backButtonHover);
        qDebug() << "Dynamic::Dynamic - Back button event filter installed.";

        connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);
        qDebug() << "Dynamic::Dynamic - Back button clicked signal connected.";
    }

    // Debounce timer setup
    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400); // 400ms debounce
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);
    qDebug() << "Dynamic::Dynamic - Debounce timer setup complete.";

    debounceActive = false;
    currentSelectedVideoWidget = nullptr;

    // Debugging resource paths and supported formats
#ifdef CV_VERSION
    qDebug() << "OpenCV Version: "<< CV_VERSION;
#else
    qDebug() << "OpenCV Version: Not defined or available at compile time.";
#endif
    qDebug() << "APP PATH: Current applicationDirPath:" << QCoreApplication::applicationDirPath();
    qDebug() << "APP PATH: Current currentPath:" << QDir::currentPath();

    QDir gifDir(":/gif");
    if (gifDir.exists()) {
        qDebug() << "QRC CHECK: Directory ':/gif' exists in resources. Listing contents:";
        for (const QString& entry : gifDir.entryList(QDir::Files | QDir::NoDotAndDotDot)) {
            qDebug() << "  - " << entry << " (QFile::exists: " << QFile::exists(":/gif/" + entry) << ")";
        }
    } else {
        qWarning() << "QRC CHECK: Directory ':/gif' does NOT exist in resources (prefix not registered or wrong).";
    }

    qDebug() << "Supported Image Formats by QImageReader:";
    for (const QByteArray& format : QImageReader::supportedImageFormats()) {
        qDebug() << "  - " << format;
    }

    // Set up video players (GIFs on buttons)
    setupVideoPlayers();
    qDebug() << "Dynamic::Dynamic - setupVideoPlayers() called.";

    // Fullscreen video widget and player setup
    fullscreenVideoWidget = new QVideoWidget(this);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenVideoWidget->setStyleSheet("background-color: transparent; border: 5px solid #FFC20F; border-radius: 8px;");
    fullscreenVideoWidget->hide(); // Start hidden
    qDebug() << "Dynamic::Dynamic - Fullscreen video widget setup complete.";

    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);
    fullscreenVideoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio); // Important for video scaling
    qDebug() << "Dynamic::Dynamic - Fullscreen media player setup complete.";

    // Connect fullscreen player to loop at end of media
    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            fullscreenPlayer->setPosition(0); // Reset to beginning
            fullscreenPlayer->play();         // Play again
        }
    });
    qDebug() << "Dynamic::Dynamic - Fullscreen player media status connected for looping.";

    // Install event filters for click detection on fullscreen video and the main widget
    fullscreenVideoWidget->installEventFilter(this);
    this->installEventFilter(this);
    qDebug() << "Dynamic::Dynamic - Fullscreen video widget and main widget event filters installed.";

    qDebug() << "Dynamic::Dynamic - Constructor finished.";
}

Dynamic::~Dynamic()
{
    qDebug() << "Dynamic::~Dynamic - Destructor started.";

    // Stop all GIF movies before cleanup
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>();
            if (gifLabel) {
                QMovie* movie = gifLabel->movie();
                if (movie && movie->state() == QMovie::Running) {
                    movie->stop();
                    qDebug() << "Dynamic::~Dynamic - Stopped GIF movie for button:" << button->objectName();
                }
            }
        }
    }

    // Clean up fullscreen player and widget
    if (fullscreenPlayer) {
        fullscreenPlayer->stop();
        delete fullscreenPlayer;
        fullscreenPlayer = nullptr;
        qDebug() << "Dynamic::~Dynamic - fullscreenPlayer deleted.";
    }
    if (fullscreenVideoWidget) {
        delete fullscreenVideoWidget;
        fullscreenVideoWidget = nullptr;
        qDebug() << "Dynamic::~Dynamic - fullscreenVideoWidget deleted.";
    }

    delete ui; // Delete the generated UI object, which will delete child widgets
    qDebug() << "Dynamic::~Dynamic - Destructor finished.";
}

void Dynamic::setupVideoPlayers()
{
    qDebug() << "Dynamic::setupVideoPlayers - Started.";

    // Define video paths for actual playback
    QStringList actualVideoPaths;
    actualVideoPaths << "qrc:/videos/videos/video1.mp4"
                     << "qrc:/videos/videos/video2.mp4"
                     << "qrc:/videos/videos/video3.mp4"
                     << "qrc:/videos/videos/video4.mp4"
                     << "qrc:/videos/videos/video5.mp4";

    // Define GIF paths for thumbnails
    QStringList gifPaths;
    gifPaths << ":/gif/test1.gif"
             << ":/gif/test2.gif"
             << ":/gif/test3.gif"
             << ":/gif/test4.gif"
             << ":/gif/test5.gif";

    // List of QPushButton pointers from the UI
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    // Loop through each button to set up its GIF thumbnail and click behavior
    for (int i = 0; i < buttons.size(); ++i) {
        QPushButton* button = buttons[i];
        if (!button) {
            qWarning() << "Button at index" << i << "is null, skipping setup.";
            continue;
        }

        qDebug() << "Dynamic::setupVideoPlayers - Processing button:" << button->objectName();

        // Store the actual video path as a property on the button
        button->setProperty("actualVideoPath", actualVideoPaths.at(i));

        // Connect the button's clicked signal
        connect(button, &QPushButton::clicked, this, [this, button]() {
            if (debounceActive) return; // Prevent rapid double clicks
            debounceActive = true;
            debounceTimer->start(); // Start debounce timer
            processVideoClick(button);
        });

        // Clear button text to ensure GIF is fully visible
        button->setText("");

        // Create QLabel to display the GIF as a child of the button
        QLabel* gifLabel = new QLabel(button);
        gifLabel->setObjectName(QString("gifLabel_for_%1").arg(button->objectName()));

        // Configure QLabel for GIF
        gifLabel->setScaledContents(true);
        gifLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true); // Important: Clicks pass through to the button
        gifLabel->setMouseTracking(false);
        gifLabel->lower(); // Place behind any potential button text (though we cleared it)

        // Set initial geometry with a small margin for the button's border
        const int margin = 5;
        gifLabel->setGeometry(margin, margin, button->width() - 2 * margin, button->height() - 2 * margin);

        qDebug() << "Dynamic::setupVideoPlayers - Button" << button->objectName() << "size:" << button->size();
        qDebug() << "Dynamic::setupVideoPlayers - gifLabel geometry (initial):" << gifLabel->geometry();

        // Create and set QMovie for the GIF
        QMovie* gifMovie = new QMovie(gifPaths.at(i), QByteArray(), gifLabel);
        gifLabel->setMovie(gifMovie);

        // Debug GIF loading status
        qDebug() << "Dynamic::setupVideoPlayers - Loading GIF:" << gifPaths.at(i);
        qDebug() << "Dynamic::setupVideoPlayers - QMovie isValid:" << gifMovie->isValid();
        qDebug() << "Dynamic::setupVideoPlayers - QMovie frameCount:" << gifMovie->frameCount();
        qDebug() << "Dynamic::setupVideoPlayers - QMovie lastErrorString:" << gifMovie->lastErrorString();

        if (gifMovie->isValid()) {
            gifMovie->start(); // Start the looping GIF animation
            gifLabel->show();
            gifLabel->raise();
            qDebug() << "Dynamic::setupVideoPlayers - Successfully loaded and started GIF for" << button->objectName();
        } else {
            qWarning() << "ERROR: QMovie could not load GIF for" << button->objectName() << ":" << gifMovie->lastErrorString();
            gifLabel->setText("GIF Error"); // Display an error message if GIF fails to load
            gifLabel->setAlignment(Qt::AlignCenter);
            gifLabel->setStyleSheet("color: red; background-color: lightgray; border: 1px solid red;");
            gifLabel->show();
        }
    }
    qDebug() << "Dynamic::setupVideoPlayers - Finished.";
}

void Dynamic::onDynamicPageShown()
{
    qDebug() << "Dynamic::onDynamicPageShown - Starting GIFs and updating geometry.";

    // Ensure GIF labels are correctly sized for current button sizes
    updateGifLabelsGeometry();

    // Loop through buttons to ensure all GIFs are running and visible
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>();
            if (gifLabel) {
                QMovie* movie = gifLabel->movie();
                if (movie && movie->isValid() && movie->state() != QMovie::Running) {
                    movie->start(); // Start GIF if not already running
                    qDebug() << "Dynamic::onDynamicPageShown - Started GIF movie for:" << button->objectName();
                }
                gifLabel->show(); // Ensure label is visible
                gifLabel->raise(); // Ensure it's on top of other button elements
                qDebug() << "Dynamic::onDynamicPageShown - gifLabel " << gifLabel->objectName() << " isVisible:" << gifLabel->isVisible();
            }
        }
    }
}

void Dynamic::updateGifLabelsGeometry()
{
    qDebug() << "Dynamic::updateGifLabelsGeometry - Started.";

    // List of buttons to update
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    // Apply new geometry to each GIF label based on its parent button's current size
    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>(); // Find the child QLabel
            if (gifLabel) {
                const int margin = 5;
                QRect newGeometry(margin, margin, button->width() - 2 * margin, button->height() - 2 * margin);
                gifLabel->setGeometry(newGeometry);
                qDebug() << "Dynamic::updateGifLabelsGeometry - Updated" << button->objectName() << "gifLabel geometry to:" << newGeometry;
            }
        }
    }
    qDebug() << "Dynamic::updateGifLabelsGeometry - Finished.";
}

void Dynamic::resetPage()
{
    qDebug() << "Dynamic::resetPage - Started.";
    hideOverlayVideo(); // Hides the fullscreen video and restarts GIFs

    // Remove highlight from all buttons
    if (ui->videoButton1) applyHighlightStyle(ui->videoButton1, false);
    if (ui->videoButton2) applyHighlightStyle(ui->videoButton2, false);
    if (ui->videoButton3) applyHighlightStyle(ui->videoButton3, false);
    if (ui->videoButton4) applyHighlightStyle(ui->videoButton4, false);
    if (ui->videoButton5) applyHighlightStyle(ui->videoButton5, false);

    currentSelectedVideoWidget = nullptr; // Clear current selection
    qDebug() << "Dynamic::resetPage - Finished.";
}

void Dynamic::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event); // Call base class implementation
    qDebug() << "Dynamic::resizeEvent - Started.";

    // Update geometry for all GIF labels inside buttons
    updateGifLabelsGeometry();

    // If the fullscreen video is visible, ensure it also resizes correctly
    if (fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
        QWidget* videoGridContent = ui->videoGridContent; // This is the widget that defines the area for the fullscreen video
        if (videoGridContent) {
            fullscreenVideoWidget->setGeometry(videoGridContent->geometry());
            qDebug() << "Dynamic::resizeEvent - Resized fullscreenVideoWidget to:" << videoGridContent->geometry();
        }
    }
    qDebug() << "Dynamic::resizeEvent - Finished.";
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    // Handle mouse button press events
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            qDebug() << "Dynamic::eventFilter - MouseButtonPress detected.";

            // If the click is on the fullscreen video widget and it's visible
            if (obj == fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
                qDebug() << "Dynamic::eventFilter - Click on fullscreen video widget. Emitting videoSelectedAndConfirmed().";
                fullscreenPlayer->stop(); // Stop the preview
                emit videoSelectedAndConfirmed(); // Signal that video is confirmed for capture
                hideOverlayVideo(); // Hide the overlay and restart GIFs
                return true; // Event handled
            }

            // If the click is on the main Dynamic widget (this) while fullscreen video is visible
            // This handles clicks outside the explicit video area, back button, etc.
            if (obj == this && fullscreenVideoWidget->isVisible()) {
                QPoint clickPos = mouseEvent->pos();
                // Get geometries of relevant UI elements to exclude them from "outside" clicks
                QRect fgLabelRect = ui->fglabel ? ui->fglabel->geometry() : QRect();
                QRect backButtonRect = ui->back ? ui->back->geometry() : QRect();
                QRect videoGridContentRect = ui->videoGridContent ? ui->videoGridContent->geometry() : QRect();

                qDebug() << "Dynamic::eventFilter - Click on main widget while fullscreen visible.";
                qDebug() << "Click pos:" << clickPos << "Fullscreen rect:" << fullscreenVideoWidget->geometry()
                         << "fgLabel rect:" << fgLabelRect << "Back button rect:" << backButtonRect
                         << "Video Grid Content rect:" << videoGridContentRect;

                // If the click is outside the fullscreen video, fgLabel, back button, and video grid area
                if (!fullscreenVideoWidget->geometry().contains(clickPos) &&
                    !fgLabelRect.contains(clickPos) &&
                    !backButtonRect.contains(clickPos) &&
                    !videoGridContentRect.contains(clickPos))
                {
                    qDebug() << "Dynamic::eventFilter - Click outside interactive areas. Resetting page.";
                    resetPage(); // Dismiss overlay, restart GIFs, and reset highlights
                    return true; // Event handled
                }
            }
        }
    }
    return QWidget::eventFilter(obj, event); // Pass event to base class or next event filter
}

void Dynamic::resetDebounce()
{
    qDebug() << "Dynamic::resetDebounce - Debounce reset.";
    debounceActive = false;
}

void Dynamic::applyHighlightStyle(QObject *obj, bool highlight)
{
    qDebug() << "Dynamic::applyHighlightStyle - Applying highlight:" << highlight << "to object:" << obj->objectName();
    if (obj) {
        obj->setProperty("selected", highlight); // Set custom property for stylesheet
        if (QPushButton *button = qobject_cast<QPushButton *>(obj)) {
            button->style()->polish(button); // Repolish style to apply changes
            button->update();                // Request redraw
            qDebug() << "Dynamic::applyHighlightStyle - Polished and updated button:" << button->objectName();
        } else {
            qWarning() << "Dynamic::applyHighlightStyle - Object is not a QPushButton, cannot apply style property.";
        }
    }
}

void Dynamic::on_back_clicked()
{
    qDebug() << "Dynamic::on_back_clicked - Back button clicked.";
    if (fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
        hideOverlayVideo(); // If fullscreen video is showing, hide it first
    } else {
        emit backtoLandingPage(); // Otherwise, go back to landing page
    }
}

void Dynamic::processVideoClick(QObject *buttonObj)
{
    qDebug() << "Dynamic::processVideoClick - Started for button object:" << buttonObj->objectName();
    QPushButton* clickedButton = qobject_cast<QPushButton*>(buttonObj);
    if (!clickedButton) {
        qWarning() << "processVideoClick received null or invalid button object.";
        return;
    }

    qDebug() << "Dynamic::processVideoClick - Clicked button:" << clickedButton->objectName();

    // Remove highlight from previous selection, apply to current
    if (currentSelectedVideoWidget && currentSelectedVideoWidget != clickedButton) {
        applyHighlightStyle(currentSelectedVideoWidget, false);
    }
    currentSelectedVideoWidget = clickedButton;
    applyHighlightStyle(currentSelectedVideoWidget, true);

    QString actualVideoPath = clickedButton->property("actualVideoPath").toString();
    qDebug() << "Dynamic::processVideoClick - Actual video path:" << actualVideoPath;

    if (!actualVideoPath.isEmpty()) {
        // Stop and hide all GIFs before showing the fullscreen video
        QList<QPushButton*> buttons;
        buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
        for (QPushButton* button : buttons) {
            if (button) {
                QLabel* gifLabel = button->findChild<QLabel*>();
                if (gifLabel) {
                    QMovie* movie = gifLabel->movie();
                    if (movie && movie->state() == QMovie::Running) {
                        movie->stop();
                        qDebug() << "Dynamic::processVideoClick - Stopped GIF movie for:" << button->objectName();
                    }
                    gifLabel->hide(); // Hide the GIF label
                    qDebug() << "Dynamic::processVideoClick - Hidden GIF label " << gifLabel->objectName() << " isVisible:" << gifLabel->isVisible();
                }
            }
        }

        showOverlayVideo(actualVideoPath); // Display the fullscreen video
        fullscreenPlayer->play();
        qDebug() << "Playing actual video:" << actualVideoPath << " as an overlay.";
    } else {
        qWarning() << "Actual video path not found for clicked button:" << clickedButton->objectName();
    }
    // Debounce timer continues to run to prevent accidental double clicks on thumbnails.
    qDebug() << "Dynamic::processVideoClick - Finished.";
}

void Dynamic::onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status)
{
    Q_UNUSED(status); // Suppress unused parameter warning
    qDebug() << "Dynamic::onPlayerMediaStatusChanged - Media status changed to:" << status;
}

void Dynamic::showOverlayVideo(const QString& videoPath)
{
    qDebug() << "Dynamic::showOverlayVideo - Showing overlay for video:" << videoPath;
    QWidget* videoGridContent = ui->videoGridContent; // Assuming this is the parent area for the fullscreen video
    if (!fullscreenPlayer || !fullscreenVideoWidget || !videoGridContent) {
        qWarning() << "showOverlayVideo: Essential components are null (player, widget, or videoGridContent).";
        return;
    }

    // Position and size the fullscreen video widget to cover the desired area
    QRect targetRect = videoGridContent->geometry();
    fullscreenVideoWidget->setGeometry(targetRect);
    fullscreenPlayer->setSource(QUrl(videoPath));
    fullscreenVideoWidget->show();
    fullscreenVideoWidget->raise(); // Bring to front
    qDebug() << "Dynamic::showOverlayVideo - Fullscreen video widget shown.";
    qDebug() << "Dynamic::showOverlayVideo - fullscreenVideoWidget isVisible:" << fullscreenVideoWidget->isVisible();

    // Optionally disable the back button while the video preview is active
    QPushButton* backButton = ui->back;
    if (backButton) {
        // backButton->setEnabled(false); // Uncomment to prevent navigating back while previewing
        qDebug() << "Dynamic::showOverlayVideo - Back button state adjusted if needed.";
    }
}

void Dynamic::hideOverlayVideo()
{
    qDebug() << "Dynamic::hideOverlayVideo - Hiding overlay video.";
    if (!fullscreenPlayer || !fullscreenVideoWidget) {
        qWarning() << "hideOverlayVideo: Essential components are null (player or widget).";
        return;
    }

    fullscreenPlayer->stop();
    fullscreenPlayer->setSource(QUrl()); // Clear current media source
    fullscreenVideoWidget->hide();
    qDebug() << "Dynamic::hideOverlayVideo - Fullscreen video widget hidden.";
    qDebug() << "Dynamic::hideOverlayVideo - fullscreenVideoWidget isVisible:" << fullscreenVideoWidget->isVisible();

    // Reset selection highlight on all buttons
    if (ui->videoButton1) applyHighlightStyle(ui->videoButton1, false);
    if (ui->videoButton2) applyHighlightStyle(ui->videoButton2, false);
    if (ui->videoButton3) applyHighlightStyle(ui->videoButton3, false);
    if (ui->videoButton4) applyHighlightStyle(ui->videoButton4, false);
    if (ui->videoButton5) applyHighlightStyle(ui->videoButton5, false);

    currentSelectedVideoWidget = nullptr; // Clear selection
    qDebug() << "Dynamic::hideOverlayVideo - Selection highlight reset.";

    // Restart and show all GIF thumbnails
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>();
            if (gifLabel) {
                QMovie* movie = gifLabel->movie();
                if (movie && movie->isValid() && movie->state() != QMovie::Running) {
                    movie->start(); // Restart GIF
                    qDebug() << "Dynamic::hideOverlayVideo - Restarted GIF movie for:" << button->objectName();
                }
                gifLabel->show(); // Ensure GIF label is visible
                gifLabel->raise(); // Ensure it's on top of button elements
                qDebug() << "Dynamic::hideOverlayVideo - gifLabel " << gifLabel->objectName() << " isVisible after show/raise:" << gifLabel->isVisible();
            }
        }
    }

    // Re-enable the back button if it was disabled
    QPushButton* backButton = ui->back;
    if (backButton) {
        backButton->setEnabled(true);
        qDebug() << "Dynamic::hideOverlayVideo - Back button enabled.";
    }
    qDebug() << "Dynamic::hideOverlayVideo - Finished.";
}
