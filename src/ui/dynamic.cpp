#include "ui/dynamic.h"

#include "../../build/Desktop_Qt_6_9_2_MSVC2022_64bit-Debug/ui_dynamic.h"  // Full definition of Ui::Dynamic class
#include "ui/iconhover.h"  // Assuming this is a local class for hover effects
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
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsProxyWidget>
#include <QGraphicsItem>

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

        // Assuming Iconhover is defined and works as expected
        Iconhover* backButtonHover = new Iconhover(this); // Parent Iconhover to Dynamic widget
        ui->back->installEventFilter(backButtonHover);

        connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);
        
        // Ensure back button is visible by default (no background styling)
        ui->back->show();
        ui->back->raise();
        qDebug() << "Dynamic::Dynamic - Back button initialized with original styling";
    }

    // Debounce timer setup
    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400); // 400ms debounce
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);

    debounceActive = false;
    currentSelectedVideoWidget = nullptr;
    m_lastSelectedVideoPath = ""; // Initialize last selected path

    // Debugging resource paths and supported formats
#ifdef CV_VERSION
    qDebug() << "OpenCV Version: "<< CV_VERSION;
#else
    qDebug() << "OpenCV Version: Not defined or available at compile time.";
#endif
    // Set up video players (GIFs on buttons)
    setupVideoPlayers();

    // Create a container widget for the video and overlay
    fullscreenStackWidget = new QWidget(this);
    fullscreenStackWidget->setMinimumSize(QSize(640, 480));
    fullscreenStackWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenStackWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenStackWidget->setStyleSheet("background-color: transparent; border: 5px solid #FFC20F; border-radius: 8px;");
    fullscreenStackWidget->hide(); // Start hidden
    
    // Fullscreen video widget setup
    fullscreenVideoWidget = new QVideoWidget(fullscreenStackWidget);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenVideoWidget->setStyleSheet("background-color: transparent;");
    
    // Make video widget fill the entire container
    fullscreenVideoWidget->setGeometry(fullscreenStackWidget->rect());
    
    qDebug() << "Dynamic::Dynamic - Fullscreen stack widget setup complete.";

    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);
    fullscreenVideoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio); // Important for video scaling
    qDebug() << "Dynamic::Dynamic - Fullscreen media player setup complete.";

    // Create dedicated back button for video preview as a top-level window
    videoPreviewBackButton = new QPushButton();
    
    // Load the icon from resources
    QIcon backIcon(":/icons/Icons/normal.svg");
    if (backIcon.isNull()) {
        qWarning() << "Dynamic::Dynamic - Failed to load back button icon from :/icons/Icons/normal.svg";
        // Fallback: create a text button if icon fails
        videoPreviewBackButton->setText("←");
    } else {
        qDebug() << "Dynamic::Dynamic - Back button icon loaded successfully";
        videoPreviewBackButton->setIcon(backIcon);
        videoPreviewBackButton->setIconSize(QSize(100, 100)); // Match original back button icon size
    }
    
    // Set button properties to match the original back button exactly
    videoPreviewBackButton->setMinimumSize(QSize(100, 80));
    videoPreviewBackButton->setMaximumSize(QSize(200, 80));
    videoPreviewBackButton->setText(""); // No text, just icon
    videoPreviewBackButton->setFlat(true); // Match original button
    
    // Use transparent background styling
    videoPreviewBackButton->setStyleSheet(
        "QPushButton { "
        "    background-color: transparent; "
        "    border: none; "
        "    padding: 0px; "
        "    margin: 0px; "
        "}"
    );
    
    // Make button transparent and add hover effect
    videoPreviewBackButton->setAttribute(Qt::WA_TranslucentBackground, true);
    
    // Set window flags to ensure it stays on top but only when video is active
    videoPreviewBackButton->setWindowFlags(Qt::WindowStaysOnTopHint | Qt::FramelessWindowHint | Qt::Tool);
    
    // Connect to application focus changes to hide button when app loses focus
    connect(qApp, &QApplication::applicationStateChanged, this, [this](Qt::ApplicationState state) {
        if (state != Qt::ApplicationActive) {
            // Application is not active, hide the back button
            if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
                videoPreviewBackButton->hide();
            }
        } else {
            // Application is active, show the back button if video is playing
            if (videoPreviewBackButton && fullscreenStackWidget && fullscreenStackWidget->isVisible()) {
                videoPreviewBackButton->show();
            }
        }
    });
    
    // Also connect to focus change events
    connect(qApp, &QApplication::focusChanged, this, [this](QWidget* old, QWidget* now) {
        Q_UNUSED(old)
        if (now == nullptr || !this->isActiveWindow()) {
            // No widget has focus or this window is not active, hide the back button
            if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
                videoPreviewBackButton->hide();
            }
        } else if (this->isActiveWindow() && fullscreenStackWidget && fullscreenStackWidget->isVisible()) {
            // This window is active and video is playing, show the back button
            if (videoPreviewBackButton) {
                videoPreviewBackButton->show();
            }
        }
    });
    
    // Add hover effect using Iconhover (same as original back button)
    Iconhover* videoPreviewBackButtonHover = new Iconhover(this);
    videoPreviewBackButton->installEventFilter(videoPreviewBackButtonHover);
    
    // Install event filter on main window to catch focus events
    this->installEventFilter(this);
    
    // Position the back button (will be positioned when video is shown)
    
    // Debug: Check button properties after creation
    qDebug() << "Dynamic::Dynamic - Button created with size:" << videoPreviewBackButton->size();
    qDebug() << "Dynamic::Dynamic - Button stylesheet:" << videoPreviewBackButton->styleSheet();
    qDebug() << "Dynamic::Dynamic - Button is visible:" << videoPreviewBackButton->isVisible();
    
    videoPreviewBackButton->hide(); // Start hidden
    connect(videoPreviewBackButton, &QPushButton::clicked, this, &Dynamic::onVideoPreviewBackClicked);

    // Connect fullscreen player to loop at end of media
    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        qDebug() << "Dynamic::fullscreenPlayer - Media status changed to:" << status;
        if (status == QMediaPlayer::EndOfMedia) {
            qDebug() << "Dynamic::fullscreenPlayer - End of media reached, looping...";
            fullscreenPlayer->setPosition(0); // Reset to beginning
            fullscreenPlayer->play();         // Play again
        } else if (status == QMediaPlayer::InvalidMedia) {
            qWarning() << "Dynamic::fullscreenPlayer - Invalid media detected!";
        } else if (status == QMediaPlayer::LoadedMedia) {
            qDebug() << "Dynamic::fullscreenPlayer - Media loaded successfully";
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
                }
            }
        }
    }

    // Clean up fullscreen player and widget
    if (fullscreenPlayer) {
        fullscreenPlayer->stop();
        delete fullscreenPlayer;
        fullscreenPlayer = nullptr;
    }
    if (fullscreenVideoWidget) {
        delete fullscreenVideoWidget;
        fullscreenVideoWidget = nullptr;
    }

    delete ui; // Delete the generated UI object, which will delete child widgets
}

void Dynamic::setupVideoPlayers()
{
    // Get the application directory and navigate to build directory
    QString appDir = QCoreApplication::applicationDirPath();
    // Go up one level from debug/ to the build directory
    QString buildDir = QDir(appDir).absoluteFilePath("..");
    
    // Template videos for both preview and capture interface
    // Use the correct build directory path: build/Desktop_Qt_6_9_2_MSVC2022_64bit-Debug/templates/dynamic/
    QStringList actualVideoPaths;
    actualVideoPaths << buildDir + "/templates/dynamic/vidtemplate1.mp4"
                     << buildDir + "/templates/dynamic/vidtemplate2.mp4"
                     << buildDir + "/templates/dynamic/vidtemplate3.mp4"
                     << buildDir + "/templates/dynamic/vidtemplate4.mp4"
                     << buildDir + "/templates/dynamic/vidtemplate5.mp4";

    // Define GIF paths for thumbnails
    QStringList gifPaths;
    gifPaths << "gif templates/dynamicbg1.gif"
             << "gif templates/dynamicbg2.gif"
             << "gif templates/dynamicbg3.gif"
             << "gif templates/dynamicbg4.gif"
             << "gif templates/dynamicbg5.gif";

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
        QString videoPath = actualVideoPaths.at(i);
        button->setProperty("actualVideoPath", videoPath);
        
        // Verify the video file exists
        if (QFile::exists(videoPath)) {
            qDebug() << "Dynamic::setupVideoPlayers - Video file exists:" << videoPath;
        } else {
            qWarning() << "Dynamic::setupVideoPlayers - Video file NOT found:" << videoPath;
        }

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

        // Try to find the GIF file in various possible locations
        QString originalGifPath = gifPaths.at(i);
        QStringList possiblePaths;
        possiblePaths << originalGifPath
                      << QDir::currentPath() + "/" + originalGifPath
                      << QCoreApplication::applicationDirPath() + "/" + originalGifPath
                      << QCoreApplication::applicationDirPath() + "/../" + originalGifPath
                      << QCoreApplication::applicationDirPath() + "/../../" + originalGifPath
                      << "../" + originalGifPath
                      << "../../" + originalGifPath
                      << "../../../" + originalGifPath;
        
        // Add the known working path as a fallback
        QString knownPath = QString("C:/Users/dorot/Documents/qt-brbooth/gif templates/dynamicbg%1.gif").arg(i + 1);
        possiblePaths << knownPath;
        qDebug() << "Dynamic::setupVideoPlayers - Added known working path:" << knownPath;
        
        // Try to find project root by looking for qt-brbooth.pro file
        QString projectRoot;
        QStringList searchDirs;
        searchDirs << QDir::currentPath()
                   << QCoreApplication::applicationDirPath()
                   << QCoreApplication::applicationDirPath() + "/.."
                   << QCoreApplication::applicationDirPath() + "/../.."
                   << "../"
                   << "../../"
                   << "../../../";
        
        for (const QString& searchDir : searchDirs) {
            if (QFile::exists(searchDir + "/qt-brbooth.pro")) {
                projectRoot = searchDir;
                qDebug() << "Dynamic::setupVideoPlayers - Found project root at:" << projectRoot;
                break;
            }
        }
        
        if (!projectRoot.isEmpty()) {
            possiblePaths.prepend(projectRoot + "/" + originalGifPath);
            qDebug() << "Dynamic::setupVideoPlayers - Added project root path:" << projectRoot + "/" + originalGifPath;
        }
        
        // Debug: Print all paths we're going to try
        qDebug() << "Dynamic::setupVideoPlayers - All paths to try for GIF" << (i + 1) << ":";
        for (int j = 0; j < possiblePaths.size(); ++j) {
            qDebug() << "  " << j << ":" << possiblePaths[j] << "(exists:" << QFile::exists(possiblePaths[j]) << ")";
        }
        
        QString validGifPath;
        for (const QString& path : possiblePaths) {
            if (QFile::exists(path)) {
                validGifPath = path;
                qDebug() << "Dynamic::setupVideoPlayers - Found GIF at:" << validGifPath;
                break;
            }
        }
        
        if (validGifPath.isEmpty()) {
            qWarning() << "Dynamic::setupVideoPlayers - GIF file not found for" << button->objectName() << "in any location:";
            for (const QString& path : possiblePaths) {
                qWarning() << "  -" << path;
            }
            // Create a placeholder with button number
            gifLabel->setText(QString("GIF %1\nMissing").arg(i + 1));
            gifLabel->setAlignment(Qt::AlignCenter);
            gifLabel->setStyleSheet("color: white; background-color: #FF6B6B; border-radius: 8px; font-weight: bold;");
            gifLabel->show();
        } else {
            // Create and set QMovie for the GIF
            QMovie* gifMovie = new QMovie(validGifPath, QByteArray(), gifLabel);
            gifLabel->setMovie(gifMovie);

            // Debug GIF loading status
            qDebug() << "Dynamic::setupVideoPlayers - Loading GIF:" << validGifPath;
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
                gifLabel->setText(QString("GIF %1\nError").arg(i + 1)); // Display an error message if GIF fails to load
                gifLabel->setAlignment(Qt::AlignCenter);
                gifLabel->setStyleSheet("color: white; background-color: #FFB74D; border-radius: 8px; font-weight: bold;");
                gifLabel->show();
            }
        }
    }
    qDebug() << "Dynamic::setupVideoPlayers - Finished.";
}

void Dynamic::onDynamicPageShown()
{
    qDebug() << "Dynamic::onDynamicPageShown - Starting GIFs and updating geometry.";

    // Ensure back button is visible
    if (ui->back) {
        ui->back->show();
        ui->back->raise();
        qDebug() << "Dynamic::onDynamicPageShown - Back button made visible";
    }

    // Ensure GIF labels are correctly sized for current button sizes
    updateGifLabelsGeometry();
    
    // Restore previous selection if any
    restoreSelection();

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

    // Ensure back button is visible
    if (ui->back) {
        ui->back->show();
        ui->back->raise();
        qDebug() << "Dynamic::resetPage - Back button made visible";
    }

    // Remove highlight from all buttons
    if (ui->videoButton1) applyHighlightStyle(ui->videoButton1, false);
    if (ui->videoButton2) applyHighlightStyle(ui->videoButton2, false);
    if (ui->videoButton3) applyHighlightStyle(ui->videoButton3, false);
    if (ui->videoButton4) applyHighlightStyle(ui->videoButton4, false);
    if (ui->videoButton5) applyHighlightStyle(ui->videoButton5, false);

    currentSelectedVideoWidget = nullptr; // Clear current selection
    // NOTE: Do NOT clear m_lastSelectedVideoPath here to preserve selection across navigation
    // m_lastSelectedVideoPath should only be cleared when user explicitly changes selection or leaves dynamic mode
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
        QWidget* videoGridContent = this; // Use the main widget as the video area
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
                emit videoSelectedAndConfirmed(m_selectedVideoPath); // Signal that video is confirmed for capture
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
                QRect videoGridContentRect = this->geometry();

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
    
    // Handle focus events for the main window
    if (obj == this) {
        if (event->type() == QEvent::WindowActivate) {
            // Window became active, show back button if video is playing
            if (videoPreviewBackButton && fullscreenStackWidget && fullscreenStackWidget->isVisible()) {
                videoPreviewBackButton->show();
            }
        } else if (event->type() == QEvent::WindowDeactivate) {
            // Window lost focus, hide back button
            if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
                videoPreviewBackButton->hide();
            }
        } else if (event->type() == QEvent::FocusOut) {
            // Focus lost, hide back button
            if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
                videoPreviewBackButton->hide();
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
    emit backtoLandingPage(); // Always go back to landing page from template selection
}

void Dynamic::onVideoPreviewBackClicked()
{
    hideOverlayVideo(); // Hide video and return to template selection
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

    // Store the selected video path for persistence across navigation
    if (!actualVideoPath.isEmpty()) {
        // The actualVideoPath is already an absolute path from setupVideoPlayers
        m_lastSelectedVideoPath = actualVideoPath;
        qDebug() << "Dynamic::processVideoClick - Updated m_lastSelectedVideoPath to:" << m_lastSelectedVideoPath;
    }

    if (!actualVideoPath.isEmpty()) {
        // Stop and hide all GIFs before showing the fullscreen video (asynchronously)
        QTimer::singleShot(0, [this, actualVideoPath]() {
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
        });
    } else {
        qWarning() << "Actual video path not found for clicked button:" << clickedButton->objectName();
    }
    // Debounce timer continues to run to prevent accidental double clicks on thumbnails.
    qDebug() << "Dynamic::processVideoClick - Finished.";
}


void Dynamic::showOverlayVideo(const QString& videoPath)
{
    qDebug() << "Dynamic::showOverlayVideo - Showing overlay for video:" << videoPath;
    QWidget* videoGridContent = this; // Use the main widget as the video area
    if (!fullscreenPlayer || !fullscreenVideoWidget || !videoGridContent) {
        qWarning() << "showOverlayVideo: Essential components are null (player, widget, or videoGridContent).";
        return;
    }

    // Position and size the container widget to cover the desired area
    QRect targetRect = videoGridContent->geometry();
    fullscreenStackWidget->setGeometry(targetRect);
    
    // Ensure video widget fills the entire container
    fullscreenVideoWidget->setGeometry(fullscreenStackWidget->rect());
    
    // The videoPath is already absolute from setupVideoPlayers
    m_selectedVideoPath = videoPath;
    m_lastSelectedVideoPath = videoPath; // Store for persistence
    
    // Verify the file exists before trying to play it
    if (!QFile::exists(videoPath)) {
        qWarning() << "Dynamic::showOverlayVideo - Video file does not exist:" << videoPath;
        return;
    }
    
    qDebug() << "Dynamic::showOverlayVideo - Loading video from:" << videoPath;
    fullscreenPlayer->setSource(QUrl::fromLocalFile(videoPath));
    
    // Show the stack widget (which contains both video and back button)
    fullscreenStackWidget->show();
    fullscreenStackWidget->raise(); // Bring to front

    // Hide original back button and show video preview back button
    if (ui->back) {
        ui->back->hide();
    }
    
    // Position the back button at the original back button location using global coordinates
    QPoint originalBackButtonPos = ui->back->pos();
    QPoint globalPos = this->mapToGlobal(originalBackButtonPos);
    videoPreviewBackButton->move(globalPos);
    
    // Show the floating button
    videoPreviewBackButton->show();
    videoPreviewBackButton->setVisible(true);
    
    // Force the button to be on top within its parent widget
    videoPreviewBackButton->raise();
    videoPreviewBackButton->setAttribute(Qt::WA_AlwaysStackOnTop, true);
    
    // Force a repaint to ensure visibility
    videoPreviewBackButton->repaint();
    
    // Additional force-to-top using QTimer to ensure it stays on top
    QTimer::singleShot(50, [this]() {
        if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
            videoPreviewBackButton->raise();
        }
    });
    
    // Debug: Check button state
    qDebug() << "Dynamic::showOverlayVideo - Button positioned at global coordinates:" << globalPos;
    qDebug() << "Dynamic::showOverlayVideo - Button visible:" << videoPreviewBackButton->isVisible();
    qDebug() << "Dynamic::showOverlayVideo - Button geometry:" << videoPreviewBackButton->geometry();
    qDebug() << "Dynamic::showOverlayVideo - Button parent:" << videoPreviewBackButton->parent();
    qDebug() << "Dynamic::showOverlayVideo - Button size:" << videoPreviewBackButton->size();
    qDebug() << "Dynamic::showOverlayVideo - Button stylesheet:" << videoPreviewBackButton->styleSheet();
    qDebug() << "Dynamic::showOverlayVideo - Button text:" << videoPreviewBackButton->text();
    qDebug() << "Dynamic::showOverlayVideo - Fullscreen stack widget visible:" << fullscreenStackWidget->isVisible();
    qDebug() << "Dynamic::showOverlayVideo - Fullscreen stack widget geometry:" << fullscreenStackWidget->geometry();
    qDebug() << "Dynamic::showOverlayVideo - Button window flags:" << videoPreviewBackButton->windowFlags();
    qDebug() << "Dynamic::showOverlayVideo - Button attributes:" << videoPreviewBackButton->testAttribute(Qt::WA_TransparentForMouseEvents) << videoPreviewBackButton->testAttribute(Qt::WA_NoSystemBackground);
    
    // Button styling is already set in constructor, just ensure it's visible
    // The stack layout approach ensures proper layering above the video

    // Optionally disable the back button while the video preview is active
    QPushButton* backButton = ui->back;
    if (backButton) {
        // backButton->setEnabled(false); // Uncomment to prevent navigating back while previewing
        qDebug() << "Dynamic::showOverlayVideo - Back button state adjusted if needed.";
    }
}

void Dynamic::hideOverlayVideo()
{
    if (!fullscreenPlayer || !fullscreenVideoWidget) {
        qWarning() << "hideOverlayVideo: Essential components are null (player or widget).";
        return;
    }

    // Stop player and hide stack widget immediately (non-blocking)
    fullscreenPlayer->stop();
    fullscreenPlayer->setSource(QUrl()); // Clear current media source
    fullscreenStackWidget->hide();
    
    // Hide video preview back button and show original back button
    videoPreviewBackButton->hide();
    
    // Reset the button to its original position
    videoPreviewBackButton->move(0, 0);
    
    if (ui->back) {
        ui->back->show();
        ui->back->raise();
        qDebug() << "Dynamic::hideOverlayVideo - Back button restored";
    }

    // Use QTimer::singleShot to make the rest of the operations non-blocking
    QTimer::singleShot(0, [this]() {
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
        qDebug() << "Dynamic::hideOverlayVideo - Finished asynchronously.";
    });
}

void Dynamic::stopAllGifs()
{
    qDebug() << "Dynamic::stopAllGifs - Stopping all GIF movies asynchronously.";
    
    // Hide video preview back button when leaving the page
    if (videoPreviewBackButton && videoPreviewBackButton->isVisible()) {
        videoPreviewBackButton->hide();
        qDebug() << "Dynamic::stopAllGifs - Hidden video preview back button";
    }
    
    // Use QTimer::singleShot to make GIF stopping non-blocking
    QTimer::singleShot(0, [this]() {
        QList<QPushButton*> buttons;
        buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
        
        for (QPushButton* button : buttons) {
            if (button) {
                QLabel* gifLabel = button->findChild<QLabel*>();
                if (gifLabel) {
                    QMovie* movie = gifLabel->movie();
                    if (movie && movie->state() == QMovie::Running) {
                        movie->stop();
                        qDebug() << "Dynamic::stopAllGifs - Stopped GIF movie for:" << button->objectName();
                    }
                }
            }
        }
        
        qDebug() << "Dynamic::stopAllGifs - All GIFs stopped asynchronously.";
    });
}

void Dynamic::restoreSelection()
{
    qDebug() << "Dynamic::restoreSelection - Restoring previous selection:" << m_lastSelectedVideoPath;
    
    if (m_lastSelectedVideoPath.isEmpty()) {
        qDebug() << "Dynamic::restoreSelection - No previous selection to restore";
        return;
    }
    
    // Find the button that corresponds to the last selected video path
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
    
    for (QPushButton* button : buttons) {
        if (button) {
            QString buttonVideoPath = button->property("actualVideoPath").toString();
            if (!buttonVideoPath.isEmpty()) {
                // The buttonVideoPath is already absolute from setupVideoPlayers
                if (buttonVideoPath == m_lastSelectedVideoPath) {
                    // Found the matching button, restore its selection
                    applyHighlightStyle(button, true);
                    currentSelectedVideoWidget = button;
                    qDebug() << "Dynamic::restoreSelection - Restored selection for button:" << button->objectName();
                    return;
                }
            }
        }
    }
    
    qDebug() << "Dynamic::restoreSelection - Could not find matching button for path:" << m_lastSelectedVideoPath;
}


