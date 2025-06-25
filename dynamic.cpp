#include "dynamic.h"
#include "ui_dynamic.h"
#include "iconhover.h" // Assuming this exists and is correctly implemented
#include <QStyle>
#include <QRegularExpression> // Not used directly in provided code, but kept if used elsewhere
#include <QVBoxLayout> // Not directly used for placeholder content, but kept if used elsewhere
#include <QStackedLayout>
#include <QDebug>
#include <QPixmap>
#include <QPalette> // Not directly used in provided code, but kept if used elsewhere
#include <QMouseEvent>
#include <QFile> // For checking file existence (Crucial for QRC debugging)
#include <QDir> // For current path debugging
#include <QImageReader> // REQUIRED for checking supported image formats

Dynamic::Dynamic(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic)
{
    ui->setupUi(this);

    // Ensure back button icon is set correctly
    QIcon backIcon(":/icons/Icons/normal.svg");
    if (backIcon.isNull()) {
        qWarning() << "WARNING: Back button icon ':/icons/Icons/normal.svg' not found! Setting text fallback.";
        ui->back->setText("Back"); // Fallback text
    }
    ui->back->setIcon(backIcon);
    ui->back->setIconSize(QSize(100, 100));
    ui->back->setFlat(true); // Often makes icons look better on push buttons

    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400); // Debounce interval (0.4 seconds)
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);

    debounceActive = false;
    currentSelectedVideoWidget = nullptr; // Initialize

    setupVideoPlayers();
}

Dynamic::~Dynamic()
{
    // Clean up media players
    // Iterate over a copy of keys or use take to avoid issues if items are removed during iteration
    for (const QString& key : videoPlayers.keys()) {
        QMediaPlayer* player = videoPlayers.take(key); // take removes and returns
        if (player) {
            player->stop();
            delete player;
        }
    }
    videoPlayers.clear(); // Ensure map is empty

    // QVideoWidget and QLabel are parented to the placeholder,
    // so they will be deleted when the placeholder is deleted.
    // We only need to clear the maps.
    videoWidgets.clear();
    thumbnailLabels.clear();
    videoLayouts.clear();

    delete ui;
}

void Dynamic::setupVideoPlayers()
{
    QList<QWidget*> videoPlaceholders;
    // Collect video placeholders from UI
    if (ui->videoPlaceholder1) videoPlaceholders << ui->videoPlaceholder1; else qWarning() << "videoPlaceholder1 not found in UI.";
    if (ui->videoPlaceholder2) videoPlaceholders << ui->videoPlaceholder2; else qWarning() << "videoPlaceholder2 not found in UI.";
    if (ui->videoPlaceholder3) videoPlaceholders << ui->videoPlaceholder3; else qWarning() << "videoPlaceholder3 not found in UI.";
    if (ui->videoPlaceholder4) videoPlaceholders << ui->videoPlaceholder4; else qWarning() << "videoPlaceholder4 not found in UI.";
    if (ui->videoPlaceholder5) videoPlaceholders << ui->videoPlaceholder5; else qWarning() << "videoPlaceholder5 not found in UI.";

    QStringList videoPaths;
    videoPaths << "qrc:/videos/videos/video1.mp4"
               << "qrc:/videos/videos/video2.mp4"
               << "qrc:/videos/videos/video3.mp4"
               << "qrc:/videos/videos/video4.mp4"
               << "qrc:/videos/videos/video5.mp4";

    QStringList thumbnailPaths;
    thumbnailPaths << "qrc:/images/pics/dynamic1.png"
                   << "qrc:/images/pics/dynamic2.png"
                   << "qrc:/images/pics/dynamic3.png"
                   << "qrc:/images/pics/dynamic4.png"
                   << "qrc:/images/pics/dynamic5.png";

    // Check for size consistency and determine loop limit
    int numPlaceholders = videoPlaceholders.size();
    int numVideos = videoPaths.size();
    int numThumbnails = thumbnailPaths.size();

    if (numPlaceholders != numVideos || numPlaceholders != numThumbnails) {
        qWarning() << "WARNING: Mismatch in number of placeholders, videos, or thumbnails. Placeholders:" << numPlaceholders
                   << "Videos:" << numVideos << "Thumbnails:" << numThumbnails;
    }

    // FIX: Changed qMin call for compatibility with older C++ standards
    int loopLimit = qMin(numPlaceholders, qMin(numVideos, numThumbnails));

    // --- Diagnostic: Print current application path (unrelated to QRC, but can be useful) ---
    qDebug() << "Application current path:" << QDir::currentPath();
    // --- NEW DIAGNOSTIC: Log supported image formats for QPixmap ---
    qDebug() << "Supported Image Read Formats:" << QImageReader::supportedImageFormats();

    for (int i = 0; i < loopLimit; ++i) {
        QWidget* placeholder = videoPlaceholders.at(i);
        if (!placeholder) {
            // This case should ideally be caught by initial checks, but good to have a safeguard
            qWarning() << "Video placeholder at index" << i << "is null in loop. Skipping.";
            continue;
        }

        qDebug() << "Setting up video player for placeholder:" << placeholder->objectName() << "at index" << i;
        qDebug() << "Placeholder geometry:" << placeholder->geometry();

        QStackedLayout *stackedLayout = new QStackedLayout(placeholder);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        // Ensure all widgets in the stack occupy the same space
        stackedLayout->setStackingMode(QStackedLayout::StackAll);

        QMediaPlayer *player = new QMediaPlayer(this);
        QVideoWidget *videoWidget = new QVideoWidget(placeholder);
        player->setVideoOutput(videoWidget);

        // --- Video file existence check (keeping this, as it's separate from pixmap) ---
        if (!QFile(videoPaths.at(i)).exists()) {
            qWarning() << "ERROR: Video file does not exist in resources or path is incorrect:" << videoPaths.at(i);
            // Optionally, display an error message on the videoWidget here.
        }
        player->setSource(QUrl(videoPaths.at(i)));

        // Determine target size based on placeholder's actual size or a default
        const QSize targetSize = (placeholder->size().isValid() && !placeholder->size().isEmpty()) ?
                                     placeholder->size() : QSize(425, 305);

        videoWidget->setMinimumSize(targetSize);
        videoWidget->setMaximumSize(targetSize);
        videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        videoWidget->setAttribute(Qt::WA_StyledBackground, true);
        videoWidget->setProperty("selected", false); // Custom property for styling
        videoWidget->setObjectName(QString("videoWidget%1").arg(i + 1));
        videoWidget->hide(); // Start hidden, thumbnail will be visible initially


        QLabel *thumbnailLabel = new QLabel(placeholder);
        thumbnailLabel->setMinimumSize(targetSize);
        thumbnailLabel->setMaximumSize(targetSize);
        thumbnailLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        thumbnailLabel->setAlignment(Qt::AlignCenter); // Center the pixmap/text
        thumbnailLabel->setScaledContents(true); // Allow QLabel to scale its contents
        thumbnailLabel->setObjectName(QString("thumbnailLabel%1").arg(i + 1));

        // --- IMPORTANT CHANGE: Directly attempt QPixmap load, rely on isNull() ---
        QPixmap pixmap(thumbnailPaths.at(i));

        qDebug() << "Attempting to load thumbnail from path:" << thumbnailPaths.at(i);
        if (!pixmap.isNull()) {
            // --- NEW DIAGNOSTIC: Check if pixmap has valid dimensions ---
            if (pixmap.width() <= 0 || pixmap.height() <= 0) {
                qWarning() << "WARNING: QPixmap loaded but has invalid dimensions (0 width/height) for:" << thumbnailPaths.at(i);
                thumbnailLabel->setStyleSheet("background-color: orange; border: 3px solid yellow; color: white; font-weight: bold; font-size: 14px;");
                thumbnailLabel->setText("THUMBNAIL INVALID SIZE\nCHECK IMAGE CONTENT");
            } else {
                qDebug() << "Thumbnail loaded successfully. Original size:" << pixmap.size();
                // Scale pixmap to fit within the target size, expanding to fill but maintaining aspect ratio
                thumbnailLabel->setPixmap(pixmap.scaled(targetSize, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
                thumbnailLabel->setStyleSheet(""); // Clear any temporary debug stylesheet
                qDebug() << "ThumbnailLabel pixmap set. Scaled pixmap size:" << thumbnailLabel->pixmap(Qt::ReturnByValue).size();
            }
        } else {
            // This is the crucial warning we're trying to debug
            qWarning() << "ERROR: QPixmap could not load thumbnail from path:" << thumbnailPaths.at(i);
            // Re-adding QFile::exists() here as a secondary check, purely for debugging output
            if (!QFile(thumbnailPaths.at(i)).exists()) {
                qWarning() << "CONFIRMATION (QFile::exists()): Thumbnail file does NOT exist at QRC path:" << thumbnailPaths.at(i);
            } else {
                qWarning() << "DEBUG: QFile::exists() reports file IS there, but QPixmap::isNull() is true. Potential plugin issue or corrupted image?";
            }
            // Enhanced debug style for failed thumbnails
            thumbnailLabel->setStyleSheet("background-color: red; border: 3px solid yellow; color: white; font-weight: bold; font-size: 16px;");
            thumbnailLabel->setText("THUMBNAIL LOAD FAILED\nCHECK .QRC PATH / PLUGINS"); // Clear text cue
        }

        stackedLayout->addWidget(videoWidget);
        stackedLayout->addWidget(thumbnailLabel); // Thumbnail is added last, so it's on top by default

        // Set the layout on the placeholder
        placeholder->setLayout(stackedLayout);

        // Store references
        QString widgetName = videoWidget->objectName();
        videoPlayers.insert(widgetName, player);
        videoWidgets.insert(widgetName, videoWidget);
        thumbnailLabels.insert(widgetName, thumbnailLabel);
        videoLayouts.insert(widgetName, stackedLayout);

        // Install event filter on the PARENT PLACEHOLDER (QWidget)
        // This ensures clicks on the entire region are caught, regardless of which child is visible.
        placeholder->installEventFilter(this);

        videoWidget->setFocusPolicy(Qt::NoFocus); // VideoWidget doesn't need focus for playback
        videoWidget->style()->polish(videoWidget); // Apply selected property style (for the custom "selected" property)

        // Ensure thumbnail is visible at start
        showThumbnail(videoWidget, true);
        qDebug() << "Initial state: Showing thumbnail for" << videoWidget->objectName();
        qDebug() << "ThumbnailLabel final size in setup:" << thumbnailLabel->size();
    }
}

void Dynamic::resetPage()
{
    qDebug() << "resetPage() called.";
    // Ensure all videos are reset, not just the previously selected one
    for (QVideoWidget* widget : videoWidgets.values()) {
        if (widget) {
            applyHighlightStyle(widget, false);
            QMediaPlayer* player = videoPlayers.value(widget->objectName());
            if (player) {
                if (player->playbackState() == QMediaPlayer::PlayingState || player->playbackState() == QMediaPlayer::PausedState) {
                    player->stop();
                    qDebug() << "Stopped player for" << widget->objectName() << "during full reset.";
                }
                player->setPosition(0); // Reset position to start for all players
            }
            showThumbnail(widget, true);
            qDebug() << "Showing thumbnail for" << widget->objectName() << "during full reset.";
        }
    }
    currentSelectedVideoWidget = nullptr; // Clear selection after resetting all

    resetDebounce();
    debounceTimer->stop();
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            // The event filter is installed on the placeholder QWidget itself.
            QWidget* clickedPlaceholder = qobject_cast<QWidget*>(obj);

            // Check if the clicked object is one of our managed video placeholders.
            // Using objectName prefix is a common way to identify them.
            if (!clickedPlaceholder || !clickedPlaceholder->objectName().startsWith("videoPlaceholder")) {
                return QWidget::eventFilter(obj, event); // Not a placeholder we manage, pass to base class
            }

            QVideoWidget *targetVideoWidget = nullptr;
            // Find the QVideoWidget associated with this placeholder
            for (const QString& name : videoWidgets.keys()) {
                QVideoWidget* vw = videoWidgets.value(name);
                if (vw && vw->parentWidget() == clickedPlaceholder) {
                    targetVideoWidget = vw;
                    break;
                }
            }

            if (targetVideoWidget) {
                qDebug() << "Click detected on placeholder:" << clickedPlaceholder->objectName()
                << ". Identified target video widget:" << targetVideoWidget->objectName();

                if (debounceActive) {
                    qDebug() << "Debounce active. Processing click immediately (rapid click scenario).";
                    processVideoClick(targetVideoWidget);
                    return true;
                } else {
                    debounceActive = true;
                    debounceTimer->start();
                    qDebug() << "Debounce started. Processing click.";
                    processVideoClick(targetVideoWidget);
                    return true;
                }
            } else {
                qWarning() << "Click on placeholder:" << clickedPlaceholder->objectName()
                << " but could not find an associated QVideoWidget.";
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void Dynamic::resetDebounce()
{
    qDebug() << "Debounce timer timed out. Debounce reset.";
    debounceActive = false;
}

void Dynamic::applyHighlightStyle(QObject *obj, bool highlight)
{
    if (obj) {
        obj->setProperty("selected", highlight);
        if (QWidget* widget = qobject_cast<QWidget*>(obj)) {
            widget->style()->polish(widget);
            widget->update(); // Request a repaint
            qDebug() << "Applied highlight style to" << obj->objectName() << ", highlight:" << highlight;
        }
    }
}

void Dynamic::on_back_clicked()
{
    qDebug() << "Back button clicked.";
    resetPage(); // Ensure all videos are stopped and thumbnails shown
    emit backtoLandingPage();
}

void Dynamic::processVideoClick(QObject *videoWidgetObj)
{
    // Casting to QVideoWidget* is safe now because currentSelectedVideoWidget is QVideoWidget*
    QVideoWidget* clickedVideoWidget = qobject_cast<QVideoWidget*>(videoWidgetObj);
    if (!clickedVideoWidget) {
        qWarning() << "processVideoClick received null or invalid object. This should not happen.";
        return;
    }

    QMediaPlayer* player = videoPlayers.value(clickedVideoWidget->objectName());
    if (!player) {
        qWarning() << "No media player found for" << clickedVideoWidget->objectName();
        return;
    }

    qDebug() << "Processing click for:" << clickedVideoWidget->objectName();

    if (clickedVideoWidget == currentSelectedVideoWidget) {
        // Double-click/re-click on the already selected video
        qDebug() << "Double-click/re-click detected on" << clickedVideoWidget->objectName() << ". Deselecting and stopping.";
        applyHighlightStyle(clickedVideoWidget, false);
        currentSelectedVideoWidget = nullptr; // Clear selection

        if (player->playbackState() == QMediaPlayer::PlayingState || player->playbackState() == QMediaPlayer::PausedState) {
            player->stop();
        }
        player->setPosition(0); // Reset video to start
        showThumbnail(clickedVideoWidget, true); // Show thumbnail again
        emit videoSelectedTwice();

        debounceTimer->stop(); // Stop debounce timer immediately as the action is complete
        resetDebounce();       // Reset debounce state
    } else {
        // Single click or new selection
        qDebug() << "Single click/new selection for" << clickedVideoWidget->objectName();

        if (currentSelectedVideoWidget) {
            // Deselect and stop the previously selected video
            qDebug() << "Deselecting previous video:" << currentSelectedVideoWidget->objectName();
            applyHighlightStyle(currentSelectedVideoWidget, false);
            QMediaPlayer* previousPlayer = videoPlayers.value(currentSelectedVideoWidget->objectName());
            if (previousPlayer) {
                if (previousPlayer->playbackState() == QMediaPlayer::PlayingState || previousPlayer->playbackState() == QMediaPlayer::PausedState) {
                    previousPlayer->stop();
                    qDebug() << "Stopped previous player.";
                }
                previousPlayer->setPosition(0); // Reset previous video to start
            }
            showThumbnail(currentSelectedVideoWidget, true); // Show previous thumbnail
        }

        // Select and play the new video
        applyHighlightStyle(clickedVideoWidget, true);
        currentSelectedVideoWidget = clickedVideoWidget;

        showThumbnail(clickedVideoWidget, false); // Hide thumbnail and show video
        qDebug() << "Now playing/showing video for:" << clickedVideoWidget->objectName();

        if (player->playbackState() == QMediaPlayer::StoppedState || player->playbackState() == QMediaPlayer::PausedState) {
            player->play();
        } else if (player->playbackState() == QMediaPlayer::PlayingState) {
            // If it's already playing (e.g., clicked itself after being selected), restart from beginning
            player->stop();
            player->setPosition(0);
            player->play();
        }
    }
}

void Dynamic::showThumbnail(QObject *videoWidgetObj, bool show)
{
    if (!videoWidgetObj) {
        qWarning() << "showThumbnail called with null object.";
        return;
    }

    QString widgetName = videoWidgetObj->objectName();
    QLabel* thumbnail = thumbnailLabels.value(widgetName);
    QVideoWidget* videoWidget = videoWidgets.value(widgetName);
    QStackedLayout* stackedLayout = videoLayouts.value(widgetName);

    if (thumbnail && videoWidget && stackedLayout) {
        if (show) {
            stackedLayout->setCurrentWidget(thumbnail);
            thumbnail->show();
            videoWidget->hide();
            qDebug() << "Showing thumbnail for" << widgetName;
        } else {
            stackedLayout->setCurrentWidget(videoWidget);
            videoWidget->show();
            thumbnail->hide();
            qDebug() << "Showing video for" << widgetName;
        }
        // Force update of the parent placeholder to ensure layout changes are applied
        if (QWidget* parentWidget = qobject_cast<QWidget*>(stackedLayout->parent())) {
            parentWidget->update();
            parentWidget->repaint();
        }
    } else {
        qWarning() << "Could not find all necessary widgets/layouts for" << widgetName << "to showThumbnail.";
        qWarning() << "Thumbnail:" << (thumbnail ? "Found" : "Not Found");
        qWarning() << "VideoWidget:" << (videoWidget ? "Found" : "Not Found");
        qWarning() << "StackedLayout:" << (stackedLayout ? "Found" : "Not Found");
    }
}
