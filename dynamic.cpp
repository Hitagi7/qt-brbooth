#include "dynamic.h"
#include "ui_dynamic.h"
#include "iconhover.h"
#include <QStyle>
#include <QStackedLayout>
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
#include <QVBoxLayout> // Still useful for general layout understanding

Dynamic::Dynamic(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic)
    , fullscreenVideoWidget(nullptr)
    , fullscreenPlayer(nullptr)
{
    ui->setupUi(this);

    QIcon backIcon(":/icons/Icons/normal.svg");
    if (backIcon.isNull()) {
        qWarning() << "WARNING: Back button icon ':/icons/Icons/normal.svg' not found! Setting text fallback.";
    }
    ui->back->setIcon(backIcon);
    ui->back->setIconSize(QSize(100, 100));
    ui->back->setFlat(true);

    Iconhover* backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);

    debounceActive = false;
    currentSelectedVideoWidget = nullptr;

#ifdef CV_VERSION
    qDebug() << "OpenCV Version: "<< CV_VERSION;
#else
    qDebug() << "OpenCV Version: Not defined or available at compile time.";
#endif

    qDebug() << "APP PATH: Current applicationDirPath:" << QCoreApplication::applicationDirPath();
    qDebug() << "APP PATH: Current currentPath:" << QDir::currentPath();
    qDebug() << "QRC CHECK: QFile::exists('qrc:/gif/test1.gif') at constructor start:" << QFile::exists("qrc:/gif/test1.gif");

    QDir gifDir(":/gif");
    if (gifDir.exists()) {
        qDebug() << "QRC CHECK: Directory ':/gif' exists in resources. Listing contents:";
        for (const QString& entry : gifDir.entryList(QDir::Files | QDir::NoDotAndDotDot)) {
            qDebug() << "  - QRC file found:" << entry;
        }
    } else {
        qWarning() << "QRC CHECK: Directory ':/gif' does NOT exist in resources (prefix not registered or wrong).";
    }

    setupVideoPlayers();

    // --- Setup for the "large" video display as an overlay ---
    // Parent the fullscreen video widget directly to 'this' (the Dynamic widget)
    fullscreenVideoWidget = new QVideoWidget(this);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenVideoWidget->setStyleSheet("background-color: black; border: 5px solid #FFC20F; border-radius: 8px;");
    fullscreenVideoWidget->hide(); // Start hidden

    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);

    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            fullscreenPlayer->setPosition(0);
            fullscreenPlayer->play();
        }
    });

    fullscreenVideoWidget->installEventFilter(this);
    this->installEventFilter(this); // To capture clicks on the 'Dynamic' widget itself

    // Initial state: Back button should be enabled as GIFs are shown.
    if (ui->back) {
        ui->back->setEnabled(true);
    }
}

Dynamic::~Dynamic()
{
    for (const QString& key : videoPlayers.keys()) {
        QMediaPlayer* player = videoPlayers.take(key);
        if (player) {
            player->stop();
            delete player;
        }
    }
    videoPlayers.clear();

    videoWidgets.clear();
    for (QStackedLayout* layout : videoLayouts.values()) {
        if (layout) {
            if (!layout->parentWidget()) {
                delete layout;
            }
        }
    }
    videoLayouts.clear();

    if (fullscreenPlayer) {
        fullscreenPlayer->stop();
        delete fullscreenPlayer;
        fullscreenPlayer = nullptr;
    }
    if (fullscreenVideoWidget) {
        delete fullscreenVideoWidget; // It's parented to 'this', so this is correct
        fullscreenVideoWidget = nullptr;
    }

    delete ui;
}

void Dynamic::setupVideoPlayers()
{
    QList<QWidget*> videoPlaceholders;
    if (ui->videoPlaceholder1) videoPlaceholders << ui->videoPlaceholder1; else qWarning() << "videoPlaceholder1 not found in UI.";
    if (ui->videoPlaceholder2) videoPlaceholders << ui->videoPlaceholder2; else qWarning() << "videoPlaceholder2 not found in UI.";
    if (ui->videoPlaceholder3) videoPlaceholders << ui->videoPlaceholder3; else qWarning() << "videoPlaceholder3 not found in UI.";
    if (ui->videoPlaceholder4) videoPlaceholders << ui->videoPlaceholder4; else qWarning() << "videoPlaceholder4 not found in UI.";
    if (ui->videoPlaceholder5) videoPlaceholders << ui->videoPlaceholder5; else qWarning() << "videoPlaceholder5 not found in UI.";

    QStringList videoPreviewPaths;
    videoPreviewPaths << "qrc:/gif/test1.gif"
                      << "qrc:/gif/test2.gif"
                      << "qrc:/gif/test3.gif"
                      << "qrc:/gif/test4.gif"
                      << "qrc:/gif/test5.gif";

    QStringList actualVideoPaths;
    actualVideoPaths << "qrc:/videos/videos/video1.mp4"
                     << "qrc:/videos/videos/video2.mp4"
                     << "qrc:/videos/videos/video3.mp4"
                     << "qrc:/videos/videos/video4.mp4"
                     << "qrc:/videos/videos/video5.mp4";


    int numPlaceholders = videoPlaceholders.size();
    int numPreviews = videoPreviewPaths.size();
    int numActualVideos = actualVideoPaths.size();

    if (numPlaceholders != numPreviews || numPlaceholders != numActualVideos) {
        qWarning() << "WARNING: Mismatch in number of placeholders, preview GIFs, or actual videos. Placeholders:" << numPlaceholders
                   << "Previews:" << numPreviews << "Actual Videos:" << numActualVideos;
    }

    int loopLimit = qMin(numPlaceholders, qMin(numPreviews, numActualVideos));

    qDebug() << "Application supported Image Read Formats (in Dynamic::setupVideoPlayers):" << QImageReader::supportedImageFormats();

    for (int i = 0; i < loopLimit; ++i) {
        QWidget* placeholder = videoPlaceholders.at(i);
        if (!placeholder) {
            qWarning() << "Video placeholder at index" << i << "is null in loop. Skipping.";
            continue;
        }

        qDebug() << "Setting up video player for placeholder:" << placeholder->objectName() << "at index" << i;
        qDebug() << "Placeholder geometry:" << placeholder->geometry();

        QStackedLayout *stackedLayout = new QStackedLayout(placeholder);
        stackedLayout->setContentsMargins(0, 0, 0, 0);
        stackedLayout->setStackingMode(QStackedLayout::StackAll);

        QMediaPlayer *player = new QMediaPlayer(this);
        QVideoWidget *videoWidget = new QVideoWidget(placeholder);
        player->setVideoOutput(videoWidget);

        if (!QFile(videoPreviewPaths.at(i)).exists()) {
            qWarning() << "ERROR: Preview GIF file does not exist in resources or path is incorrect:" << videoPreviewPaths.at(i);
        }
        player->setSource(QUrl(videoPreviewPaths.at(i)));

        connect(player, &QMediaPlayer::mediaStatusChanged, this, [player](QMediaPlayer::MediaStatus status) {
            if (status == QMediaPlayer::EndOfMedia) {
                player->setPosition(0);
                player->play();
            }
        });

        videoWidget->setMinimumSize(placeholder->minimumSize());
        videoWidget->setMaximumSize(placeholder->maximumSize());
        videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        videoWidget->setAttribute(Qt::WA_StyledBackground, true);
        videoWidget->setProperty("selected", false);
        videoWidget->setObjectName(QString("videoWidget%1").arg(i + 1));
        videoWidget->setProperty("actualVideoPath", actualVideoPaths.at(i));

        stackedLayout->addWidget(videoWidget);
        placeholder->setLayout(stackedLayout);

        QString widgetName = videoWidget->objectName();
        player->setObjectName(QString("mediaPlayer%1").arg(i + 1));
        videoPlayers.insert(widgetName, player);
        videoWidgets.insert(widgetName, videoWidget);
        videoLayouts.insert(widgetName, stackedLayout);

        placeholder->installEventFilter(this);

        videoWidget->setFocusPolicy(Qt::NoFocus);
        videoWidget->style()->polish(videoWidget);

        videoWidget->show();
        player->play();
    }
}

void Dynamic::resetPage()
{
    hideOverlayVideo(); // This will hide the fullscreen video and enable back button

    // Restart GIF previews in the now-visible grid
    for (QVideoWidget* widget : videoWidgets.values()) {
        if (widget) {
            applyHighlightStyle(widget, false);
            QMediaPlayer* player = videoPlayers.value(widget->objectName());
            if (player) {
                if (player->playbackState() == QMediaPlayer::PlayingState || player->playbackState() == QMediaPlayer::PausedState) {
                    player->stop();
                }
                player->setPosition(0);
                player->play();
            }
            // videoGridContent and its children (placeholders) are never hidden in this overlay approach
            // so no need to show them explicitly here.
        }
    }
    currentSelectedVideoWidget = nullptr;

    resetDebounce();
    debounceTimer->stop();
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {

            // Case 1: Click on the fullscreen video widget itself
            if (obj == fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
                fullscreenPlayer->stop();
                emit showCapturePage();
                return true;
            }

            // Case 2: Click on the Dynamic widget itself (the background)
            // if the fullscreen video is visible, and the click wasn't on the fullscreen video, header, or back button
            if (obj == this && fullscreenVideoWidget->isVisible()) {
                // Get the click position relative to the Dynamic widget
                QPoint clickPos = mouseEvent->pos();

                // Get geometries of elements that should NOT trigger a reset
                // These geometries are relative to the Dynamic widget
                QRect fullscreenVideoRect = fullscreenVideoWidget->geometry();
                QRect fgLabelRect = ui->fglabel->geometry();
                QRect backButtonRect = ui->back->geometry();

                // Check if the click is outside all interactive elements that should prevent a reset
                if (!fullscreenVideoRect.contains(clickPos) &&
                    !fgLabelRect.contains(clickPos) &&
                    !backButtonRect.contains(clickPos)) {
                    resetPage(); // Reset the page, hiding the overlay video
                    return true;
                }
            }

            // Case 3: Click on one of the small video placeholders (only if fullscreen video is NOT visible)
            // This ensures clicks on GIFs don't trigger anything if the large video is already up.
            if (!fullscreenVideoWidget->isVisible()) {
                QWidget* clickedPlaceholder = qobject_cast<QWidget*>(obj);
                if (clickedPlaceholder && clickedPlaceholder->objectName().startsWith("videoPlaceholder")) {
                    QVideoWidget *targetVideoWidget = nullptr;
                    for (const QString& name : videoWidgets.keys()) {
                        QVideoWidget* vw = videoWidgets.value(name);
                        if (vw && vw->parentWidget() == clickedPlaceholder) {
                            targetVideoWidget = vw;
                            break;
                        }
                    }

                    if (targetVideoWidget) {
                        if (debounceActive) {
                            return true;
                        } else {
                            debounceActive = true;
                            debounceTimer->start();
                            processVideoClick(targetVideoWidget);
                            return true;
                        }
                    } else {
                        qWarning() << "Click on placeholder:" << clickedPlaceholder->objectName()
                        << " but could not find an associated QVideoWidget.";
                    }
                }
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void Dynamic::resetDebounce()
{
    debounceActive = false;
}

void Dynamic::applyHighlightStyle(QObject *obj, bool highlight)
{
    if (obj) {
        obj->setProperty("selected", highlight);
        if (QWidget* widget = qobject_cast<QWidget*>(obj)) {
            widget->style()->polish(widget);
            widget->update();
        }
    }
}

void Dynamic::on_back_clicked()
{
    // If the large overlay video is currently visible, hide it.
    if (fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
        hideOverlayVideo(); // Hide video, enable back button
    } else {
        // Otherwise, go back to the landing page.
        emit backtoLandingPage();
    }
}

void Dynamic::processVideoClick(QObject *videoWidgetObj)
{
    QVideoWidget* clickedVideoWidget = qobject_cast<QVideoWidget*>(videoWidgetObj);
    if (!clickedVideoWidget) {
        qWarning() << "processVideoClick received null or invalid object.";
        return;
    }

    QMediaPlayer* player = videoPlayers.value(clickedVideoWidget->objectName());
    if (!player) {
        qWarning() << "No media player found for" << clickedVideoWidget->objectName();
        return;
    }

    QString actualVideoPath = clickedVideoWidget->property("actualVideoPath").toString();
    if (!actualVideoPath.isEmpty()) {
        // Stop all small preview players immediately
        for (QMediaPlayer* p : videoPlayers.values()) {
            if (p->playbackState() == QMediaPlayer::PlayingState) {
                p->stop();
            }
            p->setPosition(0);
        }

        // Do NOT hide ui->videoGridContent here. It remains visible underneath.

        showOverlayVideo(actualVideoPath); // Pass the path to the show function
        fullscreenPlayer->play();
        qDebug() << "Playing actual video:" << actualVideoPath << " as an overlay.";
    } else {
        qWarning() << "Actual video path not found for clicked widget:" << clickedVideoWidget->objectName();
    }

    debounceTimer->stop();
    resetDebounce();
}

void Dynamic::onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status)
{
    Q_UNUSED(status);
}

// showOverlayVideo now takes the video path and uses videoGridContent's geometry
void Dynamic::showOverlayVideo(const QString& videoPath)
{
    if (!fullscreenPlayer || !fullscreenVideoWidget || !ui->videoGridContent) {
        qWarning() << "showOverlayVideo: Essential components (player, video widget, or videoGridContent) are null.";
        return;
    }

    // Get the geometry of the videoGridContent widget, which contains all your GIFs.
    // This geometry is relative to the 'Dynamic' widget itself.
    QRect targetRect = ui->videoGridContent->geometry();

    // Set the fullscreenVideoWidget's geometry to match the videoGridContent's geometry
    fullscreenVideoWidget->setGeometry(targetRect);

    // Set the source and play the video
    fullscreenPlayer->setSource(QUrl(videoPath));

    fullscreenVideoWidget->show(); // Make the video widget visible
    fullscreenVideoWidget->raise(); // Bring it to the front, on top of all other siblings (like videoGridContent)

    if (ui->back) {
        ui->back->setEnabled(false); // Disable the back button
    }
}

// hideOverlayVideo no longer takes arguments
void Dynamic::hideOverlayVideo()
{
    if (!fullscreenPlayer || !fullscreenVideoWidget) {
        qWarning() << "hideOverlayVideo: Essential components are null.";
        return;
    }

    fullscreenPlayer->stop();
    fullscreenPlayer->setSource(QUrl()); // Clear the media source
    fullscreenVideoWidget->hide(); // Hide the video widget

    // ui->videoGridContent remains visible, no need to show it here.

    // Restart GIF previews in the now-visible grid
    for (QVideoWidget* widget : videoWidgets.values()) {
        if (widget) {
            applyHighlightStyle(widget, false);
            QMediaPlayer* player = videoPlayers.value(widget->objectName());
            if (player) {
                player->setPosition(0);
                player->play();
            }
        }
    }
    currentSelectedVideoWidget = nullptr;

    if (ui->back) {
        ui->back->setEnabled(true); // Re-enable the back button
    }
}
