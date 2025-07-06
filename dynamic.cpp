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
#include <QVBoxLayout>
#include <QEvent>

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
    fullscreenVideoWidget = new QVideoWidget(this);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenVideoWidget->setStyleSheet("background-color: transparent; border: 5px solid #FFC20F; border-radius: 8px;");
    fullscreenVideoWidget->hide();

    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);
    fullscreenVideoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio);

    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            fullscreenPlayer->setPosition(0);
            fullscreenPlayer->play();
        }
    });

    fullscreenVideoWidget->installEventFilter(this);
    this->installEventFilter(this);

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
    videoPlaceholders.clear();
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
        delete fullscreenVideoWidget;
        fullscreenVideoWidget = nullptr;
    }

    delete ui;
}

void Dynamic::setupVideoPlayers()
{
    QList<QWidget*> placeholders;
    if (ui->videoPlaceholder1) placeholders << ui->videoPlaceholder1; else qWarning() << "videoPlaceholder1 not found in UI.";
    if (ui->videoPlaceholder2) placeholders << ui->videoPlaceholder2; else qWarning() << "videoPlaceholder2 not found in UI.";
    if (ui->videoPlaceholder3) placeholders << ui->videoPlaceholder3; else qWarning() << "videoPlaceholder3 not found in UI.";
    if (ui->videoPlaceholder4) placeholders << ui->videoPlaceholder4; else qWarning() << "videoPlaceholder4 not found in UI.";
    if (ui->videoPlaceholder5) placeholders << ui->videoPlaceholder5; else qWarning() << "videoPlaceholder5 not found in UI.";

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

    int loopLimit = qMin(placeholders.size(), qMin(videoPreviewPaths.size(), actualVideoPaths.size()));

    for (int i = 0; i < loopLimit; ++i) {
        QWidget* placeholder = placeholders.at(i);
        if (!placeholder) {
            qWarning() << "Video placeholder at index" << i << "is null in loop. Skipping.";
            continue;
        }

        // SIMPLIFIED APPROACH: Apply hover styling directly to the placeholder widget
        placeholder->setMouseTracking(true);
        placeholder->setAttribute(Qt::WA_Hover, true);
        placeholder->setProperty("hovered", false);
        placeholder->setProperty("actualVideoPath", actualVideoPaths.at(i));

        // Set base style for placeholder - this will work reliably
        placeholder->setStyleSheet(
            "QWidget {"
            "  background-color: transparent;"
            "  border: 5px solid transparent;"
            "  border-radius: 8px;"
            "}"
            );

        qDebug() << "Setting up video player for placeholder:" << placeholder->objectName() << "at index" << i;

        QStackedLayout *stackedLayout = new QStackedLayout(placeholder);
        stackedLayout->setContentsMargins(5, 5, 5, 5); // Leave space for border
        stackedLayout->setStackingMode(QStackedLayout::StackAll);

        QMediaPlayer *player = new QMediaPlayer(this);
        QVideoWidget *videoWidget = new QVideoWidget(placeholder);
        player->setVideoOutput(videoWidget);

        if (!QFile(videoPreviewPaths.at(i)).exists()) {
            qWarning() << "ERROR: Preview GIF file does not exist:" << videoPreviewPaths.at(i);
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
        videoWidget->setObjectName(QString("videoWidget%1").arg(i + 1));
        videoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio);

        // DON'T style the video widget - leave it completely alone

        stackedLayout->addWidget(videoWidget);
        placeholder->setLayout(stackedLayout);

        QString widgetName = videoWidget->objectName();
        player->setObjectName(QString("mediaPlayer%1").arg(i + 1));
        videoPlayers.insert(widgetName, player);
        videoWidgets.insert(widgetName, videoWidget);
        videoPlaceholders.insert(widgetName, placeholder); // Store placeholder reference
        videoLayouts.insert(widgetName, stackedLayout);

        // Install event filter ONLY on placeholder - this is key!
        placeholder->installEventFilter(this);
        qDebug() << "Event filter installed on placeholder:" << placeholder->objectName();

        videoWidget->setFocusPolicy(Qt::NoFocus);
        videoWidget->show();
        player->play();
    }
}

// Apply hover styling to placeholder widget
void Dynamic::setPlaceholderHoverState(QWidget* placeholder, bool hovered)
{
    if (!placeholder) return;

    placeholder->setProperty("hovered", hovered);

    if (hovered) {
        placeholder->setStyleSheet(
            "QWidget {"
            "  background-color: rgba(255, 194, 15, 0.1);"
            "  border: 5px solid #FFC20F;"
            "  border-radius: 8px;"
            "}"
            );
        qDebug() << "Applied hover style to placeholder:" << placeholder->objectName();
    } else {
        placeholder->setStyleSheet(
            "QWidget {"
            "  background-color: transparent;"
            "  border: 5px solid transparent;"
            "  border-radius: 8px;"
            "}"
            );
        qDebug() << "Removed hover style from placeholder:" << placeholder->objectName();
    }

    placeholder->style()->polish(placeholder);
    placeholder->update();
}

void Dynamic::resetPage()
{
    hideOverlayVideo();

    // Reset hover states on placeholders
    for (QWidget* placeholder : videoPlaceholders.values()) {
        if (placeholder) {
            setPlaceholderHoverState(placeholder, false);
        }
    }

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
        }
    }
    currentSelectedVideoWidget = nullptr;

    resetDebounce();
    debounceTimer->stop();
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    // Handle hover on placeholder widgets - this should work reliably
    for (QWidget* placeholder : videoPlaceholders.values()) {
        if (obj == placeholder) {
            if (event->type() == QEvent::Enter) {
                qDebug() << "DEBUG: Mouse entered placeholder" << placeholder->objectName();
                setPlaceholderHoverState(placeholder, true);
                return false;
            } else if (event->type() == QEvent::Leave) {
                qDebug() << "DEBUG: Mouse left placeholder" << placeholder->objectName();
                setPlaceholderHoverState(placeholder, false);
                return false;
            }
        }
    }

    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {

            if (obj == fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
                fullscreenPlayer->stop();
                qDebug() << "Emitting showCapturePage() signal.";
                emit showCapturePage();
                hideOverlayVideo();
                return true;
            }

            if (obj == this && fullscreenVideoWidget->isVisible()) {
                QPoint clickPos = mouseEvent->pos();
                QRect fgLabelRect = ui->fglabel->geometry();
                QRect backButtonRect = ui->back->geometry();

                if (!fullscreenVideoWidget->geometry().contains(clickPos) &&
                    !fgLabelRect.contains(clickPos) &&
                    !backButtonRect.contains(clickPos)) {
                    resetPage();
                    return true;
                }
            }

            // Handle clicks on placeholder widgets
            if (!fullscreenVideoWidget->isVisible()) {
                for (QWidget* placeholder : videoPlaceholders.values()) {
                    if (obj == placeholder) {
                        // Find the associated video widget
                        QVideoWidget *targetVideoWidget = nullptr;
                        for (const QString& name : videoWidgets.keys()) {
                            QVideoWidget* vw = videoWidgets.value(name);
                            if (vw && videoPlaceholders.value(name) == placeholder) {
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
                        }
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
        if (QWidget *widget = qobject_cast<QWidget *>(obj)) {
            widget->style()->polish(widget);
            widget->update();
        }
    }
}

void Dynamic::on_back_clicked()
{
    if (fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
        hideOverlayVideo();
    } else {
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

    // Get the actual video path from the placeholder widget
    QWidget* placeholder = videoPlaceholders.value(clickedVideoWidget->objectName());
    QString actualVideoPath;
    if (placeholder) {
        actualVideoPath = placeholder->property("actualVideoPath").toString();
    }

    if (!actualVideoPath.isEmpty()) {
        for (QMediaPlayer* p : videoPlayers.values()) {
            if (p->playbackState() == QMediaPlayer::PlayingState) {
                p->stop();
            }
            p->setPosition(0);
        }

        showOverlayVideo(actualVideoPath);
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

void Dynamic::showOverlayVideo(const QString& videoPath)
{
    if (!fullscreenPlayer || !fullscreenVideoWidget || !ui->videoGridContent) {
        qWarning() << "showOverlayVideo: Essential components are null.";
        return;
    }

    QRect targetRect = ui->videoGridContent->geometry();
    fullscreenVideoWidget->setGeometry(targetRect);
    fullscreenPlayer->setSource(QUrl(videoPath));
    fullscreenVideoWidget->show();
    fullscreenVideoWidget->raise();

    if (ui->back) {
        ui->back->setEnabled(false);
    }
}

void Dynamic::hideOverlayVideo()
{
    if (!fullscreenPlayer || !fullscreenVideoWidget) {
        qWarning() << "hideOverlayVideo: Essential components are null.";
        return;
    }

    fullscreenPlayer->stop();
    fullscreenPlayer->setSource(QUrl());
    fullscreenVideoWidget->hide();

    // Reset hover states
    for (QWidget* placeholder : videoPlaceholders.values()) {
        if (placeholder) {
            setPlaceholderHoverState(placeholder, false);
        }
    }

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
        ui->back->setEnabled(true);
    }
}
