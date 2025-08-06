#include "dynamic.h"
#include <QDebug>  // For debugging
#include <QPixmap> // For loading images for thumbnails
#include <QRegularExpression>
#include <QStyle>
#include <QVBoxLayout> // For arranging video
#include <QHBoxLayout>
#include <QTabWidget>
#include "iconhover.h"
#ifdef TFLITE_AVAILABLE
#include "tflite_segmentation_widget.h"
#endif
#include "ui_dynamic.h"

Dynamic::Dynamic(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic)
{
    ui->setupUi(this);

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);

    debounceActive = false;
    currentSelectedVideoWidget = nullptr;

    setupVideoPlayers();
    setupTFLiteSegmentation();
}

Dynamic::~Dynamic()
{
    for (QMediaPlayer *player : videoPlayers.values()) {
        if (player) {
            player->stop();
            delete player;
        }
    }
    videoPlayers.clear();
    videoWidgets.clear();

    for (QLabel *label : thumbnailLabels.values()) {
        if (label) {
            delete label;
        }
    }
    thumbnailLabels.clear();

    delete ui;
}

void Dynamic::setupVideoPlayers()
{
    QList<QWidget *> videoPlaceholders;
    videoPlaceholders << ui->videoPlaceholder1 << ui->videoPlaceholder2 << ui->videoPlaceholder3
                      << ui->videoPlaceholder4 << ui->videoPlaceholder5;

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

    for (int i = 0; i < videoPlaceholders.size(); ++i) {
        QWidget *placeholder = videoPlaceholders.at(i);
        if (!placeholder) {
            qWarning() << "Video placeholder" << i + 1 << "not found.";
            continue;
        }

        QMediaPlayer *player = new QMediaPlayer(this);
        QVideoWidget *videoWidget = new QVideoWidget(placeholder);
        player->setVideoOutput(videoWidget);

        // Loads the video
        player->setSource(QUrl(videoPaths.at(i)));

        videoWidget->setMinimumSize(QSize(425, 305));
        videoWidget->setMaximumSize(QSize(425, 305));
        videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        videoWidget->setAttribute(Qt::WA_StyledBackground, true);
        videoWidget->setProperty("selected", false);

        videoWidget->setObjectName(QString("videoWidget%1").arg(i + 1));

        QVBoxLayout *layout = new QVBoxLayout(placeholder);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addWidget(videoWidget);

        // New: Create and configure QLabel for thumbnail
        QLabel *thumbnailLabel = new QLabel(placeholder);
        QPixmap pixmap(thumbnailPaths.at(i));
        if (!pixmap.isNull()) {
            thumbnailLabel->setPixmap(pixmap.scaled(videoWidget->size(),
                                                    Qt::KeepAspectRatioByExpanding,
                                                    Qt::SmoothTransformation));
        } else {
            qWarning() << "Could not load thumbnail:" << thumbnailPaths.at(i);
        }
        thumbnailLabel->setAlignment(Qt::AlignCenter);
        thumbnailLabel->setScaledContents(true);
        thumbnailLabel->setObjectName(QString("thumbnailLabel%1").arg(i + 1));
        layout->addWidget(thumbnailLabel);

        placeholder->setLayout(layout);

        QString widgetName = videoWidget->objectName();
        videoPlayers.insert(widgetName, player);
        videoWidgets.insert(widgetName, videoWidget);
        thumbnailLabels.insert(widgetName, thumbnailLabel);

        // Allow video to capture clicks
        videoWidget->installEventFilter(this);
        videoWidget->setFocusPolicy(Qt::NoFocus);
        videoWidget->style()->polish(videoWidget);

        // Initially show the thumbnail
        showThumbnail(videoWidget, true);
    }
}

void Dynamic::resetPage()
{
    if (currentSelectedVideoWidget) {
        applyHighlightStyle(currentSelectedVideoWidget, false);
        QMediaPlayer *player = videoPlayers.value(currentSelectedVideoWidget->objectName());
        if (player && player->playbackState() == QMediaPlayer::PlayingState) {
            player->stop();
        }
        showThumbnail(currentSelectedVideoWidget, true);
    }
    currentSelectedVideoWidget = nullptr;

    for (QVideoWidget *widget : videoWidgets.values()) {
        if (widget) {
            applyHighlightStyle(widget, false);
            QMediaPlayer *player = videoPlayers.value(widget->objectName());
            if (player && player->playbackState() == QMediaPlayer::PlayingState) {
                player->stop();
            }
            showThumbnail(widget, true);
        }
    }

    resetDebounce();
    debounceTimer->stop();
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
        QVideoWidget *videoWidget = qobject_cast<QVideoWidget *>(obj);

        if (videoWidget && mouseEvent->button() == Qt::LeftButton) {
            // Check if the event source is one of our video widgets
            if (videoWidgets.values().contains(videoWidget)) {
                showThumbnail(videoWidget, false); // Hide thumbnail on click

                if (debounceActive) {
                    processVideoClick(videoWidget);
                    return true;
                } else {
                    // First click
                    debounceActive = true;
                    debounceTimer->start();
                    processVideoClick(videoWidget);
                    return true;
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

// Stops Video on back click
void Dynamic::on_back_clicked()
{
    if (currentSelectedVideoWidget) {
        applyHighlightStyle(currentSelectedVideoWidget, false);
        QMediaPlayer *player = videoPlayers.value(currentSelectedVideoWidget->objectName());
        if (player && player->playbackState() == QMediaPlayer::PlayingState) {
            player->stop();
        }
        showThumbnail(currentSelectedVideoWidget, true); // Show thumbnail when going back
    }
    currentSelectedVideoWidget = nullptr;
    emit backtoLandingPage();
}

void Dynamic::processVideoClick(QObject *videoWidgetObj)
{
    if (!videoWidgetObj) {
        return;
    }

    QVideoWidget *clickedVideoWidget = qobject_cast<QVideoWidget *>(videoWidgetObj);
    if (!clickedVideoWidget) {
        return;
    }

    QMediaPlayer *player = videoPlayers.value(clickedVideoWidget->objectName());
    if (!player) {
        qWarning() << "No media player found for" << clickedVideoWidget->objectName();
        return;
    }

    if (clickedVideoWidget == currentSelectedVideoWidget) {
        // Double-click
        applyHighlightStyle(clickedVideoWidget, false);
        currentSelectedVideoWidget = nullptr;
        player->stop();
        player->setPosition(0);
        showThumbnail(clickedVideoWidget, true); // Show thumbnail after double-click
        emit videoSelectedTwice();
        debounceTimer->stop();
        resetDebounce();
    } else {
        // Single click or first click of a potential double-click
        if (currentSelectedVideoWidget) {
            applyHighlightStyle(currentSelectedVideoWidget, false);
            QMediaPlayer *previousPlayer = videoPlayers.value(
                currentSelectedVideoWidget->objectName());
            if (previousPlayer && previousPlayer->playbackState() == QMediaPlayer::PlayingState) {
                previousPlayer->stop();
            }
            showThumbnail(currentSelectedVideoWidget,
                          true); // Show thumbnail for the previously selected video
        }

        applyHighlightStyle(clickedVideoWidget, true);
        currentSelectedVideoWidget = clickedVideoWidget;

        showThumbnail(clickedVideoWidget, false); // Hide thumbnail before playing

        // Play the video
        if (player->playbackState() == QMediaPlayer::StoppedState
            || player->playbackState() == QMediaPlayer::PausedState) {
            player->play();
        } else if (player->playbackState() == QMediaPlayer::PlayingState) {
            player->stop();
            player->setPosition(0);
            player->play();
        }
    }
}

// New helper function to control thumbnail visibility and layering
void Dynamic::showThumbnail(QObject *videoWidgetObj, bool show)
{
    if (!videoWidgetObj)
        return;

    QString widgetName = videoWidgetObj->objectName();
    QLabel *thumbnail = thumbnailLabels.value(widgetName);
    QVideoWidget *videoWidget = qobject_cast<QVideoWidget *>(videoWidgetObj);

    if (thumbnail && videoWidget) {
        thumbnail->setVisible(show);
        if (show) {
            thumbnail->raise();
        } else {
            videoWidget->raise();
        }
    }
}

void Dynamic::setupTFLiteSegmentation()
{
#ifdef TFLITE_AVAILABLE
    // Create tab widget to organize content
    m_tabWidget = new QTabWidget(this);
    
    // Create the original video content widget
    QWidget *videoContentWidget = new QWidget();
    QVBoxLayout *videoLayout = new QVBoxLayout(videoContentWidget);
    
    // Move all existing video placeholders to the video content widget
    QList<QWidget *> videoPlaceholders;
    videoPlaceholders << ui->videoPlaceholder1 << ui->videoPlaceholder2 << ui->videoPlaceholder3
                      << ui->videoPlaceholder4 << ui->videoPlaceholder5;
    
    for (QWidget *placeholder : videoPlaceholders) {
        if (placeholder && placeholder->parent() == this) {
            placeholder->setParent(videoContentWidget);
            videoLayout->addWidget(placeholder);
        }
    }
    
    // Add back button to video content
    videoLayout->addWidget(ui->back);
    
    // Create TFLite segmentation widget
    m_segmentationWidget = new TFLiteSegmentationWidget();
    
    // Add tabs
    m_tabWidget->addTab(videoContentWidget, "Videos");
    m_tabWidget->addTab(m_segmentationWidget, "TFLite Segmentation");
    
    // Replace the main layout with tab widget
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(m_tabWidget);
    
    // Connect segmentation signals
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::segmentationStarted,
            this, [this]() {
                qDebug() << "TFLite segmentation started";
            });
    
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::segmentationStopped,
            this, [this]() {
                qDebug() << "TFLite segmentation stopped";
            });
    
    connect(m_segmentationWidget, &TFLiteSegmentationWidget::segmentationError,
            this, [this](const QString &error) {
                qWarning() << "TFLite segmentation error:" << error;
            });
#else
    // Fallback: Use OpenCV-based segmentation or just show videos
    qDebug() << "TensorFlow Lite not available. Using fallback segmentation.";
    // The original video layout remains unchanged
#endif
}
