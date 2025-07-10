#include "dynamic.h"
#include "ui_dynamic.h" // This MUST be included for Ui::Dynamic to be defined
#include "iconhover.h"
#include <QStyle>
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
#include <QFrame> // For fglabel

Dynamic::Dynamic(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic) // Initialize the Ui::Dynamic pointer
    , fullscreenVideoWidget(nullptr)
    , fullscreenPlayer(nullptr)
{
    qDebug() << "Dynamic::Dynamic - Constructor started.";
    ui->setupUi(this); // This call populates the 'ui' object with widgets from dynamic.ui

    // Now, ui->back, ui->videoButton1, etc., are available.

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

        Iconhover* backButtonHover = new Iconhover(this);
        ui->back->installEventFilter(backButtonHover);
        qDebug() << "Dynamic::Dynamic - Back button event filter installed.";

        connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);
        qDebug() << "Dynamic::Dynamic - Back button clicked signal connected.";
    }

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Dynamic::resetDebounce);
    qDebug() << "Dynamic::Dynamic - Debounce timer setup complete.";

    debounceActive = false;
    currentSelectedVideoWidget = nullptr;

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

    setupVideoPlayers(); // This function will now handle each button individually
    qDebug() << "Dynamic::Dynamic - setupVideoPlayers() called.";

    fullscreenVideoWidget = new QVideoWidget(this);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenVideoWidget->setStyleSheet("background-color: transparent; border: 5px solid #FFC20F; border-radius: 8px;");
    fullscreenVideoWidget->hide();
    qDebug() << "Dynamic::Dynamic - Fullscreen video widget setup complete.";

    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);
    fullscreenVideoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio);
    qDebug() << "Dynamic::Dynamic - Fullscreen media player setup complete.";

    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            fullscreenPlayer->setPosition(0);
            fullscreenPlayer->play();
        }
    });
    qDebug() << "Dynamic::Dynamic - Fullscreen player media status connected.";

    fullscreenVideoWidget->installEventFilter(this);
    this->installEventFilter(this);
    qDebug() << "Dynamic::Dynamic - Fullscreen video widget and main widget event filters installed.";

    qDebug() << "Dynamic::Dynamic - Constructor finished.";
}

Dynamic::~Dynamic()
{
    qDebug() << "Dynamic::~Dynamic - Destructor started.";

    // No need to iterate through gifMovies/gifLabels maps here anymore
    // QMovie and QLabel objects, being children of QPushButtons, will be
    // automatically deleted when their parent QPushButton is deleted (which happens when 'ui' is deleted).
    // However, it's good practice to explicitly stop movies if they are running.
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

    delete ui; // Delete the generated UI object, which will delete child widgets (and their children like QLabel/QMovie)
    qDebug() << "Dynamic::~Dynamic - Destructor finished.";
}

void Dynamic::setupVideoPlayers()
{
    qDebug() << "Dynamic::setupVideoPlayers - Started.";

    // Video paths
    QStringList actualVideoPaths;
    actualVideoPaths << "qrc:/videos/videos/video1.mp4"
                     << "qrc:/videos/videos/video2.mp4"
                     << "qrc:/videos/videos/video3.mp4"
                     << "qrc:/videos/videos/video4.mp4"
                     << "qrc:/videos/videos/video5.mp4";

    // GIF paths
    QStringList gifPaths;
    gifPaths << ":/gif/test1.gif"
             << ":/gif/test2.gif"
             << ":/gif/test3.gif"
             << ":/gif/test4.gif"
             << ":/gif/test5.gif";

    // List of buttons
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    // Setup each button
    for (int i = 0; i < buttons.size(); ++i) {
        QPushButton* button = buttons[i];
        if (!button) {
            qWarning() << "Button at index" << i << "is null, skipping.";
            continue;
        }

        qDebug() << "Dynamic::setupVideoPlayers - Processing button:" << button->objectName();

        // Set video path property
        button->setProperty("actualVideoPath", actualVideoPaths.at(i));

        // Connect click signal
        connect(button, &QPushButton::clicked, this, [this, button]() {
            if (debounceActive) return;
            debounceActive = true;
            debounceTimer->start();
            processVideoClick(button);
        });

        // Clear button text if you want the GIF to be the primary visual
        button->setText("");

        // Create QLabel for GIF directly as a child of the button
        QLabel* gifLabel = new QLabel(button);
        gifLabel->setObjectName(QString("gifLabel_for_%1").arg(button->objectName()));

        // Set up the label properties
        gifLabel->setScaledContents(true);
        // Important: Make it transparent for mouse events so button clicks go through
        gifLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        gifLabel->setMouseTracking(false);
        gifLabel->lower(); // Put it behind the button's text (if any)

        // Set initial geometry (will be updated in resizeEvent if necessary)
        // IMPORTANT: Make the GIF label slightly smaller than the button
        // to allow the button's border to show on hover/selection.
        gifLabel->setGeometry(5, 5, button->width() - 10, button->height() - 10);

        // Debug current geometry
        qDebug() << "Dynamic::setupVideoPlayers - Button" << button->objectName() << "size:" << button->size();
        qDebug() << "Dynamic::setupVideoPlayers - gifLabel geometry:" << gifLabel->geometry();

        // Create QMovie for GIF
        QMovie* gifMovie = new QMovie(gifPaths.at(i), QByteArray(), gifLabel); // Parent to gifLabel
        gifLabel->setMovie(gifMovie); // Assign movie to label

        // Debug GIF loading
        qDebug() << "Dynamic::setupVideoPlayers - Loading GIF:" << gifPaths.at(i);
        qDebug() << "Dynamic::setupVideoPlayers - QMovie isValid:" << gifMovie->isValid();
        qDebug() << "Dynamic::setupVideoPlayers - QMovie frameCount:" << gifMovie->frameCount();
        qDebug() << "Dynamic::setupVideoPlayers - QMovie lastErrorString:" << gifMovie->lastErrorString();

        if (gifMovie->isValid()) {
            // Start the movie, it will loop automatically
            gifMovie->start();

            // Show and raise the label (it's already a child of the button)
            gifLabel->show();
            gifLabel->raise(); // Ensure it's on top of other button elements, but under text if any

            qDebug() << "Dynamic::setupVideoPlayers - Successfully loaded GIF for" << button->objectName();
            qDebug() << "Dynamic::setupVideoPlayers - gifLabel isVisible:" << gifLabel->isVisible();
            qDebug() << "Dynamic::setupVideoPlayers - gifLabel parent:" << gifLabel->parent()->objectName();

        } else {
            qWarning() << "ERROR: QMovie could not load GIF for" << button->objectName() << ":" << gifMovie->lastErrorString();

            // Show error message
            gifLabel->setText("GIF Error");
            gifLabel->setAlignment(Qt::AlignCenter);
            gifLabel->setStyleSheet("color: red; background-color: lightgray; border: 1px solid red;");
            gifLabel->show();

            // No need to delete gifMovie here, it's owned by gifLabel, which is owned by button.
            // If the movie is invalid, it just won't play.
        }
    }

    qDebug() << "Dynamic::setupVideoPlayers - Finished.";
}

void Dynamic::onDynamicPageShown()
{
    qDebug() << "Dynamic::onDynamicPageShown - Starting GIFs and updating geometry.";

    // Update geometry for all GIF labels
    updateGifLabelsGeometry();

    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>();
            if (gifLabel) {
                QMovie* movie = gifLabel->movie();
                if (movie && movie->isValid() && movie->state() != QMovie::Running) {
                    movie->start();
                    qDebug() << "Dynamic::onDynamicPageShown - Started GIF movie for:" << button->objectName();
                }
                gifLabel->show(); // Ensure label is visible
                gifLabel->raise(); // Ensure it's on top
                qDebug() << "Dynamic::onDynamicPageShown - gifLabel " << gifLabel->objectName() << " isVisible:" << gifLabel->isVisible();
            }
        }
    }
}

void Dynamic::updateGifLabelsGeometry()
{
    qDebug() << "Dynamic::updateGifLabelsGeometry - Started.";

    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;

    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>(); // Find the child QLabel
            if (gifLabel) {
                QRect newGeometry(5, 5, button->width() - 10, button->height() - 10);
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
    hideOverlayVideo(); // This will handle showing GIFs again after hiding the overlay

    // Ensure all highlights are removed
    if (ui->videoButton1) applyHighlightStyle(ui->videoButton1, false);
    if (ui->videoButton2) applyHighlightStyle(ui->videoButton2, false);
    if (ui->videoButton3) applyHighlightStyle(ui->videoButton3, false);
    if (ui->videoButton4) applyHighlightStyle(ui->videoButton4, false);
    if (ui->videoButton5) applyHighlightStyle(ui->videoButton5, false);

    currentSelectedVideoWidget = nullptr;

    // GIFs should be restarted/shown by hideOverlayVideo or onDynamicPageShown logic
    qDebug() << "Dynamic::resetPage - Finished.";
}

void Dynamic::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    qDebug() << "Dynamic::resizeEvent - Started.";

    // Update geometry for all GIF labels
    updateGifLabelsGeometry();

    qDebug() << "Dynamic::resizeEvent - Finished.";
}

bool Dynamic::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            qDebug() << "Dynamic::eventFilter - MouseButtonPress detected.";

            if (obj == fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
                qDebug() << "Dynamic::eventFilter - Click on fullscreen video widget.";
                fullscreenPlayer->stop();
                qDebug() << "Emitting showCapturePage() signal.";
                emit showCapturePage();
                hideOverlayVideo(); // This will restart GIFs
                return true;
            }

            QFrame* fgLabel = ui->fglabel;
            QPushButton* backButton = ui->back;
            QWidget* videoGridContent = ui->videoGridContent;

            if (obj == this && fullscreenVideoWidget->isVisible()) {
                QPoint clickPos = mouseEvent->pos();
                QRect fgLabelRect = fgLabel ? fgLabel->geometry() : QRect();
                QRect backButtonRect = backButton ? backButton->geometry() : QRect();
                QRect videoGridContentRect = videoGridContent ? videoGridContent->geometry() : QRect();

                qDebug() << "Dynamic::eventFilter - Click on main widget while fullscreen visible.";
                qDebug() << "Click pos:" << clickPos << "Fullscreen rect:" << fullscreenVideoWidget->geometry() << "fgLabel rect:" << fgLabelRect << "Back button rect:" << backButtonRect << "Video Grid Content rect:" << videoGridContentRect;

                if (!fullscreenVideoWidget->geometry().contains(clickPos) &&
                    !fgLabelRect.contains(clickPos) &&
                    !backButtonRect.contains(clickPos) &&
                    !videoGridContentRect.contains(clickPos))
                {
                    qDebug() << "Dynamic::eventFilter - Click outside fullscreen, fgLabel, back button, and video grid content. Resetting page.";
                    resetPage(); // This will restart GIFs
                    return true;
                }
            }
        }
    }
    return QWidget::eventFilter(obj, event);
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
        obj->setProperty("selected", highlight);
        if (QPushButton *button = qobject_cast<QPushButton *>(obj)) {
            button->style()->polish(button);
            button->update();
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
        hideOverlayVideo(); // This will restart GIFs
    } else {
        emit backtoLandingPage();
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

    if (currentSelectedVideoWidget && currentSelectedVideoWidget != clickedButton) {
        applyHighlightStyle(currentSelectedVideoWidget, false);
    }
    currentSelectedVideoWidget = clickedButton;
    applyHighlightStyle(currentSelectedVideoWidget, true);

    QString actualVideoPath = clickedButton->property("actualVideoPath").toString();
    qDebug() << "Dynamic::processVideoClick - Actual video path:" << actualVideoPath;

    if (!actualVideoPath.isEmpty()) {
        // Stop and hide all GIFs
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
                    gifLabel->hide();
                    qDebug() << "Dynamic::processVideoClick - Hidden GIF label " << gifLabel->objectName() << " isVisible:" << gifLabel->isVisible();
                }
            }
        }

        showOverlayVideo(actualVideoPath);
        fullscreenPlayer->play();
        qDebug() << "Playing actual video:" << actualVideoPath << " as an overlay.";
    } else {
        qWarning() << "Actual video path not found for clicked button:" << clickedButton->objectName();
    }

    debounceTimer->stop();
    resetDebounce();
    qDebug() << "Dynamic::processVideoClick - Finished.";
}

void Dynamic::onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status)
{
    Q_UNUSED(status);
    qDebug() << "Dynamic::onPlayerMediaStatusChanged - Media status changed to:" << status;
}

void Dynamic::showOverlayVideo(const QString& videoPath)
{
    qDebug() << "Dynamic::showOverlayVideo - Showing overlay for video:" << videoPath;
    QWidget* videoGridContent = ui->videoGridContent;
    if (!fullscreenPlayer || !fullscreenVideoWidget || !videoGridContent) {
        qWarning() << "showOverlayVideo: Essential components are null (player, widget, or videoGridContent).";
        return;
    }

    QRect targetRect = videoGridContent->geometry();
    fullscreenVideoWidget->setGeometry(targetRect);
    fullscreenPlayer->setSource(QUrl(videoPath));
    fullscreenVideoWidget->show();
    fullscreenVideoWidget->raise();
    qDebug() << "Dynamic::showOverlayVideo - Fullscreen video widget shown.";
    qDebug() << "Dynamic::showOverlayVideo - fullscreenVideoWidget isVisible:" << fullscreenVideoWidget->isVisible();

    QPushButton* backButton = ui->back;
    if (backButton) {
        backButton->setEnabled(false);
        qDebug() << "Dynamic::showOverlayVideo - Back button disabled.";
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
    fullscreenPlayer->setSource(QUrl());
    fullscreenVideoWidget->hide();
    qDebug() << "Dynamic::hideOverlayVideo - Fullscreen video widget hidden.";
    qDebug() << "Dynamic::hideOverlayVideo - fullscreenVideoWidget isVisible:" << fullscreenVideoWidget->isVisible();

    // Reset selection highlight
    if (ui->videoButton1) applyHighlightStyle(ui->videoButton1, false);
    if (ui->videoButton2) applyHighlightStyle(ui->videoButton2, false);
    if (ui->videoButton3) applyHighlightStyle(ui->videoButton3, false);
    if (ui->videoButton4) applyHighlightStyle(ui->videoButton4, false);
    if (ui->videoButton5) applyHighlightStyle(ui->videoButton5, false);

    currentSelectedVideoWidget = nullptr;
    qDebug() << "Dynamic::hideOverlayVideo - Selection highlight reset.";

    // Restart and show all GIFs
    QList<QPushButton*> buttons;
    buttons << ui->videoButton1 << ui->videoButton2 << ui->videoButton3 << ui->videoButton4 << ui->videoButton5;
    for (QPushButton* button : buttons) {
        if (button) {
            QLabel* gifLabel = button->findChild<QLabel*>();
            if (gifLabel) {
                QMovie* movie = gifLabel->movie();
                if (movie && movie->isValid() && movie->state() != QMovie::Running) {
                    movie->start();
                    qDebug() << "Dynamic::hideOverlayVideo - Restarted GIF movie for:" << button->objectName();
                }
                gifLabel->show();
                gifLabel->raise();
                qDebug() << "Dynamic::hideOverlayVideo - gifLabel " << gifLabel->objectName() << " isVisible after show/raise:" << gifLabel->isVisible();
            }
        }
    }

    QPushButton* backButton = ui->back;
    if (backButton) {
        backButton->setEnabled(true);
        qDebug() << "Dynamic::hideOverlayVideo - Back button enabled.";
    }
    qDebug() << "Dynamic::hideOverlayVideo - Finished.";
}
