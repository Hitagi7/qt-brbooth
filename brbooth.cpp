#include "brbooth.h"
#include "background.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"
#include "ui_brbooth.h"
#include "videotemplate.h"
#include <QStyle>
#include <QStackedLayout>
#include <QDebug>
#include <QPixmap>
#include <QMouseEvent>
#include <QFile>
#include <QDir>
#include <QImageReader>
#include <QTimer>
#include <QResource>
#include <QCoreApplication>
#include <QByteArray>
#include <QScreen>
#include <QApplication>
#include <QVBoxLayout>
#include <QMediaPlayer>
#include <QVideoWidget> // Keep this include for QMediaPlayer's internal use if needed
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsVideoItem>


BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    qDebug() << "OpenCV Version: " << CV_VERSION;
    ui->setupUi(this);
    this->setStyleSheet("QMainWindow#BRBooth {"
                        "    background-image: url(:/images/pics/bg.jpg);"
                        "    background-repeat: no-repeat;"
                        "    background-position: center;"
                        "    background-size: cover;"
                        "}");

    dynamicMediaPlayer = new QMediaPlayer(this);

    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");
    QLabel* chillTextLabel = ui->landingpage->findChild<QLabel*>("chillTextLabel");

    dynamicVideoView = new QGraphicsView(this);
    dynamicVideoScene = new QGraphicsScene(dynamicVideoView);
    dynamicVideoItem = new QGraphicsVideoItem();

    dynamicVideoView->setScene(dynamicVideoScene);
    dynamicVideoScene->addItem(dynamicVideoItem);

    dynamicVideoItem->setAspectRatioMode(Qt::IgnoreAspectRatio);

    if (!dynamicButton) {
        qWarning() << "CRITICAL ERROR: dynamicButton not found in UI. Video will not play.";
        dynamicVideoView->setGeometry(0, 0, 649, 468);
        dynamicVideoView->setParent(ui->centralwidget);
    } else {
        dynamicVideoView->setParent(dynamicButton);
        dynamicVideoView->setGeometry(0, 0, 649, 468);

        dynamicVideoView->setStyleSheet("background-color: transparent; border: none;");
        // Ensure dynamicVideoView passes ALL mouse events through, including hover.
        // It's crucial to also set mouseTracking to false if it's not explicitly needed
        // for internal QGraphicsView interactions.
        dynamicVideoView->setAttribute(Qt::WA_TransparentForMouseEvents);
        dynamicVideoView->setMouseTracking(false); // Explicitly disable mouse tracking for the view

        if (chillTextLabel) {
            chillTextLabel->hide();
            chillTextLabel->raise();
            chillTextLabel->setGeometry(0, 0, 649, 468);
            // Ensure chillTextLabel also passes mouse events through
            chillTextLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
            chillTextLabel->setMouseTracking(false); // Explicitly disable mouse tracking for the label
        }

        // Ensure the button itself is set to track mouse events for hover effects
        dynamicButton->setMouseTracking(true);
    }

    dynamicMediaPlayer->setVideoOutput(dynamicVideoItem);
    dynamicMediaPlayer->setSource(QUrl("qrc:/gif/test.gif"));

    connect(dynamicMediaPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            dynamicMediaPlayer->setPosition(0);
            dynamicMediaPlayer->play();
        }
    });

    connect(dynamicMediaPlayer, &QMediaPlayer::errorOccurred, this, [this](QMediaPlayer::Error error){
        qWarning() << "Dynamic video error:" << error << dynamicMediaPlayer->errorString();
    });

    foregroundPage = ui->forepage;
    foregroundPageIndex = ui->stackedWidget->indexOf(foregroundPage);
    landingPageIndex = ui->stackedWidget->indexOf(ui->landingpage);

    dynamicPage = qobject_cast<Dynamic*>(ui->dynamicpage);
    if (!dynamicPage) {
        dynamicPage = new Dynamic(this);
        ui->stackedWidget->addWidget(dynamicPage);
    }
    dynamicPageIndex = ui->stackedWidget->indexOf(dynamicPage);

    backgroundPage = new Background(this);
    ui->stackedWidget->addWidget(backgroundPage);
    backgroundPageIndex = ui->stackedWidget->indexOf(backgroundPage);

    capturePage = new Capture(this);
    ui->stackedWidget->addWidget(capturePage);
    capturePageIndex = ui->stackedWidget->indexOf(capturePage);

    //Add widget for final output page
    finalOutputPage = new Final(this);
    ui->stackedWidget->addWidget(finalOutputPage);
    finalOutputPageIndex = ui->stackedWidget->indexOf(finalOutputPage);

    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    dynamicMediaPlayer->play();

    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);

    if (dynamicButton) {
        connect(dynamicButton, &QPushButton::clicked, this, &BRBooth::on_dynamicButton_clicked);
    }

    if (dynamicPage) {
        connect(dynamicPage, &Dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);

        connect(dynamicPage, &Dynamic::showCapturePage, this, [this](){
            previousPageIndex = dynamicPageIndex;
            showCapturePage();
        });
    }

    if (backgroundPage) {
        connect(backgroundPage,
                &Background::backtoForegroundPage,
                this,
                &BRBooth::showForegroundPage);
        connect(backgroundPage, &Background::imageSelectedTwice, this, [this]() {
            previousPageIndex = backgroundPageIndex;
            capturePage->setCaptureMode(Capture::ImageCaptureMode);
            showCapturePage();
        });
    }

    if (capturePage) {
        connect(capturePage, &Capture::backtoPreviousPage, this, [this]() {
            if (previousPageIndex == backgroundPageIndex) {
                showBackgroundPage();
            } else if (previousPageIndex == dynamicPageIndex) {
                showDynamicPage();
            }
        });
        connect(capturePage, &Capture::showFinalOutputPage, this, &BRBooth::showFinalOutputPage);
        connect(capturePage, &Capture::imageCaptured, finalOutputPage, &Final::setImage);
        connect(capturePage, &Capture::videoRecorded, finalOutputPage, &Final::setVideo);
    }

    if (finalOutputPage) {
        connect(finalOutputPage, &Final::backToCapturePage, this, &BRBooth::showCapturePage);
        connect(finalOutputPage, &Final::backToLandingPage, this, &BRBooth::showLandingPage);
    }
    connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, [this](int index){
        if (index == landingPageIndex) {
            if (dynamicMediaPlayer->playbackState() != QMediaPlayer::PlayingState) {
                dynamicMediaPlayer->play();
            }
        } else {
            if (dynamicMediaPlayer->playbackState() == QMediaPlayer::PlayingState) {
                dynamicMediaPlayer->stop();
            }
        }

        if (index == foregroundPageIndex) {
            foregroundPage->resetPage();
        }
        if (index == backgroundPageIndex) {
            backgroundPage->resetPage();
        }
        if (index == dynamicPageIndex) {
            dynamicPage->resetPage();
        }
    });
}

BRBooth::~BRBooth()
{
    if (dynamicMediaPlayer) {
        dynamicMediaPlayer->stop();
        delete dynamicMediaPlayer;
        dynamicMediaPlayer = nullptr;
    }
    if (dynamicVideoItem) {
        delete dynamicVideoItem;
        dynamicVideoItem = nullptr;
    }
    if (dynamicVideoScene) {
        delete dynamicVideoScene;
        dynamicVideoScene = nullptr;
    }
    if (dynamicVideoView) {
        delete dynamicVideoView;
        dynamicVideoView = nullptr;
    }

    delete ui;
}

void BRBooth::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    QLabel* chillTextLabel = ui->landingpage->findChild<QLabel*>("chillTextLabel");

    if (dynamicVideoView) {
        dynamicVideoView->setGeometry(0, 0, 649, 468);
        dynamicVideoScene->setSceneRect(0, 0, dynamicVideoView->width(), dynamicVideoView->height());
        dynamicVideoItem->setSize(QSizeF(dynamicVideoView->width(), dynamicVideoView->height()));
    }

    if (chillTextLabel) {
        chillTextLabel->setGeometry(0, 0, 649, 468);
    }
}

void BRBooth::showLandingPage()
{
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    if (dynamicMediaPlayer->playbackState() != QMediaPlayer::PlayingState) {
        dynamicMediaPlayer->play();
    }
}

void BRBooth::showForegroundPage()
{
    ui->stackedWidget->setCurrentIndex(foregroundPageIndex);
    if (dynamicMediaPlayer->playbackState() == QMediaPlayer::PlayingState) {
        dynamicMediaPlayer->stop();
    }
}

void BRBooth::showDynamicPage()
{
    ui->stackedWidget->setCurrentIndex(dynamicPageIndex);
    if (dynamicMediaPlayer->playbackState() == QMediaPlayer::PlayingState) {
        dynamicMediaPlayer->stop();
    }
}

void BRBooth::showBackgroundPage()
{
    ui->stackedWidget->setCurrentIndex(backgroundPageIndex);
    if (dynamicMediaPlayer->playbackState() == QMediaPlayer::PlayingState) {
        dynamicMediaPlayer->stop();
    }
}

void BRBooth::showCapturePage()
{
    ui->stackedWidget->setCurrentIndex(capturePageIndex);
    if (dynamicMediaPlayer->playbackState() == QMediaPlayer::PlayingState) {
        dynamicMediaPlayer->stop();
    }
}

void BRBooth::showFinalOutputPage()
{
    ui->stackedWidget->setCurrentIndex(finalOutputPageIndex);
}

void BRBooth::on_staticButton_clicked()
{
    showForegroundPage();
}

void BRBooth::on_dynamicButton_clicked()
{
    showDynamicPage();
}
