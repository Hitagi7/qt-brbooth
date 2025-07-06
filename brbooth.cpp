#include "brbooth.h"
#include "background.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"
#include "ui_brbooth.h"
#include <QStyle>
#include <QDebug>
#include <QMovie>
#include <QLabel>
#include <QEvent>

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

    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");

    if (!dynamicButton) {
        qWarning() << "CRITICAL ERROR: dynamicButton not found in UI.";
        return;
    }

    // Create a QLabel to display the GIF as the button's background
    QLabel* gifLabel = new QLabel(dynamicButton);
    // IMPORTANT: Make the GIF label slightly smaller to allow border to show
    gifLabel->setGeometry(5, 5, dynamicButton->width() - 10, dynamicButton->height() - 10);
    gifLabel->setScaledContents(true);
    gifLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    gifLabel->setMouseTracking(false);
    gifLabel->lower(); // Put it behind the button's text

    // Load and play the GIF
    QMovie* gifMovie = new QMovie(":/gif/test.gif", QByteArray(), gifLabel);
    gifLabel->setMovie(gifMovie);
    gifMovie->start();

    // IMPORTANT: Ensure the button itself can receive hover events
    dynamicButton->setMouseTracking(true);
    dynamicButton->setAttribute(Qt::WA_Hover, true);

    // Style the button with text on top - using a more explicit approach
    dynamicButton->setText("CHILL");
    dynamicButton->setStyleSheet(
        "QPushButton#dynamicButton {"
        "  font-family: 'Arial Black';"
        "  font-size: 80px;"
        "  font-weight: bold;"
        "  color: white;"
        "  background-color: transparent;"
        "  border: 5px solid transparent;"  // Always have a border, just transparent normally
        "  border-radius: 8px;"
        "}"
        "QPushButton#dynamicButton:hover {"
        "  border: 5px solid #FFC20F;"
        "  border-radius: 8px;"
        "  background-color: rgba(255, 194, 15, 0.1);"  // Slight background tint on hover
        "}"
        );

    // Force the button to update its style immediately
    dynamicButton->style()->polish(dynamicButton);

    // DEBUG: Add event filter to detect hover events
    dynamicButton->installEventFilter(this);
    qDebug() << "Event filter installed on dynamicButton for hover debugging";

    // Initialize page references
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

    finalOutputPage = new Final(this);
    ui->stackedWidget->addWidget(finalOutputPage);
    finalOutputPageIndex = ui->stackedWidget->indexOf(finalOutputPage);

    ui->stackedWidget->setCurrentIndex(landingPageIndex);

    // Connect signals
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
        connect(backgroundPage, &Background::backtoForegroundPage, this, &BRBooth::showForegroundPage);
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
    delete ui;
}


void BRBooth::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    // Update the GIF label size when the window resizes
    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");
    if (dynamicButton) {
        QLabel* gifLabel = dynamicButton->findChild<QLabel*>();
        if (gifLabel) {
            // Keep the 5px margin for the border
            gifLabel->setGeometry(5, 5, dynamicButton->width() - 10, dynamicButton->height() - 10);
        }
    }
}

void BRBooth::showLandingPage()
{
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
}

void BRBooth::showForegroundPage()
{
    ui->stackedWidget->setCurrentIndex(foregroundPageIndex);
}

void BRBooth::showDynamicPage()
{
    ui->stackedWidget->setCurrentIndex(dynamicPageIndex);
}

void BRBooth::showBackgroundPage()
{
    ui->stackedWidget->setCurrentIndex(backgroundPageIndex);
}

void BRBooth::showCapturePage()
{
    ui->stackedWidget->setCurrentIndex(capturePageIndex);
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
