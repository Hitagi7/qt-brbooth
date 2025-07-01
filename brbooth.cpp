#include "brbooth.h"
#include "background.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"
#include "ui_brbooth.h"
#include "videotemplate.h"

// Boilerplate
BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    qDebug() << "OpenCV Version: " << CV_VERSION;
    ui->setupUi(this);
    qDebug() << "OpenCV Version: " << CV_VERSION;
    this->setStyleSheet("QMainWindow#BRBooth {"
                        "    background-image: url(:/images/pics/bg.jpg);"
                        "    background-repeat: no-repeat;"
                        "    background-position: center;"
                        "}");

    // Widgets are already added for static and dynamic
    foregroundPage = ui->forepage;
    foregroundPageIndex = ui->stackedWidget->indexOf(foregroundPage);
    landingPageIndex = ui->stackedWidget->indexOf(ui->landingpage);
    dynamicPage = ui->dynamicpage;
    dynamicPageIndex = ui->stackedWidget->indexOf(dynamicPage);

    // Add widget for background page
    backgroundPage = new Background(this);
    ui->stackedWidget->addWidget(backgroundPage);
    backgroundPageIndex = ui->stackedWidget->indexOf(backgroundPage);

    // Add widget for capture page
    capturePage = new Capture(this);
    ui->stackedWidget->addWidget(capturePage);
    capturePageIndex = ui->stackedWidget->indexOf(capturePage);

    //Add widget for final output page
    finalOutputPage = new Final(this);
    ui->stackedWidget->addWidget(finalOutputPage);
    finalOutputPageIndex = ui->stackedWidget->indexOf(finalOutputPage);

    // Static
    // Show landing page upon pressing back button from foreground
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);

    // Show background page upon double clicking a template
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);

    // Dynamic back button
    if (dynamicPage) {
        connect(dynamicPage, &Dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);
        // Connect to capture interface, and store the previous page index
        connect(dynamicPage, &Dynamic::videoSelectedTwice, this, [this]() {
            previousPageIndex = dynamicPageIndex; // Store dynamic page as the previous
            capturePage->setCaptureMode(Capture::VideoRecordMode);

            // ADDITION: Set a default video template for now (e.g., 10 seconds)
            // Replace this with actual template logic when you have it
            VideoTemplate defaultVideoTemplate("Default Dynamic Template", 10); // 10 seconds for dynamic
            capturePage->setVideoTemplate(defaultVideoTemplate);

            showCapturePage();
        });
    }

    // Static background back button
    if (backgroundPage) {
        connect(backgroundPage,
                &Background::backtoForegroundPage,
                this,
                &BRBooth::showForegroundPage);
        // Connect to capture interface, and store the previous page index
        connect(backgroundPage, &Background::imageSelectedTwice, this, [this]() {
            previousPageIndex = backgroundPageIndex; // Store background page as the previous
            capturePage->setCaptureMode(Capture::ImageCaptureMode);
            showCapturePage();
        });
    }

    // Capture page back button logic
    if (capturePage) {
        // Connect the back signal to a lambda that checks previousPageIndex
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

    //Final Output Page
    if (finalOutputPage) {
        connect(finalOutputPage, &Final::backToCapturePage, this, &BRBooth::showCapturePage);
        connect(finalOutputPage, &Final::backToLandingPage, this, &BRBooth::showLandingPage);
    }

    // Resets static foreground page everytime its loaded
    connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, [this](int index) {
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

