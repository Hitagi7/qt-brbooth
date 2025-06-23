#include "brbooth.h"
#include "ui_brbooth.h"
#include "foreground.h"
#include "dynamic.h"
#include "background.h"
#include "capture.h"

// Boilerplate
BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    ui->setupUi(this);
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

    // Static
    // Show landing page upon pressing back button from foreground
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);

    // Show background page upon double clicking a template
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);

    // Dynamic back button
    if (dynamicPage) {
        connect(dynamicPage, &Dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);
    }

    // Static background back button
    if (backgroundPage) {
        connect(backgroundPage, &Background::backtoLandingPage, this, &BRBooth::showForegroundPage);
        //connect(backgroundPage, &Background::imageSelectedTwice, this, &BRBooth::showCapturePage);
    }

    // Resets static foreground page everytime its loaded
    connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, [this](int index){
        if (index == foregroundPageIndex) {
            foregroundPage->resetPage();
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

void BRBooth::showdynamicPage()
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

void BRBooth::on_staticButton_clicked()
{
    showForegroundPage();
}

void BRBooth::on_dynamicButton_clicked()
{
    showdynamicPage();
}
