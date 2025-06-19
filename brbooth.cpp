#include "brbooth.h"
#include "ui_brbooth.h"
#include "foreground.h"
#include "dynamic.h"

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    ui->setupUi(this);
    this->setStyleSheet("QMainWindow#BRBooth {"
                        "   background-image: url(:/images/pics/bg.png);"
                        "   background-repeat: no-repeat;"
                        "   background-size: cover;"
                        "   background-position: center;"
                        "}");

    foregroundPage = ui->forepage;
    foregroundPageIndex = ui->stackedWidget->indexOf(foregroundPage);
    landingPageIndex = ui->stackedWidget->indexOf(ui->landingpage);
    dynamicPage = ui->dynamicpage;
    dynamicPageIndex = ui->stackedWidget->indexOf(dynamicPage);

    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);

    if (dynamicPage) {
        connect(dynamicPage, &dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);
    }
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

void BRBooth::on_staticButton_clicked()
{
    showForegroundPage();
}

void BRBooth::on_dynamicButton_clicked()
{
    showdynamicPage();
}
