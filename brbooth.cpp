#include "brbooth.h"
#include "ui_brbooth.h"
#include "foreground.h"

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
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);
}

BRBooth::~BRBooth()
{
    delete ui;
}

void BRBooth::on_pushButton_clicked()
{
    showForegroundPage();
}

void BRBooth::showLandingPage()
{
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
}

void BRBooth::showForegroundPage()
{
    ui->stackedWidget->setCurrentIndex(foregroundPageIndex);
}

