#include "brbooth.h"
#include "ui_brbooth.h"
#include "foreground.h"

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
    , foregroundPage(nullptr)
    , foregroundPageIndex(-1)
{
    ui->setupUi(this);
    this->setStyleSheet("QDialog { background-image: url(:/images/pics/bg.png); }");
    foregroundPage = new Foreground(this);
    foregroundPageIndex = ui->stackedWidget->addWidget(foregroundPage);
    ui->stackedWidget->setCurrentIndex(0);
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
    ui->stackedWidget->setCurrentIndex(0);
}

void BRBooth::showForegroundPage()
{
    ui->stackedWidget->setCurrentIndex(foregroundPageIndex);
}

