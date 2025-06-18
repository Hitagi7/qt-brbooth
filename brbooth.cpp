#include "brbooth.h"
#include "ui_brbooth.h"
#include "foreground.h"

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    ui->setupUi(this);
    this->setStyleSheet("QDialog { background-image: url(:/images/bg.png); }");
}

BRBooth::~BRBooth()
{
    delete ui;
}

void BRBooth::on_pushButton_clicked()
{
    this->hide();

    Foreground foregroundDialog(this);
    foregroundDialog.exec();

    this->show();
}

