#include "brbooth.h"
#include "ui_brbooth.h"

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
{
    ui->setupUi(this);
}

BRBooth::~BRBooth()
{
    delete ui;
}
