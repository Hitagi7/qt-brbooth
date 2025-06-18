#include "foreground.h"
#include "ui_foreground.h"

Foreground::Foreground(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::Foreground)
{
    ui->setupUi(this);
}

Foreground::~Foreground()
{
    delete ui;
}
