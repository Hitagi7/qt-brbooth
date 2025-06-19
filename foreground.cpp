#include "foreground.h"
#include "ui_foreground.h"

Foreground::Foreground(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Foreground)
{
    ui->setupUi(this);
}

Foreground::~Foreground()
{
    delete ui;
}

void Foreground::on_back_clicked()
{
    emit backtoLandingPage();
}

