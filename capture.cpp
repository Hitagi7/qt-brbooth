#include "capture.h"
#include "ui_capture.h"

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
{
    ui->setupUi(this);
}

Capture::~Capture()
{
    delete ui;
}
