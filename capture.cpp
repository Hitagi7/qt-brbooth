#include "capture.h"
#include "ui_capture.h"
#include "iconhover.h"

Capture::Capture(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Capture)
{
    ui->setupUi(this);

    //Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);
}

Capture::~Capture()
{
    delete ui;
}

void Capture::on_back_clicked()
{
    emit backtoBackgroundPage();
}

