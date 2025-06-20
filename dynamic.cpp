#include "dynamic.h"
#include "ui_dynamic.h"

Dynamic::Dynamic(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic)
{
    ui->setupUi(this);
    connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);
}

Dynamic::~Dynamic()
{
    delete ui;
}

void Dynamic::on_back_clicked()
{
    emit backtoLandingPage();
}
