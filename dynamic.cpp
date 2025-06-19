#include "dynamic.h"
#include "ui_dynamic.h"

dynamic::dynamic(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::dynamic)
{
    ui->setupUi(this);
    //connect(ui->back, &QPushButton::clicked, this, &dynamic::on_back_clicked);
}

dynamic::~dynamic()
{
    delete ui;
}
//void dynamic::on_back_clicked()
//{
//    emit backtoLandingPage();
//}
