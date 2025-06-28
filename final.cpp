#include "final.h"
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include "iconhover.h"
#include "ui_final.h"

Final::Final(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Final)
{
    ui->setupUi(this);

    //Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);
}

Final::~Final()
{
    delete ui;
}

void Final::on_back_clicked()
{
    emit backToCapturePage();
}
