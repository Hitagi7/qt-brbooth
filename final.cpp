#include "final.h"
#include "ui_final.h"
#include "iconhover.h"
#include <QStyle>
#include <QRegularExpression>
#include <QMouseEvent>

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
