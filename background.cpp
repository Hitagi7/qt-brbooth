#include "background.h"
#include "ui_background.h"

#include <QStyle>

Background::Background(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Background)
{
    ui->setupUi(this);
    connect(ui->back, &QPushButton::clicked, this, &Background::on_back_clicked);

    currentSelectedImageButton = nullptr;

    if (ui->image1) {
        connect(ui->image1, &QPushButton::clicked, this, &Background::on_image1_clicked);
        ui->image1->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image2) {
        connect(ui->image2, &QPushButton::clicked, this, &Background::on_image2_clicked);
        ui->image2->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image3) {
        connect(ui->image3, &QPushButton::clicked, this, &Background::on_image3_clicked);
        ui->image3->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image4) {
        connect(ui->image4, &QPushButton::clicked, this, &Background::on_image4_clicked);
        ui->image4->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image5) {
        connect(ui->image5, &QPushButton::clicked, this, &Background::on_image5_clicked);
        ui->image5->setFocusPolicy(Qt::NoFocus);
    }
}

Background::~Background()
{
    delete ui;
}

void Background::on_back_clicked()
{
    if (currentSelectedImageButton) {
        currentSelectedImageButton->setProperty("selected", false);
        currentSelectedImageButton->style()->polish(currentSelectedImageButton);
    }
    emit backtoLandingPage();
}

void Background::setImageSelected(QPushButton *button)
{
    if (currentSelectedImageButton && currentSelectedImageButton != button) {
        currentSelectedImageButton->setProperty("selected", false);
        currentSelectedImageButton->style()->polish(currentSelectedImageButton);
    }

    if (button) {
        button->setProperty("selected", true);
        button->style()->polish(button);
    }

    currentSelectedImageButton = button;
}

void Background::on_image1_clicked()
{
    setImageSelected(ui->image1);
}

void Background::on_image2_clicked()
{
    setImageSelected(ui->image2);
}

void Background::on_image3_clicked()
{
    setImageSelected(ui->image3);
}

void Background::on_image4_clicked()
{
    setImageSelected(ui->image4);
}

void Background::on_image5_clicked()
{
    setImageSelected(ui->image5);
}

