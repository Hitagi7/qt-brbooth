#include "foreground.h"
#include "ui_foreground.h"
#include <QStyle>

Foreground::Foreground(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Foreground)
{
    ui->setupUi(this);
    connect(ui->back, &QPushButton::clicked, this, &Foreground::on_back_clicked);

    currentSelectedImageButton = nullptr;

    if (ui->image1) {
        connect(ui->image1, &QPushButton::clicked, this, &Foreground::on_image1_clicked);
        ui->image1->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image2) {
        connect(ui->image2, &QPushButton::clicked, this, &Foreground::on_image2_clicked);
        ui->image2->setFocusPolicy(Qt::NoFocus);
    }

    // IMPORTANT: Add similar blocks for image3, image4, etc., if you have them.
    /*
    if (ui->image3) {
        connect(ui->image3, &QPushButton::clicked, this, &Foreground::on_image3_clicked);
        ui->image3->setFocusPolicy(Qt::NoFocus);
    }
    */
}

Foreground::~Foreground()
{
    delete ui;
}

void Foreground::on_back_clicked()
{
    currentSelectedImageButton->setProperty("selected", false);
    currentSelectedImageButton->style()->polish(currentSelectedImageButton);
    emit backtoLandingPage();
}

void Foreground::setImageSelected(QPushButton *button)
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

void Foreground::on_image1_clicked()
{
    setImageSelected(ui->image1);
}

void Foreground::on_image2_clicked()
{
    setImageSelected(ui->image2);
}

void Foreground::on_image3_clicked()
{
    setImageSelected(ui->image3);
}


void Foreground::on_image4_clicked()
{
    setImageSelected(ui->image4);
}


void Foreground::on_image5_clicked()
{
    setImageSelected(ui->image5);
}


void Foreground::on_image6_clicked()
{
    setImageSelected(ui->image6);
}

