#include "dynamic.h"
#include "ui_dynamic.h"
#include <QStyle> // Required for style()->polish()

Dynamic::Dynamic(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Dynamic)
{
    ui->setupUi(this);
    connect(ui->back, &QPushButton::clicked, this, &Dynamic::on_back_clicked);

    currentSelectedImageButton = nullptr; // Initialize to no selection

    // Connect image buttons and set focus policy
    if (ui->image1) {
        connect(ui->image1, &QPushButton::clicked, this, &Dynamic::on_image1_clicked);
        ui->image1->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image2) {
        connect(ui->image2, &QPushButton::clicked, this, &Dynamic::on_image2_clicked);
        ui->image2->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image3) {
        connect(ui->image3, &QPushButton::clicked, this, &Dynamic::on_image3_clicked);
        ui->image3->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image4) {
        connect(ui->image4, &QPushButton::clicked, this, &Dynamic::on_image4_clicked);
        ui->image4->setFocusPolicy(Qt::NoFocus);
    }

    if (ui->image5) {
        connect(ui->image5, &QPushButton::clicked, this, &Dynamic::on_image5_clicked);
        ui->image5->setFocusPolicy(Qt::NoFocus);
    }

}

Dynamic::~Dynamic()
{
    delete ui;
}

void Dynamic::on_back_clicked()
{
    // Clear selection when navigating back
    if (currentSelectedImageButton) {
        currentSelectedImageButton->setProperty("selected", false);
        currentSelectedImageButton->style()->polish(currentSelectedImageButton);
    }
    emit backtoLandingPage();
}

void Dynamic::setImageSelected(QPushButton *button)
{
    // Deselect the previously selected button, if any
    if (currentSelectedImageButton && currentSelectedImageButton != button) {
        currentSelectedImageButton->setProperty("selected", false);
        currentSelectedImageButton->style()->polish(currentSelectedImageButton);
    }

    // Select the new button
    if (button) {
        button->setProperty("selected", true);
        button->style()->polish(button);
    }

    // Update the currently selected button
    currentSelectedImageButton = button;
}

// Implement the individual image click handlers
void Dynamic::on_image1_clicked()
{
    setImageSelected(ui->image1);
}

void Dynamic::on_image2_clicked()
{
    setImageSelected(ui->image2);
}

void Dynamic::on_image3_clicked()
{
    setImageSelected(ui->image3);
}

void Dynamic::on_image4_clicked()
{
    setImageSelected(ui->image4);
}

void Dynamic::on_image5_clicked()
{
    setImageSelected(ui->image5);
}

