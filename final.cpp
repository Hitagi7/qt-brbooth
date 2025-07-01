#include "final.h"
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include <QVBoxLayout>
#include "iconhover.h"
#include "ui_final.h"
#include <QFileDialog> // Required for QFileDialog
#include <QMessageBox> // Required for QMessageBox

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

    //Setup for displaying captured image
    QVBoxLayout *finalWidgetLayout = qobject_cast<QVBoxLayout*>(ui->finalWidget->layout());
    if(!finalWidgetLayout){
        finalWidgetLayout = new QVBoxLayout(ui->finalWidget);
        finalWidgetLayout->setContentsMargins(0,0,0,0);
    }

    imageDisplayLabel = new QLabel(ui->finalWidget);
    imageDisplayLabel->setAlignment(Qt::AlignCenter);
    imageDisplayLabel->setScaledContents(true);

    finalWidgetLayout->addWidget(imageDisplayLabel);
    finalWidgetLayout->setStretchFactor(imageDisplayLabel,1);
}

Final::~Final()
{
    delete ui;
}

void Final::on_back_clicked()
{
    emit backToCapturePage();
}

void Final::setImage(const QPixmap &image)
{
    if (!image.isNull()) {
        imageDisplayLabel->setScaledContents(true); // Prevent stretching
        imageDisplayLabel->setPixmap(image);
        imageDisplayLabel->setText("");
    } else {
        imageDisplayLabel->clear();
    }
}

void Final::on_save_clicked()
{
    QPixmap imageToSave = imageDisplayLabel->pixmap();

    if (imageToSave.isNull()) {
        QMessageBox::warning(this, "Save Image", "No image to save.");
        return;
    }

    // Get a file name from the user
    QString fileName = QFileDialog::getSaveFileName(this, "Save Image",
                                                    QDir::homePath() + "/untitled.png", // Default path and filename
                                                    "Images (*.png *.jpg *.bmp *.gif)"); // Supported image formats

    if (!fileName.isEmpty()) {
        // Save the image
        if (imageToSave.save(fileName)) {
            QMessageBox::information(this, "Save Image", "Image saved successfully!");
        } else {
            QMessageBox::critical(this, "Save Image", "Failed to save image.");
        }
    }
    emit backToLandingPage();
}
