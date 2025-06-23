#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
#include <QPushButton> // Include QPushButton for currentSelectedImageButton
#include <QMouseEvent> //Included for Icon Hover for Back Button

QT_BEGIN_NAMESPACE

namespace Ui {
class Dynamic;
}

QT_END_NAMESPACE

class Dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit Dynamic(QWidget *parent = nullptr);
    ~Dynamic();

signals:
    void backtoLandingPage();

private slots:
    void on_back_clicked();


    void on_image1_clicked();
    void on_image2_clicked();
    void on_image3_clicked();
    void on_image4_clicked();
    void on_image5_clicked();

private:
    Ui::Dynamic *ui;
    QPushButton *currentSelectedImageButton;

    void setImageSelected(QPushButton *button);
};

#endif // DYNAMIC_H
