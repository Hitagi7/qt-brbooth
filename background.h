#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <QWidget>
#include <QPushButton>

QT_BEGIN_NAMESPACE

namespace Ui {
    class Background;
}
QT_END_NAMESPACE

class Background  : public QWidget
{
    Q_OBJECT

public:
    explicit Background (QWidget *parent = nullptr);
    ~Background ();

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
    Ui::Background *ui;
    QPushButton *currentSelectedImageButton;

    void setImageSelected(QPushButton *button);
};

#endif
