#ifndef FOREGROUND_H
#define FOREGROUND_H

#include <QWidget>
#include <QPushButton>

QT_BEGIN_NAMESPACE

namespace Ui {
    class Foreground;
}

QT_END_NAMESPACE

class Foreground : public QWidget
{
    Q_OBJECT

public:
    explicit Foreground(QWidget *parent = nullptr);
    ~Foreground();

signals:
    void backtoLandingPage();
private slots:
    void on_back_clicked();

    void on_image1_clicked();

    void on_image2_clicked();

    void on_image3_clicked();

    void on_image4_clicked();

    void on_image5_clicked();

    void on_image6_clicked();

private:
    Ui::Foreground *ui;
    QPushButton *currentSelectedImageButton;

    void setImageSelected(QPushButton *button); // Correct declaration
};

#endif // FOREGROUND_H
