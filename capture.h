#ifndef CAPTURE_H
#define CAPTURE_H

#include <QWidget>

namespace Ui {
class Capture;
}

class Capture : public QWidget
{
    Q_OBJECT

public:
    explicit Capture(QWidget *parent = nullptr);
    ~Capture();


signals:
    void backtoBackgroundPage();

private slots:
    void on_back_clicked();

private:
    Ui::Capture *ui;
};

#endif // CAPTURE_H
