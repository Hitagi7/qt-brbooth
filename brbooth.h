#ifndef BRBOOTH_H
#define BRBOOTH_H

#include <QMainWindow>
#include "foreground.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class BRBooth;
}
QT_END_NAMESPACE

class BRBooth : public QMainWindow
{
    Q_OBJECT

public:
    BRBooth(QWidget *parent = nullptr);
    ~BRBooth();

private slots:
    void on_pushButton_clicked();

    void showLandingPage();
    void showForegroundPage();

private:
    Ui::BRBooth *ui;
    Foreground *foregroundPage;
    int foregroundPageIndex;
};
#endif // BRBOOTH_H
