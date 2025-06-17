#ifndef BRBOOTH_H
#define BRBOOTH_H

#include <QMainWindow>

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

private:
    Ui::BRBooth *ui;
};
#endif // BRBOOTH_H
