#ifndef FOREGROUND_H
#define FOREGROUND_H

#include <QDialog>

QT_BEGIN_NAMESPACE
namespace Ui{
class Foreground;
}
QT_END_NAMESPACE

class Foreground : public QDialog
{
    Q_OBJECT

public:
    explicit Foreground(QWidget *parent = nullptr);
    ~Foreground();

private:
    Ui::Foreground *ui;
};

#endif // FOREGROUND_H
