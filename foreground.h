#ifndef FOREGROUND_H
#define FOREGROUND_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui{
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

private:
    Ui::Foreground *ui;
};

#endif // FOREGROUND_H
