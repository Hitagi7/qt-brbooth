#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <QWidget>
#include "foreground.h"

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

private:
    Ui::Background *ui;
};

#endif
