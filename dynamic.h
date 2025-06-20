#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>

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

private:
    Ui::Dynamic *ui;
};

#endif // DYNAMIC_H
