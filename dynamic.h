#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>

QT_BEGIN_NAMESPACE

namespace Ui {
    class dynamic;
}

QT_END_NAMESPACE

class dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit dynamic(QWidget *parent = nullptr);
    ~dynamic();

signals:
    void backtoLandingPage();
private slots:
    void on_back_clicked();

private:
    Ui::dynamic *ui;
};

#endif // DYNAMIC_H
