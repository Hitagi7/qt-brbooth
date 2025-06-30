#ifndef FINAL_H
#define FINAL_H

#include <QWidget>
#include <QPixmap>
#include <QLabel>

namespace Ui {
class Final;
}

class Final : public QWidget
{
    Q_OBJECT

public:
    explicit Final(QWidget *parent = nullptr);
    ~Final();

    void setImage(const QPixmap &image);

signals:
    void backToCapturePage();

private slots:
    void on_back_clicked();

private:
    Ui::Final *ui;
    QLabel *imageDisplayLabel;
};

#endif // FINAL_H
