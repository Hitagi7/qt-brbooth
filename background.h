#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QEvent>

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

public slots:
    void resetPage();

signals:
    void backtoForegroundPage();
    void imageSelectedTwice();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();

    void on_image1_clicked();
    void on_image2_clicked();
    void on_image3_clicked();
    void on_image4_clicked();
    void on_image5_clicked();

private:
    Ui::Background *ui;
    QPushButton *currentSelectedImageButton;
    QTimer *debounceTimer;
    bool debounceActive;

    void applyHighlightStyle(QPushButton *button, bool highlight);
    void processImageButtonClick(QPushButton *button);
};

#endif
