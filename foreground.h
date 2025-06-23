#ifndef FOREGROUND_H
#define FOREGROUND_H

#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QEvent>

QT_BEGIN_NAMESPACE

namespace Ui {
    class Foreground;
}

QT_END_NAMESPACE

class Foreground : public QWidget
{
    Q_OBJECT

public:
    explicit Foreground(QWidget *parent = nullptr);
    ~Foreground();

public slots:
    void resetPage();

signals:
    void backtoLandingPage();
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
    void on_image6_clicked();

private:
    Ui::Foreground *ui;
    QPushButton *currentSelectedImageButton;
    QTimer *debounceTimer;
    bool debounceActive;

    void applyHighlightStyle(QPushButton *button, bool highlight);
    void processImageButtonClick(QPushButton *button);
};

#endif // FOREGROUND_H
