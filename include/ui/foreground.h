#ifndef FOREGROUND_H
#define FOREGROUND_H

#include <QEvent>
#include <QMouseEvent> //Included for Icon Hover for Back Button
#include <QPushButton>
#include <QTimer>
#include <QWidget>

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
    // setter & getters for foreground template selection
    void setSelectedForeground(const QString &path);
    QString getSelectedForeground() const;

public slots:
    void resetPage();

signals:
    void backtoLandingPage();
    void imageSelectedTwice();
    void foregroundChanged(const QString &newPath);

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();

private:
    Ui::Foreground *ui;
    QPushButton *currentSelectedImageButton;
    QTimer *debounceTimer;
    bool debounceActive;

    void applyHighlightStyle(QPushButton *button, bool highlight);
    void processImageButtonClick(QPushButton *button);

    QString selectedForeground; // store user selected foreground template
};

#endif // FOREGROUND_H
