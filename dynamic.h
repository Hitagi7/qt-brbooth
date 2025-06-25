#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
#include <QPushButton> // Keep QPushButton for your 'back' button
#include <QTimer>
#include <QEvent>
#include <QMouseEvent> // Included for Icon Hover for Back Button

#include <QMediaPlayer>
#include <QVideoWidget>
#include <QMap> // To store multiple video
#include <QLabel> // New: Include QLabel for displaying thumbnails

QT_BEGIN_NAMESPACE
namespace Ui {
class Dynamic;
}
QT_END_NAMESPACE

// Forward declaration of Iconhover if it's used
class Iconhover;

class Dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit Dynamic(QWidget *parent = nullptr);
    ~Dynamic();

public slots:
    void resetPage();

signals:
    void backtoLandingPage();
    void videoSelectedTwice(); // Changed signal name to reflect video selection

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void processVideoClick(QObject *videoWIdgetObj);

private:
    Ui::Dynamic *ui;
    QObject *currentSelectedVideoWidget;
    QTimer *debounceTimer;
    bool debounceActive;

    QMap<QString, QMediaPlayer*> videoPlayers;
    QMap<QString, QVideoWidget*> videoWidgets;
    QMap<QString, QLabel*> thumbnailLabels; // New: To store QLabel for thumbnails

    void applyHighlightStyle(QObject *obj, bool highlight);
    void setupVideoPlayers();
    void showThumbnail(QObject *videoWidgetObj, bool show); // New: Helper to show/hide thumbnail
};

#endif // DYNAMIC_H
