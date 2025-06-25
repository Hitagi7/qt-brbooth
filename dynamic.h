#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QEvent>
#include <QMouseEvent>

#include <QMediaPlayer>
#include <QVideoWidget>
#include <QMap>
#include <QLabel>
#include <QStackedLayout> // Included as per your comments

QT_BEGIN_NAMESPACE
namespace Ui {
class Dynamic;
}
QT_END_NAMESPACE

class Iconhover; // Forward declaration

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
    void videoSelectedTwice();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void processVideoClick(QObject *videoWidgetObj); // Corrected parameter name for consistency

private:
    Ui::Dynamic *ui;

    // Changed to QVideoWidget* for type safety as it's always used as such
    QVideoWidget *currentSelectedVideoWidget;
    QTimer *debounceTimer;
    bool debounceActive;

    QMap<QString, QMediaPlayer*> videoPlayers;
    QMap<QString, QVideoWidget*> videoWidgets;
    QMap<QString, QLabel*> thumbnailLabels;
    QMap<QString, QStackedLayout*> videoLayouts;

    void applyHighlightStyle(QObject *obj, bool highlight);
    void setupVideoPlayers();
    void showThumbnail(QObject *videoWidgetObj, bool show);
};

#endif // DYNAMIC_H
