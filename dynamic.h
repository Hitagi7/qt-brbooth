// dynamic.h

#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QHash>
#include <QStackedLayout>
#include <QTimer>

namespace Ui {
class Dynamic;
}

class Dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit Dynamic(QWidget* parent = nullptr);
    ~Dynamic();

    // Make resetPage() public so BRBooth can call it
    void resetPage(); // <--- CHANGED FROM PRIVATE TO PUBLIC

signals:
    void backtoLandingPage();
    void showCapturePage();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status);

private: // These remain private as they are internal helper functions
    void setupVideoPlayers();
    void processVideoClick(QObject *videoWidgetObj);
    void applyHighlightStyle(QObject *obj, bool highlight);
    void showOverlayVideo(const QString& videoPath);
    void hideOverlayVideo();


    Ui::Dynamic *ui;

    QHash<QString, QMediaPlayer*> videoPlayers;
    QHash<QString, QVideoWidget*> videoWidgets;
    QHash<QString, QStackedLayout*> videoLayouts;

    QVideoWidget* fullscreenVideoWidget;
    QMediaPlayer* fullscreenPlayer;

    QTimer* debounceTimer;
    bool debounceActive;
    QVideoWidget* currentSelectedVideoWidget;
};

#endif // DYNAMIC_H

