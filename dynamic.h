// dynamic.h

#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QEvent>
#include <QMouseEvent>
#include <QPushButton>
#include <QTimer>
#include <QWidget>
#include <QLabel>
#include <QMap>
#include <QMediaPlayer>
#include <QVideoWidget> // Using QVideoWidget as per your last working code
#include <QWidget>
#include <QHash>
#include <QStackedLayout>


namespace Ui {
class Dynamic;
}

class Dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit Dynamic(QWidget* parent = nullptr);
    ~Dynamic();

    void resetPage();

signals:
    void backtoLandingPage();
    void showCapturePage();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status);

private:
    void setupVideoPlayers();
    void processVideoClick(QObject *videoWidgetObj);
    void applyHighlightStyle(QObject *obj, bool highlight);
    void showOverlayVideo(const QString& videoPath);
    void hideOverlayVideo();
    void setPlaceholderHoverState(QWidget* placeholder, bool hovered);

    QHash<QString, QWidget*> videoPlaceholders;
    Ui::Dynamic *ui;

    QHash<QString, QMediaPlayer*> videoPlayers;
    QHash<QString, QVideoWidget*> videoWidgets; // Using QVideoWidget as per your last working code
    QHash<QString, QStackedLayout*> videoLayouts;

    QVideoWidget* fullscreenVideoWidget; // Using QVideoWidget as per your last working code
    QMediaPlayer* fullscreenPlayer;

    QTimer* debounceTimer;
    bool debounceActive;
    QVideoWidget* currentSelectedVideoWidget; // Using QVideoWidget as per your last working code
};

#endif // DYNAMIC_H
