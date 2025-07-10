#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QEvent>
#include <QMouseEvent>
#include <QPushButton>
#include <QTimer>
#include <QWidget>
#include <QLabel>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QMovie>
#include <QResizeEvent>

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

public slots:
    void onDynamicPageShown();

signals:
    void backtoLandingPage();
    void showCapturePage();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status);

private:
    void updateGifLabelsGeometry();
    void setupVideoPlayers();
    void processVideoClick(QObject *buttonObj);
    void applyHighlightStyle(QObject *obj, bool highlight);
    void showOverlayVideo(const QString& videoPath);
    void hideOverlayVideo();

    Ui::Dynamic *ui;

    QVideoWidget* fullscreenVideoWidget;
    QMediaPlayer* fullscreenPlayer;

    QTimer* debounceTimer;
    bool debounceActive;
    QPushButton* currentSelectedVideoWidget;
};

#endif // DYNAMIC_H
