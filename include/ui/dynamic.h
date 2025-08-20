#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
// All these includes are fine for forward declarations or if directly used in header for members
#include <QLabel>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QMovie>
#include <QPushButton> // Required for QPushButton* member
#include <QTimer> // Required for QTimer* member
#include <QTabWidget>
#include <QVBoxLayout>

// Forward declaration for Ui::Dynamic
namespace Ui {
class Dynamic;
}

// Forward declaration of Iconhover if it's used
class Iconhover;

class Dynamic : public QWidget
{
    Q_OBJECT

public:
    explicit Dynamic(QWidget* parent = nullptr);
    ~Dynamic();

    void resetPage();
    void onDynamicPageShown(); // Slot to call when this page becomes visible
    void stopAllGifs(); // Public method to stop all GIFs

signals:
    void backtoLandingPage();
    // Renamed from showCapturePage() to more clearly indicate a confirmed selection for capture
    void videoSelectedAndConfirmed();

protected:
    // Event filter to handle clicks on the fullscreen video and background
    bool eventFilter(QObject *obj, QEvent *event) override;
    // Overridden to update GIF label geometry on resize
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void on_back_clicked();
    void resetDebounce();
    void onPlayerMediaStatusChanged(QMediaPlayer::MediaStatus status);
    void processVideoClick(QObject *buttonObj); // Handles clicks on individual video buttons

private:
    Ui::Dynamic *ui;

    QVideoWidget* fullscreenVideoWidget;
    QMediaPlayer* fullscreenPlayer;

    QTimer* debounceTimer;
    bool debounceActive;
    QPushButton* currentSelectedVideoWidget; // Stores the button that was last clicked (for highlighting)

    // Helper functions
    void updateGifLabelsGeometry();
    void setupVideoPlayers();
    void applyHighlightStyle(QObject *obj, bool highlight);
    void showOverlayVideo(const QString& videoPath);
    void hideOverlayVideo();
};

#endif // DYNAMIC_H
