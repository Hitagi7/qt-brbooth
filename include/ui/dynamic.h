#ifndef DYNAMIC_H
#define DYNAMIC_H

#include <QWidget>
#include <QString>
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
    void restoreSelection(); // Restore the previously selected video template

signals:
    void backtoLandingPage();
    // Renamed from showCapturePage() to more clearly indicate a confirmed selection for capture
    void videoSelectedAndConfirmed(const QString &videoPath);

protected:
    // Event filter to handle clicks on the fullscreen video and background
    bool eventFilter(QObject *obj, QEvent *event) override;
    // Overridden to update GIF label geometry on resize
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void on_back_clicked();
    void onVideoPreviewBackClicked(); // New slot for video preview back button
    void resetDebounce();
    void processVideoClick(QObject *buttonObj); // Handles clicks on individual video buttons

private:
    Ui::Dynamic *ui;

    QWidget* fullscreenStackWidget; // Stack widget to layer video and back button
    QVideoWidget* fullscreenVideoWidget;
    QMediaPlayer* fullscreenPlayer;
    QPushButton* videoPreviewBackButton; // Dedicated back button for video preview

    QTimer* debounceTimer;
    bool debounceActive;
    QPushButton* currentSelectedVideoWidget; // Stores the button that was last clicked (for highlighting)
    QString m_selectedVideoPath; // Absolute path for the selected video, emitted on confirm
    QString m_lastSelectedVideoPath; // Store the last selected video path for persistence

    // Helper functions
    void updateGifLabelsGeometry();
    void setupVideoPlayers();
    void applyHighlightStyle(QObject *obj, bool highlight);
    void showOverlayVideo(const QString& videoPath);
    void hideOverlayVideo();
};

#endif // DYNAMIC_H
