#ifndef FINAL_H
#define FINAL_H

#include <QGridLayout> // For the main layout of the Final widget
#include <QLabel>
#include <QList>
#include <QPixmap>
#include <QStackedLayout> // For the stacked layout to layer widgets
#include <QTimer>
#include <QWidget>

// Forward declare OpenCV classes (good practice if only used in .cpp)
// You typically include <opencv2/opencv.hpp> in the .cpp file where these are used.

QT_BEGIN_NAMESPACE
namespace Ui {
class Final;
}
QT_END_NAMESPACE

class Final : public QWidget
{
    Q_OBJECT

public:
    explicit Final(QWidget *parent = nullptr);
    ~Final();

    void setImage(const QPixmap &image);
    void setVideo(const QList<QPixmap> &frames, double fps = 30.0);
    void setForegroundOverlay(const QString &foregroundPath);

signals:
    void backToCapturePage();
    void backToLandingPage();

private slots:
    void on_back_clicked();
    void on_save_clicked();
    void playNextFrame(); // Slot to advance video playback

protected:
    void resizeEvent(QResizeEvent *event) override; // Override for handling widget resizing

private:
    Ui::Final *ui;

    QTimer *videoPlaybackTimer;
    QList<QPixmap> m_videoFrames; // Stores frames for video playback/saving
    int m_currentFrameIndex;
    double m_videoFPS; // Store the FPS for video playback

    QStackedLayout *m_stackedLayout = nullptr; // Member for the stacked layout

    QPixmap m_lastLoadedImage; // To store the original image for resizing

    QLabel *overlayImageLabel; // For pixel art game UI elements

    void saveVideoToFile(); // Helper function for saving video

    void refreshDisplay(); // Helper function to scale and display content
};

#endif // FINAL_H
