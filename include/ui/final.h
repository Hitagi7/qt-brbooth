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
    void setImageWithComparison(const QPixmap &image, const QPixmap &originalImage);
    void setVideo(const QList<QPixmap> &frames, double fps = 30.0);
    void setVideoWithComparison(const QList<QPixmap> &frames, const QList<QPixmap> &originalFrames, double fps = 30.0);
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
    void showEvent(QShowEvent *event) override; // Override to reset video to beginning when page is shown

private:
    Ui::Final *ui;

    QTimer *videoPlaybackTimer;
    QList<QPixmap> m_videoFrames; // Stores frames for video playback/saving
    QList<QPixmap> m_originalVideoFrames; // Stores original frames before lighting correction
    int m_currentFrameIndex;
    double m_videoFPS; // Store the FPS for video playback

    QStackedLayout *m_stackedLayout = nullptr; // Member for the stacked layout

    QPixmap m_lastLoadedImage; // To store the original image for resizing
    QPixmap m_originalImage;   // To store the original image without lighting correction
    bool m_hasComparisonImages; // Whether we have both versions for comparison
    bool m_hasComparisonVideos; // Whether we have both video versions for comparison

    QLabel *overlayImageLabel; // For pixel art game UI elements

    void saveVideoToFile(); // Helper function for saving video
    bool saveVideoFramesToFile(const QList<QPixmap> &frames, const QString &fileName); // Helper for saving frame list to file

    void refreshDisplay(); // Helper function to scale and display content
};

#endif // FINAL_H
