#ifndef FINAL_H
#define FINAL_H

#include <QWidget>
#include <QPixmap>
#include <QList>
#include <QTimer> // Forward declaration for QTimer
#include <QLabel> // Forward declaration for QLabel

// Forward declare OpenCV classes to avoid including heavy headers in .h if possible
// This is good practice for minimizing compile times, but for cv::Mat and cv::VideoWriter
// it's often simpler to just include <opencv2/opencv.hpp> in the .cpp file where they are used.
// For now, we'll keep it simple and include in the .cpp.

QT_BEGIN_NAMESPACE
namespace Ui { class Final; }
QT_END_NAMESPACE

class Final : public QWidget
{
    Q_OBJECT

public:
    explicit Final(QWidget *parent = nullptr);
    ~Final();

    void setImage(const QPixmap &image);
    void setVideo(const QList<QPixmap> &frames);

signals:
    void backToCapturePage();
    void backToLandingPage();

private slots:
    void on_back_clicked();
    void on_save_clicked();
    void playNextFrame(); // Slot to advance video playback

private:
    Ui::Final *ui;
    QLabel *imageDisplayLabel; // Used for both image and video display

    QTimer *videoPlaybackTimer;
    QList<QPixmap> m_videoFrames; // Stores frames for video playback/saving
    int m_currentFrameIndex;

    // New private helper function for saving video
    void saveVideoToFile();
};

#endif // FINAL_H
