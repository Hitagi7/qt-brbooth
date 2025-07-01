#ifndef FINAL_H
#define FINAL_H

#include <QWidget>
#include <QLabel> // Make sure QLabel is included
#include <QPixmap>
#include <QLabel>
#include <QList>
#include <QTimer>


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
    void on_save_clicked(); // This slot will handle saving the image

private:
    Ui::Final *ui;
    QLabel *imageDisplayLabel; // Declare imageDisplayLabel as a member
    void playNextFrame();

    //Video PLayback
    QList<QPixmap> m_videoFrames; //Stores the frames
    QTimer *videoPlaybackTimer; //Timer to advance frames
    int m_currentFrameIndex; //Current frame being displayed

};

#endif // FINAL_H
