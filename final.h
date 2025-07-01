#ifndef FINAL_H
#define FINAL_H

#include <QWidget>
#include <QPixmap>
#include <QLabel>
#include <QList>
#include <QTimer>

namespace Ui {
class Final;
}

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

private slots:
    void on_back_clicked();
    void playNextFrame();

private:
    Ui::Final *ui;
    QLabel *imageDisplayLabel;

    //Video PLayback
    QList<QPixmap> m_videoFrames; //Stores the frames
    QTimer *videoPlaybackTimer; //Timer to advance frames
    int m_currentFrameIndex; //Current frame being displayed

};

#endif // FINAL_H
