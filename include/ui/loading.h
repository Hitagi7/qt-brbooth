#ifndef LOADING_H
#define LOADING_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QPixmap>
#include <QList>

QT_BEGIN_NAMESPACE
namespace Ui { class Loading; }
QT_END_NAMESPACE

class Loading : public QWidget
{
	Q_OBJECT

public:
	explicit Loading(QWidget *parent = nullptr);
	~Loading();

    // Convenience to set message
    void setMessage(const QString &text);
    // Reset progress to 0
    void resetProgress();
    // Set progress 0-100
    void setProgress(int value);

    // Preview content shown behind the progress bar
    void setImage(const QPixmap &image);
    void setVideo(const QList<QPixmap> &frames, double fps);
    void clearPreview();
    
    // Change loading text color based on background template
    void setLoadingTextColor(const QString &templatePath);

private:
	Ui::Loading *ui;

    // Preview layer
    QLabel *m_previewLabel = nullptr;
    QList<QPixmap> m_videoFrames;
    int m_currentFrameIndex = 0;
    double m_videoFPS = 30.0;
    QTimer *m_videoTimer = nullptr;

    void ensureLayeredLayout();
    void playNextFrame();
};

#endif // LOADING_H


