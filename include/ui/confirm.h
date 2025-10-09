#ifndef CONFIRM_H
#define CONFIRM_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QPixmap>
#include <QList>

QT_BEGIN_NAMESPACE
namespace Ui { class Confirm; }
QT_END_NAMESPACE

class Confirm : public QWidget
{
	Q_OBJECT

public:
	explicit Confirm(QWidget *parent = nullptr);
	~Confirm();

	// Set the video to display (recorded frames without post-processing)
	void setVideo(const QList<QPixmap> &frames, double fps);
	
	// Clear the video preview
	void clearPreview();

signals:
	// Signal emitted when back button is clicked
	void backToCapture();
	
	// Signal emitted when confirm button is clicked
	void proceedToProcessing();

private slots:
	void onBackButtonClicked();
	void onConfirmButtonClicked();

protected:
	void resizeEvent(QResizeEvent *event) override;

private:
	Ui::Confirm *ui;

	// Video preview layer (background)
	QLabel *m_previewLabel = nullptr;
	QList<QPixmap> m_videoFrames;
	int m_currentFrameIndex = 0;
	double m_videoFPS = 30.0;
	QTimer *m_videoTimer = nullptr;

	void ensureLayeredLayout();
	void playNextFrame();
	void startVideoPlayback();
	void stopVideoPlayback();
};

#endif // CONFIRM_H

