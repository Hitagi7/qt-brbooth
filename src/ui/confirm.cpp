#include "ui/confirm.h"
#include "ui_confirm.h"
#include <QGridLayout>
#include <QStackedLayout>
#include <QTimer>
#include <QDebug>
#include <QResizeEvent>

Confirm::Confirm(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::Confirm)
{
	ui->setupUi(this);

	// Ensure solid black background regardless of parent styles
	setAttribute(Qt::WA_StyledBackground, true);
	setAutoFillBackground(true);
	{
		QPalette pal = palette();
		pal.setColor(QPalette::Window, Qt::black);
		setPalette(pal);
	}

	// Set up back button icon (same as capture interface)
	ui->backButton->setIcon(QIcon(":/icons/Icons/normal.svg"));
	ui->backButton->setIconSize(QSize(100, 100));

	// Build layered layout: preview label below existing layout
	ensureLayeredLayout();

	// Create video timer for playback
	m_videoTimer = new QTimer(this);
	connect(m_videoTimer, &QTimer::timeout, this, &Confirm::playNextFrame);

	// Connect button signals
	connect(ui->backButton, &QPushButton::clicked, this, &Confirm::onBackButtonClicked);
	connect(ui->confirmButton, &QPushButton::clicked, this, &Confirm::onConfirmButtonClicked);

	// Ensure UI elements stay on top
	if (ui->overlayWidget) {
		ui->overlayWidget->raise();
	}
	if (ui->backButton) {
		ui->backButton->raise();
	}
	if (ui->confirmButton) {
		ui->confirmButton->raise();
	}
	if (ui->confirmLabel) {
		ui->confirmLabel->raise();
	}

	qDebug() << "Confirm UI initialized";
}

Confirm::~Confirm()
{
	stopVideoPlayback();
	delete ui;
}

void Confirm::resizeEvent(QResizeEvent *event)
{
	QWidget::resizeEvent(event);
	
	// Resize the preview label to fill the entire widget
	if (m_previewLabel) {
		m_previewLabel->setGeometry(rect());
		qDebug() << "Confirm: Resized preview label to:" << m_previewLabel->size();
	}
}

void Confirm::ensureLayeredLayout()
{
	if (m_previewLabel)
		return;

	// Create preview label that fills the entire widget
	m_previewLabel = new QLabel(this);
	m_previewLabel->setScaledContents(true);
	m_previewLabel->setAlignment(Qt::AlignCenter);
	m_previewLabel->setStyleSheet("background: black;");
	m_previewLabel->setGeometry(rect()); // Fill entire widget
	
	// Ensure the overlay widget and its contents stay on top
	if (ui->overlayWidget) {
		ui->overlayWidget->raise();
	}
	
	// Lower the preview label so it's behind everything
	m_previewLabel->lower();
	m_previewLabel->show();
	
	qDebug() << "Confirm: Layered layout initialized, preview label size:" << m_previewLabel->size();
}

void Confirm::setVideo(const QList<QPixmap> &frames, double fps)
{
	if (!m_previewLabel) ensureLayeredLayout();

	stopVideoPlayback();

	m_videoFrames = frames;
	m_currentFrameIndex = 0;
	m_videoFPS = fps > 0 ? fps : 30.0;

	qDebug() << "🎬 Confirm: Set video with" << frames.size() << "frames at" << fps << "fps";
	qDebug() << "🎬 Confirm: Using FPS:" << m_videoFPS << "for playback";
	qDebug() << "🎬 Confirm: Calculated interval:" << qMax(1, static_cast<int>(1000.0 / m_videoFPS)) << "ms";

	// Display first frame immediately
	if (!m_videoFrames.isEmpty()) {
		m_previewLabel->setPixmap(m_videoFrames.first());
		m_previewLabel->show();
	}

	// Start video playback
	startVideoPlayback();
}

void Confirm::clearPreview()
{
	stopVideoPlayback();
	m_videoFrames.clear();
	m_currentFrameIndex = 0;
	if (m_previewLabel) m_previewLabel->clear();
}

void Confirm::startVideoPlayback()
{
	if (m_videoFrames.isEmpty() || !m_videoTimer) {
		return;
	}

	// Calculate interval in milliseconds based on FPS (same as Final UI)
	int interval = qMax(1, static_cast<int>(1000.0 / m_videoFPS));
	m_videoTimer->start(interval);
	qDebug() << "Confirm: Started video playback at" << m_videoFPS << "fps (interval:" << interval << "ms)";
}

void Confirm::stopVideoPlayback()
{
	if (m_videoTimer) {
		m_videoTimer->stop();
	}
}

void Confirm::playNextFrame()
{
	if (m_videoFrames.isEmpty() || !m_previewLabel) {
		stopVideoPlayback();
		return;
	}

	// Loop back to start when reaching the end
	if (m_currentFrameIndex >= m_videoFrames.size()) {
		m_currentFrameIndex = 0;
	}

	m_previewLabel->setPixmap(m_videoFrames.at(m_currentFrameIndex));
	m_currentFrameIndex++;
}

void Confirm::onBackButtonClicked()
{
	qDebug() << "Confirm: Back button clicked - returning to capture";
	stopVideoPlayback();
	emit backToCapture();
}

void Confirm::onConfirmButtonClicked()
{
	qDebug() << "Confirm: Confirm button clicked - proceeding to post-processing";
	stopVideoPlayback();
	emit proceedToProcessing();
}

