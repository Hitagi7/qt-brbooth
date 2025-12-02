#include "ui/loading.h"
#include "ui_loading.h"
#include <QGridLayout>
#include <QStackedLayout>
#include <QTimer>

Loading::Loading(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::Loading)
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

	// Center label and progress bar horizontally
	if (ui->progressBar && ui->loadingLabel) {
		ui->verticalLayout->setAlignment(ui->loadingLabel, Qt::AlignHCenter);
		ui->verticalLayout->setAlignment(ui->progressBar, Qt::AlignHCenter);
		ui->progressBar->setMaximumWidth(600);
	}

	// Styling: black background, white text, green progress bar
	setStyleSheet("QWidget#Loading { background-color: black; }"
	            "QLabel#loadingLabel { color: white; font-size: 24px; font-weight: bold; }"
	            "QProgressBar { border: 2px solid #1e1e1e; border-radius: 8px; background: #111; color: white; text-align: center; }"
	            "QProgressBar::chunk { background-color: #0bc200; }");

	// Build layered layout: preview label below existing layout
	ensureLayeredLayout();

	// Ensure progress bar starts at 0 with text hidden (we control text via label)
	ui->progressBar->setRange(0, 100);
	ui->progressBar->setValue(0);
}

Loading::~Loading()
{
	delete ui;
}

void Loading::setMessage(const QString &text)
{
	ui->loadingLabel->setText(text);
}

void Loading::resetProgress()
{
	ui->progressBar->setValue(0);
}

void Loading::setProgress(int value)
{
	ui->progressBar->setValue(value);
}

void Loading::ensureLayeredLayout()
{
	if (m_previewLabel)
		return;

	// Create preview label as a sibling layered behind the vertical layout
	m_previewLabel = new QLabel(this);
	m_previewLabel->setScaledContents(true);
	m_previewLabel->setAlignment(Qt::AlignCenter);
	m_previewLabel->setStyleSheet("background: black;");
	m_previewLabel->lower();

	// Replace the top-level layout with a StackAll stacked layout
	QGridLayout *mainLayout = new QGridLayout(this);
	mainLayout->setContentsMargins(0, 0, 0, 0);
	mainLayout->setSpacing(0);

	QStackedLayout *stack = new QStackedLayout;
	stack->setStackingMode(QStackedLayout::StackAll);
	stack->setContentsMargins(0, 0, 0, 0);
	stack->setSpacing(0);

	// Take existing central widget elements (the ui->verticalLayout belongs to this widget)
	// We can wrap a container to host the existing layout
	QWidget *overlayContainer = new QWidget(this);
	overlayContainer->setLayout(ui->verticalLayout);
	overlayContainer->setAttribute(Qt::WA_TransparentForMouseEvents, false);
	overlayContainer->setStyleSheet("background: transparent;");

	stack->addWidget(m_previewLabel);
	stack->addWidget(overlayContainer);

	mainLayout->addLayout(stack, 0, 0);
	setLayout(mainLayout);

	// Ensure overlay is on top
	overlayContainer->raise();
	m_previewLabel->lower();
}

void Loading::setImage(const QPixmap &image)
{
	if (!m_previewLabel) ensureLayeredLayout();
	m_videoFrames.clear();
	m_currentFrameIndex = 0;
	if (m_videoTimer) { m_videoTimer->stop(); }
	m_previewLabel->setPixmap(image);
	m_previewLabel->show();
}

void Loading::setVideo(const QList<QPixmap> &frames, double fps)
{
	if (!m_previewLabel) ensureLayeredLayout();
	m_videoFrames = frames;
	m_currentFrameIndex = 0;
	m_videoFPS = fps > 0 ? fps : 30.0;
	
	// For progress bar UI, display only the first frame instead of playing the video
	if (!m_videoFrames.isEmpty()) {
		m_previewLabel->setPixmap(m_videoFrames.first());
		m_previewLabel->show();
	}
	
	// Stop any existing video timer
	if (m_videoTimer) {
		m_videoTimer->stop();
	}
}

void Loading::setVideoWithComparison(const QList<QPixmap> &frames, const QList<QPixmap> &originalFrames, double fps)
{
	if (!m_previewLabel) ensureLayeredLayout();
	m_videoFrames = frames; // Store processed frames for reference
	m_currentFrameIndex = 0;
	m_videoFPS = fps > 0 ? fps : 30.0;
	
	qDebug() << "Loading: Set video with comparison - Processed:" << frames.size() 
	         << "Original:" << originalFrames.size() << "frames";
	qDebug() << "Loading: Using ORIGINAL frame as background (before lighting correction)";
	
	// Use ORIGINAL frame as background (before lighting correction is applied)
	if (!originalFrames.isEmpty()) {
		m_previewLabel->setPixmap(originalFrames.first());
		m_previewLabel->show();
		qDebug() << "Loading: Set background to original frame (pre-lighting correction)";
	} else if (!m_videoFrames.isEmpty()) {
		// Fallback to processed frame if no original available
		m_previewLabel->setPixmap(m_videoFrames.first());
		m_previewLabel->show();
		qDebug() << "Loading: Fallback to processed frame (no original available)";
	}
	
	// Stop any existing video timer
	if (m_videoTimer) {
		m_videoTimer->stop();
	}
}

void Loading::clearPreview()
{
	if (m_videoTimer) m_videoTimer->stop();
	m_videoFrames.clear();
	m_currentFrameIndex = 0;
	if (m_previewLabel) m_previewLabel->clear();
}

void Loading::playNextFrame()
{
	if (m_videoFrames.isEmpty() || !m_previewLabel) {
		if (m_videoTimer) m_videoTimer->stop();
		return;
	}
	if (m_currentFrameIndex >= m_videoFrames.size()) m_currentFrameIndex = 0;
	m_previewLabel->setPixmap(m_videoFrames.at(m_currentFrameIndex));
	m_currentFrameIndex++;
}

void Loading::setLoadingTextColor(const QString &templatePath)
{
	// Check if the background template is image6 (bg6.png) - light background
	bool isImage6 = templatePath.contains("bg6.png");
	
	if (isImage6) {
		// Black text for image6 (light background) - retain all other styles
		ui->loadingLabel->setStyleSheet("#loadingLabel { color: #000; text-align: center; font-family: \"Roboto Condensed\"; font-style: italic; font-weight: 700; font-size: 50px; }");
		qDebug() << "Loading text color changed to black for image6 background template:" << templatePath;
	} else {
		// White text for all other backgrounds (default) - retain all other styles
		ui->loadingLabel->setStyleSheet("#loadingLabel { color: #FFF; text-align: center; font-family: \"Roboto Condensed\"; font-style: italic; font-weight: 700; font-size: 50px; }");
		qDebug() << "Loading text color set to white for background template:" << templatePath;
	}
}


