#include "ui/outputpreview.h"
#include "ui_outputpreview.h"
#include "core/session_manager.h"
#include <QDebug>
#include <QIcon>
#include <QPixmap>
#include <QImage>
#include <QFileInfo>
#include <QDir>
#include <QMessageBox>
#include <QRegularExpression>
#include <QStyle>
#include <QShowEvent>
#include <QLabel>
#include <QPainter>
#include <QDateTime>
#include <QFile>
#include "ui/iconhover.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <QVideoWidget>
#include <QMediaPlayer>
#include <QUrl>
#include <QResizeEvent>
#include <QApplication>

OutputPreview::OutputPreview(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::OutputPreview)
    , m_sessionManager(nullptr)
    , m_gridLayout(nullptr)
    , debounceTimer(nullptr)
    , debounceActive(false)
    , fullscreenPreviewWidget(nullptr)
    , fullscreenVideoWidget(nullptr)
    , fullscreenPlayer(nullptr)
    , fullscreenImageLabel(nullptr)
    , previewBackButton(nullptr)
    , m_isPreviewMode(false)
    , m_previewToggleMode(false)
{
    ui->setupUi(this);

    // Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this);
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &OutputPreview::on_back_clicked);
    connect(ui->confirm, &QPushButton::clicked, this, &OutputPreview::on_confirm_clicked);
    connect(ui->previewButton, &QPushButton::toggled, this, &OutputPreview::on_previewButton_toggled);
    
    // Initialize confirm button as disabled
    updateConfirmButtonState();
    
    // Preview button is always enabled (it's a toggle)
    ui->previewButton->setEnabled(true);
    ui->previewButton->setChecked(false);

    // Setup debounce timer
    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &OutputPreview::resetDebounce);
    
    // Setup fullscreen preview widget
    fullscreenPreviewWidget = new QWidget(this);
    fullscreenPreviewWidget->setMinimumSize(QSize(640, 480));
    fullscreenPreviewWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenPreviewWidget->setAttribute(Qt::WA_StyledBackground, true);
    fullscreenPreviewWidget->setStyleSheet("background-color: black;");
    fullscreenPreviewWidget->hide();
    
    // Setup video widget for dynamic outputs
    fullscreenVideoWidget = new QVideoWidget(fullscreenPreviewWidget);
    fullscreenVideoWidget->setMinimumSize(QSize(640, 480));
    fullscreenVideoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenVideoWidget->setAspectRatioMode(Qt::IgnoreAspectRatio);
    fullscreenVideoWidget->hide();
    
    // Setup image label for static outputs
    fullscreenImageLabel = new QLabel(fullscreenPreviewWidget);
    fullscreenImageLabel->setMinimumSize(QSize(640, 480));
    fullscreenImageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    fullscreenImageLabel->setAlignment(Qt::AlignCenter);
    fullscreenImageLabel->setScaledContents(true);
    fullscreenImageLabel->hide();
    
    // Setup media player
    fullscreenPlayer = new QMediaPlayer(this);
    fullscreenPlayer->setVideoOutput(fullscreenVideoWidget);
    
    // Connect player to loop videos
    connect(fullscreenPlayer, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            fullscreenPlayer->setPosition(0);
            fullscreenPlayer->play();
        }
    });
    
    // Setup preview back button
    previewBackButton = new QPushButton(this);
    QIcon backIcon(":/icons/Icons/normal.svg");
    if (!backIcon.isNull()) {
        previewBackButton->setIcon(backIcon);
        previewBackButton->setIconSize(QSize(100, 100));
    } else {
        previewBackButton->setText("←");
    }
    previewBackButton->setMinimumSize(QSize(100, 80));
    previewBackButton->setMaximumSize(QSize(200, 80));
    previewBackButton->setText("");
    previewBackButton->setFlat(true);
    previewBackButton->setStyleSheet(
        "QPushButton { "
        "    background-color: transparent; "
        "    border: none; "
        "    padding: 0px; "
        "    margin: 0px; "
        "}"
    );
    previewBackButton->setWindowFlags(Qt::WindowStaysOnTopHint | Qt::FramelessWindowHint | Qt::Tool);
    previewBackButton->setAttribute(Qt::WA_TranslucentBackground, true);
    previewBackButton->setAttribute(Qt::WA_NoSystemBackground, true);
    previewBackButton->hide();
    
    Iconhover *previewBackButtonHover = new Iconhover(this);
    previewBackButton->installEventFilter(previewBackButtonHover);
    connect(previewBackButton, &QPushButton::clicked, this, &OutputPreview::onPreviewBackClicked);
    
    // Install event filter for click detection
    fullscreenPreviewWidget->installEventFilter(this);
    this->installEventFilter(this);

    // Get grid layout from scroll area widget contents
    QWidget *scrollWidget = ui->scrollAreaWidgetContents;
    if (scrollWidget) {
        m_gridLayout = scrollWidget->findChild<QGridLayout*>("gridLayout");
        if (!m_gridLayout) {
            qWarning() << "OutputPreview: Grid layout not found in scroll area widget contents!";
            qWarning() << "OutputPreview: Creating new grid layout";
            // Create grid layout if it doesn't exist
            m_gridLayout = new QGridLayout(scrollWidget);
            m_gridLayout->setObjectName("gridLayout");
            m_gridLayout->setSpacing(40);
            m_gridLayout->setContentsMargins(60, 20, 60, 20);
            scrollWidget->setLayout(m_gridLayout);
            qDebug() << "OutputPreview: Created new grid layout";
        } else {
            qDebug() << "OutputPreview: Grid layout found successfully";
        }
    } else {
        qWarning() << "OutputPreview: Scroll area widget contents not found!";
    }
}

OutputPreview::~OutputPreview()
{
    if (fullscreenPlayer) {
        fullscreenPlayer->stop();
    }
    delete ui;
}

void OutputPreview::setSessionManager(SessionManager *sessionManager)
{
    m_sessionManager = sessionManager;
}

void OutputPreview::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    qDebug() << "OutputPreview: showEvent called";
    
    // Ensure scroll area is visible
    if (ui->scrollArea) {
        ui->scrollArea->setVisible(true);
        ui->scrollArea->show();
    }
    if (ui->scrollAreaWidgetContents) {
        ui->scrollAreaWidgetContents->setVisible(true);
        ui->scrollAreaWidgetContents->show();
    }
    
    if (m_sessionManager) {
        qDebug() << "OutputPreview: SessionManager available, loading thumbnails";
        loadOutputThumbnails();
    } else {
        qWarning() << "OutputPreview: SessionManager not available in showEvent";
    }
}

void OutputPreview::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    
    // Update preview widget size if in preview mode
    if (m_isPreviewMode && fullscreenPreviewWidget && fullscreenPreviewWidget->isVisible()) {
        QRect widgetRect = this->geometry();
        fullscreenPreviewWidget->setGeometry(widgetRect);
        
        if (fullscreenVideoWidget && fullscreenVideoWidget->isVisible()) {
            fullscreenVideoWidget->setGeometry(fullscreenPreviewWidget->rect());
        }
        
        if (fullscreenImageLabel && fullscreenImageLabel->isVisible()) {
            fullscreenImageLabel->setGeometry(fullscreenPreviewWidget->rect());
            // Update image pixmap to fit new size
            if (!m_previewFilePath.isEmpty() && !isVideoFile(m_previewFilePath)) {
                QPixmap pixmap(m_previewFilePath);
                if (!pixmap.isNull()) {
                    QSize labelSize = fullscreenImageLabel->size();
                    QPixmap scaled = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                    fullscreenImageLabel->setPixmap(scaled);
                }
            }
        }
        
        // Update back button position
        QPoint originalBackButtonPos = ui->back->pos();
        QPoint globalPos = this->mapToGlobal(originalBackButtonPos);
        previewBackButton->move(globalPos);
    }
}

void OutputPreview::loadOutputThumbnails()
{
    qDebug() << "OutputPreview::loadOutputThumbnails() called";
    
    // Clear selection when loading new thumbnails
    m_selectedFiles.clear();
    updateConfirmButtonState();
    
    if (!m_sessionManager) {
        qWarning() << "OutputPreview: SessionManager not set";
        return;
    }

    if (!m_gridLayout) {
        qWarning() << "OutputPreview: Grid layout not available - cannot load thumbnails";
        // Try to initialize it again
        QWidget *scrollWidget = ui->scrollAreaWidgetContents;
        if (scrollWidget) {
            m_gridLayout = scrollWidget->findChild<QGridLayout*>("gridLayout");
            if (!m_gridLayout) {
                m_gridLayout = new QGridLayout(scrollWidget);
                m_gridLayout->setObjectName("gridLayout");
                m_gridLayout->setSpacing(20);
                m_gridLayout->setContentsMargins(10, 10, 10, 10);
                scrollWidget->setLayout(m_gridLayout);
                qDebug() << "OutputPreview: Created grid layout in loadOutputThumbnails";
            }
        }
        if (!m_gridLayout) {
            qWarning() << "OutputPreview: Still cannot get grid layout - aborting";
            return;
        }
    }

    // Clear existing thumbnails
    clearThumbnails();

    // Get all output files from current user folder
    QString currentUserFolder = m_sessionManager->getCurrentUserFolderPath();
    qDebug() << "OutputPreview: Current user folder:" << currentUserFolder;
    
    if (currentUserFolder.isEmpty()) {
        qWarning() << "OutputPreview: Current user folder path is empty!";
        return;
    }
    
    QList<QString> outputFiles = m_sessionManager->getAllOutputFiles();
    qDebug() << "OutputPreview: Found" << outputFiles.size() << "output files in folder:" << currentUserFolder;
    
    for (const QString &file : outputFiles) {
        qDebug() << "OutputPreview: File:" << file;
    }

    if (outputFiles.isEmpty()) {
        qDebug() << "OutputPreview: No output files found in current user folder:" << currentUserFolder;
        // Don't show message box - just show empty grid
        return;
    }

    // Create thumbnail buttons in grid layout (similar to Background page)
    // Use 3 columns like the background template selection
    const int columns = 3;
    const QSize thumbnailSize(425, 305); // Same size as background template buttons

    qDebug() << "OutputPreview: Creating" << outputFiles.size() << "thumbnail buttons";
    
    for (int i = 0; i < outputFiles.size(); ++i) {
        qDebug() << "OutputPreview: Creating thumbnail button" << i << "for:" << outputFiles[i];
        createThumbnailButton(outputFiles[i], i, outputFiles.size(), columns, thumbnailSize);
    }
    
    qDebug() << "OutputPreview: Created" << m_thumbnailButtons.size() << "thumbnail buttons";
    
    // Update scroll area widget contents size based on number of buttons
    if (m_gridLayout && ui->scrollAreaWidgetContents) {
        int rows = (outputFiles.size() + columns - 1) / columns; // Round up division for rows
        int buttonHeight = thumbnailSize.height();
        int spacing = 20;
        int totalHeight = rows * buttonHeight + (rows - 1) * spacing + 40; // Add padding
        
        ui->scrollAreaWidgetContents->setMinimumHeight(totalHeight);
        ui->scrollAreaWidgetContents->setMinimumWidth(1300); // Ensure enough width for 3 columns
        
        qDebug() << "OutputPreview: Set scroll widget height to" << totalHeight << "for" << rows << "rows";
        qDebug() << "OutputPreview: Scroll widget size:" << ui->scrollAreaWidgetContents->size();
        qDebug() << "OutputPreview: Grid layout size:" << m_gridLayout->sizeHint();
        
        // Force update and repaint
        ui->scrollAreaWidgetContents->update();
        ui->scrollAreaWidgetContents->repaint();
        ui->scrollArea->update();
        ui->scrollArea->repaint();
        
        // Ensure scroll area is visible
        ui->scrollArea->setVisible(true);
        ui->scrollAreaWidgetContents->setVisible(true);
    }
}

void OutputPreview::createThumbnailButton(const QString &filePath, int index, int totalCount, int columns, const QSize &thumbnailSize)
{
    if (!m_gridLayout) {
        qWarning() << "OutputPreview: Grid layout not available";
        return;
    }

    // Generate thumbnail - scale to fill entire button without black borders
    QPixmap thumbnail = generateThumbnail(filePath, thumbnailSize);
    
    if (thumbnail.isNull()) {
        qWarning() << "OutputPreview: Failed to generate thumbnail for:" << filePath;
        // Create a placeholder button anyway
        QWidget *scrollWidget = ui->scrollAreaWidgetContents;
        if (!scrollWidget) {
            return;
        }
        QPushButton *button = new QPushButton(scrollWidget);
        button->setMinimumSize(thumbnailSize);
        button->setMaximumSize(thumbnailSize);
        button->setText("Failed to load");
        button->setStyleSheet("QPushButton { border: 2px solid red; background-color: black; color: white; }");
        int row = index / columns;
        int col = index % columns;
        m_gridLayout->addWidget(button, row, col, Qt::AlignCenter);
        m_thumbnailButtons.append(button);
        m_buttonToFileMap[button] = filePath;
        return;
    }
    
    qDebug() << "OutputPreview: Generated thumbnail size:" << thumbnail.size() << "for file:" << filePath;

    // Create button - parent should be scroll area widget contents so it appears in scroll area
    QWidget *scrollWidget = ui->scrollAreaWidgetContents;
    if (!scrollWidget) {
        qWarning() << "OutputPreview: Cannot create button - scroll widget not found";
        return;
    }
    QPushButton *button = new QPushButton(scrollWidget);
    button->setMinimumSize(thumbnailSize);
    button->setMaximumSize(thumbnailSize);
    button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    button->setCursor(Qt::PointingHandCursor);
    button->setProperty("selected", false);
    button->setFocusPolicy(Qt::NoFocus);
    button->setText(""); // Clear any text
    button->setVisible(true);
    button->setEnabled(true);

    // Save thumbnail to a temporary file and use it as background-image in stylesheet
    // This approach matches template buttons and ensures hover works properly
    QString tempPath = QDir::temp().absoluteFilePath(QString("thumb_%1_%2.png").arg(index).arg(QDateTime::currentMSecsSinceEpoch()));
    if (!thumbnail.save(tempPath)) {
        qWarning() << "OutputPreview: Failed to save thumbnail to temp file:" << tempPath;
    }
    
    // Store temp path for cleanup later
    m_tempThumbnailFiles.append(tempPath);

    // Determine if static (image) or dynamic (video) based on file extension
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    bool isStatic = (extension == "png" || extension == "jpg" || extension == "jpeg");
    
    // Create label to indicate static/dynamic type
    QLabel *typeLabel = new QLabel(button);
    typeLabel->setText(isStatic ? "STATIC" : "DYNAMIC");
    typeLabel->setStyleSheet(
        "QLabel {"
        "    background-color: rgba(0, 0, 0, 180);"
        "    color: white;"
        "    font-weight: bold;"
        "    font-size: 10px;"
        "    padding: 4px 8px;"
        "    border-radius: 4px;"
        "}"
    );
    typeLabel->setAlignment(Qt::AlignCenter);
    typeLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    typeLabel->setAttribute(Qt::WA_NoMousePropagation, true);
    typeLabel->lower(); // Place behind button border
    // Position at bottom-left of button
    typeLabel->setGeometry(10, thumbnailSize.height() - 30, 70, 25);
    typeLabel->show();
    qDebug() << "OutputPreview: Added" << (isStatic ? "STATIC" : "DYNAMIC") << "label for:" << filePath;

    // Style sheet matching template selection - use background-image like template buttons
    // This ensures hover works properly since there's no child widget blocking events
    QString normalizedPath = QDir::toNativeSeparators(tempPath);
    // Escape backslashes for stylesheet
    normalizedPath.replace("\\", "/");
    QString styleSheet = QString(
        "QPushButton {"
        "    border: none;"
        "    background-image: url(%1);"
        "    background-repeat: no-repeat;"
        "    background-position: center;"
        "    background-size: cover;"
        "    background-color: transparent;"
        "    border-radius: 8px;"
        "}"
        "QPushButton:hover {"
        "    border: 5px solid #FFC20F;"
        "    border-radius: 8px;"
        "}"
        "QPushButton[selected=\"true\"] {"
        "    border: 5px solid #0BC200;"
        "    border-radius: 8px;"
        "}"
    ).arg(normalizedPath);
    button->setStyleSheet(styleSheet);

    // Install event filter for click handling
    button->installEventFilter(this);

    // Add to grid layout
    // Special handling: if only 2 items, place 2nd in center using column stretch
    // Otherwise: 1st=left, 2nd=center, 3rd=right, 4th=left (below), etc.
    int row = index / columns;
    int col;
    
    if (totalCount == 2) {
        // Special case: 2 items - first left, second centered
        if (index == 0) {
            col = 0;
            // Set column stretch to center the second item
            // Equal stretch on columns 0 and 2, no stretch on column 1
            m_gridLayout->setColumnStretch(0, 1);
            m_gridLayout->setColumnStretch(1, 0);
            m_gridLayout->setColumnStretch(2, 1);
        } else { // index == 1
            col = 1;
        }
    } else {
        // Normal placement
        col = index % columns;
        // Reset column stretch for normal layout (equal spacing)
        if (index == 0) {
            m_gridLayout->setColumnStretch(0, 0);
            m_gridLayout->setColumnStretch(1, 0);
            m_gridLayout->setColumnStretch(2, 0);
        }
    }
    
    // Set alignment based on column: 0=left, 1=center, 2=right
    Qt::Alignment alignment = Qt::AlignTop;
    if (col == 0) {
        alignment |= Qt::AlignLeft;
    } else if (col == 1) {
        alignment |= Qt::AlignHCenter;
    } else { // col == 2
        alignment |= Qt::AlignRight;
    }
    
    m_gridLayout->addWidget(button, row, col, alignment);

    // Store button and file mapping
    m_thumbnailButtons.append(button);
    m_buttonToFileMap[button] = filePath;

    // Make sure button is visible and raised
    button->show();
    button->raise();
    button->update();

    qDebug() << "OutputPreview: Created thumbnail button" << index << "at row" << row << "col" << col 
             << "for:" << filePath << "button visible:" << button->isVisible();
}

QPixmap OutputPreview::generateThumbnail(const QString &filePath, const QSize &size)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();

    if (extension == "png" || extension == "jpg" || extension == "jpeg") {
        // Load image and create thumbnail
        QPixmap pixmap(filePath);
        if (pixmap.isNull()) {
            qWarning() << "OutputPreview: Failed to load image:" << filePath;
            return QPixmap();
        }
        
        qDebug() << "OutputPreview: Loaded image size:" << pixmap.size() << "target size:" << size;
        
        // Scale to fill entire size maintaining aspect ratio (no black borders)
        QPixmap scaled = pixmap.scaled(size, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        
        // Crop to exact size if needed (center crop)
        if (scaled.width() > size.width() || scaled.height() > size.height()) {
            int x = (scaled.width() - size.width()) / 2;
            int y = (scaled.height() - size.height()) / 2;
            scaled = scaled.copy(x, y, size.width(), size.height());
        }
        
        qDebug() << "OutputPreview: Created thumbnail size:" << scaled.size();
        return scaled;
    }
    else if (extension == "avi" || extension == "mp4") {
        // Extract first frame from video
        cv::VideoCapture cap(filePath.toStdString());
        if (!cap.isOpened()) {
            qWarning() << "OutputPreview: Failed to open video:" << filePath;
            return QPixmap();
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            qWarning() << "OutputPreview: Failed to read frame from video:" << filePath;
            cap.release();
            return QPixmap();
        }
        cap.release();

        // Convert cv::Mat to QPixmap
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        QImage qImage(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(qImage.copy());

        qDebug() << "OutputPreview: Extracted video frame size:" << pixmap.size() << "target size:" << size;
        
        // Scale to fill entire size maintaining aspect ratio (no black borders)
        QPixmap scaled = pixmap.scaled(size, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        
        // Crop to exact size if needed (center crop)
        if (scaled.width() > size.width() || scaled.height() > size.height()) {
            int x = (scaled.width() - size.width()) / 2;
            int y = (scaled.height() - size.height()) / 2;
            scaled = scaled.copy(x, y, size.width(), size.height());
        }
        
        qDebug() << "OutputPreview: Created video thumbnail size:" << scaled.size();
        return scaled;
    }

    return QPixmap();
}

void OutputPreview::clearThumbnails()
{
    qDebug() << "OutputPreview: Clearing" << m_thumbnailButtons.size() << "thumbnails";
    
    // Clear selection
    m_selectedFiles.clear();
    m_buttonToFileMap.clear();

    // Remove all buttons from layout and delete them
    if (m_gridLayout) {
        for (QPushButton *button : m_thumbnailButtons) {
            m_gridLayout->removeWidget(button);
            button->deleteLater();
        }
    } else {
        // If no grid layout, just delete buttons directly
        for (QPushButton *button : m_thumbnailButtons) {
            button->deleteLater();
        }
    }
    m_thumbnailButtons.clear();
    
    // Clean up temporary thumbnail files
    for (const QString &tempFile : m_tempThumbnailFiles) {
        QFile::remove(tempFile);
        qDebug() << "OutputPreview: Removed temp file:" << tempFile;
    }
    m_tempThumbnailFiles.clear();
    
    qDebug() << "OutputPreview: Cleared all thumbnails";
}

bool OutputPreview::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QPushButton *button = qobject_cast<QPushButton *>(obj);
        if (button && m_thumbnailButtons.contains(button)) {
            if (debounceActive) {
                return true;
            } else {
                debounceActive = true;
                debounceTimer->start();
                processThumbnailClick(button);
                return true;
            }
        }
        
        // Handle clicks on fullscreen preview widget to exit preview (but keep toggle mode ON)
        if (m_isPreviewMode && (obj == fullscreenPreviewWidget || obj == fullscreenVideoWidget || obj == fullscreenImageLabel)) {
            qDebug() << "OutputPreview: Click on preview widget - exiting fullscreen preview";
            hideFullscreenPreview();
            return true;
        }
    }
    
    return QWidget::eventFilter(obj, event);
}

void OutputPreview::processThumbnailClick(QPushButton *button)
{
    if (!button || !m_buttonToFileMap.contains(button)) {
        return;
    }

    QString filePath = m_buttonToFileMap[button];
    
    // If preview toggle mode is ON, show fullscreen preview (like dynamic page)
    if (m_previewToggleMode) {
        qDebug() << "OutputPreview: Preview toggle mode ON - showing fullscreen preview for:" << filePath;
        showFullscreenPreview(filePath);
        return;
    }
    
    // Otherwise, toggle selection (normal behavior)
    // Normalize path for comparison (use QDir to handle path differences)
    QString normalizedPath = QDir::toNativeSeparators(QFileInfo(filePath).absoluteFilePath());
    
    // Check if already selected (m_selectedFiles now contains normalized paths)
    bool isSelected = m_selectedFiles.contains(normalizedPath);

    if (isSelected) {
        // Deselect - remove normalized path
        m_selectedFiles.removeAll(normalizedPath);
        applyHighlightStyle(button, false);
    } else {
        // Select - add normalized path to ensure consistent comparison
        m_selectedFiles.append(normalizedPath);
        applyHighlightStyle(button, true);
    }

    qDebug() << "OutputPreview: Selection changed. Selected files:" << m_selectedFiles.size();
    for (const QString &selected : m_selectedFiles) {
        qDebug() << "OutputPreview: Selected file:" << selected;
    }
    
    // Update confirm button state based on selection
    updateConfirmButtonState();
}

void OutputPreview::applyHighlightStyle(QPushButton *button, bool highlight)
{
    if (button) {
        button->setProperty("selected", highlight);
        // Force style repolish to apply the selected property change
        button->style()->unpolish(button);
        button->style()->polish(button);
        button->update();
        button->repaint();
        qDebug() << "OutputPreview: Applied highlight" << highlight << "to button, selected property:" << button->property("selected");
    }
}

void OutputPreview::resetDebounce()
{
    debounceActive = false;
}

void OutputPreview::resetPage()
{
    hideFullscreenPreview();
    m_previewToggleMode = false;
    ui->previewButton->setChecked(false);
    clearThumbnails();
    resetDebounce();
    debounceTimer->stop();
}

void OutputPreview::on_back_clicked()
{
    qDebug() << "OutputPreview: Back button clicked - returning to final page";
    resetPage();
    emit backToFinalPage();
}

void OutputPreview::on_confirm_clicked()
{
    if (!m_sessionManager) {
        qWarning() << "OutputPreview: SessionManager not set";
        return;
    }

    qDebug() << "OutputPreview: Confirming selection. Selected files count:" << m_selectedFiles.size();
    for (const QString &selected : m_selectedFiles) {
        qDebug() << "OutputPreview: Selected file:" << selected;
    }

    // Don't allow confirmation if button is disabled
    if (!ui->confirm->isEnabled() || m_selectedFiles.isEmpty()) {
        return;
    }

    // Get all files in current user folder
    QList<QString> allFiles = m_sessionManager->getAllOutputFiles();
    qDebug() << "OutputPreview: Total files in folder:" << allFiles.size();

    // Delete unselected files (use normalized path comparison)
    // Note: m_selectedFiles now contains normalized paths, so direct comparison works
    int deletedCount = 0;
    for (const QString &filePath : allFiles) {
        QString normalizedFilePath = QDir::toNativeSeparators(QFileInfo(filePath).absoluteFilePath());
        bool isSelected = m_selectedFiles.contains(normalizedFilePath);
        
        if (!isSelected) {
            if (QFile::remove(filePath)) {
                deletedCount++;
                qDebug() << "OutputPreview: Deleted unselected file:" << filePath;
            } else {
                qWarning() << "OutputPreview: Failed to delete file:" << filePath;
            }
        } else {
            qDebug() << "OutputPreview: Keeping selected file:" << filePath;
        }
    }

    qDebug() << "OutputPreview: Deleted" << deletedCount << "unselected files";
    qDebug() << "OutputPreview: Kept" << m_selectedFiles.size() << "selected files";

    // Create new user folder for next user
    m_sessionManager->createNewUserFolder();

    // Reset page
    resetPage();

    // Navigate back to landing page
    emit backToLandingPage();
}

void OutputPreview::updateConfirmButtonState()
{
    bool hasSelection = !m_selectedFiles.isEmpty();
    ui->confirm->setEnabled(hasSelection);
    // Preview button is always enabled (it's a toggle)
    
    // Style the confirm button based on enabled state
    // When disabled: light green background, light green border, lighter green on hover
    // When enabled: normal green background, black border, brighter green on hover
    QString styleSheet = QString(
        "QPushButton {"
        "    background-color: %1;"
        "    color: white;"
        "    font-weight: bold;"
        "    font-size: 18px;"
        "    border: 2px solid %2;"
        "    border-radius: 9px;"
        "    padding: 10px;"
        "}"
        "QPushButton:hover {"
        "    background-color: %3;"
        "}"
        "QPushButton:disabled {"
        "    background-color: %4;"
        "    border: 2px solid %5;"
        "    color: rgba(255, 255, 255, 150);"
        "}"
        "QPushButton:disabled:hover {"
        "    background-color: %6;"
        "}"
    ).arg(hasSelection ? "#0BC200" : "#7FD97F")  // Normal bg: green if enabled, light green if disabled
     .arg(hasSelection ? "#020202" : "#7FD97F")   // Border: black if enabled, light green if disabled
     .arg(hasSelection ? "#0DE600" : "#9FE99F")   // Hover bg: brighter green if enabled, lighter green if disabled
     .arg("#7FD97F")                               // Disabled bg: light green
     .arg("#7FD97F")                               // Disabled border: light green
     .arg("#9FE99F");                              // Disabled hover: lighter green
    
    ui->confirm->setStyleSheet(styleSheet);
    
    qDebug() << "OutputPreview: Confirm button" << (hasSelection ? "enabled" : "disabled") << "with" << m_selectedFiles.size() << "selected files";
}

void OutputPreview::on_previewButton_toggled(bool checked)
{
    m_previewToggleMode = checked;
    qDebug() << "OutputPreview: Preview toggle mode" << (checked ? "ON" : "OFF");
    
    // If turning OFF preview mode and we're in fullscreen preview, exit it
    if (!checked && m_isPreviewMode) {
        hideFullscreenPreview();
    }
}

void OutputPreview::onPreviewBackClicked()
{
    qDebug() << "OutputPreview: Preview back button clicked";
    hideFullscreenPreview();
}

bool OutputPreview::isVideoFile(const QString &filePath)
{
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    return (extension == "avi" || extension == "mp4");
}

void OutputPreview::showFullscreenPreview(const QString &filePath)
{
    if (!QFile::exists(filePath)) {
        qWarning() << "OutputPreview: Preview file does not exist:" << filePath;
        return;
    }
    
    m_previewFilePath = filePath;
    m_isPreviewMode = true;
    
    // Hide normal UI elements
    ui->scrollArea->hide();
    ui->previewLabel->hide();
    ui->back->hide();
    ui->confirm->hide();
    ui->previewButton->hide();
    
    // Position and show preview widget
    QRect widgetRect = this->geometry();
    fullscreenPreviewWidget->setGeometry(widgetRect);
    fullscreenPreviewWidget->show();
    fullscreenPreviewWidget->raise();
    
    if (isVideoFile(filePath)) {
        // Show video widget
        fullscreenImageLabel->hide();
        fullscreenVideoWidget->setGeometry(fullscreenPreviewWidget->rect());
        fullscreenVideoWidget->show();
        fullscreenVideoWidget->raise();
        
        // Load and play video
        fullscreenPlayer->setSource(QUrl::fromLocalFile(filePath));
        fullscreenPlayer->play();
        qDebug() << "OutputPreview: Playing video:" << filePath;
    } else {
        // Show image label
        fullscreenVideoWidget->hide();
        if (fullscreenPlayer) {
            fullscreenPlayer->stop();
        }
        
        fullscreenImageLabel->setGeometry(fullscreenPreviewWidget->rect());
        QPixmap pixmap(filePath);
        if (!pixmap.isNull()) {
            // Scale pixmap to fit label while maintaining aspect ratio
            QSize labelSize = fullscreenImageLabel->size();
            QPixmap scaled = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            fullscreenImageLabel->setPixmap(scaled);
            fullscreenImageLabel->show();
            fullscreenImageLabel->raise();
            qDebug() << "OutputPreview: Showing image:" << filePath;
        } else {
            qWarning() << "OutputPreview: Failed to load image:" << filePath;
        }
    }
    
    // Position and show preview back button
    QPoint originalBackButtonPos = ui->back->pos();
    QPoint globalPos = this->mapToGlobal(originalBackButtonPos);
    previewBackButton->move(globalPos);
    previewBackButton->setAttribute(Qt::WA_TranslucentBackground, true);
    previewBackButton->setAttribute(Qt::WA_NoSystemBackground, true);
    previewBackButton->show();
    previewBackButton->raise();
    previewBackButton->setAttribute(Qt::WA_AlwaysStackOnTop, true);
    
    qDebug() << "OutputPreview: Fullscreen preview shown for:" << filePath;
}

void OutputPreview::hideFullscreenPreview()
{
    m_isPreviewMode = false;
    
    // Stop video if playing
    if (fullscreenPlayer) {
        fullscreenPlayer->stop();
        fullscreenPlayer->setSource(QUrl());
    }
    
    // Hide preview widgets
    fullscreenPreviewWidget->hide();
    fullscreenVideoWidget->hide();
    fullscreenImageLabel->hide();
    previewBackButton->hide();
    
    // Show normal UI elements
    ui->scrollArea->show();
    ui->previewLabel->show();
    ui->back->show();
    ui->confirm->show();
    ui->previewButton->show();
    
    qDebug() << "OutputPreview: Fullscreen preview hidden";
}
