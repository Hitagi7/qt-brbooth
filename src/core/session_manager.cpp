#include "core/session_manager.h"
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QPixmap>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

SessionManager::SessionManager(QObject *parent)
    : QObject(parent)
    , m_currentUserNumber(0)
    , m_outputCounter(0)
    , m_initialized(false)
{
}

SessionManager::~SessionManager()
{
}

void SessionManager::initializeSession()
{
    if (m_initialized) {
        qWarning() << "SessionManager: Session already initialized";
        return;
    }

    // Create sessions directory in Downloads folder
    QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    if (downloadsPath.isEmpty()) {
        downloadsPath = "C:/Downloads"; // Fallback for Windows
    }

    // Create session folder with timestamp
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    m_sessionFolderPath = QDir(downloadsPath).filePath("sessions/session_" + timestamp);
    
    QDir dir;
    if (!dir.mkpath(m_sessionFolderPath)) {
        qWarning() << "SessionManager: Failed to create session folder:" << m_sessionFolderPath;
        return;
    }

    qDebug() << "SessionManager: Created session folder:" << m_sessionFolderPath;

    // Create first user folder (before setting m_initialized = true)
    createNewUserFolder();
    
    // Verify user folder was created before marking as initialized
    if (m_currentUserFolderPath.isEmpty()) {
        qWarning() << "SessionManager: Failed to create user folder during initialization";
        return;
    }
    
    m_initialized = true;
    qDebug() << "SessionManager: Session initialized successfully";
    qDebug() << "SessionManager: Current user folder:" << m_currentUserFolderPath;
}

void SessionManager::createNewUserFolder()
{
    // Allow creation during initialization (m_initialized might be false during init)
    if (m_sessionFolderPath.isEmpty()) {
        qWarning() << "SessionManager: Cannot create user folder - session folder path is empty";
        return;
    }

    m_currentUserNumber++;
    m_outputCounter = 0; // Reset counter for new user

    m_currentUserFolderPath = QDir(m_sessionFolderPath).filePath("user_" + QString::number(m_currentUserNumber));
    
    qDebug() << "SessionManager: Creating user folder:" << m_currentUserFolderPath;
    
    QDir dir;
    if (!dir.mkpath(m_currentUserFolderPath)) {
        qWarning() << "SessionManager: Failed to create user folder:" << m_currentUserFolderPath;
        m_currentUserFolderPath.clear(); // Clear path on failure
        return;
    }

    qDebug() << "SessionManager: Successfully created user folder:" << m_currentUserFolderPath;
}

QString SessionManager::saveOutput(const QPixmap &image, const QString &extension)
{
    qDebug() << "SessionManager::saveOutput called - initialized:" << m_initialized 
             << "user folder:" << m_currentUserFolderPath
             << "image size:" << image.size()
             << "image isNull:" << image.isNull();
    
    if (!m_initialized || m_currentUserFolderPath.isEmpty()) {
        qWarning() << "SessionManager: Cannot save output - session not initialized or folder path empty";
        qWarning() << "SessionManager: Initialized:" << m_initialized << "Folder:" << m_currentUserFolderPath;
        return QString();
    }

    if (image.isNull() || image.size().isEmpty()) {
        qWarning() << "SessionManager: Cannot save - image is null or empty";
        return QString();
    }

    QString fileName = generateOutputFileName(extension);
    QString filePath = QDir(m_currentUserFolderPath).filePath(fileName);

    qDebug() << "SessionManager: Attempting to save to:" << filePath;
    qDebug() << "SessionManager: Image size:" << image.size() << "Image format valid:" << !image.isNull();

    // Ensure directory exists
    QDir dir;
    if (!dir.exists(m_currentUserFolderPath)) {
        qWarning() << "SessionManager: User folder does not exist, creating:" << m_currentUserFolderPath;
        if (!dir.mkpath(m_currentUserFolderPath)) {
            qWarning() << "SessionManager: Failed to create user folder:" << m_currentUserFolderPath;
            return QString();
        }
    }

    if (!image.save(filePath)) {
        qWarning() << "SessionManager: Failed to save image to:" << filePath;
        qWarning() << "SessionManager: Check file permissions and disk space";
        return QString();
    }

    qDebug() << "SessionManager: Successfully saved output to:" << filePath;
    
    // Verify file was created
    QFileInfo fileInfo(filePath);
    if (fileInfo.exists()) {
        qDebug() << "SessionManager: Image file verified - size:" << fileInfo.size() << "bytes";
    } else {
        qWarning() << "SessionManager: Image file was not created at:" << filePath;
    }
    
    m_outputCounter++;
    
    return filePath;
}

QString SessionManager::saveVideo(const QList<QPixmap> &frames, double fps)
{
    if (!m_initialized || m_currentUserFolderPath.isEmpty()) {
        qWarning() << "SessionManager: Cannot save video - session not initialized";
        return QString();
    }

    if (frames.isEmpty()) {
        qWarning() << "SessionManager: No frames to save";
        return QString();
    }

    QString fileName = generateOutputFileName("avi");
    QString filePath = QDir(m_currentUserFolderPath).filePath(fileName);

    qDebug() << "SessionManager: Attempting to save video to:" << filePath;
    qDebug() << "SessionManager: Video has" << frames.size() << "frames";

    // Get frame size from first pixmap
    QSize frameSize = frames.first().size();
    int width = frameSize.width();
    int height = frameSize.height();

    // Round FPS to standard rate
    double frameRate = fps;
    if (frameRate <= 26.0) {
        frameRate = 25.0;
    } else if (frameRate <= 35.0) {
        frameRate = 30.0;
    } else if (frameRate <= 55.0) {
        frameRate = 50.0;
    } else {
        frameRate = 60.0;
    }

    qDebug() << "SessionManager: Saving video with" << frames.size() << "frames at" << frameRate << "FPS";

    cv::VideoWriter videoWriter;
    bool opened = videoWriter.open(
        filePath.toStdString(),
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        frameRate,
        cv::Size(width, height)
    );

    if (!opened) {
        qWarning() << "SessionManager: Failed to open video writer for:" << filePath;
        return QString();
    }

    // Convert QPixmaps to cv::Mat and write frames
    for (const QPixmap &pixmap : frames) {
        QImage image = pixmap.toImage();
        if (image.format() != QImage::Format_RGB888) {
            image = image.convertToFormat(QImage::Format_RGB888);
        }

        cv::Mat frame(image.height(), image.width(), CV_8UC3, 
                     const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::Mat frameBGR;
        cv::cvtColor(frame, frameBGR, cv::COLOR_RGB2BGR);
        
        videoWriter.write(frameBGR);
    }

    videoWriter.release();
    qDebug() << "SessionManager: Successfully saved video to:" << filePath;
    
    // Verify file was created
    QFileInfo fileInfo(filePath);
    if (fileInfo.exists()) {
        qDebug() << "SessionManager: Video file verified - size:" << fileInfo.size() << "bytes";
    } else {
        qWarning() << "SessionManager: Video file was not created at:" << filePath;
    }
    
    m_outputCounter++;
    
    return filePath;
}

QList<QString> SessionManager::getAllOutputFiles() const
{
    QList<QString> files;

    qDebug() << "SessionManager::getAllOutputFiles - initialized:" << m_initialized 
             << "user folder:" << m_currentUserFolderPath;

    if (!m_initialized || m_currentUserFolderPath.isEmpty()) {
        qWarning() << "SessionManager: Cannot get files - session not initialized or folder path empty";
        return files;
    }

    QDir dir(m_currentUserFolderPath);
    if (!dir.exists()) {
        qWarning() << "SessionManager: User folder does not exist:" << m_currentUserFolderPath;
        qWarning() << "SessionManager: Attempting to create folder...";
        if (!dir.mkpath(m_currentUserFolderPath)) {
            qWarning() << "SessionManager: Failed to create user folder:" << m_currentUserFolderPath;
        } else {
            qDebug() << "SessionManager: Created user folder:" << m_currentUserFolderPath;
        }
        return files;
    }

    qDebug() << "SessionManager: User folder exists:" << m_currentUserFolderPath;
    qDebug() << "SessionManager: Listing all files in folder...";
    
    // List ALL files first for debugging
    QFileInfoList allFiles = dir.entryInfoList(QDir::Files, QDir::Name);
    qDebug() << "SessionManager: Total files in folder:" << allFiles.size();
    for (const QFileInfo &fileInfo : allFiles) {
        qDebug() << "SessionManager: Found file:" << fileInfo.fileName() << "extension:" << fileInfo.suffix();
    }

    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.jpeg" << "*.avi" << "*.mp4";
    
    QFileInfoList fileInfoList = dir.entryInfoList(filters, QDir::Files, QDir::Name);
    
    qDebug() << "SessionManager: Found" << fileInfoList.size() << "matching files in folder:" << m_currentUserFolderPath;
    
    for (const QFileInfo &fileInfo : fileInfoList) {
        QString absPath = fileInfo.absoluteFilePath();
        files.append(absPath);
        qDebug() << "SessionManager: Found file:" << absPath;
    }

    return files;
}

QString SessionManager::generateOutputFileName(const QString &extension) const
{
    QString counter = QString::number(m_outputCounter + 1).rightJustified(3, '0');
    return QString("output_%1.%2").arg(counter).arg(extension);
}

