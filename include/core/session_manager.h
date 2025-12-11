#ifndef SESSION_MANAGER_H
#define SESSION_MANAGER_H

#include <QString>
#include <QDir>
#include <QDateTime>
#include <QList>
#include <QObject>
#include <QPixmap>

class SessionManager : public QObject
{
    Q_OBJECT

public:
    explicit SessionManager(QObject *parent = nullptr);
    ~SessionManager();

    // Initialize session folder structure
    void initializeSession();

    // Create new user folder (called after user confirms selection)
    void createNewUserFolder();

    // Get current session folder path
    QString getSessionFolderPath() const { return m_sessionFolderPath; }

    // Get current user folder path
    QString getCurrentUserFolderPath() const { return m_currentUserFolderPath; }

    // Save output file (image or video) with sequential naming
    QString saveOutput(const QPixmap &image, const QString &extension = "png");
    QString saveVideo(const QList<QPixmap> &frames, double fps = 30.0);

    // Get all output files in current user folder
    QList<QString> getAllOutputFiles() const;

    // Get current user number
    int getCurrentUserNumber() const { return m_currentUserNumber; }

    // Check if session is initialized
    bool isInitialized() const { return m_initialized; }

private:
    QString m_sessionFolderPath;
    QString m_currentUserFolderPath;
    int m_currentUserNumber;
    int m_outputCounter;
    bool m_initialized;

    // Generate unique filename with sequential counter
    QString generateOutputFileName(const QString &extension) const;
};

#endif // SESSION_MANAGER_H

