#ifndef OUTPUTPREVIEW_H
#define OUTPUTPREVIEW_H

#include <QWidget>
#include <QPushButton>
#include <QTimer>
#include <QList>
#include <QMap>
#include <QPixmap>
#include <QGridLayout>

QT_BEGIN_NAMESPACE

namespace Ui {
class OutputPreview;
}

QT_END_NAMESPACE

class SessionManager;

class OutputPreview : public QWidget
{
    Q_OBJECT

public:
    explicit OutputPreview(QWidget *parent = nullptr);
    ~OutputPreview();

    // Set session manager reference
    void setSessionManager(SessionManager *sessionManager);

    // Load and display thumbnails from current user folder
    void loadOutputThumbnails();

public slots:
    void resetPage();

signals:
    void backToFinalPage();
    void backToLandingPage();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;
    void showEvent(QShowEvent *event) override;

private slots:
    void on_back_clicked();
    void on_confirm_clicked();
    void resetDebounce();

private:
    Ui::OutputPreview *ui;
    SessionManager *m_sessionManager;
    
    QGridLayout *m_gridLayout;
    QList<QPushButton*> m_thumbnailButtons;
    QList<QString> m_selectedFiles;
    QMap<QPushButton*, QString> m_buttonToFileMap;
    QList<QString> m_tempThumbnailFiles; // Store temp file paths for cleanup
    
    QTimer *debounceTimer;
    bool debounceActive;

    void applyHighlightStyle(QPushButton *button, bool highlight);
    void processThumbnailClick(QPushButton *button);
    void createThumbnailButton(const QString &filePath, int index, int totalCount);
    QPixmap generateThumbnail(const QString &filePath, const QSize &size);
    void clearThumbnails();
    void updateConfirmButtonState();
};

#endif // OUTPUTPREVIEW_H

