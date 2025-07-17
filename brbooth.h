#ifndef BRBOOTH_H
#define BRBOOTH_H

#include <QMainWindow>
#include <QStackedWidget> // Make sure this is included for QStackedWidget members
#include <QThread> // For cameraThread

// Forward declarations to avoid circular dependencies and unnecessary includes
class Background;
class Capture;
class Dynamic;
class Final;
class Foreground;
class Camera; // Forward declare Camera worker

namespace Ui {
class BRBooth;
}

class BRBooth : public QMainWindow
{
    Q_OBJECT

public:
    explicit BRBooth(QWidget *parent = nullptr);
    ~BRBooth();

public slots:
    void showLandingPage();
    void showForegroundPage();
    void showDynamicPage();
    void showBackgroundPage();
    void showCapturePage();
    void showFinalOutputPage();

private slots:
    void on_staticButton_clicked();
    void on_dynamicButton_clicked();

signals:
    void startCameraWorker(); // Signal to start the camera worker's operations
    void stopCameraWorker();  // Signal to stop the camera worker's operations

private:
    Ui::BRBooth *ui;

    // Pointers to your page widgets
    Foreground *foregroundPage;
    Background *backgroundPage;
    Capture *capturePage;
    Dynamic *dynamicPage;
    Final *finalOutputPage;

    // Page indices (stored for easy reference)
    int landingPageIndex;
    int foregroundPageIndex;
    int dynamicPageIndex;
    int backgroundPageIndex;
    int capturePageIndex;
    int finalOutputPageIndex;

    // This new variable will correctly track the page to return to
    int lastVisitedPageIndex;

    // Camera threading components
    QThread *cameraThread;
    Camera *cameraWorker;

    QStackedWidget *stackedWidget;
};

#endif // BRBOOTH_H
