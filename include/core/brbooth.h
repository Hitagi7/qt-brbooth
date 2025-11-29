#ifndef BRBOOTH_H
#define BRBOOTH_H

#include <QMainWindow>
#include <QStackedWidget> // Make sure this is included for QStackedWidget
#include <QThread> // For camera threading
#include <QMovie> // For GIF management
#include <QLabel> // For global loading label


// Forward declarations to avoid circular includes and speed up compilation
class Foreground;
class Background;
class Capture;
class Dynamic;
class Final;
class Loading;
class Confirm;
class Idle;
class Camera; // Forward declare Camera worker class

QT_BEGIN_NAMESPACE
namespace Ui { class BRBooth; }
QT_END_NAMESPACE

class BRBooth : public QMainWindow
{
    Q_OBJECT

public:
    BRBooth(QWidget *parent = nullptr);
    ~BRBooth();
    
    // CUDA test function
    void testCudaFunctionality();

protected:
    void resizeEvent(QResizeEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    // Slots for page transitions
    void showLandingPage();
    void showForegroundPage();
    void showDynamicPage();
    void showBackgroundPage();
    void showCapturePage();
    void showFinalOutputPage();
    void showIdlePage();
    
    // GIF management
    void startLandingPageGif();
    void stopLandingPageGif();
    
    // Idle mode management
    void onIdleTimeout();
    void resetIdleTimer();
    void startIdleTimer();
    void stopIdleTimer();
    


    // Slots for button clicks on the landing page
    void on_staticButton_clicked();
    void on_dynamicButton_clicked();

signals:
    // Signals to control the camera worker
    void startCameraWorker();
    void stopCameraWorker();

private:
    Ui::BRBooth *ui;

    // Camera threading components
    QThread *cameraThread;
    Camera *cameraWorker;

    // Pointers to the different pages
    Foreground *foregroundPage;
    Background *backgroundPage;
    Dynamic *dynamicPage;
    Capture *capturePage;
    Final *finalOutputPage;
    Loading *loadingPage;
    Confirm *confirmPage;
    Idle *idlePage;

    // Indices for the stacked widget pages
    int landingPageIndex;
    int foregroundPageIndex;
    int dynamicPageIndex;
    int backgroundPageIndex;
    int capturePageIndex;
    int loadingPageIndex;
    int confirmPageIndex;
    int finalOutputPageIndex;
    int idlePageIndex;

    // To keep track of the page visited before going to Capture or Final, for "back" navigation
    int lastVisitedPageIndex;
    
    // GIF management
    QMovie* m_landingPageGifMovie;
    
    // Transition tracking
    bool m_transitioningToCapture;
    
    // Idle mode tracking
    QTimer *m_idleTimer;
    int m_pageBeforeIdle;
    bool m_isIdleModeActive;
    bool m_idleTimerEnabled;
};
#endif // BRBOOTH_H
