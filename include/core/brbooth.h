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
    void testOpenGLFunctionality();

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
    
    // GIF management
    void startLandingPageGif();
    void stopLandingPageGif();
    


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

    // Indices for the stacked widget pages
    int landingPageIndex;
    int foregroundPageIndex;
    int dynamicPageIndex;
    int backgroundPageIndex;
    int capturePageIndex;
    int finalOutputPageIndex;

    // To keep track of the page visited before going to Capture or Final, for "back" navigation
    int lastVisitedPageIndex;
    
    // GIF management
    QMovie* m_landingPageGifMovie;
    
    // Transition tracking
    bool m_transitioningToCapture;
};
#endif // BRBOOTH_H
