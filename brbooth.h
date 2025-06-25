#ifndef BRBOOTH_H
#define BRBOOTH_H

#include <QMainWindow>

#include "opencv2/core/core.hpp"

#include "foreground.h"
#include "dynamic.h"
#include "background.h"
#include "capture.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class BRBooth;
}

QT_END_NAMESPACE

class BRBooth : public QMainWindow
{
    Q_OBJECT

public:
    BRBooth(QWidget *parent = nullptr);
    ~BRBooth();

private slots:
    void showLandingPage();
    void showForegroundPage();
    void showDynamicPage();
    void showBackgroundPage();
    void showCapturePage();
    void on_staticButton_clicked();
    void on_dynamicButton_clicked();

private:
    Ui::BRBooth *ui;
    Foreground *foregroundPage;
    int foregroundPageIndex;
    int landingPageIndex;
    Dynamic *dynamicPage;
    int dynamicPageIndex;
    Background *backgroundPage;
    int backgroundPageIndex;
    Capture *capturePage;
    int capturePageIndex;
    int previousPageIndex;
};
#endif // BRBOOTH_H
