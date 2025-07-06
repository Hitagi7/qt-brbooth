#ifndef BRBOOTH_H
#define BRBOOTH_H
#include <QMainWindow>
#include <QDebug>
#include <QMediaPlayer>
#include <QVideoWidget> // Keep this include as QMediaPlayer might still internally reference it
#include <QResizeEvent>
#include <QGraphicsView> // ADDED: For displaying QGraphicsScene
#include <QGraphicsScene> // ADDED: For holding QGraphicsVideoItem
#include <QGraphicsVideoItem> // ADDED: For video playback with aspect ratio control

#include "opencv2/core/core.hpp"
#include "background.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"


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
     void showFinalOutputPage();
     void on_staticButton_clicked();
     void on_dynamicButton_clicked();



protected:
    void resizeEvent(QResizeEvent *event) override;

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
     Final *finalOutputPage;
     int finalOutputPageIndex;

     QMediaPlayer *dynamicMediaPlayer;
     // CHANGED: Replaced QVideoWidget with QGraphicsView, QGraphicsScene, and QGraphicsVideoItem
     QGraphicsView *dynamicVideoView;
     QGraphicsScene *dynamicVideoScene;
     QGraphicsVideoItem *dynamicVideoItem;

};

#endif // BRBOOTH_H
