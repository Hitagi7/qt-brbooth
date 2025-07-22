#include "brbooth.h"
#include "background.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"
#include "ui_brbooth.h"
#include "videotemplate.h"
#include "camera.h"
#include <QDebug>
#include <opencv2/opencv.hpp> // Keep if CV_VERSION is directly used or other OpenCV types/functions are used

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
    , cameraThread(new QThread(this)) // Initialize cameraThread first
    , cameraWorker(new Camera())      // Initialize cameraWorker second
    , lastVisitedPageIndex(0)         // Initialize lastVisitedPageIndex (will be overwritten by initial showLandingPage)
{
    qDebug() << "OpenCV Version: " << CV_VERSION;
    ui->setupUi(this);
    setCentralWidget(ui->stackedWidget);

    this->setStyleSheet("QMainWindow#BRBooth {"
                        "   background-color: white);"
                        "   background-repeat: no-repeat;"
                        "   background-position: center;");

    qDebug() << "OpenCV Version: " << CV_VERSION;
    this->setStyleSheet("QMainWindow#BRBooth {"
                        "    background-color: white;"
                        "    background-repeat: no-repeat;"
                        "    background-position: center;"

                        "}");

    // ================== CAMERA THREADING SETUP ==================
    cameraWorker->moveToThread(cameraThread);

    connect(this, &BRBooth::startCameraWorker, cameraWorker, &Camera::startCamera);
    connect(this, &BRBooth::stopCameraWorker, cameraWorker, &Camera::stopCamera);

    connect(cameraThread, &QThread::started, this, [this]() {
        cameraWorker->setDesiredCameraProperties(1280, 720, 60.0);
        emit startCameraWorker();
    });

    cameraThread->start();
    qDebug() << "BRBooth: Camera thread started.";
    // ============================================================

    // Get pointers to UI-defined pages (added in Qt Designer's stacked widget)
    foregroundPage = ui->forepage;
    dynamicPage = ui->dynamicpage;

    // Get indices of all pages
    landingPageIndex = ui->stackedWidget->indexOf(ui->landingpage);
    foregroundPageIndex = ui->stackedWidget->indexOf(foregroundPage);
    dynamicPageIndex = ui->stackedWidget->indexOf(dynamicPage);

    // Create new pages and add them to the stacked widget
    backgroundPage = new Background(this);
    ui->stackedWidget->addWidget(backgroundPage);
    backgroundPageIndex = ui->stackedWidget->indexOf(backgroundPage);

    // Pass the camera worker and thread to Capture's constructor
    capturePage = new Capture(this, foregroundPage, cameraWorker, cameraThread);
    ui->stackedWidget->addWidget(capturePage);
    capturePageIndex = ui->stackedWidget->indexOf(capturePage);

    finalOutputPage = new Final(this);
    ui->stackedWidget->addWidget(finalOutputPage);
    finalOutputPageIndex = ui->stackedWidget->indexOf(finalOutputPage);

    // Set initial page - this uses the showLandingPage() slot to set the initial view
    showLandingPage();

    // Connect signals for page navigation and actions
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);

    if (dynamicPage) {
        connect(dynamicPage, &Dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);
        connect(dynamicPage, &Dynamic::videoSelectedTwice, this, [this]() {
            capturePage->setCaptureMode(Capture::VideoRecordMode);
            VideoTemplate defaultVideoTemplate("Default Dynamic Template", 10);
            capturePage->setVideoTemplate(defaultVideoTemplate);
            showCapturePage(); // This call will now correctly store the dynamic page index as lastVisited
        });
    }

    if (backgroundPage) {
        connect(backgroundPage, &Background::backtoForegroundPage, this, &BRBooth::showForegroundPage);
        connect(backgroundPage, &Background::imageSelectedTwice, this, [this]() {
            capturePage->setCaptureMode(Capture::ImageCaptureMode);
            showCapturePage(); // This call will now correctly store the background page index as lastVisited
        });
    }

    if (capturePage) {
        // Handle the 'back' action from the Capture page
        connect(capturePage, &Capture::backtoPreviousPage, this, [this]() {

            // This is the core logic for the 'back' button from Capture
            // lastVisitedPageIndex should correctly hold the index of the page *before* Capture.
            if (lastVisitedPageIndex == backgroundPageIndex) {
                showBackgroundPage();
            } else if (lastVisitedPageIndex == dynamicPageIndex) {
                showDynamicPage(); // Will transition to the correct page
            } else {
                showLandingPage(); // Fallback if lastVisitedPageIndex is unexpected
            }
        });
        connect(capturePage, &Capture::showFinalOutputPage, this, &BRBooth::showFinalOutputPage);
        connect(capturePage, &Capture::imageCaptured, finalOutputPage, &Final::setImage);
        connect(capturePage, &Capture::videoRecorded, finalOutputPage, &Final::setVideo);
    }

    if (finalOutputPage) {
        connect(finalOutputPage, &Final::backToCapturePage, this, &BRBooth::showCapturePage);
        connect(finalOutputPage, &Final::backToLandingPage, this, &BRBooth::showLandingPage);
    }

    // Resets pages when they are loaded (useful for clearing selections etc.)
    connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, [this](int index) {
        qDebug() << "DEBUG: Stacked widget current index changed to:" << index;
        if (index == foregroundPageIndex) {
            foregroundPage->resetPage();
        }
        if (index == backgroundPageIndex) {
            backgroundPage->resetPage();
        }
        if (index == dynamicPageIndex) {
            dynamicPage->resetPage();
        }
    });
}

BRBooth::~BRBooth()
{
    // Clean up camera thread and worker in BRBooth destructor
    emit stopCameraWorker(); // Signal to worker to stop
    cameraThread->quit();    // Tell thread to exit event loop
    cameraThread->wait();    // Wait for thread to finish
    delete cameraWorker;
    delete cameraThread;
    delete ui;
}

// =====================================================================
// Navigation Slots - lastVisitedPageIndex is updated ONLY when moving
// FORWARD to Capture or Final, right before setCurrentIndex.
// =====================================================================

void BRBooth::showLandingPage()
{
    ui->stackedWidget->setCurrentIndex(landingPageIndex);
}

void BRBooth::showForegroundPage()
{
    ui->stackedWidget->setCurrentIndex(foregroundPageIndex);
}

void BRBooth::showDynamicPage()
{
    ui->stackedWidget->setCurrentIndex(dynamicPageIndex);
}

void BRBooth::showBackgroundPage()
{
    ui->stackedWidget->setCurrentIndex(backgroundPageIndex);
}

void BRBooth::showCapturePage()
{
    // *** CRITICAL FIX HERE ***
    // When we call showCapturePage(), the *current* page is the one we want to remember
    // as the "last visited" (the page to go back to).
    lastVisitedPageIndex = ui->stackedWidget->currentIndex();
    qDebug() << "DEBUG: showCapturePage() called. Setting index to:" << capturePageIndex << ". SAVED lastVisitedPageIndex (page we just came from):" << lastVisitedPageIndex;
    ui->stackedWidget->setCurrentIndex(capturePageIndex);
}

void BRBooth::showFinalOutputPage()
{
    // *** CRITICAL FIX HERE ***
    // Similar to Capture, when moving to Final, save the current page (Capture)
    // as the last visited one to return to.
    lastVisitedPageIndex = ui->stackedWidget->currentIndex();
    ui->stackedWidget->setCurrentIndex(finalOutputPageIndex);
}



void BRBooth::on_staticButton_clicked()
{
    showForegroundPage();
}

void BRBooth::on_dynamicButton_clicked()
{
    showDynamicPage();
}
