#include "brbooth.h"
#include <QDebug>
#include "background.h"
#include "camera.h"
#include "capture.h"
#include "dynamic.h"
#include "final.h"
#include "foreground.h"
#include "ui_brbooth.h"
#include <QDebug>
#include <opencv2/opencv.hpp> // Using CV_VERSION for debug
#include <QMovie>   // For GIF animation
#include <QLabel>   // To display QMovie
#include <QStyle>   // For QStyle::polish
#include <QResizeEvent> // For resizeEvent override
#include "videotemplate.h"
#include <opencv2/opencv.hpp> // Keep if CV_VERSION is directly used or other OpenCV types/functions are used

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
    , cameraThread(new QThread(this)) // Initialize cameraThread first
    , cameraWorker(new Camera())      // Initialize cameraWorker second
    , lastVisitedPageIndex(
          0) // Initialize lastVisitedPageIndex (will be overwritten by initial showLandingPage)
{
    qDebug() << "OpenCV Version: " << CV_VERSION;
    ui->setupUi(this);
    setCentralWidget(ui->stackedWidget); // Set the stacked widget as the central widget

    // Set the main window background style
    this->setStyleSheet("QMainWindow#BRBooth {"
                        "    background-color: white;"
                        "    background-repeat: no-repeat;"
                        "    background-position: center;"
                        "}");

    // ================== GIF Implementation for dynamicButton on Landing Page ==================
    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");

    if (!dynamicButton) {
        qWarning() << "CRITICAL ERROR: dynamicButton not found in UI.";
        // You might want to disable dynamic features or show an error message gracefully here
    } else {
        // Create a QLabel to display the GIF as the button's background
        QLabel* gifLabel = new QLabel(dynamicButton);
        // IMPORTANT: Make the GIF label slightly smaller to allow button's border to show
        const int margin = 5;
        gifLabel->setGeometry(margin, margin, dynamicButton->width() - 2 * margin, dynamicButton->height() - 2 * margin);
        gifLabel->setScaledContents(true);
        // Make the GIF label transparent to mouse events so the button can still be clicked
        gifLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        gifLabel->setMouseTracking(false); // Not needed for the GIF label itself
        gifLabel->lower(); // Put it behind the button's text

        // Load and play the GIF from resources (assuming :/gif/test.gif is in your Qt resource file)
        QMovie* gifMovie = new QMovie(":/gif/gif templates/dynamicbg3.gif", QByteArray(), gifLabel);
        if (!gifMovie->isValid()) {
            qWarning() << "Failed to load GIF: :/gif/test.gif" << gifMovie->lastErrorString();
            delete gifMovie; // Clean up if GIF is invalid
            gifMovie = nullptr;
        } else {
            gifLabel->setMovie(gifMovie);
            gifMovie->start();
            qDebug() << "GIF loaded and started for dynamicButton.";
        }

        // Ensure the button itself can receive hover events
        dynamicButton->setMouseTracking(true);
        dynamicButton->setAttribute(Qt::WA_Hover, true);

        // Style the button with text on top - using a more explicit approach
        dynamicButton->setText("CHILL"); // Set the text for the button
        dynamicButton->setStyleSheet(
            "QPushButton#dynamicButton {"
            "  font-family: 'Arial Black';"
            "  font-size: 80px;"
            "  font-weight: bold;"
            "  color: white;"
            "  background-color: transparent;" // Make button background transparent to show GIF
            "  border: 5px solid transparent;"  // Always have a border, just transparent normally
            "  border-radius: 8px;"
            "}"
            "QPushButton#dynamicButton:hover {"
            "  border: 5px solid #FFC20F;" // Yellow border on hover
            "  border-radius: 8px;"
            "  background-color: rgba(255, 194, 15, 0.1);" // Slight background tint on hover
            "}"
            );

        // Force the button to update its style immediately
        dynamicButton->style()->polish(dynamicButton);
    }
    // ==========================================================================

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
    // =============================================================================

    // Get pointers to UI-defined pages (added in Qt Designer's stacked widget)
    foregroundPage = ui->forepage;
    dynamicPage = ui->dynamicpage; // Assuming ui->dynamicpage is already of type Dynamic* or compatible

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

    // Set initial page
    showLandingPage();

    // Connect signals for page navigation and actions
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);

    if (dynamicPage) {
        connect(dynamicPage, &Dynamic::backtoLandingPage, this, &BRBooth::showLandingPage);
        // CRITICAL CHANGE: Connect to the new signal 'videoSelectedAndConfirmed' from Dynamic
        connect(dynamicPage, &Dynamic::videoSelectedAndConfirmed, this, [this]() {
            capturePage->setCaptureMode(Capture::VideoRecordMode);
            // You might want to pass the actual selected video template information from Dynamic here
            // For now, using a default placeholder if Dynamic doesn't pass specific template info back.
            VideoTemplate defaultVideoTemplate("Default Dynamic Template", 10);
            capturePage->setVideoTemplate(defaultVideoTemplate);
            showCapturePage(); // This call will now correctly store the dynamic page index as lastVisited
        });
        // Call this slot when the dynamic page is shown to ensure GIFs start
        connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, [this](int index){
            if (index == dynamicPageIndex) {
                dynamicPage->onDynamicPageShown();
            }
        });
    }

    if (backgroundPage) {
        connect(backgroundPage,
                &Background::backtoForegroundPage,
                this,
                &BRBooth::showForegroundPage);
        connect(backgroundPage, &Background::imageSelectedTwice, this, [this]() {
            capturePage->setCaptureMode(Capture::ImageCaptureMode);
            showCapturePage(); // This call will now correctly store the background page index as lastVisited
        });
    }

    if (capturePage) {
        // Handle the 'back' action from the Capture page
        connect(capturePage, &Capture::backtoPreviousPage, this, [this]() {
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
        connect(capturePage, &Capture::foregroundPathChanged, finalOutputPage, &Final::setForegroundOverlay);
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
            dynamicPage->resetPage(); // Dynamic page reset will also hide overlay and restart GIFs
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

// Override resizeEvent to handle GIF label resizing for the landing page button
void BRBooth::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event); // Call base class implementation

    // Update the GIF label size when the window resizes
    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");
    if (dynamicButton) {
        QLabel* gifLabel = dynamicButton->findChild<QLabel*>(); // Find the QLabel child
        if (gifLabel) {
            const int margin = 5;
            gifLabel->setGeometry(margin, margin, dynamicButton->width() - 2 * margin, dynamicButton->height() - 2 * margin);
        }
    }
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
    // CRITICAL FIX: When we call showCapturePage(), the *current* page is the one we want to remember
    // as the "last visited" (the page to go back to).
    lastVisitedPageIndex = ui->stackedWidget->currentIndex();
    qDebug() << "DEBUG: showCapturePage() called. Setting index to:" << capturePageIndex
             << ". SAVED lastVisitedPageIndex (page we just came from):" << lastVisitedPageIndex;
    
    // Pass the current foreground template to the final interface
    if (foregroundPage && finalOutputPage) {
        QString currentForegroundPath = foregroundPage->getSelectedForeground();
        if (!currentForegroundPath.isEmpty()) {
            finalOutputPage->setForegroundOverlay(currentForegroundPath);
        }
    }
    
    ui->stackedWidget->setCurrentIndex(capturePageIndex);
}

void BRBooth::showFinalOutputPage()
{
    // CRITICAL FIX: Similar to Capture, when moving to Final, save the current page (Capture)
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
