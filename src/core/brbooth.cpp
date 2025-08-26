#include "core/brbooth.h"
#include <QApplication>
#include "ui/background.h"
#include "core/camera.h"
#include "core/capture.h"
#include "ui/dynamic.h"
#include "ui/final.h"
#include "ui/foreground.h"
#include "ui_brbooth.h"
#include <QThread>
#include <QDebug>
#include <opencv2/opencv.hpp> // Using CV_VERSION for debug
#include <opencv2/cudaimgproc.hpp> // For CUDA image processing functions
#include <opencv2/cudafilters.hpp> // For CUDA filter functions
#include <opencv2/cudawarping.hpp> // For CUDA resize and other warping functions
#include <QMessageBox>
#include <QTimer>
#include <QStyle>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QShortcut>
#include "core/videotemplate.h"
#include <chrono>

BRBooth::BRBooth(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::BRBooth)
    , cameraThread(new QThread(this)) // Initialize cameraThread first
    , cameraWorker(new Camera())      // Initialize cameraWorker second
    , lastVisitedPageIndex(
          0) // Initialize lastVisitedPageIndex (will be overwritten by initial showLandingPage)
{
    qDebug() << "OpenCV Version: " << CV_VERSION;
    
    // Check CUDA availability at startup
    int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
    if (cudaDevices > 0) {
        qDebug() << "🎮 CUDA GPU Acceleration: ENABLED (" << cudaDevices << " device(s) found)";
        cv::cuda::setDevice(0);
        cv::cuda::DeviceInfo deviceInfo(0);
        qDebug() << "   GPU: " << deviceInfo.name();
        qDebug() << "   Memory: " << deviceInfo.totalMemory() / (1024*1024) << " MB";
    } else {
        qDebug() << "⚠️  CUDA GPU Acceleration: DISABLED (no CUDA devices found)";
    }
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

        // Load and play the GIF from file system
        QString gifPath = "gif templates/dynamicbg3.gif";
        
        // Debug: Print current working directory and check if file exists
        qDebug() << "Current working directory:" << QDir::currentPath();
        qDebug() << "Application directory:" << QCoreApplication::applicationDirPath();
        qDebug() << "Looking for GIF at:" << QDir::currentPath() + "/" + gifPath;
        qDebug() << "File exists:" << QFile::exists(gifPath);
        
        // Check if we're in a build directory (debug/release) and try to find project root
        QString currentDir = QDir::currentPath();
        QString appDir = QCoreApplication::applicationDirPath();
        qDebug() << "Current dir contains 'debug' or 'release':" << (currentDir.contains("debug") || currentDir.contains("release"));
        qDebug() << "App dir contains 'debug' or 'release':" << (appDir.contains("debug") || appDir.contains("release"));
        
        // Try alternative paths if the direct path doesn't work
        QStringList possiblePaths;
        possiblePaths << gifPath
                      << QDir::currentPath() + "/" + gifPath
                      << QCoreApplication::applicationDirPath() + "/" + gifPath
                      << QCoreApplication::applicationDirPath() + "/../" + gifPath
                      << QCoreApplication::applicationDirPath() + "/../../" + gifPath
                      << "../" + gifPath
                      << "../../" + gifPath
                      << "../../../" + gifPath;
        
        // Add the known working path as a fallback
        QString knownPath = "C:/Users/dorot/Documents/qt-brbooth/gif templates/dynamicbg3.gif";
        possiblePaths << knownPath;
        qDebug() << "Added known working path:" << knownPath;
        
        // Try to find project root by looking for qt-brbooth.pro file
        QString projectRoot;
        QStringList searchDirs;
        searchDirs << QDir::currentPath()
                   << QCoreApplication::applicationDirPath()
                   << QCoreApplication::applicationDirPath() + "/.."
                   << QCoreApplication::applicationDirPath() + "/../.."
                   << "../"
                   << "../../"
                   << "../../../";
        
        for (const QString& searchDir : searchDirs) {
            if (QFile::exists(searchDir + "/qt-brbooth.pro")) {
                projectRoot = searchDir;
                qDebug() << "Found project root at:" << projectRoot;
                break;
            }
        }
        
        if (!projectRoot.isEmpty()) {
            possiblePaths.prepend(projectRoot + "/" + gifPath);
            qDebug() << "Added project root path:" << projectRoot + "/" + gifPath;
        }
        
        // Debug: Print all paths we're going to try
        qDebug() << "All paths to try for GIF:";
        for (int i = 0; i < possiblePaths.size(); ++i) {
            qDebug() << "  " << i << ":" << possiblePaths[i] << "(exists:" << QFile::exists(possiblePaths[i]) << ")";
        }
        
        QString validPath;
        for (const QString& path : possiblePaths) {
            if (QFile::exists(path)) {
                validPath = path;
                qDebug() << "Found GIF at:" << validPath;
                break;
            }
        }
        
        if (validPath.isEmpty()) {
            qWarning() << "GIF file not found in any of the expected locations:";
            for (const QString& path : possiblePaths) {
                qWarning() << "  -" << path;
            }
            // Create a placeholder colored label instead
            gifLabel->setStyleSheet("background-color: #FF6B6B; border-radius: 8px;");
            gifLabel->setText("GIF\nMissing");
            gifLabel->setAlignment(Qt::AlignCenter);
        } else {
            QMovie* gifMovie = new QMovie(validPath, QByteArray(), gifLabel);
            if (!gifMovie->isValid()) {
                qWarning() << "Failed to load GIF:" << validPath << gifMovie->lastErrorString();
                delete gifMovie; // Clean up if GIF is invalid
                gifMovie = nullptr;
                // Create a placeholder
                gifLabel->setStyleSheet("background-color: #FFB74D; border-radius: 8px;");
                gifLabel->setText("GIF\nError");
                gifLabel->setAlignment(Qt::AlignCenter);
            } else {
                gifLabel->setMovie(gifMovie);
                // Don't start immediately - will be started when landing page is shown
                qDebug() << "GIF loaded for dynamicButton from:" << validPath << "(not started yet)";
            }
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

    // Set camera properties but don't start immediately
    cameraWorker->setDesiredCameraProperties(1280, 720, 60.0);
    
    // Start the camera thread but don't start camera capture yet
    cameraThread->start();
    qDebug() << "BRBooth: Camera thread started (camera not yet active).";
    
    // Ensure camera is stopped initially
    emit stopCameraWorker();
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
    
    // Initialize landing page GIF management
    m_landingPageGifMovie = nullptr;
    
    // Start the landing page GIF since we're on the landing page initially
    startLandingPageGif();
    
    // Install event filter to handle window focus changes
    installEventFilter(this);

    // Connect signals for page navigation and actions
    connect(foregroundPage, &Foreground::backtoLandingPage, this, &BRBooth::showLandingPage);
    connect(foregroundPage, &Foreground::imageSelectedTwice, this, &BRBooth::showBackgroundPage);
    
    // Add keyboard shortcut for CUDA testing (Ctrl+T)
    QShortcut *cudaTestShortcut = new QShortcut(QKeySequence("Ctrl+T"), this);
    connect(cudaTestShortcut, &QShortcut::activated, this, &BRBooth::testCudaFunctionality);

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
        
        // Start camera ONLY for capture page (as requested by user)
        if (index == capturePageIndex) {
            qDebug() << "📹 Starting camera for Capture page (index:" << index << ")...";
            emit startCameraWorker();
        } else {
            // Stop camera when leaving capture page
            qDebug() << "📹 Stopping camera (leaving capture page, going to index:" << index << ")...";
            emit stopCameraWorker();
        }
        
        // Manage GIFs based on page changes
        if (index == landingPageIndex) {
            startLandingPageGif();
        } else {
            stopLandingPageGif();
        }
        
        if (index == dynamicPageIndex) {
            dynamicPage->resetPage(); // Dynamic page reset will also hide overlay and restart GIFs
        } else if (dynamicPage) {
            dynamicPage->stopAllGifs(); // Stop GIFs when leaving dynamic page
        }
        
        if (index == foregroundPageIndex) {
            foregroundPage->resetPage();
        }
        if (index == backgroundPageIndex) {
            backgroundPage->resetPage();
        }
        if (index == capturePageIndex) {
            capturePage->enableHandDetectionForCapture(); // Enable hand detection for capture page
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

void BRBooth::startLandingPageGif()
{
    QPushButton* dynamicButton = ui->landingpage->findChild<QPushButton*>("dynamicButton");
    if (dynamicButton) {
        QLabel* gifLabel = dynamicButton->findChild<QLabel*>();
        if (gifLabel) {
            // Check if there's already a movie set on the label
            QMovie* existingMovie = gifLabel->movie();
            if (existingMovie && existingMovie->isValid()) {
                // Use the existing movie
                m_landingPageGifMovie = existingMovie;
                if (m_landingPageGifMovie->state() != QMovie::Running) {
                    m_landingPageGifMovie->start();
                    qDebug() << "Landing page GIF started (existing movie)";
                }
            } else if (!m_landingPageGifMovie) {
                // Create new movie if none exists
                // Find the GIF file
                QString gifPath = "gif templates/dynamicbg3.gif";
                QStringList possiblePaths;
                possiblePaths << gifPath
                              << QDir::currentPath() + "/" + gifPath
                              << QCoreApplication::applicationDirPath() + "/" + gifPath
                              << QCoreApplication::applicationDirPath() + "/../" + gifPath
                              << QCoreApplication::applicationDirPath() + "/../../" + gifPath
                              << "../" + gifPath
                              << "../../" + gifPath
                              << "../../../" + gifPath
                              << "C:/Users/dorot/Documents/qt-brbooth/gif templates/dynamicbg3.gif";
                
                QString validPath;
                for (const QString& path : possiblePaths) {
                    if (QFile::exists(path)) {
                        validPath = path;
                        break;
                    }
                }
                
                if (!validPath.isEmpty()) {
                    m_landingPageGifMovie = new QMovie(validPath, QByteArray(), gifLabel);
                    if (m_landingPageGifMovie->isValid()) {
                        gifLabel->setMovie(m_landingPageGifMovie);
                        m_landingPageGifMovie->start();
                        qDebug() << "Landing page GIF started from:" << validPath;
                    } else {
                        delete m_landingPageGifMovie;
                        m_landingPageGifMovie = nullptr;
                    }
                }
            }
        }
    }
}

void BRBooth::stopLandingPageGif()
{
    if (m_landingPageGifMovie) {
        m_landingPageGifMovie->stop();
        delete m_landingPageGifMovie;
        m_landingPageGifMovie = nullptr;
        qDebug() << "Landing page GIF stopped";
    }
}

bool BRBooth::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == this) {
        if (event->type() == QEvent::WindowActivate) {
            qDebug() << "Window activated - resuming normal operation";
            // Restart camera if we're on capture page and camera was stopped
            if (ui->stackedWidget->currentIndex() == capturePageIndex && cameraWorker && !cameraWorker->isCameraOpen()) {
                qDebug() << "Restarting camera after window activation";
                cameraWorker->startCamera();
            }
        } else if (event->type() == QEvent::WindowDeactivate) {
            qDebug() << "Window deactivated - camera pause disabled to prevent freezing";
            // Disabled camera pause to prevent freezing issues
            // emit stopCameraWorker();
        }
    }
    return QMainWindow::eventFilter(obj, event);
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
    
    // Camera is now started by page change handler for all relevant pages
    
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

void BRBooth::testCudaFunctionality()
{
    qDebug() << "=== BRBooth CUDA Test Function ===";
    
    try {
        // Check if CUDA is available
        int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
        if (cudaDevices == 0) {
            qDebug() << "No CUDA devices found";
            return;
        }
        
        qDebug() << "CUDA devices found:" << cudaDevices;
        
        // Set device
        cv::cuda::setDevice(0);
        
        // Create a test image
        cv::Mat testImage(480, 640, CV_8UC3, cv::Scalar(100, 150, 200));
        
        // Test GPU upload
        cv::cuda::GpuMat gpuImage;
        gpuImage.upload(testImage);
        qDebug() << "✓ GPU upload successful";
        
        // Test CUDA image processing operations
        cv::cuda::GpuMat gpuGray, gpuBlurred, gpuResized;
        
        // Color conversion
        cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
        qDebug() << "✓ CUDA color conversion successful";
        
        // Gaussian blur
        cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(gpuGray.type(), gpuGray.type(), cv::Size(15, 15), 2.0);
        gaussianFilter->apply(gpuGray, gpuBlurred);
        qDebug() << "✓ CUDA Gaussian blur successful";
        
        // Resize
        cv::cuda::resize(gpuBlurred, gpuResized, cv::Size(320, 240));
        qDebug() << "✓ CUDA resize successful";
        
        // Download result
        cv::Mat result;
        gpuResized.download(result);
        qDebug() << "✓ GPU download successful";
        
        // Test performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
            gaussianFilter->apply(gpuGray, gpuBlurred);
            cv::cuda::resize(gpuBlurred, gpuResized, cv::Size(320, 240));
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "✓ Performance test: 100 iterations completed in" << duration.count() << "ms";
        
        qDebug() << "=== All CUDA tests passed! ===";
        
    } catch (const cv::Exception& e) {
        qDebug() << "✗ CUDA test failed:" << e.what();
    } catch (const std::exception& e) {
        qDebug() << "✗ CUDA test failed with exception:" << e.what();
    }
}
