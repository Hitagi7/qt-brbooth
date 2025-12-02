#include <QApplication>
#include <QDebug>
#include <QFontDatabase>
#include "core/brbooth.h"
#include "core/system_monitor.h"
#include "core/videotemplate.h"
#include <csignal>
#include <cstdlib>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <windows.h>

// Global pointer to system monitor for crash handler
static SystemMonitor *g_systemMonitor = nullptr;
static BRBooth *g_brbooth = nullptr;

// Crash handler function
void crashHandler(int signal)
{
    qDebug() << "CRASH DETECTED! Signal:" << signal;
    qDebug() << "Attempting to save statistics...";

    if (g_systemMonitor) {
        g_systemMonitor->saveStatisticsToText();
        qDebug() << "Statistics saved to text file";
    }

    // Exit
    std::exit(1);
}

// Windows exception handler
LONG WINAPI exceptionHandler(EXCEPTION_POINTERS *exceptionInfo)
{
    qDebug() << "WINDOWS EXCEPTION DETECTED! Code:"
             << exceptionInfo->ExceptionRecord->ExceptionCode;
    qDebug() << "Attempting to save statistics...";

    if (g_systemMonitor) {
        g_systemMonitor->saveStatisticsToText();
        qDebug() << "Statistics saved to text file";
    }

    return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    int fontId = QFontDatabase::addApplicationFont(
        "::/fonts/Fonts/static/RobotoCondensed-BoldItalic.ttf");
    if (fontId == -1) {
        qWarning() << "Failed to load RobotoCondensed-BoldItalic.ttf from resources.";
    } else {
        qDebug() << "Font loaded successfully. Font ID:" << fontId;
    }

    // GPU Detection Test
    qDebug() << "=== GPU Acceleration Check ===";

    // Check OpenCV version and build info
    qDebug() << "OpenCV Version:" << CV_VERSION;
    qDebug() << "OpenCV Build Information:";
    qDebug() << cv::getBuildInformation().c_str();

    // Check OpenCL
    bool openclAvailable = false;
    try {
        if (cv::ocl::useOpenCL()) {
            openclAvailable = true;
            qDebug() << "OpenCL GPU acceleration available!";
            cv::ocl::setUseOpenCL(true);

            // Get OpenCL device info
            cv::ocl::Context context = cv::ocl::Context::getDefault();
            if (!context.empty()) {
                size_t deviceCount = context.ndevices();
                qDebug() << "OpenCL devices found:" << deviceCount;
                for (size_t i = 0; i < deviceCount; i++) {
                    cv::ocl::Device device = context.device(i);
                    qDebug() << "OpenCL Device" << i << ":" << device.name().c_str();
                    qDebug() << "  - Vendor:" << device.vendorName().c_str();
                    qDebug() << "  - Version:" << device.version().c_str();
                    qDebug() << "  - Global Memory:" << device.globalMemSize() / (1024 * 1024) << "MB";
                    qDebug() << "  - Compute Units:" << device.maxComputeUnits();
                }
            }

            // Test basic OpenCL functionality
            qDebug() << "Testing OpenCL functionality...";
            try {
                // Create a simple test image
                cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(100, 150, 200));

                // Upload to GPU using UMat
                cv::UMat gpuImage;
                testImage.copyTo(gpuImage);
                qDebug() << "  ✓ GPU upload successful";

                // Test OpenCL image processing
                cv::UMat gpuGray;
                cv::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
                qDebug() << "  ✓ OpenCL color conversion successful";

                // Test OpenCL filtering
                cv::UMat gpuBlurred;
                cv::GaussianBlur(gpuGray, gpuBlurred, cv::Size(5, 5), 1.0);
                qDebug() << "  ✓ OpenCL Gaussian blur successful";

                // Download result
                cv::Mat result;
                gpuBlurred.copyTo(result);
                qDebug() << "  ✓ GPU download successful";

                qDebug() << "  ✓ All OpenCL tests passed!";

            } catch (const cv::Exception &e) {
                qDebug() << "  ✗ OpenCL functionality test failed:" << e.what();
            }

        } else {
            qDebug() << "OpenCL not available";
        }
    } catch (const cv::Exception &e) {
        qDebug() << "OpenCL not available:" << e.what();
    }

    // Check DNN backends
    qDebug() << "Available DNN backends:";
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends
        = cv::dnn::getAvailableBackends();
    for (auto backend : backends) {
        qDebug() << "  - Backend:" << static_cast<int>(backend.first)
                 << "Target:" << static_cast<int>(backend.second);
    }

    // Check for OpenCL DNN backend
    bool openclDnnAvailable = false;
    for (auto backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_OPENCV
            && backend.second == cv::dnn::DNN_TARGET_OPENCL) {
            openclDnnAvailable = true;
            break;
        }
    }
    if (openclDnnAvailable) {
        qDebug() << "✓ OpenCL DNN backend available!";
    } else {
        qDebug() << "✗ OpenCL DNN backend not available";
    }

    qDebug() << "=== End GPU Check ===";
    qDebug() << "Summary: OpenCL Available:" << openclAvailable << "| OpenCL DNN:" << openclDnnAvailable;

    qRegisterMetaType<VideoTemplate>("Video Template");

    // Set up crash handlers
    std::signal(SIGABRT, crashHandler);
    std::signal(SIGSEGV, crashHandler);
    std::signal(SIGFPE, crashHandler);
    std::signal(SIGILL, crashHandler);
    SetUnhandledExceptionFilter(exceptionHandler);

    BRBooth w;
    g_brbooth = &w;

    // Get system monitor from BRBooth for crash handler
    g_systemMonitor = w.getSystemMonitor();

    w.showFullScreen();
    int result = a.exec();

    // Clean up
    g_brbooth = nullptr;
    g_systemMonitor = nullptr;

    return result;
}
