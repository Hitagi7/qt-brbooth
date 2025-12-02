#include <QApplication>
#include <QFontDatabase>
#include "core/brbooth.h"
#include "core/videotemplate.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include "core/system_monitor.h"
#include <csignal>
#include <cstdlib>
#include <windows.h>

// Global pointer to system monitor for crash handler
static SystemMonitor* g_systemMonitor = nullptr;

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
LONG WINAPI exceptionHandler(EXCEPTION_POINTERS* exceptionInfo)
{
    qDebug() << "WINDOWS EXCEPTION DETECTED! Code:" << exceptionInfo->ExceptionRecord->ExceptionCode;
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
    
    // Check CUDA
    bool cudaAvailable = false;
    try {
        int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
        qDebug() << "CUDA devices found:" << cudaDevices;
        if (cudaDevices > 0) {
            cudaAvailable = true;
            qDebug() << "CUDA GPU acceleration available!";
            
            // Get CUDA device info
            for (int i = 0; i < cudaDevices; i++) {
                cv::cuda::setDevice(i);
                cv::cuda::DeviceInfo devInfo(i);
                qDebug() << "CUDA Device" << i << ":" << devInfo.name();
                qDebug() << "  - Compute Capability:" << devInfo.majorVersion() << "." << devInfo.minorVersion();
                qDebug() << "  - Memory:" << devInfo.totalMemory() / (1024*1024) << "MB";
                qDebug() << "  - Multi-Processor Count:" << devInfo.multiProcessorCount();
            }
            
            // Set device to 0 for testing
            cv::cuda::setDevice(0);
            qDebug() << "CUDA device set to 0";
            
            // Test basic CUDA functionality
            qDebug() << "Testing CUDA functionality...";
            try {
                // Create a simple test image
                cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(100, 150, 200));
                
                // Upload to GPU
                cv::cuda::GpuMat gpuImage;
                gpuImage.upload(testImage);
                qDebug() << "  ✓ GPU upload successful";
                
                // Test CUDA image processing
                cv::cuda::GpuMat gpuGray;
                cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
                qDebug() << "  ✓ CUDA color conversion successful";
                
                // Test CUDA filtering
                cv::cuda::GpuMat gpuBlurred;
                cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(gpuGray.type(), gpuGray.type(), cv::Size(5, 5), 1.0);
                filter->apply(gpuGray, gpuBlurred);
                qDebug() << "  ✓ CUDA Gaussian blur successful";
                
                // Download result
                cv::Mat result;
                gpuBlurred.download(result);
                qDebug() << "  ✓ GPU download successful";
                
                qDebug() << "  ✓ All CUDA tests passed!";
                
            } catch (const cv::Exception& e) {
                qDebug() << "  ✗ CUDA functionality test failed:" << e.what();
            }
            
        } else {
            qDebug() << "No CUDA devices found";
        }
    } catch (const cv::Exception& e) {
        qDebug() << "CUDA not available:" << e.what();
    }
    
    // Check OpenCL
    if (cv::ocl::useOpenCL()) {
        qDebug() << "OpenCL available for GPU acceleration";
        cv::ocl::setUseOpenCL(true);
    } else {
        qDebug() << "OpenCL not available";
    }
    
    // Check DNN backends
    qDebug() << "Available DNN backends:";
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = cv::dnn::getAvailableBackends();
    for (auto backend : backends) {
        qDebug() << "  - Backend:" << static_cast<int>(backend.first) << "Target:" << static_cast<int>(backend.second);
    }
    
    // Check for CUDA DNN backend specifically
    bool cudaDnnAvailable = false;
    for (auto backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_CUDA && backend.second == cv::dnn::DNN_TARGET_CUDA) {
            cudaDnnAvailable = true;
            break;
        }
    }
    if (cudaDnnAvailable) {
        qDebug() << "✓ CUDA DNN backend available!";
    } else {
        qDebug() << "✗ CUDA DNN backend not available";
    }
    
    qDebug() << "=== End GPU Check ===";
    qDebug() << "Summary: CUDA Available:" << cudaAvailable << "| CUDA DNN:" << cudaDnnAvailable;

    qRegisterMetaType<VideoTemplate>("Video Template");
    
    // Set up crash handlers
    std::signal(SIGABRT, crashHandler);
    std::signal(SIGSEGV, crashHandler);
    std::signal(SIGFPE, crashHandler);
    std::signal(SIGILL, crashHandler);
    SetUnhandledExceptionFilter(exceptionHandler);
    
    BRBooth w;
    
    // Get system monitor from BRBooth for crash handler
    g_systemMonitor = w.getSystemMonitor();
    
    w.showFullScreen();
    int result = a.exec();
    
    // Clean up
    g_systemMonitor = nullptr;
    
    return result;
}
