#include <QApplication>
#include <QFontDatabase>
#include "core/brbooth.h"
#include "core/videotemplate.h"
#include "core/amd_gpu_verifier.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

    // AMD GPU Detection and Verification
    qDebug() << "=== AMD GPU Acceleration Check ===";
    
    // Check OpenCV version and build info
    qDebug() << "OpenCV Version:" << CV_VERSION;
    qDebug() << "OpenCV Build Information:";
    qDebug() << cv::getBuildInformation().c_str();
    
    // Initialize AMD GPU verification
    bool amdGPUAvailable = AMDGPUVerifier::initialize();
    
    if (amdGPUAvailable) {
        qDebug() << "🎮 AMD GPU acceleration available!";
        
        // Get detailed GPU information
        AMDGPUVerifier::GPUInfo gpuInfo = AMDGPUVerifier::getGPUInfo();
        qDebug() << "GPU Name:" << gpuInfo.name;
        qDebug() << "GPU Vendor:" << gpuInfo.vendor;
        qDebug() << "GPU Version:" << gpuInfo.version;
        qDebug() << "GPU Memory:" << gpuInfo.totalMemory / (1024*1024) << "MB";
        qDebug() << "Compute Units:" << gpuInfo.computeUnits;
        
        // Test OpenCL acceleration
        if (AMDGPUVerifier::testOpenCLAcceleration()) {
            qDebug() << "✓ OpenCL acceleration working!";
        } else {
            qDebug() << "⚠ OpenCL acceleration not available";
        }
        
        // Test OpenGL acceleration
        if (AMDGPUVerifier::testOpenGLAcceleration()) {
            qDebug() << "✓ OpenGL acceleration working!";
        } else {
            qDebug() << "⚠ OpenGL acceleration limited";
        }
        
    } else {
        qDebug() << "⚠ No AMD GPU found, falling back to CPU processing";
        qDebug() << "This may impact performance for image processing tasks";
    }
    
    // Check OpenCL availability
    if (cv::ocl::useOpenCL()) {
        qDebug() << "OpenCL available for GPU acceleration";
        cv::ocl::setUseOpenCL(true);
    } else {
        qDebug() << "OpenCL not available in this OpenCV build";
    }
    
    // Check DNN backends
    qDebug() << "Available DNN backends:";
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = cv::dnn::getAvailableBackends();
    for (auto backend : backends) {
        qDebug() << "  - Backend:" << static_cast<int>(backend.first) << "Target:" << static_cast<int>(backend.second);
    }
    
    // Check for OpenCL DNN backend specifically
    bool openclDnnAvailable = false;
    for (auto backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_OPENCV && backend.second == cv::dnn::DNN_TARGET_OPENCL) {
            openclDnnAvailable = true;
            break;
        }
    }
    if (openclDnnAvailable) {
        qDebug() << "✓ OpenCL DNN backend available!";
    } else {
        qDebug() << "✗ OpenCL DNN backend not available";
    }
    
    qDebug() << "=== End AMD GPU Check ===";
    qDebug() << "Summary: AMD GPU Available:" << amdGPUAvailable << "| OpenCL DNN:" << openclDnnAvailable;

    qRegisterMetaType<VideoTemplate>("Video Template");
    BRBooth w;
    w.showFullScreen();
    return a.exec();
}
