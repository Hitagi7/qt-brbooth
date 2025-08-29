#ifndef AMD_GPU_VERIFIER_H
#define AMD_GPU_VERIFIER_H

#include <QString>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class AMDGPUVerifier
{
public:
    struct GPUInfo {
        QString name;
        QString vendor;
        QString version;
        QString driverVersion;
        size_t totalMemory;
        int computeUnits;
        bool isAMD;
        bool isOpenCLCompatible;
        bool isOpenGLCompatible;
    };

    static bool initialize();
    static GPUInfo getGPUInfo();
    static bool isAMDGPUAvailable();
    static bool testOpenCLAcceleration();
    static bool testOpenGLAcceleration();
    static void printGPUStatus();
    static QString getGPUStatusString();

private:
    static GPUInfo m_gpuInfo;
    static bool m_initialized;
    static bool m_amdGPUAvailable;
    static bool m_openCLAvailable;
    static bool m_openGLAvailable;
};

#endif // AMD_GPU_VERIFIER_H
