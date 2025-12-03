#include "core/system_monitor.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QThread>
#include <windows.h>
#include <pdh.h>
#include <psapi.h>

// Link required libraries
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")

// NVIDIA Management Library (NVML) for GPU monitoring
// Note: This requires NVML library. If not available, we'll use Windows Performance Counters as fallback
#ifdef NVML_AVAILABLE
#include <nvml.h>
#pragma comment(lib, "nvml.lib")
#endif

SystemMonitor::SystemMonitor(QObject *parent)
    : QObject(parent)
    , m_timer(new QTimer(this))
    , m_cpuQueryHandle(nullptr)
    , m_cpuCounterHandle(nullptr)
    , m_gpuQueryHandle(nullptr)
    , m_gpuCounterHandle(nullptr)
    , m_lastCpuIdleTime(0)
    , m_lastCpuKernelTime(0)
    , m_lastCpuUserTime(0)
    , m_cpuTimesInitialized(false)
    , m_activeGpuMethod(GPU_METHOD_NONE)
    , m_latestFPS(0.0)  // Initialize volatile FPS variable
    , m_peakMemoryGB(0.0)
    , m_initialized(false)
    , m_monitoring(false)
{
    // Initialize statistics
    m_lastStats.cpuUsage = 0.0;
    m_lastStats.gpuUsage = 0.0;
    m_lastStats.peakMemoryGB = 0.0;
    m_lastStats.systemMemoryUsageGB = 0.0;
    m_lastStats.systemMemoryTotalGB = 0.0;
    m_lastStats.systemMemoryAvailableGB = 0.0;
    m_lastStats.averageFPS = 0.0;
    m_lastStats.accuracy = 0.0;
    m_lastStats.timestamp = QDateTime::currentDateTime();
    
    m_peakStats = m_lastStats;
    m_averageStats = m_lastStats;
    
    // Initialize average tracking variables
    m_cpuSum = 0.0;
    m_gpuSum = 0.0;
    m_memorySum = 0.0;
    m_systemMemorySum = 0.0;
    m_fpsSum = 0.0;
    m_sampleCount = 0;
    
    // Initialize FPS tracking (volatile variable, thread-safe on x86/x64)
    m_latestFPS = 0.0;
    m_fpsTrackingTimer.start();
    
    connect(m_timer, &QTimer::timeout, this, &SystemMonitor::collectStatistics);
}

SystemMonitor::~SystemMonitor()
{
    stopMonitoring();
    
    // Clean up Windows Performance Counters
    if (m_cpuCounterHandle) {
        PdhRemoveCounter(m_cpuCounterHandle);
        m_cpuCounterHandle = nullptr;
    }
    if (m_cpuQueryHandle) {
        PdhCloseQuery(m_cpuQueryHandle);
        m_cpuQueryHandle = nullptr;
    }
    if (m_gpuCounterHandle) {
        PdhRemoveCounter(m_gpuCounterHandle);
        m_gpuCounterHandle = nullptr;
    }
    if (m_gpuQueryHandle) {
        PdhCloseQuery(m_gpuQueryHandle);
        m_gpuQueryHandle = nullptr;
    }
    
    // DXGI cleanup removed - using PDH for AMD/NVIDIA/Intel GPU monitoring
}

bool SystemMonitor::initialize()
{
    if (m_initialized) {
        return true;
    }
    
    // Initialize CPU monitoring using PDH (Performance Data Helper)
    PDH_STATUS status = PdhOpenQuery(nullptr, 0, &m_cpuQueryHandle);
    if (status != ERROR_SUCCESS) {
        qWarning() << "SystemMonitor: Failed to open CPU query:" << status;
        return false;
    }
    
    // Add CPU counter - using "\Processor(_Total)\% Processor Time"
    status = PdhAddCounter(m_cpuQueryHandle, L"\\Processor(_Total)\\% Processor Time", 0, &m_cpuCounterHandle);
    if (status != ERROR_SUCCESS) {
        qWarning() << "SystemMonitor: Failed to add CPU counter:" << status;
        PdhCloseQuery(m_cpuQueryHandle);
        m_cpuQueryHandle = nullptr;
        return false;
    }
    
    // Collect initial sample for CPU
    PdhCollectQueryData(m_cpuQueryHandle);
    
    // Initialize GPU monitoring - try multiple methods in order of preference
    // 1. NVML (NVIDIA GPUs only, most accurate)
    // 2. PDH (works with NVIDIA, AMD, Intel GPUs - best for AMD systems)
    
    #ifdef NVML_AVAILABLE
    nvmlReturn_t nvmlStatus = nvmlInit();
    if (nvmlStatus == NVML_SUCCESS) {
        // Test if we can actually get GPU usage
        nvmlDevice_t device;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result == NVML_SUCCESS) {
            nvmlUtilization_t utilization;
            result = nvmlDeviceGetUtilizationRates(device, &utilization);
            if (result == NVML_SUCCESS) {
                m_activeGpuMethod = GPU_METHOD_NVML;
                qDebug() << "SystemMonitor: NVML initialized successfully for GPU monitoring (NVIDIA GPU detected)";
            }
        }
    }
    #endif
    
    // Use PDH for GPU monitoring (works excellently with AMD, NVIDIA, and Intel GPUs)
    // PDH is the recommended method for AMD GPUs on Windows
    if (m_activeGpuMethod == GPU_METHOD_NONE) {
        status = PdhOpenQuery(nullptr, 0, &m_gpuQueryHandle);
        if (status == ERROR_SUCCESS) {
            // Try multiple GPU counter paths - AMD GPUs typically use these paths
            // Order matters: try most specific first, then fallback to generic
            const wchar_t* gpuPaths[] = {
                L"\\GPU Engine(*engtype_3D*)\\Utilization Percentage",      // 3D graphics (most common)
                L"\\GPU Engine(*engtype_Compute*)\\Utilization Percentage", // Compute workloads
                L"\\GPU Engine(*engtype_VideoDecode*)\\Utilization Percentage", // Video decoding
                L"\\GPU Engine(*engtype_VideoEncode*)\\Utilization Percentage", // Video encoding
                L"\\GPU Engine(*)\\Utilization Percentage"                  // All engines (fallback)
            };
            
            bool foundCounter = false;
            for (const wchar_t* path : gpuPaths) {
                status = PdhAddCounter(m_gpuQueryHandle, path, 0, &m_gpuCounterHandle);
                if (status == ERROR_SUCCESS) {
                    // Collect initial sample
                    PdhCollectQueryData(m_gpuQueryHandle);
                    m_activeGpuMethod = GPU_METHOD_PDH;
                    qDebug() << "SystemMonitor: Using PDH for GPU monitoring (AMD/NVIDIA/Intel compatible)";
                    qDebug() << "SystemMonitor: GPU counter path:" << QString::fromWCharArray(path);
                    foundCounter = true;
                    break;
                }
            }
            
            if (!foundCounter) {
                qWarning() << "SystemMonitor: GPU monitoring not available - GPU performance counters may not be enabled";
                qWarning() << "SystemMonitor: Try running as administrator or enable GPU performance counters in Windows";
                PdhCloseQuery(m_gpuQueryHandle);
                m_gpuQueryHandle = nullptr;
            }
        } else {
            qWarning() << "SystemMonitor: Failed to open PDH query for GPU monitoring";
        }
    }
    
    m_initialized = true;
    qDebug() << "SystemMonitor: Initialized successfully";
    return true;
}

void SystemMonitor::startMonitoring(int intervalMs)
{
    if (!m_initialized && !initialize()) {
        qWarning() << "SystemMonitor: Failed to initialize, cannot start monitoring";
        return;
    }
    
    if (m_monitoring) {
        stopMonitoring();
    }
    
    m_timer->setInterval(intervalMs);
    m_timer->start();
    m_monitoring = true;
    
    // Collect initial statistics
    collectStatistics();
    
    qDebug() << "SystemMonitor: Started monitoring with interval" << intervalMs << "ms";
}

void SystemMonitor::stopMonitoring()
{
    if (m_timer->isActive()) {
        m_timer->stop();
    }
    m_monitoring = false;
    qDebug() << "SystemMonitor: Stopped monitoring";
}

void SystemMonitor::collectStatistics()
{
    // Read FPS from volatile variable (thread-safe read on x86/x64)
    double currentFPS = m_latestFPS;
    
    // Now acquire the main mutex for the rest of the statistics
    QMutexLocker locker(&m_mutex);
    
    Statistics stats;
    stats.timestamp = QDateTime::currentDateTime();
    
    // Collect CPU usage
    stats.cpuUsage = getCpuUsage();
    
    // Collect GPU usage
    stats.gpuUsage = getGpuUsage();
    
    // Collect memory usage and update peak (process memory)
    double currentMemory = getMemoryUsage();
    stats.peakMemoryGB = m_peakMemoryGB;
    if (currentMemory > m_peakMemoryGB) {
        m_peakMemoryGB = currentMemory;
        stats.peakMemoryGB = m_peakMemoryGB;
    }
    
    // Collect system-wide memory information
    getSystemMemoryInfo(stats.systemMemoryTotalGB, stats.systemMemoryUsageGB, stats.systemMemoryAvailableGB);
    
    // Use the FPS value we read atomically
    // If FPS is 0 or invalid, keep the last known good value
    if (currentFPS > 0.0) {
        stats.averageFPS = currentFPS;
    } else if (m_lastStats.averageFPS > 0.0) {
        stats.averageFPS = m_lastStats.averageFPS; // Use last known good value
    } else {
        stats.averageFPS = 0.0; // No FPS data available yet
    }
    
    // Calculate accuracy from samples
    if (m_accuracySamples.isEmpty()) {
        stats.accuracy = 0.0;
    } else {
        double sum = 0.0;
        for (double sample : m_accuracySamples) {
            sum += sample;
        }
        stats.accuracy = sum / m_accuracySamples.size();
    }
    
    // Update last statistics
    m_lastStats = stats;
    
    // Update peak statistics
    if (stats.cpuUsage > m_peakStats.cpuUsage) m_peakStats.cpuUsage = stats.cpuUsage;
    if (stats.gpuUsage > m_peakStats.gpuUsage) m_peakStats.gpuUsage = stats.gpuUsage;
    if (stats.peakMemoryGB > m_peakStats.peakMemoryGB) m_peakStats.peakMemoryGB = stats.peakMemoryGB;
    if (stats.systemMemoryUsageGB > m_peakStats.systemMemoryUsageGB) m_peakStats.systemMemoryUsageGB = stats.systemMemoryUsageGB;
    if (stats.averageFPS > m_peakStats.averageFPS) m_peakStats.averageFPS = stats.averageFPS;
    if (stats.accuracy > m_peakStats.accuracy) m_peakStats.accuracy = stats.accuracy;
    
    // Update average statistics
    m_cpuSum += stats.cpuUsage;
    m_gpuSum += stats.gpuUsage;
    m_memorySum += currentMemory;
    m_systemMemorySum += stats.systemMemoryUsageGB;
    m_fpsSum += stats.averageFPS;
    m_sampleCount++;
    
    if (m_sampleCount > 0) {
        m_averageStats.cpuUsage = m_cpuSum / m_sampleCount;
        m_averageStats.gpuUsage = m_gpuSum / m_sampleCount;
        m_averageStats.peakMemoryGB = m_memorySum / m_sampleCount;
        m_averageStats.systemMemoryUsageGB = m_systemMemorySum / m_sampleCount;
        m_averageStats.systemMemoryTotalGB = stats.systemMemoryTotalGB; // Total doesn't change
        m_averageStats.systemMemoryAvailableGB = stats.systemMemoryAvailableGB; // Available changes frequently
        m_averageStats.averageFPS = m_fpsSum / m_sampleCount;
        m_averageStats.accuracy = stats.accuracy; // Keep current accuracy average
        m_averageStats.timestamp = stats.timestamp;
    }
    
    // Log statistics to application output
    qDebug() << "========================================";
    qDebug() << "=== SYSTEM MONITORING STATISTICS ===";
    qDebug() << "Timestamp:" << stats.timestamp.toString("yyyy-MM-dd hh:mm:ss");
    qDebug() << "--- LAST STATISTICS ---";
    qDebug() << "CPU Usage:" << QString::number(stats.cpuUsage, 'f', 2) << "%";
    qDebug() << "GPU Usage:" << QString::number(stats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Process Memory Usage:" << QString::number(stats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "System Memory Usage:" << QString::number(stats.systemMemoryUsageGB, 'f', 2) << "GB /" << QString::number(stats.systemMemoryTotalGB, 'f', 2) << "GB";
    qDebug() << "System Memory Available:" << QString::number(stats.systemMemoryAvailableGB, 'f', 2) << "GB";
    qDebug() << "Average FPS:" << QString::number(stats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "Accuracy:" << QString::number(stats.accuracy, 'f', 2) << "%";
    qDebug() << "--- AVERAGE STATISTICS ---";
    qDebug() << "Average CPU Usage:" << QString::number(m_averageStats.cpuUsage, 'f', 2) << "%";
    qDebug() << "Average GPU Usage:" << QString::number(m_averageStats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Average Process Memory Usage:" << QString::number(m_averageStats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "Average System Memory Usage:" << QString::number(m_averageStats.systemMemoryUsageGB, 'f', 2) << "GB";
    qDebug() << "Average FPS:" << QString::number(m_averageStats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "--- PEAK STATISTICS ---";
    qDebug() << "Peak CPU Usage:" << QString::number(m_peakStats.cpuUsage, 'f', 2) << "%";
    qDebug() << "Peak GPU Usage:" << QString::number(m_peakStats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Peak Process Memory Usage:" << QString::number(m_peakStats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "Peak System Memory Usage:" << QString::number(m_peakStats.systemMemoryUsageGB, 'f', 2) << "GB";
    qDebug() << "Peak Average FPS:" << QString::number(m_peakStats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "Peak Accuracy:" << QString::number(m_peakStats.accuracy, 'f', 2) << "%";
    qDebug() << "========================================";
    
    emit statisticsUpdated(stats);
}

double SystemMonitor::getCpuUsage()
{
    // Try GetSystemTimes first (more accurate)
    double cpuUsage = getCpuUsageSystemTimes();
    if (cpuUsage >= 0.0) {
        return cpuUsage;
    }
    
    // Fallback to PDH
    return getCpuUsagePDH();
}

double SystemMonitor::getCpuUsagePDH()
{
    if (!m_cpuQueryHandle || !m_cpuCounterHandle) {
        return 0.0;
    }
    
    PDH_STATUS status = PdhCollectQueryData(m_cpuQueryHandle);
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }
    
    PDH_FMT_COUNTERVALUE value;
    status = PdhGetFormattedCounterValue(m_cpuCounterHandle, PDH_FMT_DOUBLE, nullptr, &value);
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }
    
    return value.doubleValue;
}

double SystemMonitor::getCpuUsageSystemTimes()
{
    FILETIME idleTime, kernelTime, userTime;
    
    if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        return -1.0; // Error indicator
    }
    
    // Convert FILETIME to qint64 (100-nanosecond intervals)
    ULARGE_INTEGER idle, kernel, user;
    idle.LowPart = idleTime.dwLowDateTime;
    idle.HighPart = idleTime.dwHighDateTime;
    kernel.LowPart = kernelTime.dwLowDateTime;
    kernel.HighPart = kernelTime.dwHighDateTime;
    user.LowPart = userTime.dwLowDateTime;
    user.HighPart = userTime.dwHighDateTime;
    
    qint64 totalIdle = idle.QuadPart;
    qint64 totalKernel = kernel.QuadPart;
    qint64 totalUser = user.QuadPart;
    
    if (!m_cpuTimesInitialized) {
        // First call - just store values
        m_lastCpuIdleTime = totalIdle;
        m_lastCpuKernelTime = totalKernel;
        m_lastCpuUserTime = totalUser;
        m_cpuTimesInitialized = true;
        return 0.0; // Can't calculate percentage on first call
    }
    
    // Calculate difference
    qint64 idleDiff = totalIdle - m_lastCpuIdleTime;
    qint64 kernelDiff = totalKernel - m_lastCpuKernelTime;
    qint64 userDiff = totalUser - m_lastCpuUserTime;
    
    // Total CPU time = kernel + user
    qint64 totalDiff = kernelDiff + userDiff;
    
    if (totalDiff == 0) {
        return 0.0; // Avoid division by zero
    }
    
    // CPU usage = (total - idle) / total * 100
    double cpuUsage = ((double)(totalDiff - idleDiff) / (double)totalDiff) * 100.0;
    
    // Update stored values
    m_lastCpuIdleTime = totalIdle;
    m_lastCpuKernelTime = totalKernel;
    m_lastCpuUserTime = totalUser;
    
    // Clamp to valid range
    return qBound(0.0, cpuUsage, 100.0);
}

double SystemMonitor::getGpuUsage()
{
    switch (m_activeGpuMethod) {
        case GPU_METHOD_NVML:
            return getGpuUsageNVML();
        case GPU_METHOD_PDH:
            return getGpuUsagePDH();
        default:
            return 0.0;
    }
}

double SystemMonitor::getGpuUsageNVML()
{
    #ifdef NVML_AVAILABLE
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result == NVML_SUCCESS) {
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result == NVML_SUCCESS) {
            return static_cast<double>(utilization.gpu);
        }
    }
    #endif
    return 0.0;
}

double SystemMonitor::getGpuUsageDXGI()
{
    // DXGI method removed - using PDH for AMD/NVIDIA/Intel GPU monitoring
    // PDH provides accurate GPU utilization for all GPU vendors on Windows
    // This method is kept for interface compatibility but always returns 0
    return 0.0;
}

double SystemMonitor::getGpuUsagePDH()
{
    if (!m_gpuQueryHandle || !m_gpuCounterHandle) {
        return 0.0;
    }
    
    PDH_STATUS status = PdhCollectQueryData(m_gpuQueryHandle);
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }
    
    PDH_FMT_COUNTERVALUE value;
    status = PdhGetFormattedCounterValue(m_gpuCounterHandle, PDH_FMT_DOUBLE, nullptr, &value);
    if (status != ERROR_SUCCESS) {
        return 0.0;
    }
    
    // PDH returns utilization as a percentage (0-100)
    // Clamp to valid range to ensure accuracy
    double gpuUsage = qBound(0.0, value.doubleValue, 100.0);
    
    return gpuUsage;
}

double SystemMonitor::getMemoryUsage()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // Convert bytes to GB
        double memoryGB = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0 * 1024.0);
        return memoryGB;
    }
    return 0.0;
}

void SystemMonitor::getSystemMemoryInfo(double& totalGB, double& usedGB, double& availableGB)
{
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&memStatus)) {
        // Convert bytes to GB
        totalGB = static_cast<double>(memStatus.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        availableGB = static_cast<double>(memStatus.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
        usedGB = totalGB - availableGB;
    } else {
        totalGB = 0.0;
        usedGB = 0.0;
        availableGB = 0.0;
    }
}

void SystemMonitor::updatePeakMemory()
{
    double currentMemory = getMemoryUsage();
    if (currentMemory > m_peakMemoryGB) {
        m_peakMemoryGB = currentMemory;
    }
}

SystemMonitor::Statistics SystemMonitor::getCurrentStatistics() const
{
    QMutexLocker locker(&m_mutex);
    return m_lastStats;
}

SystemMonitor::Statistics SystemMonitor::getLastStatistics() const
{
    QMutexLocker locker(&m_mutex);
    return m_lastStats;
}

SystemMonitor::Statistics SystemMonitor::getAverageStatistics() const
{
    QMutexLocker locker(&m_mutex);
    return m_averageStats;
}

SystemMonitor::Statistics SystemMonitor::getPeakStatistics() const
{
    QMutexLocker locker(&m_mutex);
    return m_peakStats;
}

void SystemMonitor::updateFPS(double fps)
{
    // This method can be called from any thread via Qt::QueuedConnection
    // Simple volatile write - thread-safe on x86/x64, no locks, no crashes
    // Volatile ensures the write is not optimized away and is visible to other threads
    // Direct assignment to volatile double is safe on x86/x64 (atomic for 8-byte aligned doubles)
    if (fps >= 0.0 && fps < 1000.0) { // Sanity check: FPS should be reasonable (allow 0.0)
        // Direct assignment to volatile is safe - compiler ensures atomicity for aligned doubles
        m_latestFPS = fps;
    }
}

void SystemMonitor::resetFPSTracking()
{
    // Simple volatile write
    m_latestFPS = 0.0;
    m_fpsTrackingTimer.restart();
}

void SystemMonitor::updateAccuracy(double detectionConfidence)
{
    QMutexLocker locker(&m_mutex);
    
    // Add new sample
    m_accuracySamples.append(detectionConfidence * 100.0); // Convert to percentage
    
    // Keep only the most recent samples
    if (m_accuracySamples.size() > MAX_ACCURACY_SAMPLES) {
        m_accuracySamples.removeFirst();
    }
}

void SystemMonitor::resetAccuracyTracking()
{
    QMutexLocker locker(&m_mutex);
    m_accuracySamples.clear();
}

void SystemMonitor::saveStatisticsToDocx(const QString& filePath) const
{
    // Read FPS from volatile variable (thread-safe read)
    double latestFPS = m_latestFPS;
    
    if (latestFPS == 0.0) {
        latestFPS = m_lastStats.averageFPS; // Fallback to last known value
    }
    
    QMutexLocker locker(&m_mutex);
    
    QString outputPath = filePath;
    if (outputPath.isEmpty()) {
        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            // Fallback for systems without standard download path
            downloadsPath = "C:/Downloads";
        }
        
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath);
        }
        outputPath = downloadsPath + "/crash_report_" + timestamp + ".docx";
    }
    
    // Create a simple DOCX file (DOCX is a ZIP file containing XML)
    // For simplicity, we'll create a basic DOCX structure
    // Note: This is a minimal implementation. For full DOCX support, consider using a library.
    
    // Create temporary directory for DOCX contents
    QDir tempDir = QDir::temp();
    QString tempDocxDir = tempDir.absoluteFilePath("docx_temp_" + QString::number(QDateTime::currentMSecsSinceEpoch()));
    QDir().mkpath(tempDocxDir);
    
    // Create [Content_Types].xml
    QString contentTypes = R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>)";
    
    QFile contentTypesFile(tempDocxDir + "/[Content_Types].xml");
    if (contentTypesFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&contentTypesFile);
        out << contentTypes;
        contentTypesFile.close();
    }
    
    // Create _rels/.rels
    QDir().mkpath(tempDocxDir + "/_rels");
    QString rels = R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>)";
    
    QFile relsFile(tempDocxDir + "/_rels/.rels");
    if (relsFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&relsFile);
        out << rels;
        relsFile.close();
    }
    
    // Create word/document.xml
    QDir().mkpath(tempDocxDir + "/word");
    QString documentXml = QString(R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>System Monitoring Statistics Report</w:t></w:r></w:p>
<w:p><w:r><w:t>Generated: %1</w:t></w:r></w:p>
<w:p><w:r><w:t></w:t></w:r></w:p>
<w:p><w:r><w:t>Last Statistics:</w:t></w:r></w:p>
<w:p><w:r><w:t>Timestamp: %2</w:t></w:r></w:p>
<w:p><w:r><w:t>CPU Usage: %3%</w:t></w:r></w:p>
<w:p><w:r><w:t>GPU Usage: %4%</w:t></w:r></w:p>
<w:p><w:r><w:t>Process Memory Usage: %5 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>System Memory Usage: %6 GB / %7 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>System Memory Available: %8 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>Average FPS: %9 FPS</w:t></w:r></w:p>
<w:p><w:r><w:t>Accuracy: %10%</w:t></w:r></w:p>
<w:p><w:r><w:t></w:t></w:r></w:p>
<w:p><w:r><w:t>Average Statistics:</w:t></w:r></w:p>
<w:p><w:r><w:t>Average CPU Usage: %11%</w:t></w:r></w:p>
<w:p><w:r><w:t>Average GPU Usage: %12%</w:t></w:r></w:p>
<w:p><w:r><w:t>Average Process Memory Usage: %13 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>Average System Memory Usage: %14 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>Average FPS: %15 FPS</w:t></w:r></w:p>
<w:p><w:r><w:t></w:t></w:r></w:p>
<w:p><w:r><w:t>Peak Statistics:</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak CPU Usage: %16%</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak GPU Usage: %17%</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak Process Memory Usage: %18 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak System Memory Usage: %19 GB</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak Average FPS: %20 FPS</w:t></w:r></w:p>
<w:p><w:r><w:t>Peak Accuracy: %21%</w:t></w:r></w:p>
</w:body>
</w:document>)")
        .arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"))
        .arg(m_lastStats.timestamp.toString("yyyy-MM-dd hh:mm:ss"))
        .arg(QString::number(m_lastStats.cpuUsage, 'f', 2))
        .arg(QString::number(m_lastStats.gpuUsage, 'f', 2))
        .arg(QString::number(m_lastStats.peakMemoryGB, 'f', 2))
        .arg(QString::number(m_lastStats.systemMemoryUsageGB, 'f', 2))
        .arg(QString::number(m_lastStats.systemMemoryTotalGB, 'f', 2))
        .arg(QString::number(m_lastStats.systemMemoryAvailableGB, 'f', 2))
        .arg(QString::number(latestFPS, 'f', 2))
        .arg(QString::number(m_lastStats.accuracy, 'f', 2))
        .arg(QString::number(m_averageStats.cpuUsage, 'f', 2))
        .arg(QString::number(m_averageStats.gpuUsage, 'f', 2))
        .arg(QString::number(m_averageStats.peakMemoryGB, 'f', 2))
        .arg(QString::number(m_averageStats.systemMemoryUsageGB, 'f', 2))
        .arg(QString::number(m_averageStats.averageFPS, 'f', 2))
        .arg(QString::number(m_peakStats.cpuUsage, 'f', 2))
        .arg(QString::number(m_peakStats.gpuUsage, 'f', 2))
        .arg(QString::number(m_peakStats.peakMemoryGB, 'f', 2))
        .arg(QString::number(m_peakStats.systemMemoryUsageGB, 'f', 2))
        .arg(QString::number(m_peakStats.averageFPS, 'f', 2))
        .arg(QString::number(m_peakStats.accuracy, 'f', 2));
    
    QFile docFile(tempDocxDir + "/word/document.xml");
    if (docFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&docFile);
        out << documentXml;
        docFile.close();
    }
    
    // Create word/_rels/document.xml.rels
    QDir().mkpath(tempDocxDir + "/word/_rels");
    QString docRels = R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>)";
    
    QFile docRelsFile(tempDocxDir + "/word/_rels/document.xml.rels");
    if (docRelsFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&docRelsFile);
        out << docRels;
        docRelsFile.close();
    }
    
    // Note: DOCX files are ZIP archives containing XML files
    // For a complete implementation, we would need a ZIP library (like QuaZip or minizip)
    // For now, we'll save as a text file instead
    // The XML structure is created above if you want to implement ZIP compression later
    qWarning() << "SystemMonitor: DOCX generation requires ZIP library. Saving as text file instead.";
    QString textPath = outputPath;
    textPath.replace(".docx", ".txt");
    saveStatisticsToText(textPath);
    
    // Clean up temp directory
    QDir(tempDocxDir).removeRecursively();
}

void SystemMonitor::saveStatisticsToText(const QString& filePath) const
{
    // Read FPS from volatile variable (thread-safe read)
    double latestFPS = m_latestFPS;
    
    qDebug() << "saveStatisticsToText: Read FPS:" << latestFPS;
    
    if (latestFPS == 0.0) {
        latestFPS = m_lastStats.averageFPS; // Fallback to last known value
        qDebug() << "saveStatisticsToText: Using fallback FPS:" << latestFPS;
    }
    
    QMutexLocker locker(&m_mutex);
    
    QString outputPath = filePath;
    if (outputPath.isEmpty()) {
        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            // Fallback for systems without standard download path
            downloadsPath = "C:/Downloads";
        }
        
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath);
        }
        outputPath = downloadsPath + "/crash_report_" + timestamp + ".txt";
    }
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "SystemMonitor: Failed to open file for writing:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    out << "========================================\n";
    out << "SYSTEM MONITORING STATISTICS REPORT\n";
    out << "========================================\n\n";
    out << "Generated: " << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << "\n\n";
    
    out << "LAST STATISTICS:\n";
    out << "----------------\n";
    out << "Timestamp: " << m_lastStats.timestamp.toString("yyyy-MM-dd hh:mm:ss") << "\n";
    out << "CPU Usage: " << QString::number(m_lastStats.cpuUsage, 'f', 2) << "%\n";
    out << "GPU Usage: " << QString::number(m_lastStats.gpuUsage, 'f', 2) << "%\n";
    out << "Process Memory Usage: " << QString::number(m_lastStats.peakMemoryGB, 'f', 2) << " GB\n";
    out << "System Memory Usage: " << QString::number(m_lastStats.systemMemoryUsageGB, 'f', 2) << " GB / " << QString::number(m_lastStats.systemMemoryTotalGB, 'f', 2) << " GB\n";
    out << "System Memory Available: " << QString::number(m_lastStats.systemMemoryAvailableGB, 'f', 2) << " GB\n";
    out << "Average FPS: " << QString::number(latestFPS, 'f', 2) << " FPS\n";
    out << "Accuracy: " << QString::number(m_lastStats.accuracy, 'f', 2) << "%\n\n";
    
    out << "AVERAGE STATISTICS:\n";
    out << "----------------\n";
    out << "Average CPU Usage: " << QString::number(m_averageStats.cpuUsage, 'f', 2) << "%\n";
    out << "Average GPU Usage: " << QString::number(m_averageStats.gpuUsage, 'f', 2) << "%\n";
    out << "Average Process Memory Usage: " << QString::number(m_averageStats.peakMemoryGB, 'f', 2) << " GB\n";
    out << "Average System Memory Usage: " << QString::number(m_averageStats.systemMemoryUsageGB, 'f', 2) << " GB\n";
    out << "Average FPS: " << QString::number(m_averageStats.averageFPS, 'f', 2) << " FPS\n\n";
    
    out << "PEAK STATISTICS:\n";
    out << "----------------\n";
    out << "Peak CPU Usage: " << QString::number(m_peakStats.cpuUsage, 'f', 2) << "%\n";
    out << "Peak GPU Usage: " << QString::number(m_peakStats.gpuUsage, 'f', 2) << "%\n";
    out << "Peak Process Memory Usage: " << QString::number(m_peakStats.peakMemoryGB, 'f', 2) << " GB\n";
    out << "Peak System Memory Usage: " << QString::number(m_peakStats.systemMemoryUsageGB, 'f', 2) << " GB\n";
    out << "Peak Average FPS: " << QString::number(m_peakStats.averageFPS, 'f', 2) << " FPS\n";
    out << "Peak Accuracy: " << QString::number(m_peakStats.accuracy, 'f', 2) << "%\n";
    out << "========================================\n";
    
    file.close();
    qDebug() << "SystemMonitor: Statistics saved to:" << outputPath;
}

