#include "core/system_monitor.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStandardPaths>
#include <QCoreApplication>
#include <QtCore/qmath.h>
#include <algorithm>
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
    , m_latestFPS(0.0)  // Initialize volatile FPS variable
    , m_peakMemoryGB(0.0)
    , m_initialized(false)
    , m_monitoring(false)
{
    // Initialize statistics
    m_lastStats.cpuUsage = 0.0;
    m_lastStats.gpuUsage = 0.0;
    m_lastStats.peakMemoryGB = 0.0;
    m_lastStats.averageFPS = 0.0;
    m_lastStats.timestamp = QDateTime::currentDateTime();
    
    m_peakStats = m_lastStats;
    m_averageStats = m_lastStats;
    
    // Initialize average tracking variables
    m_cpuSum = 0.0;
    m_gpuSum = 0.0;
    m_memorySum = 0.0;
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
    
    // Initialize GPU monitoring - try NVML first, then fallback to Windows Performance Counters
    #ifdef NVML_AVAILABLE
    nvmlReturn_t nvmlStatus = nvmlInit();
    if (nvmlStatus == NVML_SUCCESS) {
        qDebug() << "SystemMonitor: NVML initialized successfully";
    } else {
        qWarning() << "SystemMonitor: NVML initialization failed, using Windows Performance Counters";
    }
    #endif
    
    // Try to initialize GPU monitoring using Windows Performance Counters
    // This may not work on all systems, but it's a fallback
    status = PdhOpenQuery(nullptr, 0, &m_gpuQueryHandle);
    if (status == ERROR_SUCCESS) {
        // Try to add GPU counter - this path may vary by system
        // Common paths: "\GPU Engine(*)\Utilization Percentage" or "\GPU Engine(*engtype_3D*)\Utilization Percentage"
        status = PdhAddCounter(m_gpuQueryHandle, L"\\GPU Engine(*engtype_3D*)\\Utilization Percentage", 0, &m_gpuCounterHandle);
        if (status != ERROR_SUCCESS) {
            // Try alternative path
            status = PdhAddCounter(m_gpuQueryHandle, L"\\GPU Engine(*)\\Utilization Percentage", 0, &m_gpuCounterHandle);
            if (status != ERROR_SUCCESS) {
                qWarning() << "SystemMonitor: GPU counter not available, GPU monitoring may be limited";
                m_gpuCounterHandle = nullptr;
            }
        }
        if (m_gpuCounterHandle) {
            PdhCollectQueryData(m_gpuQueryHandle);
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
    
    qDebug() << "========================================";
    qDebug() << "SystemMonitor::collectStatistics() - Read m_latestFPS:" << currentFPS;
    
    // Now acquire the main mutex for the rest of the statistics
    QMutexLocker locker(&m_mutex);
    
    Statistics stats;
    stats.timestamp = QDateTime::currentDateTime();
    
    // Collect CPU usage
    stats.cpuUsage = getCpuUsage();
    
    // Collect GPU usage
    stats.gpuUsage = getGpuUsage();
    
    // Collect memory usage and update peak
    double currentMemory = getMemoryUsage();
    stats.peakMemoryGB = m_peakMemoryGB;
    if (currentMemory > m_peakMemoryGB) {
        m_peakMemoryGB = currentMemory;
        stats.peakMemoryGB = m_peakMemoryGB;
    }
    
    // Use the FPS value we read atomically
    // IMPORTANT: Always use currentFPS if it's valid, don't fall back to old values
    stats.averageFPS = currentFPS;
    
    qDebug() << "SystemMonitor::collectStatistics() - Storing FPS:" << stats.averageFPS 
             << "(will be saved to m_lastStats.averageFPS for crash reports)";
    qDebug() << "========================================";
    
    // Update last statistics
    m_lastStats = stats;
    
    // Update peak statistics
    if (stats.cpuUsage > m_peakStats.cpuUsage) m_peakStats.cpuUsage = stats.cpuUsage;
    if (stats.gpuUsage > m_peakStats.gpuUsage) m_peakStats.gpuUsage = stats.gpuUsage;
    if (stats.peakMemoryGB > m_peakStats.peakMemoryGB) m_peakStats.peakMemoryGB = stats.peakMemoryGB;
    if (stats.averageFPS > m_peakStats.averageFPS) m_peakStats.averageFPS = stats.averageFPS;
    
    // Update average statistics
    m_cpuSum += stats.cpuUsage;
    m_gpuSum += stats.gpuUsage;
    m_memorySum += currentMemory;
    m_fpsSum += stats.averageFPS;
    m_sampleCount++;
    
    if (m_sampleCount > 0) {
        m_averageStats.cpuUsage = m_cpuSum / m_sampleCount;
        m_averageStats.gpuUsage = m_gpuSum / m_sampleCount;
        m_averageStats.peakMemoryGB = m_memorySum / m_sampleCount;
        m_averageStats.averageFPS = m_fpsSum / m_sampleCount;
        m_averageStats.timestamp = stats.timestamp;
    }
    
    // Log statistics to application output
    qDebug() << "========================================";
    qDebug() << "=== SYSTEM MONITORING STATISTICS ===";
    qDebug() << "Timestamp:" << stats.timestamp.toString("yyyy-MM-dd hh:mm:ss");
    qDebug() << "--- LAST STATISTICS ---";
    qDebug() << "CPU Usage:" << QString::number(stats.cpuUsage, 'f', 2) << "%";
    qDebug() << "GPU Usage:" << QString::number(stats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Peak Memory Usage:" << QString::number(stats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "Average FPS:" << QString::number(stats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "--- AVERAGE STATISTICS ---";
    qDebug() << "Average CPU Usage:" << QString::number(m_averageStats.cpuUsage, 'f', 2) << "%";
    qDebug() << "Average GPU Usage:" << QString::number(m_averageStats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Average Memory Usage:" << QString::number(m_averageStats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "Average FPS:" << QString::number(m_averageStats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "--- PEAK STATISTICS ---";
    qDebug() << "Peak CPU Usage:" << QString::number(m_peakStats.cpuUsage, 'f', 2) << "%";
    qDebug() << "Peak GPU Usage:" << QString::number(m_peakStats.gpuUsage, 'f', 2) << "%";
    qDebug() << "Peak Memory Usage:" << QString::number(m_peakStats.peakMemoryGB, 'f', 2) << "GB";
    qDebug() << "Peak Average FPS:" << QString::number(m_peakStats.averageFPS, 'f', 2) << "FPS";
    qDebug() << "========================================";
    
    emit statisticsUpdated(stats);
}

double SystemMonitor::getCpuUsage()
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

double SystemMonitor::getGpuUsage()
{
    #ifdef NVML_AVAILABLE
    // Try NVML first
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
    
    // Fallback to Windows Performance Counters
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
    
    return value.doubleValue;
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

void SystemMonitor::updateFPS(double fps)
{
    qDebug() << "SystemMonitor::updateFPS() ENTERED with fps:" << fps << "this pointer:" << (void*)this;
    qDebug() << "SystemMonitor: Address of m_latestFPS:" << (void*)&m_latestFPS;
    
    // Simple volatile write - thread-safe on x86/x64, no locks, no crashes
    // Volatile ensures the write is not optimized away and is visible to other threads
    // Cap at reasonable maximum (200 FPS) to prevent outliers from skewing statistics
    double cappedFPS = (fps > 200.0) ? 200.0 : fps;
    if (cappedFPS > 0.0 && cappedFPS <= 200.0) { // Sanity check: FPS should be reasonable (0-200 range)
        qDebug() << "SystemMonitor: About to write to m_latestFPS, current value:" << m_latestFPS;
        
        // Use memcpy as a safer alternative to direct assignment
        double tempFPS = cappedFPS;
        std::memcpy(const_cast<double*>(&m_latestFPS), &tempFPS, sizeof(double));
        
        qDebug() << "SystemMonitor: FPS updated to:" << cappedFPS << "(original:" << fps << ") Verification read:" << m_latestFPS;
    } else {
        qDebug() << "SystemMonitor: Invalid FPS value rejected:" << fps;
    }
    
    qDebug() << "SystemMonitor::updateFPS() EXITING";
}

void SystemMonitor::resetFPSTracking()
{
    // Simple volatile write
    m_latestFPS = 0.0;
    m_fpsTrackingTimer.restart();
}

void SystemMonitor::saveStatisticsToText(const QString& filePath) const
{
    QMutexLocker locker(&m_mutex);
    
    // Use AVERAGE FPS from the entire session, not the last instant FPS
    // This provides a more accurate representation of system performance
    // (last FPS can be misleading if measured during page transitions)
    double sessionAverageFPS = m_averageStats.averageFPS;
    double lastInstantFPS = m_lastStats.averageFPS;
    
    qDebug() << "saveStatisticsToText: Session Average FPS:" << sessionAverageFPS 
             << "Last Instant FPS:" << lastInstantFPS;
    
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
    out << "Peak Memory Usage: " << QString::number(m_lastStats.peakMemoryGB, 'f', 2) << " GB\n";
    out << "Last Instant FPS: " << QString::number(lastInstantFPS, 'f', 2) << " FPS\n";
    out << "Session Average FPS: " << QString::number(sessionAverageFPS, 'f', 2) << " FPS\n\n";
    
    out << "PEAK STATISTICS:\n";
    out << "----------------\n";
    out << "Peak CPU Usage: " << QString::number(m_peakStats.cpuUsage, 'f', 2) << "%\n";
    out << "Peak GPU Usage: " << QString::number(m_peakStats.gpuUsage, 'f', 2) << "%\n";
    out << "Peak Memory Usage: " << QString::number(m_peakStats.peakMemoryGB, 'f', 2) << " GB\n";
    out << "Peak Average FPS: " << QString::number(m_peakStats.averageFPS, 'f', 2) << " FPS\n";
    out << "========================================\n";
    
    file.close();
    qDebug() << "SystemMonitor: Statistics saved to:" << outputPath;
}

