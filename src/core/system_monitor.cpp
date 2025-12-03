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
// Define NVML types and structures locally to avoid header dependency
// This allows dynamic loading without requiring nvml.h at compile time

// NVML return codes
typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

// NVML temperature sensors
typedef enum nvmlTemperatureSensors_enum {
    NVML_TEMPERATURE_GPU = 0
} nvmlTemperatureSensors_t;

// NVML device handle (opaque pointer)
typedef struct nvmlDevice_st* nvmlDevice_t;

// NVML utilization structure
typedef struct nvmlUtilization_st {
    unsigned int gpu;       // GPU utilization (%)
    unsigned int memory;    // Memory utilization (%)
} nvmlUtilization_t;

// NVML memory structure
typedef struct nvmlMemory_st {
    unsigned long long total;      // Total memory (bytes)
    unsigned long long free;       // Free memory (bytes)
    unsigned long long used;       // Used memory (bytes)
} nvmlMemory_t;

// NVML function pointers for dynamic loading
typedef nvmlReturn_t (*nvmlInit_v2_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_t)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);

static nvmlInit_v2_t nvmlInit_v2_ptr = nullptr;
static nvmlShutdown_t nvmlShutdown_ptr = nullptr;
static nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex_ptr = nullptr;
static nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates_ptr = nullptr;
static nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo_ptr = nullptr;
static nvmlDeviceGetTemperature_t nvmlDeviceGetTemperature_ptr = nullptr;

static HMODULE nvmlModule = nullptr;

SystemMonitor::SystemMonitor(QObject *parent)
    : QObject(parent)
    , m_timer(new QTimer(this))
    , m_cpuQueryHandle(nullptr)
    , m_cpuCounterHandle(nullptr)
    , m_gpuQueryHandle(nullptr)
    , m_gpuCounterHandle(nullptr)
    , m_latestFPS(0.0)  // Initialize volatile FPS variable
    , m_peakMemoryGB(0.0)
    , m_nvmlInitialized(false)
    , m_nvmlDevice(nullptr)
    , m_initialized(false)
    , m_monitoring(false)
{
    // Initialize statistics
    m_lastStats.cpuUsage = 0.0;
    m_lastStats.gpuUsage = 0.0;
    m_lastStats.gpuMemoryUsedGB = 0.0;
    m_lastStats.gpuMemoryTotalGB = 0.0;
    m_lastStats.gpuMemoryUsage = 0.0;
    m_lastStats.gpuTemperature = 0.0;
    m_lastStats.systemMemoryUsedGB = 0.0;
    m_lastStats.systemMemoryTotalGB = 0.0;
    m_lastStats.systemMemoryUsage = 0.0;
    m_lastStats.processMemoryGB = 0.0;
    m_lastStats.peakProcessMemoryGB = 0.0;
    m_lastStats.averageFPS = 0.0;
    m_lastStats.timestamp = QDateTime::currentDateTime();
    m_lastStats.gpuMetricsAvailable = false;
    
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
    
    // Shutdown NVML
    shutdownNVML();
    
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

bool SystemMonitor::initializeNVML()
{
    // Try to dynamically load NVML library
    nvmlModule = LoadLibraryA("nvml.dll");
    if (!nvmlModule) {
        qWarning() << "SystemMonitor: Failed to load nvml.dll - GPU metrics will not be available";
        return false;
    }
    
    // Load NVML functions
    nvmlInit_v2_ptr = (nvmlInit_v2_t)GetProcAddress(nvmlModule, "nvmlInit_v2");
    nvmlShutdown_ptr = (nvmlShutdown_t)GetProcAddress(nvmlModule, "nvmlShutdown");
    nvmlDeviceGetHandleByIndex_ptr = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(nvmlModule, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetUtilizationRates_ptr = (nvmlDeviceGetUtilizationRates_t)GetProcAddress(nvmlModule, "nvmlDeviceGetUtilizationRates");
    nvmlDeviceGetMemoryInfo_ptr = (nvmlDeviceGetMemoryInfo_t)GetProcAddress(nvmlModule, "nvmlDeviceGetMemoryInfo");
    nvmlDeviceGetTemperature_ptr = (nvmlDeviceGetTemperature_t)GetProcAddress(nvmlModule, "nvmlDeviceGetTemperature");
    
    if (!nvmlInit_v2_ptr || !nvmlDeviceGetHandleByIndex_ptr) {
        qWarning() << "SystemMonitor: Failed to load NVML functions";
        FreeLibrary(nvmlModule);
        nvmlModule = nullptr;
        return false;
    }
    
    // Initialize NVML
    nvmlReturn_t result = nvmlInit_v2_ptr();
    if (result != NVML_SUCCESS) {
        qWarning() << "SystemMonitor: NVML initialization failed with error:" << result;
        FreeLibrary(nvmlModule);
        nvmlModule = nullptr;
        return false;
    }
    
    // Get handle to first GPU device
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex_ptr(0, &device);
    if (result != NVML_SUCCESS) {
        qWarning() << "SystemMonitor: Failed to get GPU device handle:" << result;
        nvmlShutdown_ptr();
        FreeLibrary(nvmlModule);
        nvmlModule = nullptr;
        return false;
    }
    
    // Store device handle
    m_nvmlDevice = new nvmlDevice_t(device);
    m_nvmlInitialized = true;
    
    qDebug() << "SystemMonitor: NVML initialized successfully - Accurate GPU monitoring enabled";
    return true;
}

void SystemMonitor::shutdownNVML()
{
    if (m_nvmlInitialized && nvmlShutdown_ptr) {
        nvmlShutdown_ptr();
        m_nvmlInitialized = false;
    }
    
    if (m_nvmlDevice) {
        delete static_cast<nvmlDevice_t*>(m_nvmlDevice);
        m_nvmlDevice = nullptr;
    }
    
    if (nvmlModule) {
        FreeLibrary(nvmlModule);
        nvmlModule = nullptr;
    }
}

bool SystemMonitor::initialize()
{
    if (m_initialized) {
        return true;
    }
    
    qDebug() << "SystemMonitor: Initializing comprehensive system monitoring...";
    
    // Initialize CPU monitoring using PDH (Performance Data Helper)
    PDH_STATUS status = PdhOpenQuery(nullptr, 0, &m_cpuQueryHandle);
    if (status != ERROR_SUCCESS) {
        qWarning() << "SystemMonitor: Failed to open CPU query:" << status;
        return false;
    }
    
    // Add CPU counter - using "\Processor(_Total)\% Processor Time" for SYSTEM-WIDE CPU usage
    status = PdhAddCounter(m_cpuQueryHandle, L"\\Processor(_Total)\\% Processor Time", 0, &m_cpuCounterHandle);
    if (status != ERROR_SUCCESS) {
        qWarning() << "SystemMonitor: Failed to add CPU counter:" << status;
        PdhCloseQuery(m_cpuQueryHandle);
        m_cpuQueryHandle = nullptr;
        return false;
    }
    
    // Collect initial sample for CPU
    PdhCollectQueryData(m_cpuQueryHandle);
    qDebug() << "SystemMonitor: CPU monitoring initialized (system-wide)";
    
    // Initialize NVML for accurate GPU monitoring
    if (initializeNVML()) {
        qDebug() << "SystemMonitor: GPU monitoring initialized with NVML (system-wide, accurate)";
    } else {
        qWarning() << "SystemMonitor: NVML not available - GPU metrics will not be available";
        qWarning() << "SystemMonitor: Make sure NVIDIA drivers are installed and nvml.dll is accessible";
    }
    
    m_initialized = true;
    qDebug() << "SystemMonitor: Initialization complete";
    qDebug() << "SystemMonitor: - CPU Usage: System-wide (via PDH)";
    qDebug() << "SystemMonitor: - GPU Usage: System-wide (via NVML)" << (m_nvmlInitialized ? "[ACTIVE]" : "[UNAVAILABLE]");
    qDebug() << "SystemMonitor: - GPU Memory: System-wide (via NVML)" << (m_nvmlInitialized ? "[ACTIVE]" : "[UNAVAILABLE]");
    qDebug() << "SystemMonitor: - System Memory: System-wide (via GlobalMemoryStatusEx)";
    qDebug() << "SystemMonitor: - Process Memory: Application-specific (via GetProcessMemoryInfo)";
    
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
    stats.gpuMetricsAvailable = m_nvmlInitialized;
    
    // ===== COLLECT CPU USAGE (SYSTEM-WIDE) =====
    stats.cpuUsage = getCpuUsage();
    
    // ===== COLLECT GPU METRICS (SYSTEM-WIDE) =====
    stats.gpuUsage = getGpuUsage();
    stats.gpuMemoryUsedGB = getGpuMemoryUsedGB();
    stats.gpuMemoryTotalGB = getGpuMemoryTotalGB();
    if (stats.gpuMemoryTotalGB > 0) {
        stats.gpuMemoryUsage = (stats.gpuMemoryUsedGB / stats.gpuMemoryTotalGB) * 100.0;
    } else {
        stats.gpuMemoryUsage = 0.0;
    }
    stats.gpuTemperature = getGpuTemperature();
    
    // ===== COLLECT SYSTEM MEMORY (SYSTEM-WIDE) =====
    stats.systemMemoryUsedGB = getSystemMemoryUsedGB();
    stats.systemMemoryTotalGB = getSystemMemoryTotalGB();
    if (stats.systemMemoryTotalGB > 0) {
        stats.systemMemoryUsage = (stats.systemMemoryUsedGB / stats.systemMemoryTotalGB) * 100.0;
    } else {
        stats.systemMemoryUsage = 0.0;
    }
    
    // ===== COLLECT PROCESS MEMORY (APPLICATION-SPECIFIC) =====
    double currentProcessMemory = getProcessMemoryUsage();
    stats.processMemoryGB = currentProcessMemory;
    stats.peakProcessMemoryGB = m_peakMemoryGB;
    if (currentProcessMemory > m_peakMemoryGB) {
        m_peakMemoryGB = currentProcessMemory;
        stats.peakProcessMemoryGB = m_peakMemoryGB;
    }
    
    // ===== COLLECT FPS =====
    stats.averageFPS = currentFPS;
    
    // Update last statistics
    m_lastStats = stats;
    
    // Update peak statistics
    if (stats.cpuUsage > m_peakStats.cpuUsage) m_peakStats.cpuUsage = stats.cpuUsage;
    if (stats.gpuUsage > m_peakStats.gpuUsage) m_peakStats.gpuUsage = stats.gpuUsage;
    if (stats.gpuMemoryUsage > m_peakStats.gpuMemoryUsage) m_peakStats.gpuMemoryUsage = stats.gpuMemoryUsage;
    if (stats.systemMemoryUsage > m_peakStats.systemMemoryUsage) m_peakStats.systemMemoryUsage = stats.systemMemoryUsage;
    if (stats.processMemoryGB > m_peakStats.processMemoryGB) m_peakStats.processMemoryGB = stats.processMemoryGB;
    if (stats.gpuTemperature > m_peakStats.gpuTemperature) m_peakStats.gpuTemperature = stats.gpuTemperature;
    if (stats.averageFPS > m_peakStats.averageFPS) m_peakStats.averageFPS = stats.averageFPS;
    
    // Update average statistics
    m_cpuSum += stats.cpuUsage;
    m_gpuSum += stats.gpuUsage;
    m_memorySum += stats.systemMemoryUsage;
    m_fpsSum += stats.averageFPS;
    m_sampleCount++;
    
    if (m_sampleCount > 0) {
        m_averageStats.cpuUsage = m_cpuSum / m_sampleCount;
        m_averageStats.gpuUsage = m_gpuSum / m_sampleCount;
        m_averageStats.systemMemoryUsage = m_memorySum / m_sampleCount;
        m_averageStats.averageFPS = m_fpsSum / m_sampleCount;
        m_averageStats.timestamp = stats.timestamp;
    }
    
    // Log comprehensive statistics to application output
    qDebug() << "========================================";
    qDebug() << "=== COMPREHENSIVE SYSTEM MONITORING ===";
    qDebug() << "Timestamp:" << stats.timestamp.toString("yyyy-MM-dd hh:mm:ss");
    qDebug() << "";
    qDebug() << "--- CPU USAGE (SYSTEM-WIDE) ---";
    qDebug() << "Current:" << QString::number(stats.cpuUsage, 'f', 1) << "%";
    qDebug() << "Average:" << QString::number(m_averageStats.cpuUsage, 'f', 1) << "%";
    qDebug() << "Peak:" << QString::number(m_peakStats.cpuUsage, 'f', 1) << "%";
    qDebug() << "";
    
    if (stats.gpuMetricsAvailable) {
        qDebug() << "--- GPU USAGE (SYSTEM-WIDE) ---";
        qDebug() << "Current GPU Utilization:" << QString::number(stats.gpuUsage, 'f', 1) << "%";
        qDebug() << "Average GPU Utilization:" << QString::number(m_averageStats.gpuUsage, 'f', 1) << "%";
        qDebug() << "Peak GPU Utilization:" << QString::number(m_peakStats.gpuUsage, 'f', 1) << "%";
        qDebug() << "";
        qDebug() << "--- GPU MEMORY (SYSTEM-WIDE) ---";
        qDebug() << "GPU Memory Used:" << QString::number(stats.gpuMemoryUsedGB, 'f', 2) << "GB /" 
                 << QString::number(stats.gpuMemoryTotalGB, 'f', 2) << "GB";
        qDebug() << "GPU Memory Usage:" << QString::number(stats.gpuMemoryUsage, 'f', 1) << "%";
        qDebug() << "GPU Temperature:" << QString::number(stats.gpuTemperature, 'f', 0) << "°C";
        qDebug() << "";
    } else {
        qDebug() << "--- GPU METRICS ---";
        qDebug() << "Status: UNAVAILABLE (NVML not initialized)";
        qDebug() << "";
    }
    
    qDebug() << "--- SYSTEM MEMORY (SYSTEM-WIDE) ---";
    qDebug() << "System RAM Used:" << QString::number(stats.systemMemoryUsedGB, 'f', 2) << "GB /" 
             << QString::number(stats.systemMemoryTotalGB, 'f', 2) << "GB";
    qDebug() << "System Memory Usage:" << QString::number(stats.systemMemoryUsage, 'f', 1) << "%";
    qDebug() << "";
    qDebug() << "--- PROCESS MEMORY (APPLICATION-SPECIFIC) ---";
    qDebug() << "Process Memory Used:" << QString::number(stats.processMemoryGB, 'f', 2) << "GB";
    qDebug() << "Peak Process Memory:" << QString::number(stats.peakProcessMemoryGB, 'f', 2) << "GB";
    qDebug() << "";
    qDebug() << "--- PERFORMANCE ---";
    qDebug() << "Current FPS:" << QString::number(stats.averageFPS, 'f', 1) << "FPS";
    qDebug() << "Average FPS:" << QString::number(m_averageStats.averageFPS, 'f', 1) << "FPS";
    qDebug() << "Peak FPS:" << QString::number(m_peakStats.averageFPS, 'f', 1) << "FPS";
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
    if (!m_nvmlInitialized || !m_nvmlDevice || !nvmlDeviceGetUtilizationRates_ptr) {
        return 0.0;
    }
    
    nvmlDevice_t device = *static_cast<nvmlDevice_t*>(m_nvmlDevice);
    nvmlUtilization_t utilization;
    
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates_ptr(device, &utilization);
    if (result == NVML_SUCCESS) {
        return static_cast<double>(utilization.gpu);
    }
    
    return 0.0;
}

double SystemMonitor::getGpuMemoryUsedGB()
{
    if (!m_nvmlInitialized || !m_nvmlDevice || !nvmlDeviceGetMemoryInfo_ptr) {
        return 0.0;
    }
    
    nvmlDevice_t device = *static_cast<nvmlDevice_t*>(m_nvmlDevice);
    nvmlMemory_t memory;
    
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo_ptr(device, &memory);
    if (result == NVML_SUCCESS) {
        // Convert bytes to GB
        return static_cast<double>(memory.used) / (1024.0 * 1024.0 * 1024.0);
    }
    
    return 0.0;
}

double SystemMonitor::getGpuMemoryTotalGB()
{
    if (!m_nvmlInitialized || !m_nvmlDevice || !nvmlDeviceGetMemoryInfo_ptr) {
        return 0.0;
    }
    
    nvmlDevice_t device = *static_cast<nvmlDevice_t*>(m_nvmlDevice);
    nvmlMemory_t memory;
    
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo_ptr(device, &memory);
    if (result == NVML_SUCCESS) {
        // Convert bytes to GB
        return static_cast<double>(memory.total) / (1024.0 * 1024.0 * 1024.0);
    }
    
    return 0.0;
}

double SystemMonitor::getGpuTemperature()
{
    if (!m_nvmlInitialized || !m_nvmlDevice || !nvmlDeviceGetTemperature_ptr) {
        return 0.0;
    }
    
    nvmlDevice_t device = *static_cast<nvmlDevice_t*>(m_nvmlDevice);
    unsigned int temperature;
    
    nvmlReturn_t result = nvmlDeviceGetTemperature_ptr(device, NVML_TEMPERATURE_GPU, &temperature);
    if (result == NVML_SUCCESS) {
        return static_cast<double>(temperature);
    }
    
    return 0.0;
}

double SystemMonitor::getSystemMemoryUsedGB()
{
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&memInfo)) {
        // Total physical memory - available physical memory = used memory
        DWORDLONG usedMemory = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
        // Convert bytes to GB
        return static_cast<double>(usedMemory) / (1024.0 * 1024.0 * 1024.0);
    }
    
    return 0.0;
}

double SystemMonitor::getSystemMemoryTotalGB()
{
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (GlobalMemoryStatusEx(&memInfo)) {
        // Convert bytes to GB
        return static_cast<double>(memInfo.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    }
    
    return 0.0;
}

double SystemMonitor::getProcessMemoryUsage()
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
    double currentMemory = getProcessMemoryUsage();
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
    
    double sessionAverageFPS = m_averageStats.averageFPS;
    double lastInstantFPS = m_lastStats.averageFPS;
    
    QString outputPath = filePath;
    if (outputPath.isEmpty()) {
        QString downloadsPath = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
        if (downloadsPath.isEmpty()) {
            downloadsPath = "C:/Downloads";
        }
        
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        QDir dir;
        if (!dir.exists(downloadsPath)) {
            dir.mkpath(downloadsPath);
        }
        outputPath = downloadsPath + "/system_stats_" + timestamp + ".txt";
    }
    
    QFile file(outputPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "SystemMonitor: Failed to open file for writing:" << outputPath;
        return;
    }
    
    QTextStream out(&file);
    out << "================================================================\n";
    out << "    COMPREHENSIVE SYSTEM MONITORING STATISTICS REPORT\n";
    out << "================================================================\n\n";
    out << "Generated: " << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << "\n";
    out << "Report Type: System-Wide Performance Metrics\n\n";
    
    out << "================================================================\n";
    out << "CURRENT STATISTICS (Last Sample)\n";
    out << "================================================================\n";
    out << "Timestamp: " << m_lastStats.timestamp.toString("yyyy-MM-dd hh:mm:ss") << "\n\n";
    
    out << "CPU (System-Wide):\n";
    out << "  - Usage: " << QString::number(m_lastStats.cpuUsage, 'f', 1) << "%\n\n";
    
    if (m_lastStats.gpuMetricsAvailable) {
        out << "GPU (System-Wide via NVML):\n";
        out << "  - Utilization: " << QString::number(m_lastStats.gpuUsage, 'f', 1) << "%\n";
        out << "  - Memory Used: " << QString::number(m_lastStats.gpuMemoryUsedGB, 'f', 2) << " GB / " 
            << QString::number(m_lastStats.gpuMemoryTotalGB, 'f', 2) << " GB\n";
        out << "  - Memory Usage: " << QString::number(m_lastStats.gpuMemoryUsage, 'f', 1) << "%\n";
        out << "  - Temperature: " << QString::number(m_lastStats.gpuTemperature, 'f', 0) << " °C\n\n";
    } else {
        out << "GPU Metrics: UNAVAILABLE (NVML not initialized)\n\n";
    }
    
    out << "System Memory (System-Wide):\n";
    out << "  - Used: " << QString::number(m_lastStats.systemMemoryUsedGB, 'f', 2) << " GB / " 
        << QString::number(m_lastStats.systemMemoryTotalGB, 'f', 2) << " GB\n";
    out << "  - Usage: " << QString::number(m_lastStats.systemMemoryUsage, 'f', 1) << "%\n\n";
    
    out << "Process Memory (Application-Specific):\n";
    out << "  - Current: " << QString::number(m_lastStats.processMemoryGB, 'f', 2) << " GB\n";
    out << "  - Peak: " << QString::number(m_lastStats.peakProcessMemoryGB, 'f', 2) << " GB\n\n";
    
    out << "Performance:\n";
    out << "  - Current FPS: " << QString::number(lastInstantFPS, 'f', 1) << " FPS\n\n";
    
    out << "================================================================\n";
    out << "AVERAGE STATISTICS (Session)\n";
    out << "================================================================\n";
    out << "CPU Usage: " << QString::number(m_averageStats.cpuUsage, 'f', 1) << "%\n";
    out << "GPU Usage: " << QString::number(m_averageStats.gpuUsage, 'f', 1) << "%\n";
    out << "System Memory Usage: " << QString::number(m_averageStats.systemMemoryUsage, 'f', 1) << "%\n";
    out << "Average FPS: " << QString::number(sessionAverageFPS, 'f', 1) << " FPS\n\n";
    
    out << "================================================================\n";
    out << "PEAK STATISTICS (Session)\n";
    out << "================================================================\n";
    out << "CPU Usage: " << QString::number(m_peakStats.cpuUsage, 'f', 1) << "%\n";
    out << "GPU Usage: " << QString::number(m_peakStats.gpuUsage, 'f', 1) << "%\n";
    out << "GPU Memory Usage: " << QString::number(m_peakStats.gpuMemoryUsage, 'f', 1) << "%\n";
    out << "GPU Temperature: " << QString::number(m_peakStats.gpuTemperature, 'f', 0) << " °C\n";
    out << "System Memory Usage: " << QString::number(m_peakStats.systemMemoryUsage, 'f', 1) << "%\n";
    out << "Process Memory: " << QString::number(m_peakStats.processMemoryGB, 'f', 2) << " GB\n";
    out << "FPS: " << QString::number(m_peakStats.averageFPS, 'f', 1) << " FPS\n";
    out << "================================================================\n";
    
    file.close();
    qDebug() << "SystemMonitor: Comprehensive statistics saved to:" << outputPath;
}

