#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QMutex>
#include <QString>
#include <QDateTime>
#include <QAtomicInteger>
#include <memory>

// Forward declarations
struct SystemStats;

class SystemMonitor : public QObject
{
    Q_OBJECT

public:
    explicit SystemMonitor(QObject *parent = nullptr);
    ~SystemMonitor();

    // Initialization
    bool initialize();
    void startMonitoring(int intervalMs = 5000); // Default 5 seconds
    void stopMonitoring();

    // Statistics retrieval
    struct Statistics {
        // CPU Metrics (System-wide)
        double cpuUsage;          // CPU usage (%)
        
        // GPU Metrics (System-wide)
        double gpuUsage;          // GPU usage (%)
        double gpuMemoryUsedGB;   // GPU memory used (GB)
        double gpuMemoryTotalGB;  // GPU memory total (GB)
        double gpuMemoryUsage;    // GPU memory usage (%)
        double gpuTemperature;    // GPU temperature (Celsius)
        
        // System Memory Metrics (System-wide)
        double systemMemoryUsedGB;   // System RAM used (GB)
        double systemMemoryTotalGB;  // System RAM total (GB)
        double systemMemoryUsage;    // System memory usage (%)
        
        // Process Memory Metrics (Application-specific)
        double processMemoryGB;      // Process memory usage (GB)
        double peakProcessMemoryGB;  // Peak process memory usage (GB)
        
        // Performance Metrics
        double averageFPS;        // Average FPS (frames per second)
        
        // Metadata
        QDateTime timestamp;      // When these stats were collected
        bool gpuMetricsAvailable; // Whether GPU metrics are available
    };

    Statistics getCurrentStatistics() const;
    Statistics getLastStatistics() const;

    // FPS tracking
    void updateFPS(double fps);
    void resetFPSTracking();

    // Crash handling
    void saveStatisticsToText(const QString& filePath = QString()) const;

signals:
    void statisticsUpdated(const Statistics& stats);

private slots:
    void collectStatistics();

private:
    // Windows-specific monitoring functions
    double getCpuUsage();
    double getGpuUsage();
    double getGpuMemoryUsedGB();
    double getGpuMemoryTotalGB();
    double getGpuTemperature();
    double getSystemMemoryUsedGB();
    double getSystemMemoryTotalGB();
    double getProcessMemoryUsage();
    void updatePeakMemory();
    
    // NVML initialization
    bool initializeNVML();
    void shutdownNVML();

    // Internal state
    QTimer* m_timer;
    mutable QMutex m_mutex;  // Mutable to allow locking in const methods
    Statistics m_lastStats;
    Statistics m_peakStats;
    Statistics m_averageStats;
    
    // Windows-specific handles
    void* m_cpuQueryHandle;
    void* m_cpuCounterHandle;
    void* m_gpuQueryHandle;
    void* m_gpuCounterHandle;
    
    // FPS tracking - using simple volatile double (thread-safe reads/writes on x86/x64)
    // Atomic operations were causing crashes, using volatile as safer alternative
    volatile double m_latestFPS;
    QElapsedTimer m_fpsTrackingTimer;
    
    // Average statistics tracking
    double m_cpuSum;
    double m_gpuSum;
    double m_memorySum;
    double m_fpsSum;
    int m_sampleCount;
    
    // Memory tracking
    double m_peakMemoryGB;
    
    // NVML state
    bool m_nvmlInitialized;
    void* m_nvmlDevice;  // nvmlDevice_t stored as void* to avoid header dependency
    
    // Initialization flag
    bool m_initialized;
    bool m_monitoring;
};

#endif // SYSTEM_MONITOR_H

