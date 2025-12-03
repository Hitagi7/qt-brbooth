#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QMutex>
#include <QString>
#include <QDateTime>
#include <memory>
#include <cstring>

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
        double cpuUsage;          // CPU usage (%)
        double gpuUsage;          // GPU usage (%)
        double peakMemoryGB;      // Peak memory usage (GB) - process memory
        double systemMemoryUsageGB; // System-wide memory usage (GB)
        double systemMemoryTotalGB; // Total system memory (GB)
        double systemMemoryAvailableGB; // Available system memory (GB)
        double averageFPS;        // Average FPS
        double accuracy;          // Accuracy metric (%)
        QDateTime timestamp;      // When these stats were collected
    };

    Statistics getCurrentStatistics() const;
    Statistics getLastStatistics() const;
    Statistics getAverageStatistics() const;
    Statistics getPeakStatistics() const;

    // FPS tracking (thread-safe)
    Q_INVOKABLE void updateFPS(double fps);  // Q_INVOKABLE allows Qt::QueuedConnection
    void resetFPSTracking();

    // Accuracy tracking
    void updateAccuracy(double detectionConfidence);
    void resetAccuracyTracking();

    // Crash handling
    void saveStatisticsToDocx(const QString& filePath = QString()) const;
    void saveStatisticsToText(const QString& filePath = QString()) const;

signals:
    void statisticsUpdated(const Statistics& stats);

private slots:
    void collectStatistics();

private:
    // Windows-specific monitoring functions
    double getCpuUsage();
    double getGpuUsage();
    double getMemoryUsage(); // Process memory
    void getSystemMemoryInfo(double& totalGB, double& usedGB, double& availableGB);
    void updatePeakMemory();
    
    // GPU monitoring methods (tries multiple approaches)
    double getGpuUsageNVML();      // NVIDIA NVML (NVIDIA GPUs only)
    double getGpuUsageDXGI();      // Placeholder (not used - PDH preferred)
    double getGpuUsagePDH();       // Windows Performance Counters (AMD/NVIDIA/Intel - best for AMD)
    
    // CPU monitoring methods
    double getCpuUsagePDH();       // Performance Data Helper
    double getCpuUsageSystemTimes(); // GetSystemTimes (more accurate)

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
    
    // CPU monitoring using GetSystemTimes
    qint64 m_lastCpuIdleTime;
    qint64 m_lastCpuKernelTime;
    qint64 m_lastCpuUserTime;
    bool m_cpuTimesInitialized;
    
    // GPU monitoring method tracking
    enum GpuMonitoringMethod {
        GPU_METHOD_NONE = 0,
        GPU_METHOD_NVML = 1,
        GPU_METHOD_PDH = 2
    };
    GpuMonitoringMethod m_activeGpuMethod;
    
    // FPS tracking (volatile for thread-safe access)
    volatile double m_latestFPS;
    QElapsedTimer m_fpsTrackingTimer;
    
    // Average tracking variables
    double m_cpuSum;
    double m_gpuSum;
    double m_memorySum;
    double m_systemMemorySum;
    double m_fpsSum;
    int m_sampleCount;
    
    // Accuracy tracking
    QList<double> m_accuracySamples;
    static const int MAX_ACCURACY_SAMPLES = 100;
    
    // Memory tracking
    double m_peakMemoryGB;
    
    // Initialization flag
    bool m_initialized;
    bool m_monitoring;
};

#endif // SYSTEM_MONITOR_H

