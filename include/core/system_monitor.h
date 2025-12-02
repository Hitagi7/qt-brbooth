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
        double cpuUsage;          // CPU usage (%)
        double gpuUsage;          // GPU usage (%)
        double peakMemoryGB;      // Peak memory usage (GB)
        double averageFPS;        // Average FPS (frames per second)
        QDateTime timestamp;      // When these stats were collected
    };

    Statistics getCurrentStatistics() const;
    Statistics getLastStatistics() const;

    // FPS tracking
    void updateFPS(double fps);
    void resetFPSTracking();

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
    double getMemoryUsage();
    void updatePeakMemory();

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
    
    // Initialization flag
    bool m_initialized;
    bool m_monitoring;
};

#endif // SYSTEM_MONITOR_H

