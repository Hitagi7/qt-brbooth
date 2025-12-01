#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QMutex>
#include <QString>
#include <QDateTime>
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
        double accuracy;          // Accuracy metric (%)
        QDateTime timestamp;      // When these stats were collected
    };

    Statistics getCurrentStatistics() const;
    Statistics getLastStatistics() const;

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
    double getMemoryUsage();
    void updatePeakMemory();

    // Internal state
    QTimer* m_timer;
    mutable QMutex m_mutex;  // Mutable to allow locking in const methods
    Statistics m_lastStats;
    Statistics m_peakStats;
    
    // Windows-specific handles
    void* m_cpuQueryHandle;
    void* m_cpuCounterHandle;
    void* m_gpuQueryHandle;
    void* m_gpuCounterHandle;
    
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

