#ifndef UI_MANAGER_H
#define UI_MANAGER_H

#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QCheckBox>
#include <QTimer>
#include <QStackedLayout>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QDebug>

class UiManager : public QObject
{
    Q_OBJECT

public:
    explicit UiManager(QObject *parent = nullptr);
    virtual ~UiManager();

    void setupUi(QWidget *parent, QLabel *videoLabel, QWidget *overlayWidget);
    void setupDebugDisplay(QWidget *parent);
    void setupStackedLayout(QWidget *parent, QLabel *videoLabel, QWidget *overlayWidget, QLabel *overlayImageLabel);
    void updateOverlayStyles(QPushButton *backButton, QPushButton *captureButton, QSlider *slider, QWidget *overlayWidget);
    void setupCountdownLabel(QWidget *parent, QLabel *&countdownLabel);
    void setupSlider(QSlider *slider);
    void setupButtons(QPushButton *backButton, QPushButton *captureButton);
    void updateDebugDisplay(int fps, bool personDetected, int detectionCount, double avgConfidence,
                            bool showBoundingBoxes, bool isProcessingFrame);
    void showBoundingBoxNotification(QWidget *parent, bool show);
    void showSegmentationNotification(QWidget *parent, bool show);
    void repositionDebugPanel(QWidget *parent, QWidget *debugPanel);

Q_SIGNALS:
    void boundingBoxToggled(bool show);
    void segmentationToggled(bool show);

public Q_SLOTS:
    void onBoundingBoxCheckBoxToggled(bool checked);
    void onSegmentationCheckBoxToggled(bool checked);
    void updateDebugInfo();

private:
    QWidget *debugWidget;
    QLabel *fpsLabel;
    QLabel *detectionLabel;
    QLabel *debugLabel;
    QCheckBox *boundingBoxCheckBox;
    QCheckBox *segmentationCheckBox;
    QTimer *debugUpdateTimer;
    QStackedLayout *stackedLayout;

    void createDebugWidget(QWidget *parent);
    void createDebugLabels();
    void createDebugCheckboxes();
    void createKeyboardShortcutsInfo();
};

#endif // UI_MANAGER_H
