#include "ui/ui_manager.h"

#include <QDebug>
#include <QShortcut>
#include <QTimer>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QWidget>
#include <QCoreApplication>

UiManager::UiManager(QObject *parent)
    : QObject(parent)
    , debugWidget(nullptr)
    , fpsLabel(nullptr)
    , detectionLabel(nullptr)
    , debugLabel(nullptr)
    , boundingBoxCheckBox(nullptr)

    , debugUpdateTimer(nullptr)
    , stackedLayout(nullptr)
{
    qDebug() << "UiManager constructor called";
}

UiManager::~UiManager() {}

void UiManager::setupUi(QWidget *parent, QLabel *videoLabel, QWidget *overlayWidget)
{
    Q_UNUSED(parent);
    Q_UNUSED(videoLabel);
    Q_UNUSED(overlayWidget);
}

void UiManager::setupDebugDisplay(QWidget *parent)
{
    createDebugWidget(parent);
    createDebugLabels();
    createDebugCheckboxes();

    if (!debugUpdateTimer) {
        debugUpdateTimer = new QTimer(this);
        connect(debugUpdateTimer, &QTimer::timeout, this, &UiManager::updateDebugInfo);
        debugUpdateTimer->start(1000);
    }
}

void UiManager::setupStackedLayout(QWidget *parent, QLabel *videoLabel, QWidget *overlayWidget, QLabel *overlayImageLabel)
{
    Q_UNUSED(parent);
    Q_UNUSED(videoLabel);
    Q_UNUSED(overlayWidget);
    Q_UNUSED(overlayImageLabel);
}

void UiManager::updateOverlayStyles(QPushButton *backButton, QPushButton *captureButton, QSlider *slider, QWidget *overlayWidget)
{
    Q_UNUSED(backButton);
    Q_UNUSED(captureButton);
    Q_UNUSED(slider);
    Q_UNUSED(overlayWidget);
}

void UiManager::setupCountdownLabel(QWidget *parent, QLabel *&countdownLabel)
{
    Q_UNUSED(parent);
    Q_UNUSED(countdownLabel);
}

void UiManager::setupSlider(QSlider *slider)
{
    Q_UNUSED(slider);
}

void UiManager::setupButtons(QPushButton *backButton, QPushButton *captureButton)
{
    Q_UNUSED(backButton);
    Q_UNUSED(captureButton);
}

void UiManager::updateDebugDisplay(int fps, bool personDetected, int detectionCount, double avgConfidence,
                                   bool showBoundingBoxes, bool isProcessingFrame)
{
    Q_UNUSED(fps);
    Q_UNUSED(personDetected);
    Q_UNUSED(detectionCount);
    Q_UNUSED(avgConfidence);
    Q_UNUSED(showBoundingBoxes);
    Q_UNUSED(isProcessingFrame);
}

void UiManager::showBoundingBoxNotification(QWidget *parent, bool show)
{
    Q_UNUSED(parent);
    Q_UNUSED(show);
}



void UiManager::repositionDebugPanel(QWidget *parent, QWidget *debugPanel)
{
    Q_UNUSED(parent);
    Q_UNUSED(debugPanel);
}

void UiManager::onBoundingBoxCheckBoxToggled(bool checked)
{
    Q_UNUSED(checked);
}

void UiManager::onSegmentationCheckBoxToggled(bool checked)
{
    Q_UNUSED(checked);
}



void UiManager::updateDebugInfo()
{
    // No-op stub; real implementation can update labels
}

void UiManager::createDebugWidget(QWidget *parent)
{
    if (!debugWidget) {
        debugWidget = new QWidget(parent);
    }
}

void UiManager::createDebugLabels()
{
    if (!fpsLabel) fpsLabel = new QLabel(debugWidget);
    if (!detectionLabel) detectionLabel = new QLabel(debugWidget);
    if (!debugLabel) debugLabel = new QLabel(debugWidget);
}

void UiManager::createDebugCheckboxes()
{
    if (!boundingBoxCheckBox) {
        boundingBoxCheckBox = new QCheckBox(debugWidget);
        connect(boundingBoxCheckBox, &QCheckBox::toggled, this, &UiManager::onBoundingBoxCheckBoxToggled);
    }

}

void UiManager::createKeyboardShortcutsInfo()
{
    // No-op stub; real implementation can create keyboard shortcuts info
}


