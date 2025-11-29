#include "ui/idle.h"
#include "ui_idle.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QResizeEvent>
#include <QDebug>

Idle::Idle(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Idle)
    , m_currentStep(0)
{
    ui->setupUi(this);
    
    // Set up walkthrough timer (cycle through steps every 3 seconds)
    m_walkthroughTimer = new QTimer(this);
    connect(m_walkthroughTimer, &QTimer::timeout, this, &Idle::updateWalkthroughStep);
    
    // Set up walkthrough labels
    setupWalkthroughLabels();
    
    qDebug() << "Idle mode initialized";
}

Idle::~Idle()
{
    delete ui;
}

void Idle::setupWalkthroughLabels()
{
    // Main vertical container to center everything
    QVBoxLayout *outerLayout = new QVBoxLayout(this);
    outerLayout->setAlignment(Qt::AlignCenter);
    outerLayout->setContentsMargins(20, 20, 20, 20);
    outerLayout->setSpacing(0);
    
    outerLayout->addStretch(2);
    
    // Horizontal layout for the steps
    QHBoxLayout *stepsLayout = new QHBoxLayout();
    stepsLayout->setAlignment(Qt::AlignCenter);
    stepsLayout->setSpacing(5);
    
    // Step 1
    m_step1Label = new QLabel("🎭\n\nStep 1\n\nSelect STATIC\nor DYNAMIC", this);
    m_step1Label->setAlignment(Qt::AlignCenter);
    m_step1Label->setWordWrap(true);
    m_step1Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step1Effect = new QGraphicsOpacityEffect(m_step1Label);
    m_step1Label->setGraphicsEffect(m_step1Effect);
    stepsLayout->addWidget(m_step1Label, 1);
    
    // Arrow 1
    QLabel *arrow1 = new QLabel("➡️", this);
    arrow1->setAlignment(Qt::AlignCenter);
    arrow1->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    arrow1->setStyleSheet("QLabel { font-size: 96px; color: #0bc200; background-color: transparent; }");
    stepsLayout->addWidget(arrow1, 0);
    
    // Step 2
    m_step2Label = new QLabel("🖼️\n\nStep 2\n\nChoose your\nfavorite template", this);
    m_step2Label->setAlignment(Qt::AlignCenter);
    m_step2Label->setWordWrap(true);
    m_step2Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step2Effect = new QGraphicsOpacityEffect(m_step2Label);
    m_step2Label->setGraphicsEffect(m_step2Effect);
    stepsLayout->addWidget(m_step2Label, 1);
    
    // Arrow 2
    QLabel *arrow2 = new QLabel("➡️", this);
    arrow2->setAlignment(Qt::AlignCenter);
    arrow2->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    arrow2->setStyleSheet("QLabel { font-size: 96px; color: #0bc200; background-color: transparent; }");
    stepsLayout->addWidget(arrow2, 0);
    
    // Step 3
    m_step3Label = new QLabel("📸\n\nStep 3\n\nGet ready\nfor the camera", this);
    m_step3Label->setAlignment(Qt::AlignCenter);
    m_step3Label->setWordWrap(true);
    m_step3Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step3Effect = new QGraphicsOpacityEffect(m_step3Label);
    m_step3Label->setGraphicsEffect(m_step3Effect);
    stepsLayout->addWidget(m_step3Label, 1);
    
    // Arrow 3
    QLabel *arrow3 = new QLabel("➡️", this);
    arrow3->setAlignment(Qt::AlignCenter);
    arrow3->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    arrow3->setStyleSheet("QLabel { font-size: 96px; color: #0bc200; background-color: transparent; }");
    stepsLayout->addWidget(arrow3, 0);
    
    // Step 4
    m_step4Label = new QLabel("✨\n\nStep 4\n\nStrike a pose\nand capture!", this);
    m_step4Label->setAlignment(Qt::AlignCenter);
    m_step4Label->setWordWrap(true);
    m_step4Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step4Effect = new QGraphicsOpacityEffect(m_step4Label);
    m_step4Label->setGraphicsEffect(m_step4Effect);
    stepsLayout->addWidget(m_step4Label, 1);
    
    // Arrow 4
    QLabel *arrow4 = new QLabel("➡️", this);
    arrow4->setAlignment(Qt::AlignCenter);
    arrow4->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    arrow4->setStyleSheet("QLabel { font-size: 96px; color: #0bc200; background-color: transparent; }");
    stepsLayout->addWidget(arrow4, 0);
    
    // Step 5
    m_step5Label = new QLabel("💾\n\nStep 5\n\nSave or retake\nyour photo!", this);
    m_step5Label->setAlignment(Qt::AlignCenter);
    m_step5Label->setWordWrap(true);
    m_step5Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step5Effect = new QGraphicsOpacityEffect(m_step5Label);
    m_step5Label->setGraphicsEffect(m_step5Effect);
    stepsLayout->addWidget(m_step5Label, 1);
    
    outerLayout->addLayout(stepsLayout, 3);
    
    outerLayout->addStretch(2);
    
    // Set background style with animated gradient
    this->setStyleSheet("QWidget#Idle {"
                       "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
                       "    stop:0 #0f0c29, stop:0.4 #302b63, stop:0.8 #24243e, stop:1 #0f0c29);"
                       "}");
    
    // Initially hide all steps
    hideAllSteps();
}

void Idle::hideAllSteps()
{
    // Completely hide all steps (invisible) - used for initial state
    if (m_step1Effect) m_step1Effect->setOpacity(0.0);
    if (m_step2Effect) m_step2Effect->setOpacity(0.0);
    if (m_step3Effect) m_step3Effect->setOpacity(0.0);
    if (m_step4Effect) m_step4Effect->setOpacity(0.0);
    if (m_step5Effect) m_step5Effect->setOpacity(0.0);
}

void Idle::setStepOpacity(int stepIndex, double opacity)
{
    QGraphicsOpacityEffect *effect = nullptr;
    switch (stepIndex) {
        case 0: effect = m_step1Effect; break;
        case 1: effect = m_step2Effect; break;
        case 2: effect = m_step3Effect; break;
        case 3: effect = m_step4Effect; break;
        case 4: effect = m_step5Effect; break;
    }
    if (effect) {
        effect->setOpacity(opacity);
    }
}

void Idle::showStep(int step)
{
    QGraphicsOpacityEffect *effect = nullptr;
    QLabel *label = nullptr;
    
    switch (step) {
        case 0:
            effect = m_step1Effect;
            label = m_step1Label;
            break;
        case 1:
            effect = m_step2Effect;
            label = m_step2Label;
            break;
        case 2:
            effect = m_step3Effect;
            label = m_step3Label;
            break;
        case 3:
            effect = m_step4Effect;
            label = m_step4Label;
            break;
        case 4:
            effect = m_step5Effect;
            label = m_step5Label;
            break;
    }
    
    // Calculate responsive font sizes
    int stepFontSize = qMax(18, qMin(42, this->height() / 22));
    
    // Get responsive styles
    QString normalStyle = getStepStyle(stepFontSize);
    QString highlightStyle = getHighlightStepStyle(stepFontSize + 4);
    
    // Update all steps based on their state relative to current step
    for (int i = 0; i < 5; i++) {
        QLabel *stepLabel = nullptr;
        switch (i) {
            case 0: stepLabel = m_step1Label; break;
            case 1: stepLabel = m_step2Label; break;
            case 2: stepLabel = m_step3Label; break;
            case 3: stepLabel = m_step4Label; break;
            case 4: stepLabel = m_step5Label; break;
        }
        
        if (i < step) {
            // Previous steps: visible with lower opacity (completed)
            setStepOpacity(i, 0.5);
            if (stepLabel) stepLabel->setStyleSheet(normalStyle);
        } else if (i == step) {
            // Current step: fully visible and highlighted
            setStepOpacity(i, 1.0);
            if (stepLabel) stepLabel->setStyleSheet(highlightStyle);
        } else {
            // Future steps: invisible
            setStepOpacity(i, 0.0);
            if (stepLabel) stepLabel->setStyleSheet(normalStyle);
        }
    }
    
    // Animate the current step's appearance
    if (effect) {
        if (m_currentAnimation) {
            m_currentAnimation->stop();
            delete m_currentAnimation;
        }
        
        m_currentAnimation = new QPropertyAnimation(effect, "opacity");
        m_currentAnimation->setDuration(800);
        m_currentAnimation->setStartValue(0.0);
        m_currentAnimation->setEndValue(1.0);
        m_currentAnimation->setEasingCurve(QEasingCurve::OutCubic);
        m_currentAnimation->start();
    }
}

void Idle::updateWalkthroughStep()
{
    showStep(m_currentStep);
    m_currentStep = (m_currentStep + 1) % 5; // Cycle through 5 steps
}

void Idle::startWalkthrough()
{
    qDebug() << "Starting idle mode walkthrough animation";
    m_currentStep = 0;
    showStep(0);
    m_walkthroughTimer->start(3000); // Change step every 3 seconds
}

void Idle::stopWalkthrough()
{
    qDebug() << "Stopping idle mode walkthrough animation";
    m_walkthroughTimer->stop();
    hideAllSteps();
}

void Idle::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    updateFontSizes(); // Ensure proper sizing when shown
    startWalkthrough();
}

void Idle::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    stopWalkthrough();
}

void Idle::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    updateFontSizes();
}

void Idle::updateFontSizes()
{
    // Calculate responsive font sizes based on widget dimensions
    int screenWidth = this->width();
    int screenHeight = this->height();
    
    // Base font sizes scale with screen size
    int titleFontSize = qMax(24, qMin(56, screenHeight / 18));
    int stepFontSize = qMax(18, qMin(42, screenHeight / 22));
    Q_UNUSED(screenWidth); // Suppress unused variable warning
    
    // Update step label styles
    QString normalStyle = getStepStyle(stepFontSize);
    QString highlightStyle = getHighlightStepStyle(stepFontSize + 4);
    
    // Apply normal style to all, then re-highlight the current step
    if (m_step1Label) m_step1Label->setStyleSheet(normalStyle);
    if (m_step2Label) m_step2Label->setStyleSheet(normalStyle);
    if (m_step3Label) m_step3Label->setStyleSheet(normalStyle);
    if (m_step4Label) m_step4Label->setStyleSheet(normalStyle);
    if (m_step5Label) m_step5Label->setStyleSheet(normalStyle);
    
    // Re-highlight current step
    QLabel *currentLabel = nullptr;
    switch (m_currentStep) {
        case 0: currentLabel = m_step1Label; break;
        case 1: currentLabel = m_step2Label; break;
        case 2: currentLabel = m_step3Label; break;
        case 3: currentLabel = m_step4Label; break;
        case 4: currentLabel = m_step5Label; break;
    }
    if (currentLabel) {
        currentLabel->setStyleSheet(highlightStyle);
    }
    
    qDebug() << "Idle UI resized to:" << screenWidth << "x" << screenHeight 
             << "- Font sizes: title=" << titleFontSize 
             << ", step=" << stepFontSize;
}

QString Idle::getStepStyle(int fontSize)
{
    int borderRadius = qMax(10, fontSize / 2);
    int padding = qMax(15, fontSize / 2);
    
    return QString("QLabel {"
                  "  font-family: 'Roboto Condensed', sans-serif;"
                  "  font-size: %1px;"
                  "  font-weight: bold;"
                  "  color: white;"
                  "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                  "    stop:0 rgba(11, 194, 0, 200),"
                  "    stop:1 rgba(8, 150, 0, 200));"
                  "  border: 3px solid rgba(255, 255, 255, 100);"
                  "  border-radius: %2px;"
                  "  padding: %3px;"
                  "}").arg(fontSize).arg(borderRadius).arg(padding);
}

QString Idle::getHighlightStepStyle(int fontSize)
{
    int borderRadius = qMax(12, fontSize / 2);
    int padding = qMax(18, fontSize / 2);
    
    return QString("QLabel {"
                  "  font-family: 'Roboto Condensed', sans-serif;"
                  "  font-size: %1px;"
                  "  font-weight: bold;"
                  "  color: white;"
                  "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                  "    stop:0 rgba(11, 194, 0, 255),"
                  "    stop:1 rgba(9, 170, 0, 255));"
                  "  border: 5px solid rgba(255, 255, 255, 230);"
                  "  border-radius: %2px;"
                  "  padding: %3px;"
                  "}").arg(fontSize).arg(borderRadius).arg(padding);
}

