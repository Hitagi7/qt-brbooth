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
    // Get layouts from UI file
    QVBoxLayout *outerLayout = qobject_cast<QVBoxLayout*>(this->layout());
    if (!outerLayout) {
        qWarning() << "Could not find outerLayout in UI file";
        return;
    }
    
    outerLayout->setAlignment(Qt::AlignCenter);
    
    // Find the steps layout from UI
    QHBoxLayout *stepsLayout = nullptr;
    for (int i = 0; i < outerLayout->count(); i++) {
        QLayoutItem *item = outerLayout->itemAt(i);
        if (item && item->layout()) {
            QHBoxLayout *hLayout = qobject_cast<QHBoxLayout*>(item->layout());
            if (hLayout) {
                stepsLayout = hLayout;
                break;
            }
        }
    }
    
    if (!stepsLayout) {
        qWarning() << "Could not find stepsLayout in UI file";
        return;
    }
    
    stepsLayout->setAlignment(Qt::AlignCenter);
    
    // Get title label from UI file
    m_titleLabel = ui->titleLabel;
    
    // Step 1
    m_step1Label = new QLabel("<div style='text-align: center;'>"
                              "<div style='font-size: 64px; margin-bottom: 10px;'>🎭</div>"
                              "<div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>Step 1</div>"
                              "<div style='font-size: 20px; line-height: 1.4;'>Select STATIC<br>or DYNAMIC</div>"
                              "</div>", this);
    m_step1Label->setTextFormat(Qt::RichText);
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
    arrow1->setProperty("class", "arrowLabel");
    arrow1->setStyleSheet("QLabel {"
                         "  font-size: 96px;"
                         "  color: #0bc200;"
                         "  background-color: transparent;"
                         "  font-weight: bold;"
                         "}");
    stepsLayout->addWidget(arrow1, 0);
    
    // Step 2
    m_step2Label = new QLabel("<div style='text-align: center;'>"
                              "<div style='font-size: 64px; margin-bottom: 10px;'>🖼️</div>"
                              "<div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>Step 2</div>"
                              "<div style='font-size: 20px; line-height: 1.4;'>Choose your<br>favorite template</div>"
                              "</div>", this);
    m_step2Label->setTextFormat(Qt::RichText);
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
    arrow2->setProperty("class", "arrowLabel");
    arrow2->setStyleSheet("QLabel {"
                         "  font-size: 96px;"
                         "  color: #0bc200;"
                         "  background-color: transparent;"
                         "  font-weight: bold;"
                         "}");
    stepsLayout->addWidget(arrow2, 0);
    
    // Step 3
    m_step3Label = new QLabel("<div style='text-align: center;'>"
                              "<div style='font-size: 64px; margin-bottom: 10px;'>📸</div>"
                              "<div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>Step 3</div>"
                              "<div style='font-size: 20px; line-height: 1.4;'>Get ready<br>for the camera</div>"
                              "</div>", this);
    m_step3Label->setTextFormat(Qt::RichText);
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
    arrow3->setProperty("class", "arrowLabel");
    arrow3->setStyleSheet("QLabel {"
                         "  font-size: 96px;"
                         "  color: #0bc200;"
                         "  background-color: transparent;"
                         "  font-weight: bold;"
                         "}");
    stepsLayout->addWidget(arrow3, 0);
    
    // Step 4
    m_step4Label = new QLabel("<div style='text-align: center;'>"
                              "<div style='font-size: 64px; margin-bottom: 10px;'>✨</div>"
                              "<div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>Step 4</div>"
                              "<div style='font-size: 20px; line-height: 1.4;'>Strike a pose<br>and capture!</div>"
                              "</div>", this);
    m_step4Label->setTextFormat(Qt::RichText);
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
    arrow4->setProperty("class", "arrowLabel");
    arrow4->setStyleSheet("QLabel {"
                         "  font-size: 96px;"
                         "  color: #0bc200;"
                         "  background-color: transparent;"
                         "  font-weight: bold;"
                         "}");
    stepsLayout->addWidget(arrow4, 0);
    
    // Step 5
    m_step5Label = new QLabel("<div style='text-align: center;'>"
                              "<div style='font-size: 64px; margin-bottom: 10px;'>💾</div>"
                              "<div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>Step 5</div>"
                              "<div style='font-size: 20px; line-height: 1.4;'>Save or retake<br>your photo!</div>"
                              "</div>", this);
    m_step5Label->setTextFormat(Qt::RichText);
    m_step5Label->setAlignment(Qt::AlignCenter);
    m_step5Label->setWordWrap(true);
    m_step5Label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_step5Effect = new QGraphicsOpacityEffect(m_step5Label);
    m_step5Label->setGraphicsEffect(m_step5Effect);
    stepsLayout->addWidget(m_step5Label, 1);
    
    // Background style is already set in the .ui file
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
    // Move to next step before showing it
    m_currentStep = (m_currentStep + 1) % 5; // Cycle through 5 steps
    showStep(m_currentStep);
}

void Idle::startWalkthrough()
{
    qDebug() << "Starting idle mode walkthrough animation";
    m_currentStep = 0;
    showStep(0); // Show first step immediately
    // Start timer - when it fires, it will move to step 1
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
    int titleFontSize = qMax(32, qMin(72, screenHeight / 15));
    int stepFontSize = qMax(18, qMin(42, screenHeight / 22));
    Q_UNUSED(screenWidth); // Suppress unused variable warning
    
    // Update title font - keep same style as "Select Output Type" but scale font size
    if (m_titleLabel) {
        int scaledFontSize = qMax(40, qMin(70, screenHeight / 18));
        int scaledWidth = qMax(500, qMin(800, screenWidth / 2));
        int scaledHeight = qMax(100, qMin(150, screenHeight / 8));
        
        m_titleLabel->setMinimumSize(scaledWidth, scaledHeight);
        m_titleLabel->setMaximumSize(scaledWidth, scaledHeight);
        m_titleLabel->setStyleSheet(QString("#titleLabel {"
                                            "  color: #FFF;"
                                            "  text-align: center;"
                                            "  font-family: 'Roboto Condensed';"
                                            "  font-style: italic;"
                                            "  font-weight: 700;"
                                            "  font-size: %1px;"
                                            "  background-color: #0BC200;"
                                            "  border-radius: 9px;"
                                            "  border-bottom: 2px solid #020202;"
                                            "}").arg(scaledFontSize));
    }
    
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
    int borderRadius = qMax(20, fontSize / 2 + 8);
    int padding = qMax(25, fontSize / 2 + 10);
    
    return QString("QLabel {"
                  "  font-family: 'Roboto Condensed', 'Arial', sans-serif;"
                  "  color: white;"
                  "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                  "    stop:0 rgba(11, 194, 0, 220),"
                  "    stop:0.5 rgba(11, 194, 0, 200),"
                  "    stop:1 rgba(8, 150, 0, 220));"
                  "  border: 3px solid rgba(255, 255, 255, 120);"
                  "  border-radius: %1px;"
                  "  padding: %2px;"
                  "  min-height: 220px;"
                  "}").arg(borderRadius).arg(padding);
}

QString Idle::getHighlightStepStyle(int fontSize)
{
    int borderRadius = qMax(25, fontSize / 2 + 12);
    int padding = qMax(30, fontSize / 2 + 12);
    
    return QString("QLabel {"
                  "  font-family: 'Roboto Condensed', 'Arial', sans-serif;"
                  "  color: white;"
                  "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                  "    stop:0 rgba(11, 194, 0, 255),"
                  "    stop:0.5 rgba(15, 220, 0, 255),"
                  "    stop:1 rgba(11, 194, 0, 255));"
                  "  border: 5px solid rgba(255, 255, 255, 250);"
                  "  border-radius: %1px;"
                  "  padding: %2px;"
                  "  min-height: 240px;"
                  "}").arg(borderRadius).arg(padding);
}

