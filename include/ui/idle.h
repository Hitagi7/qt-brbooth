#ifndef IDLE_H
#define IDLE_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include <QEasingCurve>
#include <QShowEvent>
#include <QHideEvent>
#include <QResizeEvent>

QT_BEGIN_NAMESPACE
namespace Ui { class Idle; }
QT_END_NAMESPACE

class Idle : public QWidget
{
    Q_OBJECT

public:
    explicit Idle(QWidget *parent = nullptr);
    ~Idle();

    // Start/stop the walkthrough animation
    void startWalkthrough();
    void stopWalkthrough();

protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateWalkthroughStep();

private:
    Ui::Idle *ui;

    // Walkthrough animation
    QTimer *m_walkthroughTimer = nullptr;
    int m_currentStep = 0;
    
    // Labels for each step
    QLabel *m_step1Label = nullptr;
    QLabel *m_step2Label = nullptr;
    QLabel *m_step3Label = nullptr;
    QLabel *m_step4Label = nullptr;
    QLabel *m_step5Label = nullptr;
    QLabel *m_subtitleLabel = nullptr;
    QLabel *m_bottomLabel = nullptr;
    QLabel *m_titleLabel = nullptr;
    
    // Arrow labels (connect step i to step i+1)
    QLabel *m_arrow1Label = nullptr;  // Connects step 1 to step 2
    QLabel *m_arrow2Label = nullptr;  // Connects step 2 to step 3
    QLabel *m_arrow3Label = nullptr;  // Connects step 3 to step 4
    QLabel *m_arrow4Label = nullptr;  // Connects step 4 to step 5
    
    // Opacity effects for fade in/out
    QGraphicsOpacityEffect *m_step1Effect = nullptr;
    QGraphicsOpacityEffect *m_step2Effect = nullptr;
    QGraphicsOpacityEffect *m_step3Effect = nullptr;
    QGraphicsOpacityEffect *m_step4Effect = nullptr;
    QGraphicsOpacityEffect *m_step5Effect = nullptr;
    
    // Opacity effects for arrows
    QGraphicsOpacityEffect *m_arrow1Effect = nullptr;
    QGraphicsOpacityEffect *m_arrow2Effect = nullptr;
    QGraphicsOpacityEffect *m_arrow3Effect = nullptr;
    QGraphicsOpacityEffect *m_arrow4Effect = nullptr;
    
    // Animations
    QPropertyAnimation *m_currentAnimation = nullptr;

    void setupWalkthroughLabels();
    void showStep(int step);
    void showStepInternal(int step, const QString &normalStyle, const QString &highlightStyle);
    void hideAllSteps();
    void setStepOpacity(int stepIndex, double opacity);
    void setArrowOpacity(int arrowIndex, double opacity);
    void updateFontSizes();
    QString getStepStyle(int baseFontSize);
    QString getHighlightStepStyle(int baseFontSize);
};

#endif // IDLE_H
