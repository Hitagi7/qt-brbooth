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
    
    // Opacity effects for fade in/out
    QGraphicsOpacityEffect *m_step1Effect = nullptr;
    QGraphicsOpacityEffect *m_step2Effect = nullptr;
    QGraphicsOpacityEffect *m_step3Effect = nullptr;
    QGraphicsOpacityEffect *m_step4Effect = nullptr;
    QGraphicsOpacityEffect *m_step5Effect = nullptr;
    
    // Animations
    QPropertyAnimation *m_currentAnimation = nullptr;

    void setupWalkthroughLabels();
    void showStep(int step);
    void hideAllSteps();
    void setStepOpacity(int stepIndex, double opacity);
    void updateFontSizes();
    QString getStepStyle(int baseFontSize);
    QString getHighlightStepStyle(int baseFontSize);
};

#endif // IDLE_H
