#include "foreground.h"
#include "ui_foreground.h"
#include <QStyle>
#include <QRegularExpression>
#include <QMouseEvent>

Foreground::Foreground(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Foreground)
{
    ui->setupUi(this);
    connect(ui->back, &QPushButton::clicked, this, &Foreground::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Foreground::resetDebounce);

    debounceActive = false;

    QList<QPushButton*> imageButtons = findChildren<QPushButton*>(QRegularExpression("image[1-6]"));
    for (QPushButton* button : imageButtons) {
        if (button) {
            button->installEventFilter(this);
            button->setFocusPolicy(Qt::NoFocus);
            button->setProperty("selected", false);
            button->style()->polish(button);
        }
    }

    currentSelectedImageButton = nullptr;
}

Foreground::~Foreground()
{
    delete ui;
}

void Foreground::resetPage()
{
    if (currentSelectedImageButton) {
        applyHighlightStyle(currentSelectedImageButton, false);
    }
    currentSelectedImageButton = nullptr;

    QList<QPushButton*> imageButtons = findChildren<QPushButton*>(QRegularExpression("image[1-6]"));
    for (QPushButton* button : imageButtons) {
        if (button) {
            applyHighlightStyle(button, false);
        }
    }

    resetDebounce();
    debounceTimer->stop();
}

bool Foreground::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress) {
        QPushButton *button = qobject_cast<QPushButton*>(obj);
        if (button && button->objectName().startsWith("image")) {
            if (debounceActive) {
                return true;
            } else {
                debounceActive = true;
                debounceTimer->start();
                processImageButtonClick(button);
                return true;
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void Foreground::resetDebounce()
{
    debounceActive = false;
}

void Foreground::applyHighlightStyle(QPushButton *button, bool highlight)
{
    if (button) {
        button->setProperty("selected", highlight);
        button->style()->polish(button);
        button->update();
    }
}

void Foreground::on_back_clicked()
{
    if (currentSelectedImageButton) {
        applyHighlightStyle(currentSelectedImageButton, false);
    }
    currentSelectedImageButton = nullptr;
    emit backtoLandingPage();
}

void Foreground::processImageButtonClick(QPushButton *button)
{
    if (!button) {
        return;
    }

    if (button == currentSelectedImageButton) {
        applyHighlightStyle(button, false);
        currentSelectedImageButton = nullptr;
        emit imageSelectedTwice();
    } else {
        if (currentSelectedImageButton) {
            applyHighlightStyle(currentSelectedImageButton, false);
        }

        applyHighlightStyle(button, true);
        currentSelectedImageButton = button;
    }
}

void Foreground::on_image1_clicked() {}
void Foreground::on_image2_clicked() {}
void Foreground::on_image3_clicked() {}
void Foreground::on_image4_clicked() {}
void Foreground::on_image5_clicked() {}
void Foreground::on_image6_clicked() {}
