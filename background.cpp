#include "background.h"
#include "ui_background.h"
#include "iconhover.h"
#include <QStyle>
#include <QRegularExpression>
#include <QMouseEvent>

Background::Background(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Background)
{
    ui->setupUi(this);

    //Setting Up Back Icon
    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &Background::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Background::resetDebounce);

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

Background::~Background()
{
    delete ui;
}

void Background::resetPage()
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

bool Background::eventFilter(QObject *obj, QEvent *event)
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

void Background::resetDebounce()
{
    debounceActive = false;
}

void Background::applyHighlightStyle(QPushButton *button, bool highlight)
{
    if (button) {
        button->setProperty("selected", highlight);
        button->style()->polish(button);
        button->update();
    }
}

void Background::on_back_clicked()
{
    if (currentSelectedImageButton) {
        applyHighlightStyle(currentSelectedImageButton, false);
    }
    currentSelectedImageButton = nullptr;
    emit backtoForegroundPage();
}

void Background::processImageButtonClick(QPushButton *button)
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


void Background::on_image1_clicked(){}
void Background::on_image2_clicked(){}
void Background::on_image3_clicked(){}
void Background::on_image4_clicked(){}
void Background::on_image5_clicked(){}

