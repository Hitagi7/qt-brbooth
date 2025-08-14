#include "ui/foreground.h"
#include <QDebug>
#include <QDir>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include "ui/iconhover.h"
#include "ui_foreground.h"

Foreground::Foreground(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Foreground)
{
    //Setting Up Back Icon
    ui->setupUi(this);

    ui->back->setIcon(QIcon(":/icons/Icons/normal.svg"));
    ui->back->setIconSize(QSize(100, 100));

    Iconhover *backButtonHover = new Iconhover(this); // 'this' as parent for memory management
    ui->back->installEventFilter(backButtonHover);

    connect(ui->back, &QPushButton::clicked, this, &Foreground::on_back_clicked);

    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(400);
    connect(debounceTimer, &QTimer::timeout, this, &Foreground::resetDebounce);

    debounceActive = false;

    QList<QPushButton *> imageButtons = findChildren<QPushButton *>(
        QRegularExpression("image[1-6]"));
    for (QPushButton *button : imageButtons) {
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

    QList<QPushButton *> imageButtons = findChildren<QPushButton *>(
        QRegularExpression("image[1-6]"));
    for (QPushButton *button : imageButtons) {
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
        QPushButton *button = qobject_cast<QPushButton *>(obj);
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
    if (!button)
        return;

    // Set selected foreground based on objectName
    QString name = button->objectName(); // e.g., "image1"
    QString path;

    if (name == "image1")
        path = ":/foreground/templates/foreground/1.png";
    else if (name == "image2")
        path = ":/foreground/templates/foreground/2.png";
    else if (name == "image3")
        path = ":/foreground/templates/foreground/3.png";
    else if (name == "image4")
        path = ":/foreground/templates/foreground/4.png";
    else if (name == "image5")
        path = ":/foreground/templates/foreground/5.png";
    else if (name == "image6")
        path = ":/foreground/templates/foreground/6.png";

    if (!path.isEmpty()) {
        setSelectedForeground(path);
        qDebug() << "Foreground selected:" << path;
        emit foregroundChanged(getSelectedForeground());
    }

    // Handle double click selection logic
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

void Foreground::setSelectedForeground(const QString &path)
{
    selectedForeground = path;
}

QString Foreground::getSelectedForeground() const
{
    return selectedForeground;
}
