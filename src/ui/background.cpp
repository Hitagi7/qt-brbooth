#include "ui/background.h"
#include <QDebug>
#include <QDir>
#include <QMouseEvent>
#include <QRegularExpression>
#include <QStyle>
#include "ui/iconhover.h"
#include "ui_background.h"

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

bool Background::eventFilter(QObject *obj, QEvent *event)
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

    // Set selected background based on objectName
    QString name = button->objectName(); // e.g., "image1"
    QString path;

    if (name == "image1")
        path = "templates/background/bg1.png";
    else if (name == "image2")
        path = "templates/background/bg2.png";
    else if (name == "image3")
        path = "templates/background/bg3.png";
    else if (name == "image4")
        path = "templates/background/bg4.png";
    else if (name == "image5")
        path = "templates/background/bg5.png";
    else if (name == "image6")
        path = "templates/background/bg6.png";

    if (!path.isEmpty()) {
        setSelectedBackground(path);
        qDebug() << "🎯 Background selected:" << path;
        qDebug() << "🎯 Background stored:" << getSelectedBackground();
        emit backgroundChanged(getSelectedBackground());
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

void Background::setSelectedBackground(const QString &path)
{
    selectedBackground = path;
}

QString Background::getSelectedBackground() const
{
    return selectedBackground;
}

// Implement the missing on_imageX_clicked methods that are referenced by MOC
void Background::on_image1_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image1");
    if (button) {
        processImageButtonClick(button);
    }
}

void Background::on_image2_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image2");
    if (button) {
        processImageButtonClick(button);
    }
}

void Background::on_image3_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image3");
    if (button) {
        processImageButtonClick(button);
    }
}

void Background::on_image4_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image4");
    if (button) {
        processImageButtonClick(button);
    }
}

void Background::on_image5_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image5");
    if (button) {
        processImageButtonClick(button);
    }
}

void Background::on_image6_clicked()
{
    QPushButton *button = findChild<QPushButton *>("image6");
    if (button) {
        processImageButtonClick(button);
    }
}


