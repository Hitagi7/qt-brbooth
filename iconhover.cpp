#include "iconhover.h"

Iconhover::Iconhover(QObject *parent)
    : QObject{parent}
{

}
bool Iconhover::eventFilter(QObject *watched, QEvent *event)
{
    QPushButton *button = qobject_cast<QPushButton*>(watched);
    if (!button){
        return false;
    }

    if(event->type() == QEvent::Enter){
        button->setIcon(QIcon(":/icons/Icons/hover.svg"));
        return true;
    }

    if(event->type() == QEvent::Leave){
        button->setIcon(QIcon(":/icons/Icons/normal.svg"));
        return true;
    }

    return QObject::eventFilter(watched, event);
}
