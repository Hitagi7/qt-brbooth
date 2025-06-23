#ifndef ICONHOVER_H
#define ICONHOVER_H

#include <QObject>
#include <QPushButton>
#include <QEvent>
#include <QIcon>

class Iconhover : public QObject
{
    Q_OBJECT

public:
    explicit Iconhover(QObject *parent = nullptr);
    virtual bool eventFilter(QObject *watched, QEvent *event);

signals:
};

#endif // ICONHOVER_H
