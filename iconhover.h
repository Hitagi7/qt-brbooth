#ifndef ICONHOVER_H
#define ICONHOVER_H

#include <QEvent>
#include <QIcon>
#include <QObject>
#include <QPushButton>

class Iconhover : public QObject
{
    Q_OBJECT

public:
    explicit Iconhover(QObject *parent = nullptr);
    virtual bool eventFilter(QObject *watched, QEvent *event);

signals:
};

#endif // ICONHOVER_H
