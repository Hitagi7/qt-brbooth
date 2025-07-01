#ifndef VIDEOTEMPLATE_H
#define VIDEOTEMPLATE_H

#include <QString>
#include <QVariant>
#include <QDebug>


struct VideoTemplate
{
    QString name;
    int durationSeconds;

    VideoTemplate() : durationSeconds(0){}

    VideoTemplate(const QString& name, int duration)
        : name(name), durationSeconds(duration){}
};
Q_DECLARE_METATYPE(VideoTemplate)

#endif // VIDEOTEMPLATE_H
