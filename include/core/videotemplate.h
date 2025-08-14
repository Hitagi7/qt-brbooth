#ifndef VIDEOTEMPLATE_H
#define VIDEOTEMPLATE_H

#include <QDebug>
#include <QString>
#include <QVariant>

struct VideoTemplate
{
    QString name;
    int durationSeconds;

    VideoTemplate()
        : durationSeconds(0)
    {}

    VideoTemplate(const QString &name, int duration)
        : name(name)
        , durationSeconds(duration)
    {}
};
Q_DECLARE_METATYPE(VideoTemplate)

#endif // VIDEOTEMPLATE_H
