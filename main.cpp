
#include <QApplication>
#include <QFontDatabase>
#include "brbooth.h"
#include "videotemplate.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    int fontId = QFontDatabase::addApplicationFont(
        "::/fonts/Fonts/static/RobotoCondensed-BoldItalic.ttf");
    if (fontId == -1) {
        qWarning() << "Failed to load RobotoCondensed-BoldItalic.ttf from resources.";
    } else {
        qDebug() << "Font loaded successfully. Font ID:" << fontId;
    }

    qRegisterMetaType<VideoTemplate>("Video Template");
    BRBooth w;
    w.showFullScreen();
    return a.exec();
}
