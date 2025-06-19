#include "brbooth.h"
#include <QFontDatabase>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    int fontId = QFontDatabase::addApplicationFont(":/fonts/RobotoCondensed-BoldItalic.ttf");
    if (fontId == -1) {
        qWarning() << "Failed to load RobotoCondensed-BoldItalic.ttf from resources.";
    } else {
        qDebug() << "Font loaded successfully. Font ID:" << fontId;
    }

    BRBooth w;
    w.showMaximized();
    return a.exec();
    //test
}
