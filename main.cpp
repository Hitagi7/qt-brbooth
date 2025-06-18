 #include "brbooth.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    BRBooth w;
    w.show();
    return a.exec();
    //test
}
