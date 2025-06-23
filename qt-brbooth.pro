QT       += core gui
QT       += multimedia multimediawidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    background.cpp \
    capture.cpp \
    dynamic.cpp \
    foreground.cpp \
    iconhover.cpp \
    main.cpp \
    brbooth.cpp

HEADERS += \
    background.h \
    brbooth.h \
    capture.h \
    dynamic.h \
    foreground.h \
    iconhover.h

FORMS += \
    background.ui \
    brbooth.ui \
    capture.ui \
    dynamic.ui \
    foreground.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resources.qrc
