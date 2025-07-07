QT += core gui widgets

CONFIG += c++17

TARGET = yolo_example
TEMPLATE = app

# Include the main project's configuration
include(../qt-brbooth.pro)

# Override some settings for the example
TARGET = yolo_example
TEMPLATE = app

# Remove the main application files
SOURCES -= main.cpp \
    brbooth.cpp \
    background.cpp \
    capture.cpp \
    dynamic.cpp \
    final.cpp \
    foreground.cpp \
    iconhover.cpp

HEADERS -= brbooth.h \
    background.h \
    capture.h \
    dynamic.h \
    final.h \
    foreground.h \
    iconhover.h \
    videotemplate.h

FORMS -= brbooth.ui \
    background.ui \
    capture.ui \
    dynamic.ui \
    final.ui \
    foreground.ui

# Add the example source
SOURCES += yolo_example.cpp

# Remove Qt multimedia since we don't need it for this example
QT -= multimedia multimediawidgets

# Remove resource files
RESOURCES =

message("Building YOLOv5 example application")