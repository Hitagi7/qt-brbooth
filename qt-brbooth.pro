QT       += core gui
QT       += multimedia multimediawidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# Config for OpenCV
CONFIG += link_pkgconfig

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    background.cpp \
    capture.cpp \
    dynamic.cpp \
    final.cpp \
    foreground.cpp \
    iconhover.cpp \
    main.cpp \
    brbooth.cpp \
    simplepersondetector.cpp

HEADERS += \
    background.h \
    brbooth.h \
    capture.h \
    dynamic.h \
    final.h \
    foreground.h \
    iconhover.h \
    videotemplate.h \
    simplepersondetector.h

FORMS += \
    background.ui \
    brbooth.ui \
    capture.ui \
    dynamic.ui \
    final.ui \
    foreground.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resources.qrc


INCLUDEPATH += C:\opencv_build\install\include
DEPENDPATH += C:\opencv_build\install\include

OPENCV_INSTALL_DIR = C:/opencv_build/install

# Add OpenCV include paths
INCLUDEPATH += $$OPENCV_INSTALL_DIR/include \
               $$OPENCV_INSTALL_DIR/include/opencv2


LIBS += -L$$OPENCV_INSTALL_DIR/x64/vc17/lib \
        -lopencv_core4110d \
        -lopencv_highgui4110d \
        -lopencv_imgproc4110d \
        -lopencv_videoio4110d \
        -lopencv_calib3d4110d \
        -lopencv_dnn4110d \
        -lopencv_features2d4110d \
        -lopencv_flann4110d \
        -lopencv_gapi4110d \
        -lopencv_imgcodecs4110d \
        -lopencv_ml4110d \
        -lopencv_objdetect4110d \
        -lopencv_photo4110d \
        -lopencv_stitching4110d

