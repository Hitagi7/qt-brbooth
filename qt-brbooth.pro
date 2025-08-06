QT       += core gui
QT       += multimedia multimediawidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# Enable OpenMP for multi-threading optimizations
win32-msvc*: QMAKE_CXXFLAGS += /openmp
win32-msvc*: LIBS += -llibomp
win32-g++: QMAKE_CXXFLAGS += -fopenmp
win32-g++: LIBS += -lgomp
unix: QMAKE_CXXFLAGS += -fopenmp
unix: LIBS += -lgomp

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
    tflite_deeplabv3.cpp \
    tflite_segmentation_widget.cpp

HEADERS += \
    background.h \
    brbooth.h \
    capture.h \
    dynamic.h \
    final.h \
    foreground.h \
    iconhover.h \
    videotemplate.h \
    common_types.h \
    tflite_deeplabv3.h \
    tflite_segmentation_widget.h

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

# OpenCV Configuration
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

# TensorFlow Lite configuration (commented out for OpenCV fallback)
# TENSORFLOW_DIR = C:/tensorflow-2.18.1
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/core
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/kernels
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/delegates
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/delegates/gpu
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/delegates/hexagon
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/tools
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/schema
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/experimental
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/async
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/acceleration
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/string_util
# INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow/lite/error_reporter

# Define TFLite availability (using OpenCV fallback)
DEFINES += TFLITE_AVAILABLE
DEFINES += TFLITE_DEEPLABV3_ENABLED

