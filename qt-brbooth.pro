QT       += core gui
QT       += multimedia multimediawidgets
QT       += concurrent

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

# Include paths for the new directory structure
INCLUDEPATH += include \
               include/algorithms \
               include/core \
               include/ui \
               ui \
               .

# Source files organized by category
SOURCES += \
    src/main.cpp \
    src/core/brbooth.cpp \
    src/core/camera.cpp \
    src/core/capture.cpp \
    src/ui/background.cpp \
    src/ui/foreground.cpp \
    src/ui/dynamic.cpp \
    src/ui/final.cpp \
    src/ui/iconhover.cpp \
    src/ui/ui_manager.cpp \
    src/algorithms/segmentation/tflite_deeplabv3.cpp \
    src/algorithms/segmentation/tflite_segmentation_widget.cpp \
    src/algorithms/hand_detection/advanced_hand_detector.cpp \
    src/algorithms/hand_detection/mediapipe_like_hand_tracker.cpp \
    src/algorithms/person_detection/simplepersondetector.cpp \
    src/algorithms/person_detection/personsegmentation.cpp \
    src/algorithms/person_detection/segmentation_manager.cpp

HEADERS += \
    include/core/brbooth.h \
    include/core/camera.h \
    include/core/capture.h \
    include/core/videotemplate.h \
    include/core/common_types.h \
    include/ui/background.h \
    include/ui/foreground.h \
    include/ui/dynamic.h \
    include/ui/final.h \
    include/ui/iconhover.h \
    include/ui/ui_manager.h \
    include/algorithms/tflite_deeplabv3.h \
    include/algorithms/tflite_segmentation_widget.h \
    include/algorithms/advanced_hand_detector.h \
    include/algorithms/mediapipe_like_hand_tracker.h \
    include/algorithms/simplepersondetector.h \
    include/algorithms/personsegmentation.h \
    include/algorithms/segmentation_manager.h

FORMS += \
    ui/background.ui \
    ui/brbooth.ui \
    ui/capture.ui \
    ui/dynamic.ui \
    ui/final.ui \
    ui/foreground.ui

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
        -lopencv_video4110d \
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
DEFINES += DEBUG_THROTTLE

