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
    simplepersondetector.cpp \
    personsegmentation.cpp \
    optimized_detector.cpp \
    fast_segmentation.cpp \
    segmentation_manager.cpp \
    detection_manager.cpp

# Conditionally include TensorFlow Lite files if available
exists($$TFLITE_DIR/include/tensorflow/lite/interpreter.h) {
    SOURCES += tflite_deeplabv3.cpp tflite_segmentation_widget.cpp
    DEFINES += TFLITE_AVAILABLE
}

HEADERS += \
    background.h \
    brbooth.h \
    capture.h \
    dynamic.h \
    final.h \
    foreground.h \
    iconhover.h \
    videotemplate.h \
    simplepersondetector.h \
    personsegmentation.h \
    optimized_detector.h \
    fast_segmentation.h \
    segmentation_manager.h \
    detection_manager.h \
    common_types.h

# Conditionally include TensorFlow Lite headers if available
exists($$TFLITE_DIR/include/tensorflow/lite/interpreter.h) {
    HEADERS += tflite_deeplabv3.h tflite_segmentation_widget.h
}

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

# TensorFlow Lite configuration (optional)
# Set TFLITE_DIR to the path where TensorFlow Lite is installed
# If not set or not found, the project will compile without TensorFlow Lite
TFLITE_DIR = C:/tensorflow_lite

# Only add TensorFlow Lite if it's available
exists($$TFLITE_DIR/include/tensorflow/lite/interpreter.h) {
    INCLUDEPATH += $$TFLITE_DIR/include
    DEPENDPATH += $$TFLITE_DIR/include
    
    # TensorFlow Lite libraries (adjust paths and library names based on your installation)
    win32-msvc*: LIBS += -L$$TFLITE_DIR/lib \
                         -ltensorflowlite \
                         -ltensorflowlite_gpu_delegate \
                         -ltensorflowlite_hexagon_delegate
    
    # Alternative: If using vcpkg or other package managers
    # win32-msvc*: LIBS += -ltensorflowlite
} else {
    # Fallback: Use OpenCV-based segmentation
    DEFINES += USE_OPENCV_SEGMENTATION
    message(TensorFlow Lite not found. Using OpenCV-based segmentation fallback.)
}

