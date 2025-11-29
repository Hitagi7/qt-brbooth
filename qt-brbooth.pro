QT       += core gui
QT       += multimedia multimediawidgets
QT       += concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# OpenCV configuration for enhanced person detection
DEFINES += OPENCV_ENABLE_NONFREE

# Ensure MOC runs for classes with Q_OBJECT
CONFIG += moc

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
               . \
               build/Desktop_Qt_6_9_1_MSVC2022_64bit-Debug \
               build/Desktop_Qt_6_9_1_MinGW_64_bit-Debug

# Source files organized by category
SOURCES += \
    src/main.cpp \
    src/core/brbooth.cpp \
    src/core/camera.cpp \
    src/core/capture.cpp \
    src/ui/loading.cpp \
    src/ui/confirm.cpp \
    src/ui/background.cpp \
    src/ui/foreground.cpp \
    src/ui/dynamic.cpp \
    src/ui/final.cpp \
    src/ui/idle.cpp \
    src/ui/iconhover.cpp \
    src/ui/ui_manager.cpp \
    src/algorithms/hand_detection/hand_detector.cpp \
    src/algorithms/lighting_correction/lighting_corrector.cpp
    # src/algorithms/hand_detection/advanced_hand_detector.cpp \
    # src/algorithms/hand_detection/mediapipe_like_hand_tracker.cpp

HEADERS += \
    include/core/brbooth.h \
    include/core/camera.h \
    include/core/capture.h \
    include/core/videotemplate.h \
    include/core/common_types.h \
    include/ui/loading.h \
    include/ui/confirm.h \
    include/ui/background.h \
    include/ui/foreground.h \
    include/ui/dynamic.h \
    include/ui/final.h \
    include/ui/idle.h \
    include/ui/iconhover.h \
    include/ui/ui_manager.h \
    include/algorithms/hand_detection/hand_detector.h \
    include/algorithms/lighting_correction/lighting_corrector.h
    # include/algorithms/advanced_hand_detector.h \
    # include/algorithms/mediapipe_like_hand_tracker.h

FORMS += \
    ui/background.ui \
    ui/brbooth.ui \
    ui/capture.ui \
    ui/confirm.ui \
    ui/dynamic.ui \
    ui/final.ui \
    ui/idle.ui \
    ui/loading.ui \
    ui/foreground.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# Note: Video files are now loaded from external files instead of embedded resources
# To run the application, manually copy video files to the build directory:
# - Copy videos/*.mp4 to build/Desktop_Qt_6_9_1_MSVC2022_64bit-Debug/videos/
# - Copy templates/dynamic/*.mp4 to build/Desktop_Qt_6_9_1_MSVC2022_64bit-Debug/templates/dynamic/

RESOURCES += \
    resources.qrc

# OpenCV Configuration - Using CUDA-Enabled Build
INCLUDEPATH += C:\opencv_cuda\opencv_cuda_build\install\include
DEPENDPATH += C:\opencv_cuda\opencv_cuda_build\install\include

OPENCV_INSTALL_DIR = C:/opencv_cuda/opencv_cuda_build/install

# Add OpenCV include paths
INCLUDEPATH += $$OPENCV_INSTALL_DIR/include \
               $$OPENCV_INSTALL_DIR/include/opencv2

# Add CUDA include paths
INCLUDEPATH += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"

# CUDA libraries are already linked through OpenCV's CUDA build
# No need to explicitly link CUDA libraries as they're included in opencv_world4110d

# Add OpenCV library path
LIBS += -L$$OPENCV_INSTALL_DIR/x64/vc17/lib \
        -lopencv_world4110d

# cuDNN Configuration
CUDNN_INSTALL_DIR = "C:/Program Files/NVIDIA/CUDNN/v9.13"

# Add cuDNN include paths
INCLUDEPATH += "$$CUDNN_INSTALL_DIR/include/13.0"

# Add cuDNN library paths
LIBS += -L"$$CUDNN_INSTALL_DIR/lib/13.0/x64"

# Add cuDNN libraries
LIBS += -lcudnn \
        -lcudnn_adv \
        -lcudnn_cnn \
        -lcudnn_graph \
        -lcudnn_ops

# Add Windows system libraries
LIBS += -lshell32 -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32


DEFINES += DEBUG_THROTTLE
