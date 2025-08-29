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
               .

# Source files organized by category
SOURCES += \
    src/main.cpp \
    src/core/brbooth.cpp \
    src/core/camera.cpp \
    src/core/capture.cpp \
    src/core/amd_gpu_verifier.cpp \
    src/ui/background.cpp \
    src/ui/foreground.cpp \
    src/ui/dynamic.cpp \
    src/ui/final.cpp \
    src/ui/iconhover.cpp \
    src/ui/ui_manager.cpp \
    src/algorithms/hand_detection/hand_detector.cpp
    # src/algorithms/hand_detection/advanced_hand_detector.cpp \
    # src/algorithms/hand_detection/mediapipe_like_hand_tracker.cpp

HEADERS += \
    include/core/brbooth.h \
    include/core/camera.h \
    include/core/capture.h \
    include/core/videotemplate.h \
    include/core/common_types.h \
    include/core/amd_gpu_verifier.h \
    include/ui/background.h \
    include/ui/foreground.h \
    include/ui/dynamic.h \
    include/ui/final.h \
    include/ui/iconhover.h \
    include/ui/ui_manager.h \
    include/algorithms/hand_detection/hand_detector.h
    # include/algorithms/advanced_hand_detector.h \
    # include/algorithms/mediapipe_like_hand_tracker.h

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

# Note: Video files are now loaded from external files instead of embedded resources
# To run the application, manually copy video files to the build directory:
# - Copy videos/*.mp4 to build/Desktop_Qt_6_9_1_MSVC2022_64bit-Debug/videos/
# - Copy templates/dynamic/*.mp4 to build/Desktop_Qt_6_9_1_MSVC2022_64bit-Debug/templates/dynamic/

RESOURCES += \
    resources.qrc

# OpenCV Configuration - Using AMD-Enabled Build with OpenGL
INCLUDEPATH += C:\opencv_amd\opencv_amd_build\install\include
DEPENDPATH += C:\opencv_amd\opencv_amd_build\install\include

OPENCV_INSTALL_DIR = C:/opencv_amd/opencv_amd_build/install

# Add OpenCV include paths
INCLUDEPATH += $$OPENCV_INSTALL_DIR/include \
               $$OPENCV_INSTALL_DIR/include/opencv2

# Add OpenGL include paths
INCLUDEPATH += "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um" \
               "C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt"

# Add OpenCV library path
LIBS += -L$$OPENCV_INSTALL_DIR/x64/vc17/lib \
        -lopencv_world4110d

# Add OpenGL libraries
LIBS += -lopengl32 -lglu32

# Add AMD GPU libraries (if available) - Make OpenCL optional
# LIBS += -lOpenCL  # Commented out to avoid linking errors

# Add Windows system libraries
LIBS += -lshell32 -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32

# Define AMD GPU support
DEFINES += AMD_GPU_SUPPORT
DEFINES += OPENGL_ACCELERATION


DEFINES += DEBUG_THROTTLE
