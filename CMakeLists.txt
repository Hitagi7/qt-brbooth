cmake_minimum_required(VERSION 3.16)
project(qt-brbooth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Multimedia MultimediaWidgets)
find_package(OpenCV REQUIRED)

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT_PATH "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT_PATH)
    if(EXISTS "${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime found at ${ONNXRUNTIME_ROOT_PATH}")
        add_definitions(-DONNXRUNTIME_AVAILABLE)
        
        # Include directories
        include_directories("${ONNXRUNTIME_ROOT_PATH}/include")
        
        # Library directories and libraries
        if(WIN32)
            link_directories("${ONNXRUNTIME_ROOT_PATH}/lib")
            set(ONNXRUNTIME_LIBS onnxruntime)
        else()
            link_directories("${ONNXRUNTIME_ROOT_PATH}/lib")
            set(ONNXRUNTIME_LIBS onnxruntime)
        endif()
    else()
        message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_ROOT_PATH}")
        message(WARNING "YOLOv5 detection will be disabled. Please install ONNX Runtime to enable object detection.")
        set(ONNXRUNTIME_LIBS "")
    endif()
else()
    message(WARNING "ONNXRUNTIME_ROOT_PATH not set. YOLOv5 detection will be disabled.")
    message(WARNING "To enable object detection, set ONNXRUNTIME_ROOT_PATH to your ONNX Runtime installation.")
    set(ONNXRUNTIME_LIBS "")
endif()

# Source files
set(SOURCES
    main.cpp
    brbooth.cpp
    background.cpp
    capture.cpp
    dynamic.cpp
    final.cpp
    foreground.cpp
    iconhover.cpp
    src/yolo/yolov5detector.cpp
)

set(HEADERS
    brbooth.h
    background.h
    capture.h
    dynamic.h
    final.h
    foreground.h
    iconhover.h
    videotemplate.h
    src/yolo/yolov5detector.h
)

set(UI_FILES
    brbooth.ui
    background.ui
    capture.ui
    dynamic.ui
    final.ui
    foreground.ui
)

# Qt resource files
set(RESOURCE_FILES
    resources.qrc
)

# Create the main executable
qt6_add_executable(qt-brbooth ${SOURCES})

# Process Qt UI files
qt6_add_resources(qt-brbooth ${RESOURCE_FILES})

# Set target properties
set_target_properties(qt-brbooth PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.example.qt-brbooth"
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# Link libraries
target_link_libraries(qt-brbooth
    Qt6::Core
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBS}
)

# Include directories
target_include_directories(qt-brbooth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# Add example executable (optional)
option(BUILD_YOLO_EXAMPLE "Build YOLOv5 example application" ON)

if(BUILD_YOLO_EXAMPLE)
    qt6_add_executable(yolo_example
        examples/yolo_example.cpp
        src/yolo/yolov5detector.cpp
    )
    
    target_link_libraries(yolo_example
        Qt6::Core
        Qt6::Widgets
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBS}
    )
    
    target_include_directories(yolo_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    
    set_target_properties(yolo_example PROPERTIES
        MACOSX_BUNDLE TRUE
        WIN32_EXECUTABLE TRUE
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS qt-brbooth
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_YOLO_EXAMPLE)
    install(TARGETS yolo_example
        BUNDLE DESTINATION .
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Qt Version: ${Qt6_VERSION}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "  Build YOLO Example: ${BUILD_YOLO_EXAMPLE}")
message(STATUS "")

if(ONNXRUNTIME_ROOT_PATH)
    message(STATUS "YOLOv5 object detection: ENABLED")
else()
    message(STATUS "YOLOv5 object detection: DISABLED")
    message(STATUS "  To enable: cmake -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime ..")
endif()
message(STATUS "")