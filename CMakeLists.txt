cmake_minimum_required(VERSION 3.16)
project(qt-brbooth VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Multimedia MultimediaWidgets)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Try to find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS 
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
    DOC "ONNX Runtime include directory"
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS 
        /usr/local/lib
        /usr/lib
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib
    DOC "ONNX Runtime library"
)

# Set sources
set(SOURCES
    main.cpp
    brbooth.cpp
    background.cpp
    capture.cpp
    dynamic.cpp
    final.cpp
    foreground.cpp
    iconhover.cpp
    src/yolo/persondetector.cpp
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
    src/yolo/persondetector.h
)

set(UI_FILES
    background.ui
    brbooth.ui
    capture.ui
    dynamic.ui
    final.ui
    foreground.ui
)

# Process Qt files
qt6_wrap_ui(UI_HEADERS ${UI_FILES})
qt6_add_resources(RESOURCES resources.qrc)

# Create executable
qt6_add_executable(qt-brbooth
    ${SOURCES}
    ${HEADERS}
    ${UI_HEADERS}
    ${RESOURCES}
)

# Include directories
target_include_directories(qt-brbooth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src/yolo
)

# Add ONNX Runtime if found
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    target_include_directories(qt-brbooth PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(qt-brbooth PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(qt-brbooth PRIVATE HAVE_ONNXRUNTIME)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
else()
    message(WARNING "ONNX Runtime not found. Person detection will be disabled.")
endif()

# Link libraries
target_link_libraries(qt-brbooth PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
    ${OpenCV_LIBS}
)

# Compiler definitions
target_compile_definitions(qt-brbooth PRIVATE QT_NO_DEBUG_OUTPUT)

# Build example if requested
option(BUILD_EXAMPLES "Build example applications" ON)
if(BUILD_EXAMPLES AND ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    qt6_add_executable(person_detection_example
        examples/person_detection_example.cpp
        src/yolo/persondetector.cpp
        src/yolo/persondetector.h
    )
    
    target_include_directories(person_detection_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        src/yolo
    )
    
    target_link_libraries(person_detection_example PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Multimedia
        Qt6::MultimediaWidgets
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
    )
    
    target_compile_definitions(person_detection_example PRIVATE HAVE_ONNXRUNTIME)
endif()

# Installation
install(TARGETS qt-brbooth
    BUNDLE DESTINATION .
    RUNTIME DESTINATION bin
)

if(TARGET person_detection_example)
    install(TARGETS person_detection_example
        RUNTIME DESTINATION bin
    )
endif()