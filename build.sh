#!/bin/bash

# Build script for Qt Booth with YOLOv5n Person Detection
# This script automates the build process for the booth application

set -e

echo "=========================================="
echo "Qt Booth - YOLOv5n Person Detection Build"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build configuration
BUILD_DIR="build"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_EXAMPLES="${BUILD_EXAMPLES:-ON}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.16+."
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_status "CMake version: $CMAKE_VERSION"
    
    # Check Qt6
    if ! pkg-config --exists Qt6Core Qt6Widgets Qt6Multimedia; then
        print_warning "Qt6 development packages not found."
        print_warning "Please install: qt6-base-dev qt6-multimedia-dev"
    else
        QT_VERSION=$(pkg-config --modversion Qt6Core)
        print_status "Qt6 version: $QT_VERSION"
    fi
    
    # Check OpenCV
    if ! pkg-config --exists opencv4; then
        print_warning "OpenCV development packages not found."
        print_warning "Please install: libopencv-dev"
    else
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_status "OpenCV version: $OPENCV_VERSION"
    fi
    
    # Check ONNX Runtime
    if [ -f "/usr/local/include/onnxruntime_cxx_api.h" ] || [ -f "/usr/include/onnxruntime/onnxruntime_cxx_api.h" ]; then
        print_status "ONNX Runtime headers found"
    else
        print_warning "ONNX Runtime not found. Person detection will be disabled."
        print_warning "Download from: https://github.com/microsoft/onnxruntime/releases"
    fi
}

# Download YOLOv5n model if not present
download_model() {
    MODEL_DIR="models"
    MODEL_FILE="$MODEL_DIR/yolov5n.onnx"
    
    if [ ! -f "$MODEL_FILE" ]; then
        print_status "YOLOv5n model not found. Downloading..."
        mkdir -p "$MODEL_DIR"
        
        if command -v wget &> /dev/null; then
            wget -O "$MODEL_FILE" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
        elif command -v curl &> /dev/null; then
            curl -L -o "$MODEL_FILE" "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
        else
            print_warning "wget or curl not found. Please download YOLOv5n model manually:"
            print_warning "URL: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
            print_warning "Save to: $MODEL_FILE"
        fi
        
        if [ -f "$MODEL_FILE" ]; then
            print_status "YOLOv5n model downloaded successfully"
        fi
    else
        print_status "YOLOv5n model found: $MODEL_FILE"
    fi
}

# Build the project
build_project() {
    print_status "Building project..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Run CMake configuration
    print_status "Running CMake configuration..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DBUILD_EXAMPLES="$BUILD_EXAMPLES"
    
    # Build
    print_status "Compiling..."
    cmake --build . --config "$CMAKE_BUILD_TYPE" -j$(nproc 2>/dev/null || echo 4)
    
    cd ..
    print_status "Build completed successfully!"
}

# Show build results
show_results() {
    print_status "Build results:"
    
    if [ -f "$BUILD_DIR/qt-brbooth" ]; then
        echo "  ✓ Main application: $BUILD_DIR/qt-brbooth"
    else
        echo "  ✗ Main application: NOT BUILT"
    fi
    
    if [ -f "$BUILD_DIR/person_detection_example" ]; then
        echo "  ✓ Person detection example: $BUILD_DIR/person_detection_example"
    else
        echo "  ✗ Person detection example: NOT BUILT"
    fi
    
    echo ""
    print_status "To run the application:"
    echo "  cd $BUILD_DIR"
    echo "  ./qt-brbooth"
    echo ""
    print_status "To run the person detection example:"
    echo "  cd $BUILD_DIR"
    echo "  ./person_detection_example"
}

# Print usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --clean             Clean build directory before building"
    echo "  --debug             Build in Debug mode (default: Release)"
    echo "  --no-examples       Don't build examples"
    echo "  --download-model    Download YOLOv5n model if missing"
    echo ""
    echo "Environment variables:"
    echo "  CMAKE_BUILD_TYPE    Build type (Release|Debug)"
    echo "  BUILD_EXAMPLES      Build examples (ON|OFF)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build in Release mode with examples"
    echo "  $0 --debug          # Build in Debug mode"
    echo "  $0 --clean          # Clean and build"
    echo "  $0 --download-model # Download model and build"
}

# Parse command line arguments
CLEAN_BUILD=false
DOWNLOAD_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            CMAKE_BUILD_TYPE="Debug"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --download-model)
            DOWNLOAD_MODEL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "Build configuration:"
    echo "  Build type: $CMAKE_BUILD_TYPE"
    echo "  Build examples: $BUILD_EXAMPLES"
    echo "  Clean build: $CLEAN_BUILD"
    echo ""
    
    check_dependencies
    
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    if [ "$DOWNLOAD_MODEL" = true ]; then
        download_model
    fi
    
    build_project
    show_results
}

# Run main function
main "$@"