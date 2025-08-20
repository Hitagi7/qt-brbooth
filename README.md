# Qt BRBooth - Interactive Photo Booth Application

A modern, full-screen photo booth application built with Qt and OpenCV, featuring real-time image processing and hand detection capabilities.

## Overview

Qt BRBooth is an interactive photo booth application that provides a complete user experience from template selection to final photo/video output. The application runs in full-screen mode and offers both static and dynamic photo booth experiences.

## Features

### Core Functionality
- **Multi-page Interface**: Landing page with navigation to different photo booth modes
- **Static Mode**: Traditional photo booth with static backgrounds and foregrounds
- **Dynamic Mode**: Video-based backgrounds and templates
- **Real-time Camera Capture**: Live camera feed with instant preview
- **Image/Video Recording**: Capture photos or record videos with customizable templates

### Advanced Image Processing
- **Hand Detection**: Advanced hand tracking and detection capabilities
- **Background Replacement**: Dynamic background replacement and blending
- **Template System**: Customizable foreground, background, and video templates

### User Interface
- **Full-screen Experience**: Immersive full-screen interface
- **Responsive Design**: Adaptive layouts for different screen sizes
- **Custom Fonts**: Roboto Condensed font integration
- **Icon System**: SVG-based icon system with hover effects

## Project Structure

```
qt-brbooth/
├── src/                           # Source files
│   ├── main.cpp                   # Application entry point
│   ├── core/                      # Core application logic
│   │   ├── brbooth.cpp           # Main window and navigation
│   │   ├── camera.cpp            # Camera handling
│   │   └── capture.cpp           # Camera capture and processing
│   ├── ui/                        # User interface components
│   │   ├── background.cpp        # Background selection page
│   │   ├── foreground.cpp        # Foreground selection page
│   │   ├── dynamic.cpp           # Dynamic template page
│   │   ├── final.cpp             # Final output page
│   │   └── iconhover.cpp         # Icon hover effects
│   └── algorithms/                # Computer vision algorithms
│       └── hand_detection/        # Hand detection and tracking
│           ├── advanced_hand_detector.cpp
│           └── mediapipe_like_hand_tracker.cpp
├── include/                       # Header files
│   ├── core/                      # Core headers
│   │   ├── brbooth.h
│   │   ├── camera.h
│   │   ├── capture.h
│   │   ├── videotemplate.h
│   │   └── common_types.h
│   ├── ui/                        # UI headers
│   │   ├── background.h
│   │   ├── foreground.h
│   │   ├── dynamic.h
│   │   ├── final.h
│   │   ├── iconhover.h
│   │   └── ui_manager.h
│   └── algorithms/                # Algorithm headers
│       ├── advanced_hand_detector.h
│       └── mediapipe_like_hand_tracker.h
├── ui/                            # Qt Designer UI files
│   ├── background.ui
│   ├── brbooth.ui
│   ├── capture.ui
│   ├── dynamic.ui
│   ├── final.ui
│   └── foreground.ui
├── Resources/                     # Application resources
│   ├── pics/                      # Static images and backgrounds
│   ├── videos/                    # Video templates
│   ├── templates/                 # Template assets
│   │   ├── background/            # Background templates
│   │   ├── foreground/            # Foreground templates
│   │   └── dynamic/               # Dynamic video templates
│   ├── gif templates/             # Animated background templates
│   ├── Icons/                     # SVG icons
│   └── Fonts/                     # Custom fonts
├── resources.qrc                  # Resource definitions
└── qt-brbooth.pro                 # Qt project file
```

## Dependencies

### Required Libraries
- **Qt 6**: Core, GUI, Widgets, Multimedia, MultimediaWidgets, Concurrent
- **OpenCV 4.11.0**: Computer vision library for image processing
- **C++17**: Modern C++ features

### Optional Dependencies

- **OpenMP**: Multi-threading optimizations

## Building the Project

### Prerequisites
1. **Qt 6**: Install Qt 6 with MSVC 2019/2022 compiler
2. **OpenCV 4.11.0**: Build and install OpenCV with the following modules:
   - core, highgui, imgproc, videoio, video
   - calib3d, dnn, features2d, flann, gapi
   - imgcodecs, ml, objdetect, photo, stitching

### Build Instructions
1. Clone the repository
2. Open `qt-brbooth.pro` in Qt Creator
3. Configure the OpenCV installation path in the project file
4. Build the project (Release or Debug configuration)

### OpenCV Configuration
The project expects OpenCV to be installed at `C:\opencv_build\install`. Update the paths in `qt-brbooth.pro` if your installation is different:

```qmake
OPENCV_INSTALL_DIR = C:/opencv_build/install
INCLUDEPATH += $$OPENCV_INSTALL_DIR/include
LIBS += -L$$OPENCV_INSTALL_DIR/x64/vc17/lib
```

## Usage

### Application Flow
1. **Landing Page**: Choose between Static or Dynamic mode
2. **Template Selection**: Select backgrounds, foregrounds, or video templates
3. **Capture Page**: Real-time camera feed with processing options
4. **Final Output**: View and save captured photos/videos

### Controls
- **Navigation**: Use on-screen buttons to navigate between pages
- **Camera Controls**: Start/stop camera, capture images, record videos
- **Processing Options**: Toggle hand detection, adjust confidence thresholds
- **Template Selection**: Browse and select from available templates

### Processing Features
- **Hand Detection**: Track and detect hand gestures and positions
- **Performance Modes**: Adjust processing quality vs. speed
- **Confidence Thresholds**: Fine-tune detection sensitivity

## Technical Details

### Architecture
- **Multi-threaded Design**: Camera operations run in separate threads
- **Signal-Slot Communication**: Qt's event-driven architecture
- **OpenCV Integration**: Real-time image processing pipeline
- **Template System**: Modular template loading and management
- **Modular Organization**: Clean separation of concerns with dedicated directories for core, UI, and algorithms

### Performance Optimizations
- **OpenMP Support**: Multi-threading for image processing
- **GPU Acceleration**: OpenCV DNN module support
- **Memory Management**: Efficient image buffer handling
- **Processing Throttling**: Configurable processing intervals

### File Formats
- **Images**: PNG, JPG support for templates and output
- **Videos**: MP4 format for templates and recording
- **Fonts**: TTF format for custom typography
- **Icons**: SVG format for scalable graphics

## Development

### Code Organization
- **Header Files**: Organized in include/ with subdirectories for core, UI, and algorithms
- **Source Files**: Organized in src/ with matching subdirectory structure
- **UI Files**: Qt Designer layouts in ui/ directory
- **Resources**: Assets and media files in dedicated resource directories

### Key Classes
- `BRBooth`: Main application window and navigation
- `Capture`: Camera handling and image processing
- `AdvancedHandDetector`: Hand detection and tracking
- `Camera`: Camera device management

### Contributing
1. Follow Qt coding conventions
2. Use meaningful variable and function names
3. Add comments for complex algorithms
4. Test on different screen resolutions
5. Ensure proper memory management
6. Maintain the organized directory structure

## License

This project is part of a thesis work. Please contact the author for licensing information.

## Support

For issues and questions:
1. Check the build configuration
2. Verify OpenCV installation
3. Ensure Qt version compatibility
4. Review camera device permissions

---

**Note**: This application is designed for full-screen photo booth environments and requires appropriate camera hardware and display setup for optimal operation.
