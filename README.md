# qt-brbooth - Real-Time Person Segmentation Photo Booth

A Qt-based photo booth application with real-time person segmentation using OpenCV-based edge detection and silhouette formation.

## Features

- **Real-Time Person Segmentation**: Advanced edge-based silhouette detection
- **Background Replacement**: Seamless background substitution
- **Foreground Overlays**: Add custom foreground elements
- **Video Recording**: Capture videos with segmentation effects
- **Multiple Capture Modes**: Image and video capture options
- **Performance Optimization**: Multi-threaded processing for smooth real-time performance

## Technology Stack

- **Qt 6**: Cross-platform GUI framework
- **OpenCV 4.11**: Computer vision and image processing
- **C++17**: Modern C++ features
- **OpenMP**: Multi-threading optimization

## Prerequisites

### Required Software
- **Qt 6.5+** with MSVC 2019/2022 compiler
- **OpenCV 4.11** (pre-built or compiled from source)
- **Visual Studio 2019/2022** (Windows)
- **CMake 3.16+**

### System Requirements
- **Windows 10/11** (64-bit)
- **8GB RAM** minimum (16GB recommended)
- **Webcam** for real-time capture
- **OpenGL 3.3+** compatible graphics card

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd qt-brbooth
```

### 2. Install OpenCV

#### Option A: Pre-built OpenCV (Recommended)
1. Download OpenCV 4.11 from [opencv.org](https://opencv.org/releases/)
2. Extract to `C:\opencv_build\install`
3. Add `C:\opencv_build\install\x64\vc17\bin` to your system PATH

#### Option B: Build OpenCV from Source
```bash
# Follow OpenCV build instructions for Windows
# Ensure you build with MSVC 2019/2022 compatibility
```

### 3. Configure Qt Project
1. Open `qt-brbooth.pro` in Qt Creator
2. Update OpenCV paths if needed:
   ```qmake
   OPENCV_INSTALL_DIR = C:/opencv_build/install
   ```
3. Build the project

### 4. Build and Run
```bash
# Using Qt Creator
# 1. Open qt-brbooth.pro
# 2. Configure project
# 3. Build (Ctrl+B)
# 4. Run (Ctrl+R)

# Using command line
qmake qt-brbooth.pro
make
./qt-brbooth
```

## Project Structure

```
qt-brbooth/
├── src/                    # Source files
│   ├── main.cpp           # Application entry point
│   ├── brbooth.cpp        # Main application window
│   ├── capture.cpp        # Camera capture and segmentation
│   ├── tflite_deeplabv3.cpp # Segmentation algorithm
│   └── ...
├── ui/                     # Qt UI files
├── resources/              # Application resources
├── templates/              # Background/foreground templates
├── pics/                   # Sample images
└── videos/                 # Sample videos
```

## Usage

### Basic Operation
1. **Launch Application**: Run the compiled executable
2. **Camera Setup**: Ensure webcam is connected and accessible
3. **Segmentation**: Click "Start Segmentation" to begin real-time processing
4. **Capture**: Use "Capture" button for images or "Record" for videos
5. **Background**: Select backgrounds from the template library
6. **Foreground**: Add overlay elements as needed

### Segmentation Controls
- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Performance Mode**: Choose between Speed/Balanced/Quality
- **Show Segmentation**: Toggle green outline display

### Advanced Features
- **Real-time FPS**: Monitor processing performance
- **Debug Display**: View technical information
- **Template Management**: Add custom backgrounds/foregrounds

## Configuration

### OpenCV Path Configuration
If OpenCV is installed in a different location, update `qt-brbooth.pro`:
```qmake
OPENCV_INSTALL_DIR = C:/your/opencv/path
```

### Performance Tuning
Adjust segmentation parameters in `tflite_deeplabv3.cpp`:
- Edge detection thresholds
- Contour filtering criteria
- Morphological operation kernels

## Troubleshooting

### Common Issues

#### 1. OpenCV Not Found
```
Error: Cannot find OpenCV
```
**Solution**: Verify OpenCV installation path and update `qt-brbooth.pro`

#### 2. Camera Access Denied
```
Error: Cannot access camera
```
**Solution**: Check camera permissions and ensure no other applications are using the camera

#### 3. Segmentation Not Working
```
Error: No person detected
```
**Solution**: 
- Ensure good lighting
- Person should be clearly visible
- Adjust confidence threshold
- Check camera positioning

#### 4. Performance Issues
```
Error: Low FPS or lag
```
**Solution**:
- Reduce camera resolution
- Switch to "Speed" performance mode
- Close other applications
- Check system resources

### Build Issues

#### Compilation Errors
```bash
# Clean and rebuild
make clean
qmake qt-brbooth.pro
make
```

#### Missing Dependencies
```bash
# Ensure all Qt modules are installed
# Check OpenCV installation
# Verify compiler compatibility
```

## Development

### Adding New Features
1. Follow Qt coding standards
2. Use C++17 features
3. Implement proper error handling
4. Add debug output for troubleshooting

### Code Structure
- **UI Logic**: Separate from business logic
- **Segmentation**: Modular algorithm implementation
- **Resource Management**: Proper memory management
- **Error Handling**: Comprehensive error checking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## License

[Your License Here]

## Support

For issues and questions:
- Create GitHub issue
- Check troubleshooting section
- Review documentation

## Changelog

### Version 2.0 (Current)
- ✅ Replaced YOLOv5 with OpenCV-based segmentation
- ✅ Improved real-time performance
- ✅ Enhanced silhouette detection
- ✅ Better edge-based person segmentation
- ✅ Simplified deployment (no Python dependencies)

### Version 1.0
- Initial release with YOLOv5 integration 