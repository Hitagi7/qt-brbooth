# Team Setup Guide - qt-brbooth

This guide will help your team members set up the project quickly and efficiently.

## Quick Start (5 minutes)

### 1. Prerequisites Check
Before starting, ensure you have:
- ✅ **Qt 6.5+** installed with MSVC compiler
- ✅ **Visual Studio 2019/2022** (Community edition is fine)
- ✅ **Git** installed
- ✅ **8GB+ RAM** available

### 2. Clone and Setup
```bash
# Clone the repository
git clone <your-repository-url>
cd qt-brbooth

# Open in Qt Creator
qt-creator qt-brbooth.pro
```

### 3. Install OpenCV (One-time setup)
```bash
# Download OpenCV 4.11
# Extract to C:\opencv_build\install
# Add C:\opencv_build\install\x64\vc17\bin to PATH
```

### 4. Build and Run
```bash
# In Qt Creator:
# 1. Configure project (if prompted)
# 2. Build (Ctrl+B)
# 3. Run (Ctrl+R)
```

## Detailed Setup Instructions

### Step 1: Install Qt 6.5+
1. Download Qt 6.5+ from [qt.io](https://www.qt.io/download)
2. During installation, select:
   - **Qt 6.5.x** (latest stable)
   - **MSVC 2019/2022** compiler
   - **Qt Creator** IDE
   - **CMake** and **Ninja** (optional but recommended)

### Step 2: Install Visual Studio
1. Download Visual Studio 2019/2022 Community from Microsoft
2. Install with **C++ development tools**
3. Ensure **MSVC v143** compiler is included

### Step 3: Install OpenCV 4.11

#### Option A: Pre-built (Recommended)
1. Download from [opencv.org/releases](https://opencv.org/releases/)
2. Extract to `C:\opencv_build\install`
3. Add to PATH: `C:\opencv_build\install\x64\vc17\bin`

#### Option B: Build from Source
```bash
# Clone OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.11.0

# Build with CMake
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
cmake --install . --prefix C:/opencv_build/install
```

### Step 4: Configure Environment
1. **Add OpenCV to PATH**:
   - Open System Properties → Environment Variables
   - Add `C:\opencv_build\install\x64\vc17\bin` to PATH

2. **Verify Installation**:
   ```bash
   # Test OpenCV installation
   python -c "import cv2; print(cv2.__version__)"
   ```

### Step 5: Project Setup
1. **Clone Repository**:
   ```bash
   git clone <your-repository-url>
   cd qt-brbooth
   ```

2. **Open in Qt Creator**:
   - Launch Qt Creator
   - Open `qt-brbooth.pro`
   - Configure project if prompted

3. **Verify Configuration**:
   - Check that OpenCV paths are correct in `qt-brbooth.pro`
   - Ensure MSVC compiler is selected

### Step 6: Build and Test
1. **Build Project**:
   ```bash
   # In Qt Creator: Ctrl+B
   # Or command line:
   qmake qt-brbooth.pro
   make
   ```

2. **Run Application**:
   ```bash
   # In Qt Creator: Ctrl+R
   # Or command line:
   ./qt-brbooth
   ```

3. **Test Features**:
   - Camera should open
   - Segmentation should work
   - Background replacement should function

## Troubleshooting

### Common Issues and Solutions

#### 1. "OpenCV not found" Error
```bash
# Check OpenCV installation
ls C:\opencv_build\install\x64\vc17\bin\opencv_core4110d.dll

# Update qt-brbooth.pro if needed:
OPENCV_INSTALL_DIR = C:/opencv_build/install
```

#### 2. "Qt Creator cannot find compiler"
```bash
# Ensure Visual Studio is installed
# Check Qt Creator → Tools → Options → Kits
# Verify MSVC compiler is detected
```

#### 3. "Camera access denied"
```bash
# Check camera permissions
# Ensure no other apps are using camera
# Test camera in other applications
```

#### 4. "Build fails with linking errors"
```bash
# Clean and rebuild
make clean
qmake qt-brbooth.pro
make

# Check OpenCV library paths
# Verify all required DLLs are in PATH
```

#### 5. "Segmentation not working"
```bash
# Check lighting conditions
# Ensure person is clearly visible
# Adjust confidence threshold
# Check camera positioning
```

### Performance Optimization

#### For Better Performance:
1. **Reduce Camera Resolution**:
   - Lower resolution = higher FPS
   - 720p is usually sufficient

2. **Adjust Segmentation Settings**:
   - Use "Speed" performance mode
   - Lower confidence threshold
   - Reduce processing frequency

3. **System Optimization**:
   - Close unnecessary applications
   - Ensure adequate RAM (8GB+)
   - Use SSD for faster loading

## Development Workflow

### Daily Development
```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and test
# 4. Commit changes
git add .
git commit -m "Add your feature description"

# 5. Push to remote
git push origin feature/your-feature-name

# 6. Create pull request
```

### Code Standards
- Use **C++17** features
- Follow **Qt coding conventions**
- Add **proper error handling**
- Include **debug output** for troubleshooting
- Write **clear commit messages**

### Testing Checklist
Before committing:
- ✅ Application builds successfully
- ✅ Camera opens and works
- ✅ Segmentation functions properly
- ✅ Background replacement works
- ✅ Video recording functions
- ✅ No memory leaks
- ✅ Performance is acceptable

## Team Communication

### Git Workflow
- **Main branch**: Always stable and working
- **Feature branches**: For new features
- **Pull requests**: For code review
- **Issues**: For bug reports and feature requests

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code review and discussion
- **README.md**: Project documentation
- **SETUP.md**: This setup guide

## Support

### Getting Help
1. **Check this guide** first
2. **Search existing issues** on GitHub
3. **Create new issue** with detailed description
4. **Ask team members** for assistance

### Useful Resources
- [Qt Documentation](https://doc.qt.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [C++ Reference](https://en.cppreference.com/)
- [Git Documentation](https://git-scm.com/doc)

## Quick Commands Reference

```bash
# Build project
qmake qt-brbooth.pro && make

# Clean build
make clean && qmake qt-brbooth.pro && make

# Run application
./qt-brbooth

# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Check Qt version
qmake -v

# Check compiler
cl
```

## Environment Variables

Add these to your system PATH:
```
C:\opencv_build\install\x64\vc17\bin
C:\Qt\6.5.x\msvc2019_64\bin
```

## Project Structure Overview

```
qt-brbooth/
├── qt-brbooth.pro          # Qt project file
├── README.md               # Project documentation
├── SETUP.md               # This setup guide
├── main.cpp               # Application entry point
├── brbooth.cpp            # Main window
├── capture.cpp            # Camera and segmentation
├── tflite_deeplabv3.cpp   # Segmentation algorithm
├── templates/             # Background/foreground templates
├── pics/                  # Sample images
└── videos/                # Sample videos
```

---

**Need help?** Create an issue on GitHub or ask your team members! 