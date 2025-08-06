# TensorFlow Lite Setup Guide

This guide will help you set up TensorFlow Lite for the Qt BRBooth project.

## Quick Start (Recommended)

1. **Run the setup script**:
   ```bash
   setup_tflite.bat
   ```
   This will install Python dependencies and download the TFLite model.

2. **Install TensorFlow Lite C++ library**:
   ```bash
   install_tflite_cpp.bat
   ```
   Choose option 1 (download pre-built binaries) for easiest setup.

3. **Build the project**:
   ```bash
   qmake
   make
   ```

## Installation Options

### Option 1: Pre-built Binaries (Easiest)
- Download from: https://github.com/tensorflow/tensorflow/releases
- Extract to `C:\tensorflow_lite\`
- Update `TFLITE_DIR` in `qt-brbooth.pro` if needed

### Option 2: vcpkg Package Manager
```bash
vcpkg install tensorflow-lite:x64-windows
```

### Option 3: Build from Source
Follow: https://www.tensorflow.org/lite/guide/build_cmake

## Fallback Mode

If TensorFlow Lite is not available, the project will:
- Compile without TFLite dependencies
- Use OpenCV-based segmentation instead
- Show a debug message: "TensorFlow Lite not available. Using fallback segmentation."

## Troubleshooting

### Error: Cannot open include file 'tensorflow/lite/interpreter.h'
**Solution**: Install TensorFlow Lite C++ library using one of the methods above.

### Error: No matching distribution found for tensorflow
**Solution**: The script now uses the latest compatible version for your Python version.

### Error: Model conversion failed
**Solution**: The script includes fallback model creation that should work with any TensorFlow version.

## Project Structure

- `tflite_deeplabv3.h/cpp`: Core TFLite segmentation logic
- `tflite_segmentation_widget.h/cpp`: Qt UI for TFLite segmentation
- `setup_tflite.bat`: Python setup script
- `install_tflite_cpp.bat`: C++ library installer
- `download_tflite_model.py`: Model download and conversion

## Configuration

The project automatically detects TensorFlow Lite availability:
- If found: Enables TFLite segmentation with tabs
- If not found: Uses fallback mode with original video layout

## Next Steps

1. Run the setup scripts
2. Build the project
3. Launch the application
4. Go to the "Dynamic" section
5. Use the "TFLite Segmentation" tab (if available) 