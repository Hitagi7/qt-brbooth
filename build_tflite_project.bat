@echo off
echo Building Qt BrBooth with TFLite Deeplabv3 Integration...
echo.

REM Set environment variables
set TENSORFLOW_DIR=C:\tensorflow-2.18.1
set OPENCV_DIR=C:\opencv_build\install

echo Checking TensorFlow installation...
if not exist "%TENSORFLOW_DIR%" (
    echo ERROR: TensorFlow directory not found at %TENSORFLOW_DIR%
    echo Please ensure TensorFlow 2.18.1 is installed at the correct location.
    pause
    exit /b 1
)

echo Checking OpenCV installation...
if not exist "%OPENCV_DIR%" (
    echo ERROR: OpenCV directory not found at %OPENCV_DIR%
    echo Please ensure OpenCV is installed at the correct location.
    pause
    exit /b 1
)

echo Checking TFLite model...
if not exist "deeplabv3.tflite" (
    echo WARNING: deeplabv3.tflite model not found in current directory.
    echo The application will show an error when trying to load the model.
    echo.
)

echo.
echo Running qmake...
qmake qt-brbooth.pro

if %ERRORLEVEL% neq 0 (
    echo ERROR: qmake failed
    pause
    exit /b 1
)

echo.
echo Running nmake...
nmake

if %ERRORLEVEL% neq 0 (
    echo ERROR: nmake failed
    echo.
    echo Troubleshooting tips:
    echo 1. Ensure Visual Studio is installed with C++ development tools
    echo 2. Check that TensorFlow headers are available at %TENSORFLOW_DIR%\tensorflow\lite\interpreter.h
    echo 3. Verify OpenCV installation at %OPENCV_DIR%
    echo 4. Make sure Qt is properly installed and qmake is in PATH
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo.
echo To run the application:
echo 1. Copy deeplabv3.tflite to the build directory (if not already present)
echo 2. Run: qt-brbooth.exe
echo.
pause 