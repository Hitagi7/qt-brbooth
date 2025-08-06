@echo off
echo ========================================
echo TensorFlow Lite DeepLabv3 Setup Script
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"
echo Working directory: %CD%
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Installing required Python packages...
echo.
pip install tensorflow tensorflow-hub numpy

if errorlevel 1 (
    echo ERROR: Failed to install Python packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Downloading and converting DeepLabv3 model...
python download_tflite_model.py

if errorlevel 1 (
    echo ERROR: Model conversion failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Install TensorFlow Lite C++ library
echo 2. Update TFLITE_DIR path in qt-brbooth.pro if needed
echo 3. Build your Qt project
echo 4. Run the application and use the TFLite Segmentation tab
echo.
echo For detailed instructions, see TFLITE_SETUP.md
echo.
pause 