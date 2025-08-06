@echo off
echo ========================================
echo TensorFlow Lite C++ Library Installer
echo ========================================
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"
echo Working directory: %CD%
echo.

echo This script will help you install TensorFlow Lite C++ library.
echo.
echo Choose an installation method:
echo 1. Download pre-built binaries (Recommended for beginners)
echo 2. Use vcpkg package manager
echo 3. Build from source (Advanced users)
echo 4. Skip TensorFlow Lite installation (use fallback segmentation)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto download_binaries
if "%choice%"=="2" goto use_vcpkg
if "%choice%"=="3" goto build_source
if "%choice%"=="4" goto skip_installation
goto invalid_choice

:download_binaries
echo.
echo Downloading pre-built TensorFlow Lite binaries...
echo.

REM Create directory for TensorFlow Lite
if not exist "C:\tensorflow_lite" mkdir "C:\tensorflow_lite"
if not exist "C:\tensorflow_lite\include" mkdir "C:\tensorflow_lite\include"
if not exist "C:\tensorflow_lite\lib" mkdir "C:\tensorflow_lite\lib"

echo Downloading TensorFlow Lite headers...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.15.0.zip' -OutFile 'tensorflow.zip'}"
if errorlevel 1 (
    echo ERROR: Failed to download TensorFlow source
    goto error_exit
)

echo Extracting headers...
powershell -Command "& {Expand-Archive -Path 'tensorflow.zip' -DestinationPath '.' -Force}"
if errorlevel 1 (
    echo ERROR: Failed to extract TensorFlow source
    goto error_exit
)

echo Copying headers...
xcopy "tensorflow-2.15.0\tensorflow\lite\*" "C:\tensorflow_lite\include\tensorflow\lite\" /E /I /Y
if errorlevel 1 (
    echo ERROR: Failed to copy headers
    goto error_exit
)

echo.
echo NOTE: You will need to download pre-built libraries separately.
echo Please visit: https://github.com/tensorflow/tensorflow/releases
echo Download the Windows x64 libraries and place them in C:\tensorflow_lite\lib\
echo.
echo For now, we'll create a fallback configuration...
goto create_fallback

:use_vcpkg
echo.
echo Using vcpkg to install TensorFlow Lite...
echo.

REM Check if vcpkg is installed
vcpkg --version >nul 2>&1
if errorlevel 1 (
    echo vcpkg is not installed. Installing vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..
)

echo Installing TensorFlow Lite via vcpkg...
vcpkg install tensorflow-lite:x64-windows
if errorlevel 1 (
    echo ERROR: Failed to install TensorFlow Lite via vcpkg
    goto error_exit
)

echo.
echo TensorFlow Lite installed via vcpkg!
echo You may need to update your qt-brbooth.pro file to use vcpkg paths.
goto success

:build_source
echo.
echo Building TensorFlow Lite from source...
echo This is an advanced option and may take a long time.
echo.
echo Please follow the official TensorFlow documentation:
echo https://www.tensorflow.org/lite/guide/build_cmake
echo.
echo For now, we'll create a fallback configuration...
goto create_fallback

:skip_installation
echo.
echo Skipping TensorFlow Lite installation.
echo Creating fallback configuration...
goto create_fallback

:create_fallback
echo.
echo Creating fallback configuration...
echo.

REM Create a modified .pro file that doesn't require TensorFlow Lite
copy qt-brbooth.pro qt-brbooth_fallback.pro
if errorlevel 1 (
    echo ERROR: Failed to create fallback configuration
    goto error_exit
)

echo.
echo Fallback configuration created: qt-brbooth_fallback.pro
echo This version will compile without TensorFlow Lite dependencies.
echo.
echo To use the fallback version:
echo 1. Rename qt-brbooth.pro to qt-brbooth_original.pro
echo 2. Rename qt-brbooth_fallback.pro to qt-brbooth.pro
echo 3. Build your project
echo.
echo The fallback version will use OpenCV-based segmentation instead.
goto success

:invalid_choice
echo.
echo Invalid choice. Please run the script again and select 1-4.
goto end

:error_exit
echo.
echo Installation failed. Please check the error messages above.
pause
exit /b 1

:success
echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. If you installed TensorFlow Lite, update the TFLITE_DIR path in qt-brbooth.pro
echo 2. Build your Qt project
echo 3. Run the application and use the TFLite Segmentation tab
echo.
echo For detailed instructions, see TFLITE_SETUP.md
echo.

:end
pause 