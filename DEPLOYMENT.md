# Deployment Guide - Creating a Distributable .exe

This guide explains how to create a standalone .exe file that can run on other Windows machines without requiring Qt, OpenCV, or CUDA to be installed.

## Prerequisites

1. Build your project in **Release mode** with **MSVC compiler** using Qt Creator
2. The .exe will be located at: `build/Desktop_Qt_6_9_2_MSVC2022_64bit-Release/release/qt-brbooth.exe`

## Step-by-Step Deployment Process

### Step 1: Build in Release Mode with MSVC
1. Open Qt Creator
2. Select **MSVC2022 64-bit** kit (if not already selected)
3. Select **Release** build configuration
4. Build the project (Ctrl+B or Build → Build Project)
5. Verify the .exe exists in the release folder

### Step 2: Create Deployment Directory
Create a folder where you'll bundle everything (e.g., `qt-brbooth-release/`)

### Step 3: Copy Qt DLLs
Use Qt's `windeployqt` tool to automatically copy required Qt DLLs.

**Using Command Prompt:**
1. Open Command Prompt (cmd.exe)
2. Navigate to your release folder:
   ```
   cd build\Desktop_Qt_6_9_2_MSVC2022_64bit-Release\release
   ```
3. Run windeployqt:
   ```
   C:\Qt\6.9.2\msvc2022_64\bin\windeployqt.exe qt-brbooth.exe --release --compiler-runtime
   ```

**Note:** Adjust the Qt path if your installation is different. The `windeployqt.exe` tool is located in your Qt installation:
- Example: `C:\Qt\6.9.2\msvc2022_64\bin\windeployqt.exe`

This copies all Qt DLLs needed by your application.

### Step 4: Copy OpenCV DLLs
Copy the OpenCV DLL from your OpenCV installation:

**From:** `C:\opencv_cuda\opencv_cuda_build\install\x64\vc17\bin\opencv_world4110.dll`
**To:** Your deployment directory

**Note:** For Debug builds, use `opencv_world4110d.dll`. For Release builds, use `opencv_world4110.dll`.

### Step 5: Copy CUDA Runtime DLLs
Copy required CUDA DLLs from your CUDA installation:

**From:** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\`
**Required DLLs:**
- `cudart64_130.dll`
- `cublas64_130.dll` (if used)
- `cufft64_130.dll` (if used)
- `curand64_130.dll` (if used)
- `cusolver64_130.dll` (if used)
- `cusparse64_130.dll` (if used)

**To:** Your deployment directory

### Step 6: Copy cuDNN DLLs
Copy cuDNN DLLs from your cuDNN installation:

**From:** `C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0\`
**Required DLLs:**
- `cudnn64_9.dll`
- `cudnn_adv_infer64_9.dll`
- `cudnn_adv_train64_9.dll`
- `cudnn_cnn_infer64_9.dll`
- `cudnn_cnn_train64_9.dll`
- `cudnn_ops_infer64_9.dll`
- `cudnn_ops_train64_9.dll`

**To:** Your deployment directory

### Step 7: Copy TensorRT DLLs (if applicable)
If you're using TensorRT for hand detection, copy TensorRT DLLs:

**From:** Your TensorRT installation `bin/` directory
**To:** Your deployment directory

### Step 8: Copy Application Resources
Copy the following directories and files to your deployment directory:

```
qt-brbooth-release/
├── qt-brbooth.exe
├── [Qt DLLs] (copied by windeployqt)
├── [OpenCV DLLs]
├── [CUDA DLLs]
├── [cuDNN DLLs]
├── templates/
│   ├── background/
│   ├── foreground/
│   ├── dynamic/
│   └── hand_gestures/
├── pics/
├── gif templates/
├── Icons/
├── Fonts/
└── models/
    └── hand_landmark_fp16.engine
```

### Step 9: Copy Visual C++ Runtime
The application requires Microsoft Visual C++ Runtime. You have two options:

**Option A:** Install VC++ Redistributable on target machines
- Download from Microsoft: https://aka.ms/vs/17/release/vc_redist.x64.exe

**Option B:** Bundle runtime DLLs (if using `--compiler-runtime` flag with windeployqt)

### Step 10: Test the Deployment
1. Copy the entire deployment folder to a clean Windows machine (without Qt/OpenCV/CUDA)
2. Ensure the target machine has:
   - NVIDIA GPU with CUDA-capable drivers
   - Visual C++ Runtime installed
3. Run `qt-brbooth.exe`

## Important Notes

1. **GPU Requirements:** Target machines must have NVIDIA GPUs with CUDA-capable drivers installed
2. **CUDA Driver Version:** Ensure target machines have CUDA drivers compatible with CUDA 13.0
3. **File Paths:** The application expects resources in relative paths (templates/, pics/, etc.)
4. **License:** Ensure you have proper licenses for redistributing Qt, OpenCV, CUDA, and cuDNN DLLs

## Troubleshooting

### "Missing DLL" Errors
- Use Dependency Walker or `dumpbin /dependents qt-brbooth.exe` to identify missing DLLs
- Ensure all DLLs are in the same directory as the .exe

### "CUDA not found" Errors
- Verify CUDA drivers are installed on target machine
- Check CUDA driver version compatibility

### Resource Not Found Errors
- Ensure all resource directories (templates/, pics/, etc.) are copied
- Check relative paths match your application's expectations

## Creating an Installer (Optional)

For professional distribution, consider using:
- **NSIS** (Nullsoft Scriptable Install System) - Free
- **Inno Setup** - Free
- **Qt Installer Framework** - Free (from Qt)

These tools can create a single installer .exe that bundles everything.

