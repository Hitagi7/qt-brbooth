# MVC Reorganization Script for qt-brbooth
# This script reorganizes the codebase into MVC architecture

Write-Host "Starting MVC reorganization..."

# Create directory structure
$dirs = @(
    "src\models\algorithms\segmentation",
    "src\models\algorithms\edge_blending", 
    "src\models\algorithms\lighting",
    "src\models\algorithms\green_screen",
    "src\models\algorithms\hand_detection",
    "src\models\data",
    "src\models\camera",
    "src\models\gpu",
    "src\views\ui",
    "src\controllers\capture",
    "src\controllers\main",
    "include\models\algorithms\segmentation",
    "include\models\algorithms\edge_blending",
    "include\models\algorithms\lighting", 
    "include\models\algorithms\green_screen",
    "include\models\algorithms\hand_detection",
    "include\models\data",
    "include\models\camera",
    "include\models\gpu",
    "include\views\ui",
    "include\controllers\capture",
    "include\controllers\main"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "Directories created."

# Move algorithm files
Write-Host "Moving algorithm files..."
Move-Item -Path "src\core\capture_segmentation.cpp" -Destination "src\models\algorithms\segmentation\segmentation.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\capture_edge_blending.cpp" -Destination "src\models\algorithms\edge_blending\edge_blending.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\capture_lighting.cpp" -Destination "src\models\algorithms\lighting\lighting.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\capture_green_screen.cpp" -Destination "src\models\algorithms\green_screen\green_screen.cpp" -Force -ErrorAction SilentlyContinue

# Move algorithm headers
if (Test-Path "include\core\capture_edge_blending.h") {
    Move-Item -Path "include\core\capture_edge_blending.h" -Destination "include\models\algorithms\edge_blending\edge_blending.h" -Force
}

# Move lighting corrector
if (Test-Path "src\algorithms\lighting_correction\lighting_corrector.cpp") {
    Move-Item -Path "src\algorithms\lighting_correction\lighting_corrector.cpp" -Destination "src\models\algorithms\lighting\lighting_corrector.cpp" -Force
}
if (Test-Path "include\algorithms\lighting_correction\lighting_corrector.h") {
    Move-Item -Path "include\algorithms\lighting_correction\lighting_corrector.h" -Destination "include\models\algorithms\lighting\lighting_corrector.h" -Force
}

# Move hand detection
if (Test-Path "src\algorithms\hand_detection\trt_hand_landmarker.cpp") {
    Move-Item -Path "src\algorithms\hand_detection\trt_hand_landmarker.cpp" -Destination "src\models\algorithms\hand_detection\trt_hand_landmarker.cpp" -Force
}
if (Test-Path "include\algorithms\hand_detection\trt_hand_landmarker.h") {
    Move-Item -Path "include\algorithms\hand_detection\trt_hand_landmarker.h" -Destination "include\models\algorithms\hand_detection\trt_hand_landmarker.h" -Force
}

# Move model files
Write-Host "Moving model files..."
Move-Item -Path "src\core\camera.cpp" -Destination "src\models\camera\camera.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\camera.h" -Destination "include\models\camera\camera.h" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\gpu_memory_pool.cpp" -Destination "src\models\gpu\gpu_memory_pool.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\system_monitor.cpp" -Destination "src\models\system_monitor.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\system_monitor.h" -Destination "include\models\system_monitor.h" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\common_types.h" -Destination "include\models\data\common_types.h" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\videotemplate.h" -Destination "include\models\data\videotemplate.h" -Force -ErrorAction SilentlyContinue

# Move view files
Write-Host "Moving view files..."
Get-ChildItem -Path "src\ui\*.cpp" -ErrorAction SilentlyContinue | Move-Item -Destination "src\views\ui\" -Force
Get-ChildItem -Path "include\ui\*.h" -ErrorAction SilentlyContinue | Move-Item -Destination "include\views\ui\" -Force

# Move controller files
Write-Host "Moving controller files..."
Move-Item -Path "src\core\capture.cpp" -Destination "src\controllers\capture\capture.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\capture.h" -Destination "include\controllers\capture\capture.h" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\capture_dynamic.cpp" -Destination "src\controllers\capture\capture_dynamic.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "src\core\brbooth.cpp" -Destination "src\controllers\main\brbooth.cpp" -Force -ErrorAction SilentlyContinue
Move-Item -Path "include\core\brbooth.h" -Destination "include\controllers\main\brbooth.h" -Force -ErrorAction SilentlyContinue

Write-Host "File reorganization complete!"
Write-Host "Next: Update include paths in all files and update qt-brbooth.pro"
