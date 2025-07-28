# YOLOv5n Model Downloader for Windows PowerShell
# This script downloads the YOLOv5n model for the Qt BRBooth project

Write-Host "=== YOLOv5n Model Downloader ===" -ForegroundColor Green

# Create models directory
if (!(Test-Path "models")) {
    New-Item -ItemType Directory -Path "models"
    Write-Host "Created models directory" -ForegroundColor Yellow
}

$modelPath = "models/yolov5nu.onnx"

# Check if model already exists
if (Test-Path $modelPath) {
    Write-Host "Model already exists at $modelPath" -ForegroundColor Yellow
    $response = Read-Host "Do you want to re-download? (y/N)"
    if ($response -ne "y") {
        Write-Host "Skipping download." -ForegroundColor Yellow
        exit
    }
}

# Try to download the model
Write-Host "Downloading YOLOv5n model..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt" -OutFile $modelPath
    Write-Host "✅ YOLOv5n model downloaded successfully!" -ForegroundColor Green
    Write-Host "Model location: $((Get-Location).Path)/$modelPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Build your Qt project: qmake && make" -ForegroundColor White
    Write-Host "2. Run your application - it will automatically use the C++ YOLOv5 detector" -ForegroundColor White
}
catch {
    Write-Host "❌ Failed to download YOLOv5n model." -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download options:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://github.com/ultralytics/assets/releases" -ForegroundColor White
    Write-Host "2. Download yolov5n.pt from the latest release" -ForegroundColor White
    Write-Host "3. Place it in the models/ directory" -ForegroundColor White
    Write-Host "4. Rename it to yolov5nu.onnx" -ForegroundColor White
} 