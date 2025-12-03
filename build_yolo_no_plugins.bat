@echo off
REM Set all necessary paths for TensorRT and CUDA
set PATH=C:\TensorRT-8.6.1.6\bin;C:\TensorRT-8.6.1.6\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Build YOLOv8n TensorRT engine WITHOUT plugins
"C:\TensorRT-8.6.1.6\bin\trtexec.exe" --onnx=models\yolov8n.onnx --saveEngine=models\yolov8n_fp16.engine --fp16 --workspace=4096 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --noBuilderCache --verbose

echo.
echo Done! Check models\yolov8n_fp16.engine
pause


