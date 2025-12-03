@echo off
REM Set all necessary paths for TensorRT and CUDA
set PATH=C:\TensorRT-10.1.0.27\bin;C:\TensorRT-10.1.0.27\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Build YOLOv8n TensorRT engine
"C:\TensorRT-10.1.0.27\bin\trtexec.exe" --onnx=models\yolov8n.onnx --saveEngine=models\yolov8n_fp16.engine --fp16 --memPoolSize=workspace:4096 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640

echo.
echo Done! Check models\yolov8n_fp16.engine
pause

