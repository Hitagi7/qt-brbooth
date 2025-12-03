# YOLO Model Setup Instructions

This project now uses YOLO (You Only Look Once) for person detection and segmentation instead of green screen segmentation.

## Step 1: Download YOLOv8-seg Model

You need to download a YOLOv8-seg (segmentation) model in ONNX format. Here are the recommended options:

### Option A: Convert from PyTorch (EASIEST METHOD - Recommended)

This is the most reliable method. The Ultralytics package will automatically download the PyTorch model and convert it to ONNX:

1. **Install Python and pip** (if not already installed)

2. **Install Ultralytics package:**
   ```bash
   pip install ultralytics
   ```

3. **Create a Python script to convert the model:**
   
   Create a file called `convert_yolo.py` with this content:
   ```python
   from ultralytics import YOLO
   
   # This will automatically download the PyTorch model if not present
   # Choose one: 'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', etc.
   model = YOLO('yolov8n-seg.pt')  # nano - smallest, fastest
   # model = YOLO('yolov8s-seg.pt')  # small - balanced
   # model = YOLO('yolov8m-seg.pt')  # medium - better accuracy
   
   # Export to ONNX format
   model.export(format='onnx', imgsz=640, opset=11)
   
   print("Conversion complete! Check for yolov8n-seg.onnx in the current directory")
   ```

4. **Run the script:**
   ```bash
   python convert_yolo.py
   ```

5. **The ONNX file will be created** in the same directory as the script (e.g., `yolov8n-seg.onnx`)

### Option B: Download from Hugging Face (Alternative)

If you prefer to download a pre-converted model:

1. **Visit Hugging Face:**
   - Kalray's YOLOv8n-seg: https://huggingface.co/Kalray/yolov8n-seg
   - Click on "Files and versions" tab
   - Download the `.onnx` file

2. **Or use direct download** (if available):
   - Look for the "Download" button next to the `.onnx` file

### Option C: Manual Download from Ultralytics Releases

1. Visit: https://github.com/ultralytics/assets/releases
2. Look for the latest release (e.g., v8.2.0 or newer)
3. Download the `.onnx` file from the assets (if available)
4. Note: Pre-converted ONNX files may not always be available in releases

If you have PyTorch installed, you can convert a PyTorch model to ONNX:

```python
from ultralytics import YOLO

# Load YOLOv8-seg model
model = YOLO('yolov8n-seg.pt')  # or yolov8s-seg.pt, yolov8m-seg.pt, etc.

# Export to ONNX
model.export(format='onnx', imgsz=640)
```

## Step 2: Create Models Directory

The `models` directory has already been created for you. If it doesn't exist, create it:

1. Create a `models` folder in your project root directory (same level as `src`, `include`, etc.)
   ```
   qt-brbooth/
   ├── models/
   │   └── yolov8n-seg.onnx
   ├── src/
   ├── include/
   └── ...
   ```

## Step 3: Place Model File

1. **If you used Option A (Python conversion):**
   - Copy the generated `yolov8n-seg.onnx` file from where you ran the Python script
   - Paste it into the `models/` directory in your project

2. **If you downloaded from Hugging Face or other source:**
   - Copy the downloaded `.onnx` file
   - Rename it to `yolov8n-seg.onnx` (if different)
   - Place it in the `models/` directory

3. **Verify the file location:**
   - The file should be at: `qt-brbooth/models/yolov8n-seg.onnx`
   - The application will automatically look for the model at this path

## Step 4: Verify Setup

1. Build and run the application in Qt Creator
2. Check the console output - you should see:
   - `"YOLO model loaded successfully from: [path]"`
   - If the model is not found, you'll see: `"YOLO model not found at: [path]"` and it will fall back to green screen segmentation

## Model Selection Guide

- **yolov8n-seg.onnx** (Nano): ~6MB, fastest, good for real-time on CPU
- **yolov8s-seg.onnx** (Small): ~22MB, balanced speed/accuracy
- **yolov8m-seg.onnx** (Medium): ~52MB, better accuracy, requires more GPU/CPU
- **yolov8l-seg.onnx** (Large): ~87MB, high accuracy, slower
- **yolov8x-seg.onnx** (Extra Large): ~136MB, highest accuracy, slowest

**Recommendation:** Start with `yolov8n-seg.onnx` for best performance. If you need better accuracy, try `yolov8s-seg.onnx` or `yolov8m-seg.onnx`.

## Troubleshooting

### Model Not Found
- Ensure the `models/` directory exists in the project root
- Check that the model file is named exactly `yolov8n-seg.onnx` (or update the path in code)
- Verify the file is not corrupted

### Model Loading Fails
- Ensure OpenCV is built with DNN module support (should be included by default)
- Check that the ONNX file is valid (try opening it with an ONNX viewer)
- Verify OpenCV version is 4.5.0 or higher

### Performance Issues
- Use a smaller model (nano or small) for better performance
- Ensure GPU acceleration is enabled (OpenCL)
- Reduce input resolution if needed (currently set to 640x640)

## Fallback Behavior

If the YOLO model is not found or fails to load, the application will automatically fall back to green screen segmentation. This ensures the application continues to work even without the YOLO model.

## Notes

- The YOLO model detects "person" class (class 0 in COCO dataset)
- Confidence threshold is set to 0.5 (50%) by default
- Non-maximum suppression threshold is 0.4
- Model input size is 640x640 pixels

