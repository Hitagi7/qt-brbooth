# qt-brbooth with YOLOv5 Person Detection

## Setup

1. **Install Python 3.x and pip.**
   - Download from https://www.python.org/downloads/
   - During installation, check the box to "Add Python to PATH".

2. **Install dependencies:**
   ```sh
   pip install -r yolov5/requirements.txt
   ```

3. **Build the Qt project as usual in Qt Creator.**

## Usage

- The app will call the Python script in `yolov5/detect.py` for person detection.
- Make sure `yolov5n.pt` is present in the `yolov5/` folder.
- Detection results are output as JSON to stdout for easy parsing in Qt/C++.

## How Integration Works

- Qt/C++ uses `QProcess` to run the Python script:
  ```sh
  python yolov5/detect.py --weights yolov5/yolov5n.pt --source <image_path> --classes 0 --nosave
  ```
- The script outputs detection results as JSON, e.g.:
  ```json
  [
    {
      "image": "image1.jpg",
      "detections": [
        {"bbox": [x1, y1, x2, y2], "confidence": 0.95},
        ...
      ]
    }
  ]
  ```
- Qt parses this output and can display bounding boxes or use the results as needed.

## Notes
- Only the "person" class (class 0) is detected.
- You can train your own model or use the pre-trained YOLOv5n weights.
- For custom training, see the YOLOv5 documentation. 