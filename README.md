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

## Bounding Box Features

The application now includes real-time bounding box visualization for detected persons:

### Features:
- **Real-time Detection**: Bounding boxes are drawn around detected persons in real-time
- **Confidence-based Colors**: 
  - 🟢 Green: High confidence (>80%)
  - 🟡 Yellow: Medium confidence (60-80%)
  - 🔴 Red: Low confidence (<60%)
- **Person Numbering**: Each detected person is numbered (Person 1, Person 2, etc.)
- **Confidence Display**: Shows confidence percentage for each detection
- **Box Dimensions**: Displays the width x height of each bounding box
- **Toggle Control**: Checkbox to enable/disable bounding box display
- **Auto-cleanup**: Old detections are automatically cleared after 0.5 seconds

### Usage:
1. Start the camera feed
2. **Press 'B' key** or check "Show Bounding Boxes (B)" to enable visualization
3. Stand in front of the camera to see bounding boxes appear
4. **Press 'B' key** again or uncheck to hide bounding boxes

### Keyboard Shortcuts:
- **B** - Toggle bounding boxes on/off
- Shows on-screen notification when toggled

### Technical Details:
- Uses Qt's QPainter for efficient drawing
- Thread-safe detection storage with QMutex
- Asynchronous YOLO processing to maintain UI responsiveness
- Automatic scaling with camera resolution 