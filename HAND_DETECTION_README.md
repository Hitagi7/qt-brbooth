# Hand Detection Feature

## Overview
This feature adds real-time hand detection to the Qt photo booth application. When enabled, it will detect raised hands in the camera feed and display green bounding boxes around them.

## Features
- **Real-time hand detection** using color-based skin detection
- **Bounding box visualization** with confidence percentage
- **Raised hand detection** - specifically detects hands in the upper portion of the frame
- **Camera mirroring** - camera feed is automatically mirrored for natural interaction
- **Performance monitoring** - shows hand detection FPS in debug display
- **Toggle controls** - can be enabled/disabled via UI or keyboard shortcut
- **Debug tools** - save debug frames and skin masks for troubleshooting

## How to Use

### Keyboard Shortcuts
- **H key** - Toggle hand detection on/off
- **S key** - Toggle segmentation on/off
- **D key** - Toggle debug display visibility
- **F12 key** - Save debug frame for troubleshooting
- **Space** - Capture image/start recording
- **Escape** - Go back to previous page

### UI Controls
- **Hand Detection Button** - Located in the debug widget (top-left corner)
- **Confidence Threshold** - Adjustable via code (currently set to 0.5)

## Technical Details

### Detection Method
The hand detection uses a **hand-specific approach** with the following steps:

1. **Color Space Conversion** - Converts BGR to HSV for better skin color detection
2. **Region of Interest (ROI)** - Focuses on upper 60% of image to exclude faces
3. **Skin Color Masking** - Uses HSV ranges to identify skin-colored pixels
4. **Morphological Operations** - Cleans up the mask using opening and closing operations
5. **Contour Detection** - Finds hand-shaped contours in the skin mask
6. **Hand-Specific Filtering** - Applies filters based on:
   - **Area** (500-50000 pixels) - hands are smaller than faces
   - **Aspect ratio** (0.8-3.0) - hands are taller than wide
   - **Circularity** (< 0.6) - hands are less round than faces
   - **Solidity** (< 0.85) - hands have finger gaps
   - **Convexity defects** (2-8) - hands have finger separations
   - **Position** (upper 60% of frame) - raised hands only
   - **Center position** (above image center) - excludes centered faces

### Performance
- **Real-time processing** - runs on every camera frame
- **Thread-safe** - uses mutex protection for shared data
- **Optimized** - minimal impact on overall application performance

### Configuration
The detection parameters can be adjusted in `handdetector.cpp`:

```cpp
// Skin color range in HSV
cv::Scalar lowerSkin(0, 20, 70);
cv::Scalar upperSkin(20, 255, 255);

// Area filters
if (area < 1000 || area > 50000) continue;

// Aspect ratio filter
if (aspectRatio < 0.8 || aspectRatio > 2.0) continue;

// Position filter (raised hand)
if (handTop > imageHeight * 0.4) continue;
```

## Files Added/Modified

### New Files
- `handdetector.h` - Hand detection class header
- `handdetector.cpp` - Hand detection implementation
- `HAND_DETECTION_README.md` - This documentation

### Modified Files
- `capture.h` - Added hand detection members and methods
- `capture.cpp` - Integrated hand detection into camera feed
- `qt-brbooth.pro` - Added handdetector files to build system

## Future Enhancements
- **MediaPipe integration** for more accurate hand landmark detection
- **Gesture recognition** for specific hand poses
- **Left/right hand distinction**
- **Async processing** for better performance
- **Customizable detection parameters** via UI

## Troubleshooting

### Hand Not Detected
1. Ensure good lighting conditions
2. Check that your hand is in the upper portion of the frame
3. Verify hand detection is enabled (H key or UI button)
4. Press F12 to save a debug frame and check the console output
5. Look for debug images: `debug_skin_mask.png`, `debug_original_frame.png`, `debug_hsv_image.png`, `debug_roi_mask.png`
6. Check console for detection messages and confidence values
7. Try different hand positions and distances from camera
8. The system now uses hand-specific filtering to exclude faces
9. Ensure your hand is in the upper portion of the frame (above center)

### Performance Issues
1. Reduce camera resolution if needed
2. Disable segmentation if not needed
3. Check debug display for FPS information

### Build Issues
1. Ensure OpenCV is properly installed
2. Verify all files are included in the .pro file
3. Check that OpenCV includes the required modules (imgproc, highgui, etc.)
