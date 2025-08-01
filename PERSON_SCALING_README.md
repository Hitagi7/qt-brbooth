# Person Scaling Functionality

This document describes the person scaling functionality implemented in the Capture class.

## Overview

The scaling functionality allows users to scale detected persons in real-time using a vertical slider. When a person is detected (via HOG or other detection methods), the slider can be used to make the person appear smaller in the camera feed.

**Note**: HOG person detection is now implemented and working automatically.

## Features

- **Real-time scaling**: Person scaling is applied in real-time as the slider is moved
- **Range**: Scale factor ranges from 0.5x (50% smaller) to 1.0x (normal size)
- **Scale down only**: Users can only make the person smaller, not larger
- **Centered scaling**: The person is scaled from their center point to maintain natural appearance
- **Boundary protection**: Scaled regions are automatically adjusted to stay within image bounds
- **Capture integration**: Scaling is applied to both live feed and captured images

## How It Works

### Slider Mapping (Inverted Slider with Tick Intervals)
The slider moves in 10-unit tick intervals for precise control:

- **Slider at top (value 100)**: 1.0x scale (normal size, no scaling)
- **Slider at 90**: 0.95x scale (person appears 5% smaller)
- **Slider at 80**: 0.9x scale (person appears 10% smaller)
- **Slider at 70**: 0.85x scale (person appears 15% smaller)
- **Slider at 60**: 0.8x scale (person appears 20% smaller)
- **Slider at 50**: 0.75x scale (person appears 25% smaller)
- **Slider at 40**: 0.7x scale (person appears 30% smaller)
- **Slider at 30**: 0.65x scale (person appears 35% smaller)
- **Slider at 20**: 0.6x scale (person appears 40% smaller)
- **Slider at 10**: 0.55x scale (person appears 45% smaller)
- **Slider at bottom (value 0)**: 0.5x scale (person appears 50% smaller)

### Technical Implementation

1. **Person Detection**: The system uses OpenCV's HOG detector to find people in each frame
2. **Best Detection Selection**: The largest detected person is selected for scaling
3. **Scaling Application**: When the slider changes, the `applyPersonScaling()` method is called
4. **Image Processing**: The person region is extracted, scaled, and composited back into the image
5. **Real-time Updates**: Scaling is applied to each frame when a person is detected

## Usage

### How to Use

1. **Run the application** and navigate to the Capture screen
2. **Stand in front of the camera** (or have someone stand there)
3. **Move the vertical slider** to see the scaling effect in real-time
4. **HOG detection works automatically** - no buttons needed

The system automatically detects people using HOG and applies scaling based on the slider position.

### Integration with Person Detection
When implementing HOG or other person detection methods:

1. **Update detection results**: Call `setTestPersonDetection()` with the detected person's bounding box
2. **Automatic scaling**: The slider will automatically apply scaling to the detected person
3. **Real-time feedback**: Users can adjust the slider to see immediate scaling effects

#### HOG Integration (Implemented)
HOG person detection is now fully implemented:

```cpp
// HOG detection happens automatically in updateCameraFeed()
// The system automatically:
// 1. Detects people using OpenCV's HOG detector
// 2. Selects the largest detection as the best person
// 3. Applies scaling based on slider position
// 4. Updates the display in real-time
```

## Key Methods

### `detectPersonWithHOG(const QImage& image)`
- Uses OpenCV's HOG detector to find people in the image
- Converts QImage to cv::Mat for processing
- Calls `findBestPersonDetection()` to select the best person
- Updates person detection state and bounding box

### `findBestPersonDetection(const std::vector<cv::Rect>& detections)`
- Selects the largest detected person (most prominent)
- Converts cv::Rect to QRect for Qt compatibility
- Returns empty QRect if no detections found

### `applyPersonScaling(const QPixmap& originalPixmap, const QRect& personRect, double scaleFactor)`
- Extracts the person region from the original image
- Scales the region according to the scale factor
- Centers the scaled region on the original person's center
- Ensures the scaled region stays within image bounds
- Returns the modified pixmap

### `on_verticalSlider_valueChanged(int value)`
- Converts slider value (0-100) to scale factor (0.5-1.0)
- Applies scaling immediately if a person is detected
- Provides real-time feedback

## Future Enhancements

1. **Multiple Person Support**: Scale multiple detected persons independently
2. **Smooth Transitions**: Add animation for scaling changes
3. **Preset Scaling**: Add preset scaling levels (e.g., "Small", "Medium", "Large")
4. **Region Selection**: Allow users to manually select regions to scale
5. **Undo/Redo**: Add ability to undo scaling changes

## Notes

- The scaling functionality is designed to work with any person detection method
- Currently optimized for single person detection (highest confidence)
- Scaling is applied after overlay composition but before display
- Performance impact is minimal due to efficient region-based processing 