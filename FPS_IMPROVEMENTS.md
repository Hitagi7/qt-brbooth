# FPS Measurement Improvements

## What Was Changed

### Old Method (Inaccurate)
- Used instant frame-to-frame timing
- Single measurement per frame
- Very noisy and inaccurate
- Could show wildly fluctuating values

### New Method (Accurate)
- **Time-based frame counting**: Counts frames over a 1-second window
- **Exponential smoothing**: Uses EMA (Exponential Moving Average) with alpha=0.1 for stable readings
- **More accurate**: Reflects actual frame rate over time, not just instant measurements
- **Stable display**: Smooth, consistent FPS readings without wild fluctuations

## How It Works

1. **Frame Counting**: Counts every frame that gets displayed
2. **Time Window**: Measures FPS over a 1-second window (1000ms)
3. **Calculation**: `FPS = (frame_count / elapsed_time_ms) * 1000`
4. **Smoothing**: Applies exponential moving average: `newFPS = 0.1 * actual + 0.9 * old`
5. **Display**: Updates every second with smoothed value

## Third-Party Tools for Verification

### Recommended FPS Monitoring Tools:

1. **MSI Afterburner** (Free)
   - Real-time FPS overlay
   - GPU/CPU monitoring
   - Very accurate and widely used
   - Download: https://www.msi.com/Landing/afterburner

2. **Fraps** (Free/Paid)
   - Simple FPS counter
   - Benchmarking tools
   - Download: https://fraps.com/

3. **RivaTuner Statistics Server** (Free)
   - Part of MSI Afterburner
   - Highly customizable overlay
   - Very accurate measurements
   - Download: Included with MSI Afterburner

4. **NVIDIA FrameView** (Free - NVIDIA only)
   - Official NVIDIA tool
   - Accurate frame timing
   - Download: https://developer.nvidia.com/frameview

5. **PresentMon** (Free - Open Source)
   - Microsoft's frame timing tool
   - Very accurate
   - Command-line or GUI versions
   - Download: https://github.com/GameTechDev/PresentMon

### How to Use for Verification:

1. **Enable overlay** in your chosen tool
2. **Run your application** in Qt Creator
3. **Compare FPS readings**:
   - Your app's FPS (shown in debug display)
   - Third-party tool's FPS
   - They should be very close now with the new algorithm

## Technical Details

### Exponential Moving Average (EMA)
- **Alpha (α) = 0.1**: 10% weight to new measurement, 90% to previous
- **Formula**: `EMA = α × new_value + (1 - α) × previous_EMA`
- **Benefits**: Smooths out spikes while still responding to real changes

### Why This Is More Accurate

1. **Time-based**: Measures actual frames per second over time
2. **Averaging**: Reduces noise from frame timing variations
3. **Window-based**: Uses 1-second window (standard for FPS measurement)
4. **Smoothing**: Prevents display flicker from rapid changes

## Expected Results

- **More stable FPS readings**: No wild fluctuations
- **Accurate values**: Should match third-party tools closely
- **Better performance insight**: Reflects actual application performance
- **Smoother display**: FPS counter updates smoothly every second

## Debug Output

The new algorithm logs every 5 seconds:
```
FPS: Actual=30.5 Smoothed=30.2 Displayed=30
```

This shows:
- **Actual**: Raw FPS from current measurement window
- **Smoothed**: EMA-smoothed value
- **Displayed**: Rounded integer shown to user

