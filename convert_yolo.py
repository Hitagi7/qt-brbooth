#!/usr/bin/env python3
"""
YOLOv8-seg Model Converter
Converts YOLOv8-segmentation PyTorch models to ONNX format for use with OpenCV DNN.

Usage:
    python convert_yolo.py

This will download the PyTorch model (if needed) and convert it to ONNX format.
The output file will be saved in the current directory.
"""

from ultralytics import YOLO
import os
import sys

def convert_model(model_name='yolov8n-seg.pt', output_name=None):
    """
    Convert YOLOv8-segmentation model from PyTorch to ONNX format.
    
    Args:
        model_name: Name of the PyTorch model (e.g., 'yolov8n-seg.pt')
        output_name: Optional output filename (defaults to model_name with .onnx extension)
    """
    print(f"Loading YOLOv8 model: {model_name}")
    print("Note: This will download the model if it's not already cached.")
    
    try:
        # Load the model (will download if needed)
        model = YOLO(model_name)
        
        # Determine output filename
        if output_name is None:
            output_name = model_name.replace('.pt', '.onnx')
        
        print(f"\nConverting to ONNX format...")
        print(f"Output file: {output_name}")
        
        # Export to ONNX
        # imgsz=640: Input image size (640x640 pixels)
        # opset=11: ONNX opset version (compatible with OpenCV)
        model.export(format='onnx', imgsz=640, opset=11)
        
        # Check if file was created
        if os.path.exists(output_name):
            file_size = os.path.getsize(output_name) / (1024 * 1024)  # Size in MB
            print(f"\n✓ Success! ONNX model created: {output_name}")
            print(f"  File size: {file_size:.2f} MB")
            print(f"\nNext steps:")
            print(f"  1. Copy '{output_name}' to your qt-brbooth/models/ directory")
            print(f"  2. Ensure it's named 'yolov8n-seg.onnx' (or update the path in code)")
            return True
        else:
            print(f"\n✗ Error: Output file {output_name} was not created")
            return False
            
    except ImportError:
        print("\n✗ Error: ultralytics package not found!")
        print("Please install it with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8-seg Model Converter")
    print("=" * 60)
    print("\nAvailable models:")
    print("  - yolov8n-seg.pt  (nano - smallest, fastest)")
    print("  - yolov8s-seg.pt  (small - balanced)")
    print("  - yolov8m-seg.pt  (medium - better accuracy)")
    print("  - yolov8l-seg.pt  (large - high accuracy)")
    print("  - yolov8x-seg.pt  (extra large - highest accuracy)")
    print("\n" + "=" * 60)
    
    # Default to nano model (smallest and fastest)
    model_name = 'yolov8n-seg.pt'
    
    # Allow command-line argument for different model
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if not model_name.endswith('.pt'):
            model_name += '.pt'
    
    print(f"\nConverting: {model_name}")
    print("(This may take a few minutes on first run as it downloads the model)\n")
    
    success = convert_model(model_name)
    
    sys.exit(0 if success else 1)

