#!/usr/bin/env python3
"""
Simple YOLOv5 person detection using PyTorch model
"""

import sys
import os
import torch
import cv2
import numpy as np

def detect_person(image_path, model_path):
    """
    Detect person in image using YOLOv5 PyTorch model
    Returns: True if person detected, False otherwise
    """
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.eval()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not load image: {image_path}")
            return False
        
        # Run inference
        results = model(image)
        
        # Check for person class (class 0 in COCO dataset)
        detections = results.xyxy[0]  # Get detections
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0 and conf > 0.3:  # Person class with confidence > 0.3
                print(f"Person detected with confidence: {conf:.2f}")
                return True
        
        print("No person detected")
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python yolo_detect.py <image_path> <model_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    person_detected = detect_person(image_path, model_path)
    
    # Exit with code 0 if person detected, 1 if not
    sys.exit(0 if person_detected else 1) 