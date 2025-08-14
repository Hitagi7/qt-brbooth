#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <opencv2/opencv.hpp>
#include <QString>

/**
 * @brief Common data structures shared across the application
 * 
 * This file contains struct definitions that are used by multiple classes
 * and need to be available to Qt's MOC system for signals/slots.
 */

struct BoundingBox {
    int x1, y1, x2, y2;
    double confidence;
    
    BoundingBox() : x1(0), y1(0), x2(0), y2(0), confidence(0.0) {}
    BoundingBox(int x1_, int y1_, int x2_, int y2_, double conf = 1.0) 
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), confidence(conf) {}
        
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

struct OptimizedDetection {
    cv::Rect boundingBox;
    cv::Mat mask;           // Direct segmentation mask from model
    double confidence;
    QString className;
    
    OptimizedDetection() : confidence(0.0), className("person") {}
};

#endif // COMMON_TYPES_H