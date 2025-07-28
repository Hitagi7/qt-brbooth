#include "yolov5_detector.h"
#include <iostream>
#include <fstream>

YoloV5Detector::YoloV5Detector(const std::string& modelPath, float confThreshold, float nmsThreshold)
    : confThreshold(confThreshold), nmsThreshold(nmsThreshold), inputWidth(640), inputHeight(640), modelLoaded(false)
{
    std::cout << "YoloV5Detector constructor called with model path: " << modelPath << std::endl;
    
    try {
        // Check if file exists
        std::ifstream file(modelPath);
        if (!file.good()) {
            std::cerr << "Model file does not exist or is not readable: " << modelPath << std::endl;
            return;
        }
        file.close();
        
        std::cout << "Loading ONNX model from: " << modelPath << std::endl;
        
        // Load the ONNX model
        net = cv::dnn::readNetFromONNX(modelPath);
        
        if (net.empty()) {
            std::cerr << "Failed to load YOLOv5 model from: " << modelPath << std::endl;
            return;
        }
        
        std::cout << "ONNX model loaded successfully, setting backend..." << std::endl;
        
        // Set backend and target
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Initialize class names (COCO dataset)
        classNames = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"};
        
        modelLoaded = true;
        std::cout << "YOLOv5 model loaded successfully from: " << modelPath << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error loading model: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
}

YoloV5Detector::~YoloV5Detector() {
    // OpenCV handles cleanup automatically
}

bool YoloV5Detector::isModelLoaded() const {
    return modelLoaded;
}

cv::Mat YoloV5Detector::preprocess(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
    return blob;
}

std::vector<Detection> YoloV5Detector::postprocess(const cv::Mat& output, const cv::Mat& originalImage) {
    std::vector<Detection> detections;
    
    // YOLOv5 output format: [batch, 25200, 85] where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
    float* data = (float*)output.data;
    const int dimensions = 85;
    const int rows = output.size[1];
    
    float scaleX = (float)originalImage.cols / inputWidth;
    float scaleY = (float)originalImage.rows / inputHeight;
    
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        
        if (confidence >= confThreshold) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, 80, CV_32FC1, classes_scores);
            cv::Point classIdPoint;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
            
            if (max_class_score > confThreshold) {
                int centerX = (int)(data[0] * scaleX);
                int centerY = (int)(data[1] * scaleY);
                int width = (int)(data[2] * scaleX);
                int height = (int)(data[3] * scaleY);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                // Ensure bounding box is within image bounds
                left = std::max(0, left);
                top = std::max(0, top);
                width = std::min(width, originalImage.cols - left);
                height = std::min(height, originalImage.rows - top);
                
                if (width > 0 && height > 0) {
                    Detection det;
                    det.box = cv::Rect(left, top, width, height);
                    det.confidence = (float)max_class_score;
                    det.class_id = classIdPoint.x;
                    det.class_name = classNames[classIdPoint.x];
                    detections.push_back(det);
                }
            }
        }
        data += dimensions;
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (const auto& det : detections) {
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }
    
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);
    
    std::vector<Detection> result;
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }
    
    return result;
}

std::vector<Detection> YoloV5Detector::detect(const cv::Mat& image) {
    if (!modelLoaded || image.empty()) {
        return std::vector<Detection>();
    }
    
    try {
        // Preprocess
        cv::Mat blob = preprocess(image);
        
        // Forward pass
        net.setInput(blob);
        cv::Mat output = net.forward();
        
        // Postprocess
        return postprocess(output, image);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error during detection: " << e.what() << std::endl;
        return std::vector<Detection>();
    } catch (const std::exception& e) {
        std::cerr << "Error during detection: " << e.what() << std::endl;
        return std::vector<Detection>();
    }
}

bool YoloV5Detector::detectPerson(const cv::Mat& image, float minConfidence) {
    if (!modelLoaded || image.empty()) {
        return false;
    }
    
    std::vector<Detection> detections = detect(image);
    
    // Check if any person was detected with sufficient confidence
    for (const auto& det : detections) {
        if (det.class_name == "person" && det.confidence >= minConfidence) {
            return true;
        }
    }
    
    return false;
}

void YoloV5Detector::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = det.class_name + " " + std::to_string((int)(det.confidence * 100)) + "%";
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point labelPoint(det.box.x, det.box.y - 10);
        if (labelPoint.y < labelSize.height) {
            labelPoint.y = det.box.y + labelSize.height;
        }
        
        cv::rectangle(image, 
                     cv::Point(labelPoint.x, labelPoint.y - labelSize.height - 10),
                     cv::Point(labelPoint.x + labelSize.width, labelPoint.y),
                     cv::Scalar(0, 255, 0), -1);
        
        cv::putText(image, label, labelPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
} 