#ifndef YOLOV5_DETECTOR_H
#define YOLOV5_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;
};

class YoloV5Detector {
public:
    YoloV5Detector(const std::string& modelPath, float confThreshold = 0.4, float nmsThreshold = 0.5);
    ~YoloV5Detector();
    
    std::vector<Detection> detect(const cv::Mat& image);
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
    bool isModelLoaded() const;
    bool detectPerson(const cv::Mat& image, float minConfidence = 0.3f);

private:
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;
    std::vector<std::string> classNames;
    bool modelLoaded;
    
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const cv::Mat& output, const cv::Mat& originalImage);
};

#endif // YOLOV5_DETECTOR_H 