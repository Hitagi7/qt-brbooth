#include "optimized_detector.h"
#include "capture.h" // For BoundingBox struct
#include <QCoreApplication>
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QThread>
#include <algorithm>

OptimizedPersonDetector::OptimizedPersonDetector(QObject* parent)
    : QObject(parent)
    , m_modelType(YOLO_SEGMENTATION)
    , m_perfMode(REAL_TIME)
    , m_initialized(false)
    , m_processing(false)
    , m_inputSize(640, 640)
    , m_defaultConfThreshold(0.5)
    , m_defaultNMSThreshold(0.4)
    , m_scaleX(1.0)
    , m_scaleY(1.0)
    , m_avgInferenceTime(0.0)
    , m_currentFPS(30)
    , m_frameCount(0)
{
    qDebug() << "🚀 OptimizedPersonDetector: Initializing high-performance ONNX detector";
    
    // Setup class IDs for person detection
    m_classIds = {0}; // COCO class 0 = person
}

OptimizedPersonDetector::~OptimizedPersonDetector() {
    qDebug() << "✅ OptimizedPersonDetector destroyed";
}

bool OptimizedPersonDetector::initialize(ModelType modelType, PerformanceMode perfMode) {
    m_modelType = modelType;
    m_perfMode = perfMode;
    
    qDebug() << "🔧 Initializing with model type:" << modelType << "performance mode:" << perfMode;
    
    try {
        // Determine model path based on type
        QString modelPath;
        QString appDir = QCoreApplication::applicationDirPath();
        
        switch (modelType) {
            case YOLO_SEGMENTATION:
                // Try multiple possible paths for segmentation model
                modelPath = appDir + "/models/yolov5s-seg.onnx";
                if (!QFile::exists(modelPath)) {
                    modelPath = appDir + "/../../../models/yolov5s-seg.onnx";
                }
                if (!QFile::exists(modelPath)) {
                    modelPath = appDir + "/yolov5s-seg.onnx";
                }
                break;
                
            case YOLO_DETECTION:
                modelPath = appDir + "/models/yolov5s.onnx";
                if (!QFile::exists(modelPath)) {
                    modelPath = appDir + "/../../../yolov5/yolov5s.onnx";
                }
                break;
                
            case TENSORRT_OPTIMIZED:
                modelPath = appDir + "/models/yolov5s-seg.trt";
                break;
        }
        
        qDebug() << "📁 Model path:" << modelPath;
        
        if (!QFile::exists(modelPath)) {
            qWarning() << "❌ Model file not found:" << modelPath;
            qDebug() << "💡 Please download yolov5s-seg.onnx model to one of these locations:";
            qDebug() << "   -" << appDir + "/models/yolov5s-seg.onnx";
            qDebug() << "   -" << appDir + "/yolov5s-seg.onnx";
            return false;
        }
        
        // Load ONNX model
        m_net = cv::dnn::readNetFromONNX(modelPath.toStdString());
        
        if (m_net.empty()) {
            qWarning() << "❌ Failed to load ONNX model";
            return false;
        }
        
        // Set backend and target based on performance mode
        if (perfMode == REAL_TIME) {
            // Try CUDA first, fallback to CPU
            try {
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                qDebug() << "🎮 Using CUDA backend for maximum speed";
            } catch (...) {
                qDebug() << "⚠️ CUDA not available, using optimized CPU";
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } else {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        // Get output layer names
        std::vector<cv::String> layerNames = m_net.getLayerNames();
        std::vector<int> outLayers = m_net.getUnconnectedOutLayers();
        
        m_outputNames.clear();
        for (size_t i = 0; i < outLayers.size(); ++i) {
            m_outputNames.push_back(layerNames[outLayers[i] - 1]);
        }
        
        qDebug() << "📋 Output layers:" << QString::fromStdString(
            std::accumulate(m_outputNames.begin(), m_outputNames.end(), std::string{},
                [](const std::string& a, const std::string& b) {
                    return a.empty() ? b : a + ", " + b;
                }));
        
        // Warm up the model
        warmupModel();
        
        m_initialized = true;
        qDebug() << "✅ OptimizedPersonDetector initialized successfully";
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ OpenCV exception during initialization:" << e.what();
        return false;
    } catch (const std::exception& e) {
        qWarning() << "❌ Standard exception during initialization:" << e.what();
        return false;
    }
}

QList<OptimizedDetection> OptimizedPersonDetector::detectPersons(const cv::Mat& image, 
                                                                double confThreshold, 
                                                                double nmsThreshold) {
    QMutexLocker locker(&m_mutex);
    
    if (!m_initialized || image.empty()) {
        return QList<OptimizedDetection>();
    }
    
    auto start = std::chrono::steady_clock::now();
    
    QList<OptimizedDetection> detections;
    
    try {
        if (m_modelType == YOLO_SEGMENTATION) {
            detections = runYOLOSegmentation(image, confThreshold, nmsThreshold);
        } else {
            detections = runYOLODetection(image, confThreshold, nmsThreshold);
        }
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Detection error:" << e.what();
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    updatePerformanceStats(duration.count());
    
    return detections;
}

void OptimizedPersonDetector::detectPersonsAsync(const cv::Mat& image, 
                                                double confThreshold, 
                                                double nmsThreshold) {
    if (m_processing || !m_initialized) {
        return;
    }
    
    m_processing = true;
    
    // Execute detection synchronously for real-time performance
    QList<OptimizedDetection> detections = detectPersons(image, confThreshold, nmsThreshold);
    
    // Emit results immediately
    emit detectionsReady(detections);
    emit processingFinished();
    
    m_processing = false;
}

QList<OptimizedDetection> OptimizedPersonDetector::runYOLOSegmentation(const cv::Mat& image, 
                                                                      double confThreshold, 
                                                                      double nmsThreshold) {
    // Preprocess image
    cv::Mat blob = preprocessImage(image);
    
    // Set input to the network
    m_net.setInput(blob);
    
    // Run inference
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_outputNames);
    
    // Post-process results for segmentation
    return postprocessSegmentation(outputs, image, confThreshold, nmsThreshold);
}

QList<OptimizedDetection> OptimizedPersonDetector::runYOLODetection(const cv::Mat& image, 
                                                                   double confThreshold, 
                                                                   double nmsThreshold) {
    // Preprocess image
    cv::Mat blob = preprocessImage(image);
    
    // Set input to the network
    m_net.setInput(blob);
    
    // Run inference
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_outputNames);
    
    // Post-process results for detection
    return postprocessDetections(outputs, image, confThreshold, nmsThreshold);
}

cv::Mat OptimizedPersonDetector::preprocessImage(const cv::Mat& image) {
    // Calculate scale factors
    m_scaleX = static_cast<double>(image.cols) / m_inputSize.width;
    m_scaleY = static_cast<double>(image.rows) / m_inputSize.height;
    
    // Create blob from image (normalize to [0,1], swap R and B channels)
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, m_inputSize, cv::Scalar(0,0,0), true, false, CV_32F);
    
    return blob;
}

QList<OptimizedDetection> OptimizedPersonDetector::postprocessSegmentation(const std::vector<cv::Mat>& outputs,
                                                                          const cv::Mat& originalImage,
                                                                          double confThreshold,
                                                                          double nmsThreshold) {
    QList<OptimizedDetection> detections;
    
    if (outputs.size() < 2) {
        qWarning() << "❌ Insufficient outputs for segmentation model";
        return detections;
    }
    
    // YOLOv5-seg typically outputs:
    // outputs[0]: detection results [1, 25200, 117] (x,y,w,h,conf,class,mask_coeffs...)
    // outputs[1]: prototype masks [1, 32, 160, 160]
    
    cv::Mat detectOutput = outputs[0];  // Detection output
    cv::Mat protoOutput = outputs[1];   // Prototype masks
    
    // Reshape detection output if needed
    if (detectOutput.dims == 3) {
        detectOutput = detectOutput.reshape(1, detectOutput.size[1]);
    }
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<cv::Mat> maskCoeffs;
    
    // Process each detection
    for (int i = 0; i < detectOutput.rows; ++i) {
        cv::Mat row = detectOutput.row(i);
        cv::Mat scores = row.colRange(5, 85);  // Class scores
        
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        
        // Filter by confidence and class (person = 0)
        if (confidence > confThreshold && classIdPoint.x == 0) {
            float centerX = row.at<float>(0);
            float centerY = row.at<float>(1);
            float width = row.at<float>(2);
            float height = row.at<float>(3);
            
            // Convert to corner coordinates and scale to original image
            int x = static_cast<int>((centerX - width/2) * m_scaleX);
            int y = static_cast<int>((centerY - height/2) * m_scaleY);
            int w = static_cast<int>(width * m_scaleX);
            int h = static_cast<int>(height * m_scaleY);
            
            // Clamp to image bounds
            x = std::max(0, std::min(x, originalImage.cols));
            y = std::max(0, std::min(y, originalImage.rows));
            w = std::min(w, originalImage.cols - x);
            h = std::min(h, originalImage.rows - y);
            
            if (w > MIN_BOX_AREA && h > MIN_BOX_AREA) {
                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(static_cast<float>(confidence));
                classIds.push_back(0); // Person class
                
                // Extract mask coefficients (typically last 32 values)
                cv::Mat coeffs = row.colRange(85, 117);  // Adjust based on your model
                maskCoeffs.push_back(coeffs.clone());
            }
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    // Create final detections with masks
    for (int idx : indices) {
        OptimizedDetection detection;
        detection.boundingBox = boxes[idx];
        detection.confidence = confidences[idx];
        detection.className = "person";
        
        // Generate mask from prototypes and coefficients
        if (!protoOutput.empty() && idx < static_cast<int>(maskCoeffs.size())) {
            detection.mask = extractMask(protoOutput, maskCoeffs[idx], 
                                       detection.boundingBox, originalImage.size());
        }
        
        detections.append(detection);
    }
    
    qDebug() << "🎯 Segmentation: Found" << detections.size() << "persons with masks";
    return detections;
}

QList<OptimizedDetection> OptimizedPersonDetector::postprocessDetections(const std::vector<cv::Mat>& outputs,
                                                                        const cv::Mat& originalImage,
                                                                        double confThreshold,
                                                                        double nmsThreshold) {
    QList<OptimizedDetection> detections;
    
    if (outputs.empty()) {
        return detections;
    }
    
    cv::Mat detectOutput = outputs[0];
    
    // Reshape if needed
    if (detectOutput.dims == 3) {
        detectOutput = detectOutput.reshape(1, detectOutput.size[1]);
    }
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    
    // Process each detection
    for (int i = 0; i < detectOutput.rows; ++i) {
        cv::Mat row = detectOutput.row(i);
        cv::Mat scores = row.colRange(5, 85);  // Class scores
        
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        
        // Filter by confidence and class (person = 0)
        if (confidence > confThreshold && classIdPoint.x == 0) {
            float centerX = row.at<float>(0);
            float centerY = row.at<float>(1);
            float width = row.at<float>(2);
            float height = row.at<float>(3);
            
            // Convert and scale coordinates
            int x = static_cast<int>((centerX - width/2) * m_scaleX);
            int y = static_cast<int>((centerY - height/2) * m_scaleY);
            int w = static_cast<int>(width * m_scaleX);
            int h = static_cast<int>(height * m_scaleY);
            
            // Clamp to image bounds
            x = std::max(0, std::min(x, originalImage.cols));
            y = std::max(0, std::min(y, originalImage.rows));
            w = std::min(w, originalImage.cols - x);
            h = std::min(h, originalImage.rows - y);
            
            if (w > MIN_BOX_AREA && h > MIN_BOX_AREA) {
                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(static_cast<float>(confidence));
                classIds.push_back(0);
            }
        }
    }
    
    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    // Create final detections
    for (int idx : indices) {
        OptimizedDetection detection;
        detection.boundingBox = boxes[idx];
        detection.confidence = confidences[idx];
        detection.className = "person";
        // No mask for detection-only model
        
        detections.append(detection);
    }
    
    qDebug() << "🎯 Detection: Found" << detections.size() << "persons";
    return detections;
}

cv::Mat OptimizedPersonDetector::extractMask(const cv::Mat& maskProtos, 
                                            const cv::Mat& maskCoeffs, 
                                            const cv::Rect& /* bbox */, 
                                            const cv::Size& originalSize) {
    if (maskProtos.empty() || maskCoeffs.empty()) {
        return cv::Mat();
    }
    
    try {
        // Flatten prototypes and coefficients for matrix multiplication
        cv::Mat protosFlat = maskProtos.reshape(1, maskProtos.size[1]); // [32, 160*160]
        cv::Mat coeffsFlat = maskCoeffs.reshape(1, 1); // [1, 32]
        
        // Matrix multiplication: coeffs * protos = [1, 160*160]
        cv::Mat maskFlat;
        cv::gemm(coeffsFlat, protosFlat, 1.0, cv::Mat(), 0.0, maskFlat);
        
        // Reshape back to 2D mask
        cv::Mat mask = maskFlat.reshape(1, maskProtos.size[2]); // [160, 160]
        
        // Apply sigmoid activation
        cv::exp(-mask, mask);
        cv::divide(1.0, (1.0 + mask), mask);
        
        // Resize mask to original image size
        cv::Mat fullMask;
        cv::resize(mask, fullMask, originalSize, 0, 0, cv::INTER_LINEAR);
        
        // Threshold to binary mask
        cv::Mat binaryMask;
        cv::threshold(fullMask, binaryMask, 0.5, 255, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8UC1);
        
        return binaryMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "❌ Error extracting mask:" << e.what();
        return cv::Mat();
    }
}

void OptimizedPersonDetector::warmupModel() {
    qDebug() << "🔥 Warming up model...";
    
    // Create dummy input
    cv::Mat dummyImage = cv::Mat::zeros(m_inputSize, CV_8UC3);
    
    // Run a few dummy inferences
    for (int i = 0; i < 3; ++i) {
        try {
            cv::Mat blob = preprocessImage(dummyImage);
            m_net.setInput(blob);
            
            std::vector<cv::Mat> outputs;
            m_net.forward(outputs, m_outputNames);
            
        } catch (...) {
            // Ignore warmup errors
        }
    }
    
    qDebug() << "✅ Model warmup complete";
}

void OptimizedPersonDetector::updatePerformanceStats(double inferenceTime) {
    m_inferenceTimes.append(inferenceTime);
    
    // Keep only recent samples
    while (m_inferenceTimes.size() > MAX_TIMING_SAMPLES) {
        m_inferenceTimes.removeFirst();
    }
    
    // Calculate average
    if (!m_inferenceTimes.isEmpty()) {
        double sum = 0.0;
        for (double time : m_inferenceTimes) {
            sum += time;
        }
        m_avgInferenceTime = sum / m_inferenceTimes.size();
        m_currentFPS = static_cast<int>(1000.0 / m_avgInferenceTime);
    }
    
    m_frameCount++;
}

void OptimizedPersonDetector::setInputSize(int width, int height) {
    m_inputSize = cv::Size(width, height);
    qDebug() << "📐 Input size set to:" << width << "x" << height;
}

// Removed onDetectionFinished - now using synchronous processing

