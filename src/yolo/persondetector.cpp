#include "persondetector.h"
#include <QDebug>
#include <QSize>
#include <QColor>
#include <QPainter>
#include <QPen>
#include <QFont>
#include <algorithm>
#include <opencv2/imgproc.hpp>

PersonDetector::PersonDetector(const QString& modelPath, float confidenceThreshold, float nmsThreshold)
    : m_modelPath(modelPath)
    , m_confidenceThreshold(confidenceThreshold)
    , m_nmsThreshold(nmsThreshold)
    , m_initialized(false)
{
#ifdef HAVE_ONNXRUNTIME
    if (!modelPath.isEmpty()) {
        initialize(modelPath);
    }
#else
    qWarning() << "PersonDetector: ONNX Runtime not available. Person detection disabled.";
#endif
}

PersonDetector::~PersonDetector()
{
#ifdef HAVE_ONNXRUNTIME
    m_session.reset();
    m_sessionOptions.reset();
    m_env.reset();
#endif
}

bool PersonDetector::initialize(const QString& modelPath)
{
#ifdef HAVE_ONNXRUNTIME
    try {
        m_modelPath = modelPath;
        
        // Initialize ONNX Runtime environment
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PersonDetector");
        
        // Create session options
        m_sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_sessionOptions->SetIntraOpNumThreads(1);
        m_sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        std::string modelPathStd = modelPath.toStdString();
        m_session = std::make_unique<Ort::Session>(*m_env, modelPathStd.c_str(), *m_sessionOptions);
        
        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input information
        size_t numInputNodes = m_session->GetInputCount();
        if (numInputNodes != 1) {
            qWarning() << "PersonDetector: Expected 1 input, got" << numInputNodes;
            return false;
        }
        
        m_inputNames.clear();
        m_inputNames.push_back(m_session->GetInputName(0, allocator));
        
        Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        m_inputShape = inputTensorInfo.GetShape();
        
        // Expected input shape: [1, 3, 640, 640]
        if (m_inputShape.size() != 4 || m_inputShape[1] != 3 || m_inputShape[2] != 640 || m_inputShape[3] != 640) {
            qWarning() << "PersonDetector: Unexpected input shape. Expected [1,3,640,640]";
            return false;
        }
        
        // Output information
        size_t numOutputNodes = m_session->GetOutputCount();
        if (numOutputNodes != 1) {
            qWarning() << "PersonDetector: Expected 1 output, got" << numOutputNodes;
            return false;
        }
        
        m_outputNames.clear();
        m_outputNames.push_back(m_session->GetOutputName(0, allocator));
        
        Ort::TypeInfo outputTypeInfo = m_session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        m_outputShape = outputTensorInfo.GetShape();
        
        m_initialized = true;
        qDebug() << "PersonDetector: Successfully initialized with model:" << modelPath;
        qDebug() << "Input shape: [" << m_inputShape[0] << "," << m_inputShape[1] << "," << m_inputShape[2] << "," << m_inputShape[3] << "]";
        qDebug() << "Output shape: [" << m_outputShape[0] << "," << m_outputShape[1] << "]";
        
        return true;
        
    } catch (const Ort::Exception& e) {
        qWarning() << "PersonDetector: ONNX Runtime error:" << e.what();
        return false;
    } catch (const std::exception& e) {
        qWarning() << "PersonDetector: Error initializing:" << e.what();
        return false;
    }
#else
    qWarning() << "PersonDetector: ONNX Runtime not available";
    return false;
#endif
}

bool PersonDetector::isInitialized() const
{
    return m_initialized;
}

QVector<PersonDetection> PersonDetector::detectPersons(const QImage& image)
{
    if (!m_initialized) {
        qWarning() << "PersonDetector: Not initialized";
        return QVector<PersonDetection>();
    }
    
    cv::Mat cvImage = qImageToCvMat(image);
    return detectPersons(cvImage);
}

QVector<PersonDetection> PersonDetector::detectPersons(const cv::Mat& image)
{
    if (!m_initialized) {
        qWarning() << "PersonDetector: Not initialized";
        return QVector<PersonDetection>();
    }
    
#ifdef HAVE_ONNXRUNTIME
    try {
        // Preprocess image
        cv::Mat preprocessed = preprocessImage(image);
        
        // Prepare input tensor
        std::vector<float> inputData(preprocessed.rows * preprocessed.cols * preprocessed.channels());
        
        // Convert BGR to RGB and normalize
        cv::Mat rgbImage;
        cv::cvtColor(preprocessed, rgbImage, cv::COLOR_BGR2RGB);
        
        // Reshape to CHW format and normalize to [0,1]
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 640; ++h) {
                for (int w = 0; w < 640; ++w) {
                    inputData[c * 640 * 640 + h * 640 + w] = 
                        rgbImage.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }
        
        // Create input tensor
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(), m_inputShape.data(), m_inputShape.size());
        
        // Run inference
        auto outputTensors = m_session->Run(Ort::RunOptions{nullptr}, 
                                          m_inputNames.data(), &inputTensor, 1,
                                          m_outputNames.data(), 1);
        
        // Process output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        // Convert to cv::Mat for easier processing
        cv::Mat output(m_outputShape[1], m_outputShape[2], CV_32F, outputData);
        
        // Post-process to get person detections
        QVector<PersonDetection> detections = postprocessOutput(output, QSize(image.cols, image.rows));
        
        // Apply Non-Maximum Suppression
        detections = applyNMS(detections);
        
        qDebug() << "PersonDetector: Found" << detections.size() << "person(s)";
        return detections;
        
    } catch (const Ort::Exception& e) {
        qWarning() << "PersonDetector: Inference error:" << e.what();
        return QVector<PersonDetection>();
    } catch (const std::exception& e) {
        qWarning() << "PersonDetector: Error during detection:" << e.what();
        return QVector<PersonDetection>();
    }
#else
    qWarning() << "PersonDetector: ONNX Runtime not available";
    return QVector<PersonDetection>();
#endif
}

int PersonDetector::countPersons(const QImage& image)
{
    return detectPersons(image).size();
}

int PersonDetector::countPersons(const cv::Mat& image)
{
    return detectPersons(image).size();
}

QImage PersonDetector::drawDetections(const QImage& image, 
                                     const QVector<PersonDetection>& detections,
                                     bool drawConfidence)
{
    QImage result = image.copy();
    QPainter painter(&result);
    
    // Set up pen for bounding boxes
    QPen pen(QColor(0, 255, 0), 3); // Green color, 3px width
    painter.setPen(pen);
    
    // Set up font for confidence text
    QFont font("Arial", 12, QFont::Bold);
    painter.setFont(font);
    
    for (const auto& detection : detections) {
        // Draw bounding box
        painter.drawRect(detection.boundingBox);
        
        if (drawConfidence) {
            // Draw confidence text
            QString confidenceText = QString("Person: %1%").arg(
                static_cast<int>(detection.confidence * 100));
            
            QRect textRect = detection.boundingBox;
            textRect.setHeight(20);
            textRect.moveTop(detection.boundingBox.top() - 25);
            
            // Draw text background
            painter.fillRect(textRect, QColor(0, 255, 0, 180));
            
            // Draw text
            painter.setPen(QColor(0, 0, 0));
            painter.drawText(textRect, Qt::AlignCenter, confidenceText);
            painter.setPen(pen);
        }
    }
    
    return result;
}

void PersonDetector::drawDetections(cv::Mat& image, 
                                   const QVector<PersonDetection>& detections,
                                   bool drawConfidence)
{
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(image, 
                     cv::Point(detection.boundingBox.left(), detection.boundingBox.top()),
                     cv::Point(detection.boundingBox.right(), detection.boundingBox.bottom()),
                     cv::Scalar(0, 255, 0), 3);
        
        if (drawConfidence) {
            // Draw confidence text
            std::string confidenceText = "Person: " + 
                std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
            
            cv::Point textPos(detection.boundingBox.left(), detection.boundingBox.top() - 10);
            cv::putText(image, confidenceText, textPos, cv::FONT_HERSHEY_SIMPLEX, 
                       0.7, cv::Scalar(0, 255, 0), 2);
        }
    }
}

cv::Mat PersonDetector::preprocessImage(const cv::Mat& image)
{
    cv::Mat resized;
    
    // Resize to 640x640 with padding to maintain aspect ratio
    int targetSize = 640;
    float scale = std::min(static_cast<float>(targetSize) / image.cols,
                          static_cast<float>(targetSize) / image.rows);
    
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    
    cv::resize(image, resized, cv::Size(newWidth, newHeight));
    
    // Create 640x640 image with padding
    cv::Mat padded = cv::Mat::zeros(targetSize, targetSize, CV_8UC3);
    
    int offsetX = (targetSize - newWidth) / 2;
    int offsetY = (targetSize - newHeight) / 2;
    
    resized.copyTo(padded(cv::Rect(offsetX, offsetY, newWidth, newHeight)));
    
    return padded;
}

QVector<PersonDetection> PersonDetector::postprocessOutput(const cv::Mat& output, const QSize& originalSize)
{
    QVector<PersonDetection> detections;
    
    // YOLOv5 output format: [batch, num_detections, 85]
    // 85 = 4 (bbox) + 1 (objectness) + 80 (class probabilities)
    // Person class is index 0 in COCO dataset
    
    int numDetections = output.rows;
    
    for (int i = 0; i < numDetections; ++i) {
        const float* data = output.ptr<float>(i);
        
        // Extract bbox coordinates (center_x, center_y, width, height)
        float centerX = data[0];
        float centerY = data[1];
        float width = data[2];
        float height = data[3];
        
        // Extract objectness score
        float objectness = data[4];
        
        // Extract person class probability (index 5 corresponds to class 0 - person)
        float personProb = data[5];
        
        // Calculate final confidence
        float confidence = objectness * personProb;
        
        // Filter by confidence threshold and ensure it's person class
        if (confidence >= m_confidenceThreshold) {
            // Convert from center coordinates to top-left coordinates
            float x1 = centerX - width / 2.0f;
            float y1 = centerY - height / 2.0f;
            float x2 = centerX + width / 2.0f;
            float y2 = centerY + height / 2.0f;
            
            // Scale coordinates from 640x640 to original image size
            float scaleX = static_cast<float>(originalSize.width()) / 640.0f;
            float scaleY = static_cast<float>(originalSize.height()) / 640.0f;
            
            // Account for padding during preprocessing
            float targetSize = 640.0f;
            float imgScale = std::min(targetSize / originalSize.width(), 
                                    targetSize / originalSize.height());
            
            float offsetX = (targetSize - originalSize.width() * imgScale) / 2.0f;
            float offsetY = (targetSize - originalSize.height() * imgScale) / 2.0f;
            
            // Adjust for padding offset
            x1 = (x1 - offsetX) / imgScale;
            y1 = (y1 - offsetY) / imgScale;
            x2 = (x2 - offsetX) / imgScale;
            y2 = (y2 - offsetY) / imgScale;
            
            // Clamp to image boundaries
            x1 = std::max(0.0f, std::min(static_cast<float>(originalSize.width() - 1), x1));
            y1 = std::max(0.0f, std::min(static_cast<float>(originalSize.height() - 1), y1));
            x2 = std::max(0.0f, std::min(static_cast<float>(originalSize.width() - 1), x2));
            y2 = std::max(0.0f, std::min(static_cast<float>(originalSize.height() - 1), y2));
            
            QRect boundingBox(static_cast<int>(x1), static_cast<int>(y1),
                            static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
            
            detections.append(PersonDetection(boundingBox, confidence, 0));
        }
    }
    
    return detections;
}

QVector<PersonDetection> PersonDetector::applyNMS(const QVector<PersonDetection>& detections)
{
    if (detections.isEmpty()) {
        return detections;
    }
    
    // Convert to OpenCV format for NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (const auto& detection : detections) {
        boxes.push_back(cv::Rect(detection.boundingBox.x(), detection.boundingBox.y(),
                                detection.boundingBox.width(), detection.boundingBox.height()));
        scores.push_back(detection.confidence);
    }
    
    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, m_confidenceThreshold, m_nmsThreshold, indices);
    
    // Build result
    QVector<PersonDetection> result;
    for (int idx : indices) {
        result.append(detections[idx]);
    }
    
    return result;
}

cv::Mat PersonDetector::qImageToCvMat(const QImage& qimage)
{
    QImage swapped = qimage.rgbSwapped();
    return cv::Mat(swapped.height(), swapped.width(), CV_8UC4, 
                   (void*)swapped.constBits(), swapped.bytesPerLine()).clone();
}

QImage PersonDetector::cvMatToQImage(const cv::Mat& mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return qimg.rgbSwapped();
    }
    case CV_8UC3: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return qimg.rgbSwapped();
    }
    case CV_8UC1: {
        QImage qimg(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return qimg;
    }
    default:
        qWarning() << "PersonDetector: Unsupported cv::Mat format for conversion";
        return QImage();
    }
}