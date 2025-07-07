#include "yolov5detector.h"
#include <QDebug>
#include <QFileInfo>
#include <QElapsedTimer>
#include <QMutexLocker>
#include <QColor>
#include <QPainter>
#include <QPen>
#include <QFont>
#include <QApplication>

// Include ONNX Runtime headers with proper error handling
#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable: 4251)
#endif

#ifdef ONNXRUNTIME_AVAILABLE
    #include <onnxruntime_cxx_api.h>
#else
    // Fallback when ONNX Runtime is not available
    namespace Ort {
        class Env {};
        class Session {};
        class Value {};
        class MemoryInfo {};
    }
#endif

#ifdef _WIN32
    #pragma warning(pop)
#endif

// Initialize static member
QVector<QString> YOLOv5Detector::s_cocoClassNames;

YOLOv5Detector::YOLOv5Detector(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_confidenceThreshold(0.5f)
    , m_nmsThreshold(0.4f)
    , m_inputSize(640, 640)
{
    // Initialize COCO class names
    if (s_cocoClassNames.isEmpty()) {
        initializeCocoClassNames();
    }
    
    // Connect internal signals
    connect(this, &YOLOv5Detector::detectionCompleted,
            this, &YOLOv5Detector::onDetectionCompleted);
}

YOLOv5Detector::~YOLOv5Detector()
{
    QMutexLocker locker(&m_mutex);
    m_session.reset();
    m_memoryInfo.reset();
    m_env.reset();
}

bool YOLOv5Detector::initialize(const QString& modelPath)
{
    QMutexLocker locker(&m_mutex);
    
    // Check if model file exists
    QFileInfo fileInfo(modelPath);
    if (!fileInfo.exists() || !fileInfo.isFile()) {
        QString error = QString("Model file does not exist: %1").arg(modelPath);
        emit errorOccurred(error);
        qWarning() << error;
        return false;
    }
    
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        // Initialize ONNX Runtime environment
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv5Detector");
        
        // Create session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        std::wstring wModelPath = modelPath.toStdWString();
        m_session = std::make_unique<Ort::Session>(*m_env, wModelPath.c_str(), sessionOptions);
        
        // Initialize memory info
        m_memoryInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input name
        auto input_name = m_session->GetInputNameAllocated(0, allocator);
        m_inputName = QString::fromLocal8Bit(input_name.get());
        
        // Get output name  
        auto output_name = m_session->GetOutputNameAllocated(0, allocator);
        m_outputName = QString::fromLocal8Bit(output_name.get());
        
        // Get input shape to verify model
        auto input_shape = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() >= 4) {
            // Expected shape: [batch_size, channels, height, width]
            m_inputSize = QSize(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2]));
        }
        
        m_initialized = true;
        qDebug() << "YOLOv5Detector initialized successfully";
        qDebug() << "Model path:" << modelPath;
        qDebug() << "Input size:" << m_inputSize;
        qDebug() << "Input name:" << m_inputName;
        qDebug() << "Output name:" << m_outputName;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        QString error = QString("ONNX Runtime error: %1").arg(e.what());
        emit errorOccurred(error);
        qWarning() << error;
        return false;
    } catch (const std::exception& e) {
        QString error = QString("Standard exception: %1").arg(e.what());
        emit errorOccurred(error);
        qWarning() << error;
        return false;
    }
#else
    QString error = "ONNX Runtime not available. Please install ONNX Runtime and recompile.";
    emit errorOccurred(error);
    qWarning() << error;
    return false;
#endif
}

bool YOLOv5Detector::isInitialized() const
{
    QMutexLocker locker(&m_mutex);
    return m_initialized;
}

void YOLOv5Detector::setConfidenceThreshold(float threshold)
{
    QMutexLocker locker(&m_mutex);
    m_confidenceThreshold = qBound(0.0f, threshold, 1.0f);
}

float YOLOv5Detector::getConfidenceThreshold() const
{
    QMutexLocker locker(&m_mutex);
    return m_confidenceThreshold;
}

void YOLOv5Detector::setNmsThreshold(float threshold)
{
    QMutexLocker locker(&m_mutex);
    m_nmsThreshold = qBound(0.0f, threshold, 1.0f);
}

float YOLOv5Detector::getNmsThreshold() const
{
    QMutexLocker locker(&m_mutex);
    return m_nmsThreshold;
}

QVector<Detection> YOLOv5Detector::detectObjects(const QImage& image)
{
    if (image.isNull()) {
        emit errorOccurred("Input image is null");
        return QVector<Detection>();
    }
    
    cv::Mat cvImage = qImageToCvMat(image);
    return detectObjects(cvImage);
}

QVector<Detection> YOLOv5Detector::detectObjects(const cv::Mat& image)
{
    QElapsedTimer timer;
    timer.start();
    
    QMutexLocker locker(&m_mutex);
    
    if (!m_initialized) {
        emit errorOccurred("Detector not initialized");
        return QVector<Detection>();
    }
    
    if (image.empty()) {
        emit errorOccurred("Input image is empty");
        return QVector<Detection>();
    }
    
    QVector<Detection> detections;
    
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        // Store original image size
        QSize originalSize(image.cols, image.rows);
        
        // Preprocess image
        cv::Mat preprocessed = preprocessImage(image);
        
        // Prepare input tensor
        std::vector<int64_t> inputShape = {1, 3, m_inputSize.height(), m_inputSize.width()};
        size_t inputTensorSize = 1 * 3 * m_inputSize.height() * m_inputSize.width();
        
        std::vector<float> inputTensorValues(inputTensorSize);
        
        // Convert BGR to RGB and normalize to [0,1], then copy to input tensor
        for (int y = 0; y < preprocessed.rows; ++y) {
            for (int x = 0; x < preprocessed.cols; ++x) {
                cv::Vec3b pixel = preprocessed.at<cv::Vec3b>(y, x);
                // BGR to RGB conversion and normalization
                inputTensorValues[0 * m_inputSize.height() * m_inputSize.width() + y * m_inputSize.width() + x] = pixel[2] / 255.0f; // R
                inputTensorValues[1 * m_inputSize.height() * m_inputSize.width() + y * m_inputSize.width() + x] = pixel[1] / 255.0f; // G
                inputTensorValues[2 * m_inputSize.height() * m_inputSize.width() + y * m_inputSize.width() + x] = pixel[0] / 255.0f; // B
            }
        }
        
        // Create input tensor
        auto inputTensor = Ort::Value::CreateTensor<float>(
            *m_memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputShape.data(), inputShape.size());
        
        // Run inference
        const char* inputNames[] = {m_inputName.toLocal8Bit().constData()};
        const char* outputNames[] = {m_outputName.toLocal8Bit().constData()};
        
        auto outputTensors = m_session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        
        // Process output
        float* floatOutput = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Convert output to cv::Mat for easier processing
        // YOLOv5 output shape: [1, 25200, 85] (for 640x640 input)
        // 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
        int numDetections = static_cast<int>(outputShape[1]);
        int numFeatures = static_cast<int>(outputShape[2]);
        
        cv::Mat output(numDetections, numFeatures, CV_32F, floatOutput);
        
        // Post-process output
        detections = postprocessOutput(output, originalSize);
        
        // Apply NMS
        detections = applyNMS(detections);
        
    } catch (const Ort::Exception& e) {
        QString error = QString("Inference error: %1").arg(e.what());
        emit errorOccurred(error);
        qWarning() << error;
    } catch (const std::exception& e) {
        QString error = QString("Standard exception during inference: %1").arg(e.what());
        emit errorOccurred(error);
        qWarning() << error;
    }
#else
    emit errorOccurred("ONNX Runtime not available");
#endif
    
    int processingTime = static_cast<int>(timer.elapsed());
    
    // Unlock before emitting signal to avoid deadlock
    locker.unlock();
    emit detectionCompleted(detections, processingTime);
    
    return detections;
}

QImage YOLOv5Detector::drawBoundingBoxes(const QImage& image, const QVector<Detection>& detections)
{
    if (image.isNull()) {
        return image;
    }
    
    QImage result = image.copy();
    QPainter painter(&result);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Define colors for different classes
    QVector<QColor> colors = {
        QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
        QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
        QColor(128, 0, 0), QColor(0, 128, 0), QColor(0, 0, 128),
        QColor(128, 128, 0), QColor(128, 0, 128), QColor(0, 128, 128)
    };
    
    QFont font("Arial", 12, QFont::Bold);
    painter.setFont(font);
    
    for (const Detection& detection : detections) {
        // Choose color based on class ID
        QColor color = colors[detection.classId % colors.size()];
        
        // Draw bounding box
        QPen pen(color, 2);
        painter.setPen(pen);
        painter.drawRect(detection.boundingBox);
        
        // Prepare label text
        QString label = QString("%1: %.2f")
                        .arg(detection.className)
                        .arg(detection.confidence, 0, 'f', 2);
        
        // Calculate text metrics
        QFontMetrics fm(font);
        QRect textRect = fm.boundingRect(label);
        textRect.adjust(-5, -2, 5, 2);
        
        // Position text above bounding box
        QPointF textPos = detection.boundingBox.topLeft();
        textPos.setY(textPos.y() - 5);
        
        // Ensure text is within image bounds
        if (textPos.y() - textRect.height() < 0) {
            textPos.setY(detection.boundingBox.top() + textRect.height() + 5);
        }
        
        textRect.moveTo(textPos.toPoint());
        
        // Draw text background
        painter.fillRect(textRect, color);
        
        // Draw text
        painter.setPen(QPen(Qt::white));
        painter.drawText(textRect, Qt::AlignCenter, label);
    }
    
    return result;
}

void YOLOv5Detector::drawBoundingBoxes(cv::Mat& image, const QVector<Detection>& detections)
{
    if (image.empty()) {
        return;
    }
    
    // Define colors for different classes (BGR format for OpenCV)
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
        cv::Scalar(0, 0, 128), cv::Scalar(0, 128, 0), cv::Scalar(128, 0, 0),
        cv::Scalar(0, 128, 128), cv::Scalar(128, 0, 128), cv::Scalar(128, 128, 0)
    };
    
    for (const Detection& detection : detections) {
        // Choose color based on class ID
        cv::Scalar color = colors[detection.classId % colors.size()];
        
        // Convert QRectF to cv::Rect
        cv::Rect rect(static_cast<int>(detection.boundingBox.x()),
                     static_cast<int>(detection.boundingBox.y()),
                     static_cast<int>(detection.boundingBox.width()),
                     static_cast<int>(detection.boundingBox.height()));
        
        // Draw bounding box
        cv::rectangle(image, rect, color, 2);
        
        // Prepare label text
        QString label = QString("%1: %.2f")
                        .arg(detection.className)
                        .arg(detection.confidence, 0, 'f', 2);
        
        // Calculate text size
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // Position text above bounding box
        cv::Point textOrg(rect.x, rect.y - 5);
        if (textOrg.y - textSize.height < 0) {
            textOrg.y = rect.y + textSize.height + 5;
        }
        
        // Draw text background
        cv::Rect textRect(textOrg.x - 2, textOrg.y - textSize.height - 2,
                         textSize.width + 4, textSize.height + 4);
        cv::rectangle(image, textRect, color, cv::FILLED);
        
        // Draw text
        cv::putText(image, label.toStdString(), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

QVector<QString> YOLOv5Detector::getCocoClassNames()
{
    if (s_cocoClassNames.isEmpty()) {
        initializeCocoClassNames();
    }
    return s_cocoClassNames;
}

QSize YOLOv5Detector::getModelInputSize() const
{
    QMutexLocker locker(&m_mutex);
    return m_inputSize;
}

void YOLOv5Detector::onDetectionCompleted(const QVector<Detection>& detections, int processingTimeMs)
{
    qDebug() << QString("Detection completed: %1 objects detected in %2ms")
                .arg(detections.size())
                .arg(processingTimeMs);
}

cv::Mat YOLOv5Detector::preprocessImage(const cv::Mat& image)
{
    cv::Mat processed;
    
    // Resize with padding to maintain aspect ratio
    float scale = std::min(static_cast<float>(m_inputSize.width()) / image.cols,
                          static_cast<float>(m_inputSize.height()) / image.rows);
    
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    
    cv::resize(image, processed, cv::Size(newWidth, newHeight));
    
    // Add padding to reach target size
    int padX = (m_inputSize.width() - newWidth) / 2;
    int padY = (m_inputSize.height() - newHeight) / 2;
    
    cv::copyMakeBorder(processed, processed, padY, m_inputSize.height() - newHeight - padY,
                      padX, m_inputSize.width() - newWidth - padX, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return processed;
}

QVector<Detection> YOLOv5Detector::postprocessOutput(const cv::Mat& output, const QSize& originalSize)
{
    QVector<Detection> detections;
    
    float confThreshold = m_confidenceThreshold;
    
    // Calculate scale factors for coordinate conversion
    float scaleX = static_cast<float>(originalSize.width()) / m_inputSize.width();
    float scaleY = static_cast<float>(originalSize.height()) / m_inputSize.height();
    float scale = std::min(scaleX, scaleY);
    
    int padX = (m_inputSize.width() - originalSize.width() / scale) / 2;
    int padY = (m_inputSize.height() - originalSize.height() / scale) / 2;
    
    for (int i = 0; i < output.rows; ++i) {
        const float* row = output.ptr<float>(i);
        
        // Extract bounding box coordinates (center format)
        float centerX = row[0];
        float centerY = row[1];
        float width = row[2];
        float height = row[3];
        float objectConfidence = row[4];
        
        // Skip if object confidence is too low
        if (objectConfidence < confThreshold) {
            continue;
        }
        
        // Find the class with highest probability
        float maxClassScore = 0.0f;
        int bestClassId = -1;
        
        for (int j = 5; j < output.cols; ++j) {
            float classScore = row[j];
            if (classScore > maxClassScore) {
                maxClassScore = classScore;
                bestClassId = j - 5;
            }
        }
        
        // Calculate final confidence
        float finalConfidence = objectConfidence * maxClassScore;
        
        // Skip if final confidence is too low
        if (finalConfidence < confThreshold) {
            continue;
        }
        
        // Convert center format to corner format and scale back to original image
        float x1 = (centerX - width / 2.0f - padX) * scale;
        float y1 = (centerY - height / 2.0f - padY) * scale;
        float x2 = (centerX + width / 2.0f - padX) * scale;
        float y2 = (centerY + height / 2.0f - padY) * scale;
        
        // Clamp coordinates to image bounds
        x1 = qBound(0.0f, x1, static_cast<float>(originalSize.width()));
        y1 = qBound(0.0f, y1, static_cast<float>(originalSize.height()));
        x2 = qBound(0.0f, x2, static_cast<float>(originalSize.width()));
        y2 = qBound(0.0f, y2, static_cast<float>(originalSize.height()));
        
        // Create detection
        QRectF boundingBox(x1, y1, x2 - x1, y2 - y1);
        QString className = (bestClassId >= 0 && bestClassId < s_cocoClassNames.size()) 
                           ? s_cocoClassNames[bestClassId] 
                           : QString("unknown");
        
        detections.append(Detection(boundingBox, finalConfidence, bestClassId, className));
    }
    
    return detections;
}

QVector<Detection> YOLOv5Detector::applyNMS(const QVector<Detection>& detections)
{
    if (detections.isEmpty()) {
        return detections;
    }
    
    // Convert to OpenCV format for NMS
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    
    for (const Detection& detection : detections) {
        boxes.push_back(cv::Rect2f(detection.boundingBox.x(), detection.boundingBox.y(),
                                  detection.boundingBox.width(), detection.boundingBox.height()));
        scores.push_back(detection.confidence);
        classIds.push_back(detection.classId);
    }
    
    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, m_confidenceThreshold, m_nmsThreshold, indices);
    
    // Build result
    QVector<Detection> result;
    for (int idx : indices) {
        result.append(detections[idx]);
    }
    
    return result;
}

cv::Mat YOLOv5Detector::qImageToCvMat(const QImage& qimage)
{
    QImage swapped = qimage.rgbSwapped();
    return cv::Mat(swapped.height(), swapped.width(), CV_8UC3, 
                   const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
}

QImage YOLOv5Detector::cvMatToQImage(const cv::Mat& mat)
{
    switch (mat.type()) {
    case CV_8UC4: {
        QImage qimage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return qimage.rgbSwapped();
    }
    case CV_8UC3: {
        QImage qimage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return qimage.rgbSwapped();
    }
    case CV_8UC1: {
        static QVector<QRgb> sColorTable;
        if (sColorTable.isEmpty()) {
            for (int i = 0; i < 256; ++i)
                sColorTable.push_back(qRgb(i, i, i));
        }
        QImage qimage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        qimage.setColorTable(sColorTable);
        return qimage;
    }
    default:
        qWarning() << "cvMatToQImage - Mat type not handled:" << mat.type();
        return QImage();
    }
}

void YOLOv5Detector::initializeCocoClassNames()
{
    s_cocoClassNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
}

// Include the MOC file for proper Qt meta-object compilation
#include "yolov5detector.moc"