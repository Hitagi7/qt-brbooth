#include "algorithms/lighting_correction/lighting_corrector.h"
#include <QFileInfo>
#include <cmath>

LightingCorrector::LightingCorrector()
    : m_gpuAvailable(false)
    , m_initialized(false)
    , m_clipLimit(8.0)  // More aggressive CLAHE for visible effect
    , m_tileGridSize(8, 8)
    , m_gammaValue(1.5)  // More aggressive gamma correction
{
    qDebug() << "🌟 LightingCorrector: Constructor called";
    qDebug() << "🌟 CLAHE clip limit:" << m_clipLimit;
    qDebug() << "🌟 Gamma value:" << m_gammaValue;
}

LightingCorrector::~LightingCorrector()
{
    cleanup();
    qDebug() << "🌟 LightingCorrector: Destructor called";
}

bool LightingCorrector::initialize()
{
    qDebug() << "🌟 LightingCorrector: Initializing lighting correction system";
    
    try {
        // Check GPU availability
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            qDebug() << "🌟 LightingCorrector: CUDA available, initializing GPU resources";
            
            // Initialize GPU CLAHE
            m_gpuCLAHE = cv::cuda::createCLAHE();
            m_gpuCLAHE->setClipLimit(m_clipLimit);
            m_gpuCLAHE->setTilesGridSize(m_tileGridSize);
            
            // 🚀 PERFORMANCE: Pre-allocate GPU buffers for common resolutions
            int commonWidth = 1920;  // Full HD width
            int commonHeight = 1080; // Full HD height
            
            // Initialize GPU buffers with common size
            m_gpuInputBuffer.create(commonHeight, commonWidth, CV_8UC3);
            m_gpuMaskBuffer.create(commonHeight, commonWidth, CV_8UC1);
            m_gpuTemplateBuffer.create(commonHeight, commonWidth, CV_8UC3);
            m_gpuOutputBuffer.create(commonHeight, commonWidth, CV_8UC3);
            m_gpuTempBuffer1.create(commonHeight, commonWidth, CV_8UC3);
            m_gpuTempBuffer2.create(commonHeight, commonWidth, CV_8UC3);
            
            m_gpuAvailable = true;
            qDebug() << "🌟 LightingCorrector: GPU resources initialized successfully with" << commonWidth << "x" << commonHeight << "buffers";
        } else {
            qDebug() << "🌟 LightingCorrector: CUDA not available, using CPU processing";
            m_gpuAvailable = false;
        }
        
        // Initialize CPU CLAHE with performance-optimized settings
        m_cpuCLAHE = cv::createCLAHE();
        m_cpuCLAHE->setClipLimit(m_clipLimit * 0.8); // Slightly reduced for performance
        m_cpuCLAHE->setTilesGridSize(cv::Size(6, 6)); // Reduced from 8x8 for better performance
        
        m_initialized = true;
        qDebug() << "🌟 LightingCorrector: Initialization completed successfully";
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Initialization failed:" << e.what();
        m_initialized = false;
        m_gpuAvailable = false;
        return false;
    }
}

bool LightingCorrector::setReferenceTemplate(const QString &templatePath)
{
    if (!m_initialized) {
        qWarning() << "🌟 LightingCorrector: Not initialized, cannot set reference template";
        return false;
    }
    
    if (templatePath.isEmpty()) {
        qWarning() << "🌟 LightingCorrector: Empty template path provided";
        return false;
    }
    
    QFileInfo fileInfo(templatePath);
    if (!fileInfo.exists()) {
        qWarning() << "🌟 LightingCorrector: Template file does not exist:" << templatePath;
        return false;
    }
    
    try {
        // Load template image
        m_referenceTemplate = cv::imread(templatePath.toStdString());
        if (m_referenceTemplate.empty()) {
            qWarning() << "🌟 LightingCorrector: Failed to load template image:" << templatePath;
            return false;
        }
        
        m_templatePath = templatePath;
        qDebug() << "🌟 LightingCorrector: Reference template set successfully:" << templatePath;
        qDebug() << "🌟 LightingCorrector: Template size:" << m_referenceTemplate.cols << "x" << m_referenceTemplate.rows;
        
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Failed to set reference template:" << e.what();
        return false;
    }
}

cv::Mat LightingCorrector::applyGlobalLightingCorrection(const cv::Mat &inputImage)
{
    qDebug() << "🌟✨✨✨ applyGlobalLightingCorrection CALLED!";
    qDebug() << "🌟 Initialized:" << m_initialized;
    
    if (!m_initialized) {
        qWarning() << "🌟❌ LightingCorrector: Not initialized, returning original";
        return inputImage.clone();
    }
    
    if (inputImage.empty()) {
        qWarning() << "🌟❌ LightingCorrector: Empty input image for global correction";
        return cv::Mat();
    }
    
    qDebug() << "🌟 LightingCorrector: Applying global lighting correction to image size:" << inputImage.cols << "x" << inputImage.rows;
    qDebug() << "🌟 Using CLAHE clip limit:" << m_clipLimit << "Gamma:" << m_gammaValue;
    
    try {
        cv::Mat result;
        
        // 🚀 PERFORMANCE: Use GPU processing if available (with CPU fallback)
        if (m_gpuAvailable && !m_gpuInputBuffer.empty()) {
            try {
                // Upload to GPU
                cv::cuda::GpuMat gpuInput, gpuOutput;
                gpuInput.upload(inputImage);
                
                // Convert to LAB color space on GPU
                cv::cuda::cvtColor(gpuInput, gpuOutput, cv::COLOR_BGR2Lab);
                
                // Download for CPU processing (GPU split/merge not available)
                cv::Mat labImage;
                gpuOutput.download(labImage);
                
                // Apply CLAHE to L channel on CPU
                std::vector<cv::Mat> labChannels;
                cv::split(labImage, labChannels);
                m_gpuCLAHE->apply(cv::cuda::GpuMat(labChannels[0]), cv::cuda::GpuMat());
                cv::merge(labChannels, result);
                
                // Convert back to BGR on GPU
                gpuInput.upload(result);
                cv::cuda::cvtColor(gpuInput, gpuOutput, cv::COLOR_Lab2BGR);
                gpuOutput.download(result);
                
                // Apply gamma correction on CPU (fast operation)
                result = applyGammaCorrection(result, m_gammaValue * 0.8); // Reduced for performance
                
                return result;
                
            } catch (const cv::Exception& e) {
                qWarning() << "🌟 LightingCorrector: GPU correction failed, falling back to CPU:" << e.what();
                // Fall through to CPU processing
            }
        }
        
        // CPU fallback processing
        cv::Mat labImage;
        cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);
        
        // Split channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);
        
        // Apply CLAHE to L channel with reduced parameters for performance
        cv::Ptr<cv::CLAHE> fastCLAHE = cv::createCLAHE();
        fastCLAHE->setClipLimit(m_clipLimit * 0.7); // Reduced for performance
        fastCLAHE->setTilesGridSize(cv::Size(4, 4)); // Reduced from 8x8 for performance
        fastCLAHE->apply(labChannels[0], labChannels[0]);
        
        // Merge channels
        cv::merge(labChannels, result);
        
        // Convert back to BGR
        cv::cvtColor(result, result, cv::COLOR_Lab2BGR);
        
        // Apply reduced gamma correction
        result = applyGammaCorrection(result, m_gammaValue * 0.8);
        
        // qDebug() << "🌟 LightingCorrector: CPU global lighting correction completed";
        qDebug() << "🌟 Output size:" << result.cols << "x" << result.rows;
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Global correction failed:" << e.what();
        return inputImage.clone();
    }
}

cv::Mat LightingCorrector::applyGammaCorrection(const cv::Mat &input, double gamma)
{
    try {
        cv::Mat result;
        
        // Create lookup table
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar* p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
        }
        
        // Apply gamma correction
        cv::LUT(input, lookupTable, result);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Gamma correction failed:" << e.what();
        return input.clone();
    }
}

bool LightingCorrector::isGPUAvailable() const
{
    return m_gpuAvailable;
}

cv::Mat LightingCorrector::getReferenceTemplate() const
{
    return m_referenceTemplate.clone();
}

void LightingCorrector::cleanup()
{
    qDebug() << "🌟 LightingCorrector: Cleaning up resources";
    
    // Release GPU resources
    m_gpuCLAHE.release();
    m_gpuInputBuffer.release();
    m_gpuMaskBuffer.release();
    m_gpuTemplateBuffer.release();
    m_gpuOutputBuffer.release();
    m_gpuTempBuffer1.release();
    m_gpuTempBuffer2.release();
    
    // Release CPU resources
    m_cpuCLAHE.release();
    
    // Release template data
    m_referenceTemplate.release();
    
    m_initialized = false;
    m_gpuAvailable = false;
}
