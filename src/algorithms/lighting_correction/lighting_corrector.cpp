#include "algorithms/lighting_correction/lighting_corrector.h"
#include <QFileInfo>
#include <cmath>

/**
 * @brief Constructor - Initializes lighting correction parameters
 * 
 * Sets up default values for CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * and gamma correction. These parameters control how aggressively lighting correction
 * is applied to improve image visibility and quality.
 */
LightingCorrector::LightingCorrector()
    : m_gpuAvailable(false)
    , m_initialized(false)
    , m_clipLimit(8.0)      // Higher = more contrast boost
    , m_tileGridSize(8, 8)  // 8x8 grid for adaptive processing
    , m_gammaValue(1.5)     // Brightens the image (1.0 = no change)
{
}

// Cleans up resources
LightingCorrector::~LightingCorrector()
{
    cleanup();
}

/**
 * @brief Initialize the lighting correction system
 * 
 * Sets up both GPU and CPU processing pipelines:
 * - GPU: Creates CUDA CLAHE processor if CUDA is available
 * - CPU: Creates CPU CLAHE processor as fallback
 * 
 * @return true if initialization successful, false otherwise
 */
bool LightingCorrector::initialize()
{
    try {
        // Check if CUDA-enabled GPU is available
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            // Initialize GPU CLAHE (Contrast Limited Adaptive Histogram Equalization)
            // CLAHE improves local contrast while preventing over-amplification
            m_gpuCLAHE = cv::cuda::createCLAHE();
            m_gpuCLAHE->setClipLimit(m_clipLimit);
            m_gpuCLAHE->setTilesGridSize(m_tileGridSize);
            
            m_gpuAvailable = true;
            qDebug() << "LightingCorrector: GPU acceleration enabled";
        } else {
            m_gpuAvailable = false;
            qDebug() << "LightingCorrector: GPU not available, using CPU processing";
        }
        
        // Initialize CPU CLAHE as fallback
        // Uses slightly reduced parameters for better performance on CPU
        m_cpuCLAHE = cv::createCLAHE();
        m_cpuCLAHE->setClipLimit(m_clipLimit * 0.8);
        m_cpuCLAHE->setTilesGridSize(cv::Size(6, 6));
        
        m_initialized = true;
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Initialization failed:" << e.what();
        m_initialized = false;
        m_gpuAvailable = false;
        return false;
    }
}

// Load a background image to use as a lighting reference
// This helps match the person's lighting to the background
bool LightingCorrector::setReferenceTemplate(const QString &templatePath)
{
    if (!m_initialized) {
        qWarning() << "LightingCorrector: Not initialized, cannot set reference template";
        return false;
    }
    
    if (templatePath.isEmpty()) {
        qWarning() << "LightingCorrector: Empty template path provided";
        return false;
    }
    
    // Verify file exists before attempting to load
    QFileInfo fileInfo(templatePath);
    if (!fileInfo.exists()) {
        qWarning() << "LightingCorrector: Template file does not exist:" << templatePath;
        return false;
    }
    
    try {
        // Load template image using OpenCV
        m_referenceTemplate = cv::imread(templatePath.toStdString());
        if (m_referenceTemplate.empty()) {
            qWarning() << "LightingCorrector: Failed to load template image:" << templatePath;
            return false;
        }
        
        qDebug() << "LightingCorrector: Reference template loaded:" << templatePath
                 << "Size:" << m_referenceTemplate.cols << "x" << m_referenceTemplate.rows;
        
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Failed to set reference template:" << e.what();
        return false;
    }
}

/**
 * @brief Apply lighting correction to entire image
 * 
 * This method improves image lighting using a two-step process:
 * 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel in LAB color space
 *    - Works in LAB color space to preserve color while enhancing brightness/contrast
 *    - Only processes the L (lightness) channel to avoid color shifts
 * 2. Gamma correction to brighten the overall image
 * 
 * The method attempts GPU acceleration first, then falls back to CPU if GPU fails.
 * 
 * @param inputImage Input image in BGR format
 * @return Corrected image in BGR format, or original image if correction fails
 */
cv::Mat LightingCorrector::applyGlobalLightingCorrection(const cv::Mat &inputImage)
{
    if (!m_initialized) {
        qWarning() << "LightingCorrector: Not initialized, returning original image";
        return inputImage.clone();
    }
    
    if (inputImage.empty()) {
        qWarning() << "LightingCorrector: Empty input image";
        return cv::Mat();
    }
    
    try {
        cv::Mat result;
        
        // Attempt GPU-accelerated processing if available
        if (m_gpuAvailable) {
            try {
                // Upload image to GPU
                cv::cuda::GpuMat gpuInput, gpuLab;
                gpuInput.upload(inputImage);
                
                // Convert BGR to LAB color space on GPU
                // LAB separates lightness (L) from color (A, B channels)
                cv::cuda::cvtColor(gpuInput, gpuLab, cv::COLOR_BGR2Lab);
                
                // Download LAB image to CPU for channel processing
                // (OpenCV CUDA doesn't have split/merge operations)
                cv::Mat labImage;
                gpuLab.download(labImage);
                
                // Split channels so we can work on just the L (lightness) channel
                std::vector<cv::Mat> labChannels;
                cv::split(labImage, labChannels);
                
                // Upload L channel to GPU and apply CLAHE
                cv::cuda::GpuMat gpuLChannel, gpuLChannelCorrected;
                gpuLChannel.upload(labChannels[0]);
                m_gpuCLAHE->apply(gpuLChannel, gpuLChannelCorrected);
                
                // Download corrected L channel back to CPU
                gpuLChannelCorrected.download(labChannels[0]);
                
                // Merge channels back together
                cv::merge(labChannels, labImage);
                
                // Convert back to BGR on GPU
                gpuLab.upload(labImage);
                cv::cuda::cvtColor(gpuLab, gpuInput, cv::COLOR_Lab2BGR);
                gpuInput.download(result);
                
                // Apply gamma correction on CPU (fast operation, no need for GPU)
                result = applyGammaCorrection(result, m_gammaValue * 0.8);
                
                return result;
                
            } catch (const cv::Exception& e) {
                qWarning() << "LightingCorrector: GPU processing failed, falling back to CPU:" << e.what();
                // Fall through to CPU processing
            }
        }
        
        // CPU path (either no GPU or GPU failed)
        cv::Mat labImage;
        cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);
        
        // Split channels and boost contrast on L channel only
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);
        
        // Apply CLAHE to L (lightness) channel only
        // This enhances contrast without affecting color
        m_cpuCLAHE->apply(labChannels[0], labChannels[0]);
        
        // Merge channels back together
        cv::merge(labChannels, labImage);
        
        // Convert back to BGR color space
        cv::cvtColor(labImage, result, cv::COLOR_Lab2BGR);
        
        // Apply gamma correction to brighten the image
        result = applyGammaCorrection(result, m_gammaValue * 0.8);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Lighting correction failed:" << e.what();
        return inputImage.clone();
    }
}

/**
 * @brief Apply gamma correction to adjust image brightness
 * 
 * Gamma correction is a non-linear operation that adjusts the brightness of an image.
 * - gamma > 1.0: Brightens the image (makes dark areas lighter)
 * - gamma < 1.0: Darkens the image (makes bright areas darker)
 * - gamma = 1.0: No change
 * 
 * Uses a lookup table (LUT) for efficient pixel value transformation.
 * 
 * @param input Input image to correct
 * @param gamma Gamma value (typically > 1.0 for brightening)
 * @return Gamma-corrected image
 */
cv::Mat LightingCorrector::applyGammaCorrection(const cv::Mat &input, double gamma)
{
    try {
        cv::Mat result;
        
        // Build lookup table once, then apply it to all pixels (much faster)
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar* p = lookupTable.ptr();
        
        // Fill lookup table with gamma-corrected values
        // Formula: output = (input/255)^(1/gamma) * 255
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
        }
        
        cv::LUT(input, lookupTable, result);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Gamma correction failed:" << e.what();
        return input.clone();
    }
}

/**
 * @brief Check if GPU acceleration is available
 * 
 * @return true if GPU is available and initialized, false otherwise
 */
bool LightingCorrector::isGPUAvailable() const
{
    return m_gpuAvailable;
}

// Get a copy of the current reference template
cv::Mat LightingCorrector::getReferenceTemplate() const
{
    return m_referenceTemplate.clone();
}

// Free up GPU memory and reset everything
void LightingCorrector::cleanup()
{
    m_gpuCLAHE.release();
    m_cpuCLAHE.release();
    m_referenceTemplate.release();
    m_initialized = false;
    m_gpuAvailable = false;
}
