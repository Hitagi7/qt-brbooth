#include "algorithms/lighting_correction/lighting_corrector.h"
#include <QFileInfo>
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * @brief Constructor - Initializes lighting correction parameters
 * 
 * Sets up histogram matching system for lighting correction.
 */
LightingCorrector::LightingCorrector()
    : m_gpuAvailable(false)
    , m_initialized(false)
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
 * Sets up histogram matching system with GPU support if available.
 * 
 * @return true if initialization successful, false otherwise
 */
bool LightingCorrector::initialize()
{
    try {
        // Check if OpenCL-enabled GPU is available
        if (cv::ocl::useOpenCL()) {
            cv::ocl::setUseOpenCL(true);
            m_gpuAvailable = true;
            qDebug() << "LightingCorrector: OpenCL GPU acceleration enabled";
        } else {
            m_gpuAvailable = false;
            qDebug() << "LightingCorrector: GPU not available, using CPU processing";
        }
        
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
 * @brief Apply lighting correction to entire image using histogram matching
 * 
 * This method matches the histogram of the input image to the reference template's histogram.
 * If no reference template is available, it returns the original image.
 * 
 * The method works in LAB color space, matching only the L (lightness) channel to preserve
 * color information while adjusting lighting characteristics.
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
    
    // If no reference template, return original image
    if (m_referenceTemplate.empty()) {
        qDebug() << "LightingCorrector: No reference template, returning original image";
        return inputImage.clone();
    }
    
    try {
        // Resize reference template to match input image size
        cv::Mat reference = m_referenceTemplate.clone();
        if (reference.size() != inputImage.size()) {
            cv::resize(reference, reference, inputImage.size());
        }
        
        // Apply histogram matching
        return applyHistogramMatching(inputImage, reference);
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Lighting correction failed:" << e.what();
        return inputImage.clone();
    }
}

/**
 * @brief Apply histogram matching to match input histogram to reference histogram
 * 
 * This function matches the histogram of the input image to the reference image's histogram.
 * It works in LAB color space, matching only the L (lightness) channel to preserve color
 * while adjusting lighting characteristics.
 * 
 * @param input Input image in BGR format
 * @param reference Reference image in BGR format
 * @return Histogram-matched image in BGR format
 */
cv::Mat LightingCorrector::applyHistogramMatching(const cv::Mat &input, const cv::Mat &reference)
{
    try {
        // Convert both images to LAB color space
        cv::Mat inputLab, referenceLab;
        cv::cvtColor(input, inputLab, cv::COLOR_BGR2Lab);
        cv::cvtColor(reference, referenceLab, cv::COLOR_BGR2Lab);
        
        // Split channels
        std::vector<cv::Mat> inputChannels, referenceChannels;
        cv::split(inputLab, inputChannels);
        cv::split(referenceLab, referenceChannels);
        
        // Compute histograms for L channel only
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        bool uniform = true, accumulate = false;
        
        cv::Mat inputHist, referenceHist;
        cv::calcHist(&inputChannels[0], 1, 0, cv::Mat(), inputHist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&referenceChannels[0], 1, 0, cv::Mat(), referenceHist, 1, &histSize, &histRange, uniform, accumulate);
        
        // Normalize histograms
        cv::normalize(inputHist, inputHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(referenceHist, referenceHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        
        // Compute CDFs
        std::vector<float> inputCDF = computeCDF(inputHist);
        std::vector<float> referenceCDF = computeCDF(referenceHist);
        
        // Create mapping
        std::vector<uchar> mapping = createHistogramMapping(inputCDF, referenceCDF);
        
        // Apply mapping to L channel
        cv::Mat matchedL;
        cv::LUT(inputChannels[0], cv::Mat(1, 256, CV_8UC1, mapping.data()), matchedL);
        
        // Replace L channel with matched version
        inputChannels[0] = matchedL;
        
        // Merge channels back
        cv::Mat resultLab;
        cv::merge(inputChannels, resultLab);
        
        // Convert back to BGR
        cv::Mat result;
        cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "LightingCorrector: Histogram matching failed:" << e.what();
        return input.clone();
    }
}

/**
 * @brief Compute cumulative distribution function from histogram
 * 
 * @param hist Input histogram (normalized)
 * @return CDF as a vector of floats
 */
std::vector<float> LightingCorrector::computeCDF(const cv::Mat &hist)
{
    std::vector<float> cdf(256, 0.0f);
    float sum = 0.0f;
    
    for (int i = 0; i < 256; ++i) {
        sum += hist.at<float>(i);
        cdf[i] = sum;
    }
    
    return cdf;
}

/**
 * @brief Create lookup table for histogram matching
 * 
 * Maps each source pixel value to a reference pixel value based on matching CDF values.
 * 
 * @param sourceCDF CDF of source image
 * @param referenceCDF CDF of reference image
 * @return Lookup table mapping source values to reference values
 */
std::vector<uchar> LightingCorrector::createHistogramMapping(const std::vector<float> &sourceCDF, 
                                                              const std::vector<float> &referenceCDF)
{
    std::vector<uchar> mapping(256);
    
    for (int i = 0; i < 256; ++i) {
        float sourceValue = sourceCDF[i];
        
        // Find the closest reference CDF value
        int bestMatch = 0;
        float minDiff = std::abs(sourceValue - referenceCDF[0]);
        
        for (int j = 1; j < 256; ++j) {
            float diff = std::abs(sourceValue - referenceCDF[j]);
            if (diff < minDiff) {
                minDiff = diff;
                bestMatch = j;
            }
        }
        
        mapping[i] = static_cast<uchar>(bestMatch);
    }
    
    return mapping;
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
    m_referenceTemplate.release();
    m_initialized = false;
    m_gpuAvailable = false;
}
