#include "algorithms/lighting_correction/lighting_corrector.h"
#include <QFileInfo>
#include <cmath>

LightingCorrector::LightingCorrector()
    : m_enabled(true)
    , m_gpuAvailable(false)
    , m_initialized(false)
    , m_clipLimit(8.0)  // More aggressive CLAHE for visible effect
    , m_tileGridSize(8, 8)
    , m_gammaValue(1.5)  // More aggressive gamma correction
    , m_colorBalanceStrength(1.2)  // More aggressive color balance
{
    qDebug() << "🌟 LightingCorrector: Constructor called";
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
        
        // Analyze template lighting characteristics
        m_templateLightingProfile = analyzeTemplateLighting(m_referenceTemplate);
        
        m_templatePath = templatePath;
        qDebug() << "🌟 LightingCorrector: Reference template set successfully:" << templatePath;
        qDebug() << "🌟 LightingCorrector: Template size:" << m_referenceTemplate.cols << "x" << m_referenceTemplate.rows;
        
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Failed to set reference template:" << e.what();
        return false;
    }
}

// Helper: Match histogram of source to reference
static cv::Mat matchHistogram(const cv::Mat& src, const cv::Mat& ref, const cv::Mat& mask = cv::Mat())
{
    CV_Assert(src.type() == CV_8UC3 && ref.type() == CV_8UC3);

    cv::Mat srcYCrCb, refYCrCb;
    cv::cvtColor(src, srcYCrCb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(ref, refYCrCb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> srcChannels, refChannels;
    cv::split(srcYCrCb, srcChannels);
    cv::split(refYCrCb, refChannels);

    for (int i = 0; i < 3; i++) {
        // Compute histogram
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};

        cv::Mat srcHist, refHist;
        cv::calcHist(&srcChannels[i], 1, 0, mask, srcHist, 1, &histSize, &histRange, true, false);
        cv::calcHist(&refChannels[i], 1, 0, cv::Mat(), refHist, 1, &histSize, &histRange, true, false);

        // Normalize histograms
        cv::normalize(srcHist, srcHist, 0, 1, cv::NORM_MINMAX);
        cv::normalize(refHist, refHist, 0, 1, cv::NORM_MINMAX);

        // Build LUT for mapping
        std::vector<int> lut(256, 0);
        int ri = 0;
        float srcCDF = 0, refCDF = 0;

        for (int v = 0; v < 256; v++) {
            srcCDF += srcHist.at<float>(v);
            while (ri < 256 && refCDF < srcCDF) {
                refCDF += refHist.at<float>(ri++);
            }
            lut[v] = std::min(ri, 255);
        }

        // Apply LUT
        cv::Mat lutMat(1, 256, CV_8U);
        for (int v = 0; v < 256; v++) lutMat.at<uchar>(v) = static_cast<uchar>(lut[v]);
        cv::LUT(srcChannels[i], lutMat, srcChannels[i]);
    }

    cv::merge(srcChannels, srcYCrCb);
    cv::Mat matched;
    cv::cvtColor(srcYCrCb, matched, cv::COLOR_YCrCb2BGR);
    return matched;
}

cv::Mat LightingCorrector::applyPersonLightingCorrection(
    const cv::Mat &inputImage,
    const cv::Mat &personMaskIn,
    const cv::Mat &referenceTemplate)
{
    if (!m_enabled || !m_initialized) return inputImage.clone();
    if (inputImage.empty() || personMaskIn.empty() || referenceTemplate.empty())
        return inputImage.clone();

    try {
        // 🚀 PERFORMANCE OPTIMIZATION: Simplified for performance
        // Skip complex processing - just return clone for now
        
        // --- 0) Prepare copies and canonical formats ---
        cv::Mat frame;
        if (inputImage.type() != CV_8UC3) inputImage.convertTo(frame, CV_8UC3);
        else frame = inputImage.clone();

        cv::Mat mask;
        if (personMaskIn.channels() == 3) cv::cvtColor(personMaskIn, mask, cv::COLOR_BGR2GRAY);
        else mask = personMaskIn.clone();
        if (mask.type() != CV_8U) mask.convertTo(mask, CV_8U);
        double mn, mx; cv::minMaxLoc(mask, &mn, &mx);
        if (mx <= 1.0) mask *= 255;

        // --- 1) Template resize ---
        cv::Mat tmpl;
        if (referenceTemplate.size() != frame.size())
            cv::resize(referenceTemplate, tmpl, frame.size(), 0, 0, cv::INTER_AREA);
        else
            tmpl = referenceTemplate.clone();

        // 🚀 PERFORMANCE: Simplified mask cleanup 
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));

        // Simple return for performance - skip complex lighting processing
        return inputImage.clone();

    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Person lighting correction failed:" << e.what();
        return inputImage.clone();
    }
}

cv::Mat LightingCorrector::applyGlobalLightingCorrection(const cv::Mat &inputImage)
{
    if (!m_enabled || !m_initialized) {
        qDebug() << "🌟 LightingCorrector: Not enabled or not initialized, returning original";
        return inputImage.clone();
    }
    
    if (inputImage.empty()) {
        qWarning() << "🌟 LightingCorrector: Empty input image for global correction";
        return cv::Mat();
    }
    
    // 🚀 PERFORMANCE: Reduce debug logging during intensive processing
    // qDebug() << "🌟 LightingCorrector: Applying global lighting correction to image size:" << inputImage.cols << "x" << inputImage.rows;
    
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
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Global correction failed:" << e.what();
        return inputImage.clone();
    }
}


cv::Mat LightingCorrector::analyzeTemplateLighting(const cv::Mat &templateImage)
{
    try {
        // Convert to LAB and analyze L channel statistics
        cv::Mat labTemplate;
        cv::cvtColor(templateImage, labTemplate, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> channels;
        cv::split(labTemplate, channels);
        
        // Calculate lighting profile (mean, std dev, etc.)
        cv::Scalar mean, stddev;
        cv::meanStdDev(channels[0], mean, stddev);
        
        // Create a simple lighting profile
        cv::Mat profile = cv::Mat::zeros(1, 3, CV_64F);
        profile.at<double>(0) = mean[0];  // Mean lightness
        profile.at<double>(1) = stddev[0]; // Standard deviation
        profile.at<double>(2) = cv::mean(channels[0])[0]; // Overall brightness
        
        return profile;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Template analysis failed:" << e.what();
        return cv::Mat::zeros(1, 3, CV_64F);
    }
}

cv::Mat LightingCorrector::applyColorBalanceWithReference(const cv::Mat &input, 
                                                         const cv::Mat &reference, 
                                                         const cv::Mat &mask)
{
    try {
        cv::Mat result = input.clone();
        
        // Calculate mean values for person region only
        cv::Scalar personMean = cv::mean(input, mask);
        cv::Scalar referenceMean = cv::mean(reference);
        
        // Calculate correction factors
        double factorB = referenceMean[0] / (personMean[0] + 1e-6);
        double factorG = referenceMean[1] / (personMean[1] + 1e-6);
        double factorR = referenceMean[2] / (personMean[2] + 1e-6);
        
        // Limit correction factors
        factorB = std::min(std::max(factorB, 0.7), 1.3);
        factorG = std::min(std::max(factorG, 0.7), 1.3);
        factorR = std::min(std::max(factorR, 0.7), 1.3);
        
        // Apply correction only to person region
        std::vector<cv::Mat> channels;
        cv::split(result, channels);
        
        cv::Mat maskFloat;
        mask.convertTo(maskFloat, CV_32F, 1.0/255.0);
        
        // Apply color balance with mask
        cv::Mat correctedB, correctedG, correctedR;
        cv::multiply(channels[0], factorB, correctedB);
        cv::multiply(channels[1], factorG, correctedG);
        cv::multiply(channels[2], factorR, correctedR);
        
        // Blend with original using mask
        cv::Mat finalB, finalG, finalR;
        cv::multiply(channels[0], (1.0 - maskFloat), finalB);
        cv::multiply(correctedB, maskFloat, correctedB);
        cv::add(finalB, correctedB, finalB);
        
        cv::multiply(channels[1], (1.0 - maskFloat), finalG);
        cv::multiply(correctedG, maskFloat, correctedG);
        cv::add(finalG, correctedG, finalG);
        
        cv::multiply(channels[2], (1.0 - maskFloat), finalR);
        cv::multiply(correctedR, maskFloat, finalR);
        cv::add(finalR, finalR, finalR);
        
        std::vector<cv::Mat> finalChannels = {finalB, finalG, finalR};
        cv::merge(finalChannels, result);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Color balance failed:" << e.what();
        return input.clone();
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

cv::Mat LightingCorrector::createSmoothMask(const cv::Mat &binaryMask)
{
    try {
        cv::Mat smoothMask;
        
        // Apply Gaussian blur for smooth edges
        cv::GaussianBlur(binaryMask, smoothMask, cv::Size(15, 15), 0);
        
        // Apply morphological operations for better mask quality
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(smoothMask, smoothMask, cv::MORPH_CLOSE, kernel);
        
        return smoothMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Smooth mask creation failed:" << e.what();
        return binaryMask.clone();
    }
}

void LightingCorrector::setEnabled(bool enabled)
{
    m_enabled = enabled;
    qDebug() << "🌟 LightingCorrector: Lighting correction" << (enabled ? "enabled" : "disabled");
}

bool LightingCorrector::isEnabled() const
{
    return m_enabled;
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
    m_templateLightingProfile.release();
    
    m_initialized = false;
    m_gpuAvailable = false;
}
