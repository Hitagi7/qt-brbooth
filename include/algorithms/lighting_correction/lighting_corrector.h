#ifndef LIGHTING_CORRECTOR_H
#define LIGHTING_CORRECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/imgproc.hpp>
#include <QString>
#include <QDebug>

/**
 * @brief Advanced lighting correction system for photo booth applications
 * 
 * This class provides intelligent lighting correction that:
 * - Uses background template as reference for optimal lighting
 * - Applies correction only to segmented subjects (not background)
 * - Supports both GPU and CPU processing
 * - Maintains natural skin tones and prevents over-correction
 */
class LightingCorrector
{
public:
    /**
     * @brief Constructor
     */
    LightingCorrector();
    
    /**
     * @brief Destructor
     */
    ~LightingCorrector();
    
    /**
     * @brief Initialize the lighting correction system
     * @return true if initialization successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Set the reference template for lighting correction
     * @param templatePath Path to the background template image
     * @return true if template loaded successfully, false otherwise
     */
    bool setReferenceTemplate(const QString &templatePath);
    
    /**
     * @brief Apply lighting correction to segmented person only
     * @param inputImage Original image with segmented person
     * @param personMask Binary mask indicating person pixels (255=person, 0=background)
     * @param referenceTemplate Background template for lighting reference
     * @return Corrected image with improved lighting on person only
     */
    cv::Mat applyPersonLightingCorrection(const cv::Mat &inputImage, 
                                         const cv::Mat &personMask, 
                                         const cv::Mat &referenceTemplate);
    
    /**
     * @brief Apply lighting correction to entire image (fallback method)
     * @param inputImage Input image to correct
     * @return Corrected image
     */
    cv::Mat applyGlobalLightingCorrection(const cv::Mat &inputImage);
    
    /**
     * @brief Enable or disable lighting correction
     * @param enabled true to enable, false to disable
     */
    void setEnabled(bool enabled);
    
    /**
     * @brief Check if lighting correction is enabled
     * @return true if enabled, false otherwise
     */
    bool isEnabled() const;
    
    /**
     * @brief Check if GPU acceleration is available
     * @return true if GPU available, false otherwise
     */
    bool isGPUAvailable() const;
    
    /**
     * @brief Get the current reference template
     * @return Reference template image
     */
    cv::Mat getReferenceTemplate() const;
    
    /**
     * @brief Clean up resources
     */
    void cleanup();

private:
    // Core lighting correction methods
    cv::Mat analyzeTemplateLighting(const cv::Mat &templateImage);
    cv::Mat correctPersonLighting(const cv::Mat &personRegion, 
                                 const cv::Mat &templateLighting, 
                                 const cv::Mat &personMask);
    cv::Mat applyCLAHEWithReference(const cv::Mat &input, 
                                   const cv::Mat &reference, 
                                   const cv::Mat &mask);
    cv::Mat applyColorBalanceWithReference(const cv::Mat &input, 
                                          const cv::Mat &reference, 
                                          const cv::Mat &mask);
    cv::Mat applyGammaCorrection(const cv::Mat &input, double gamma);
    
    
    // Utility methods
    cv::Mat extractPersonRegion(const cv::Mat &inputImage, const cv::Mat &personMask);
    cv::Mat compositeCorrectedPerson(const cv::Mat &originalImage, 
                                    const cv::Mat &correctedPerson, 
                                    const cv::Mat &personMask);
    cv::Mat createSmoothMask(const cv::Mat &binaryMask);
    
    // Member variables
    bool m_enabled;
    bool m_gpuAvailable;
    bool m_initialized;
    
    // Reference template
    cv::Mat m_referenceTemplate;
    cv::Mat m_templateLightingProfile;
    QString m_templatePath;
    
    // GPU resources
    cv::Ptr<cv::cuda::CLAHE> m_gpuCLAHE;
    cv::cuda::GpuMat m_gpuInputBuffer;
    cv::cuda::GpuMat m_gpuMaskBuffer;
    cv::cuda::GpuMat m_gpuTemplateBuffer;
    cv::cuda::GpuMat m_gpuOutputBuffer;
    cv::cuda::GpuMat m_gpuTempBuffer1;
    cv::cuda::GpuMat m_gpuTempBuffer2;
    
    // CPU resources
    cv::Ptr<cv::CLAHE> m_cpuCLAHE;
    
    // Processing parameters
    double m_clipLimit;
    cv::Size m_tileGridSize;
    double m_gammaValue;
    double m_colorBalanceStrength;
};

#endif // LIGHTING_CORRECTOR_H
