#ifndef LIGHTING_CORRECTOR_H
#define LIGHTING_CORRECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <QString>
#include <QDebug>

/**
 * @brief Lighting correction system
 * 
 * This class provides lighting correction that:
 * - Uses background template as reference for optimal lighting
 * - Applies correction only to segmented subjects 
 * - Supports both GPU and CPU processing
 * - Maintains natural skin tones and prevents over-correction
 */
class LightingCorrector
{
public:
    LightingCorrector();
    ~LightingCorrector();
    
    /**
     * @brief Initialize the lighting correction system
     * @return true if initialization successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Apply lighting correction to entire image (fallback method)
     * @param inputImage Input image to correct
     * @return Corrected image
     */
    cv::Mat applyGlobalLightingCorrection(const cv::Mat &inputImage);
    
    // Check if GPU is available
    bool isGPUAvailable() const;
    
    // Gets the background template that's used as a reference
    cv::Mat getReferenceTemplate() const;

    // Loads the selected background image to use as a lighting reference
    bool setReferenceTemplate(const QString &templatePath);

    // Free up GPU/CPU resources
    void cleanup();

private:
    // Adjusts brightness using gamma curve
    cv::Mat applyGammaCorrection(const cv::Mat &input, double gamma);
    
    // Member variables
    bool m_gpuAvailable;
    bool m_initialized;
    cv::Mat m_referenceTemplate;  // Background image we use as lighting reference
    cv::Ptr<cv::cuda::CLAHE> m_gpuCLAHE;  // GPU contrast enhancer
    cv::Ptr<cv::CLAHE> m_cpuCLAHE;  // CPU contrast enhancer (fallback)
    
    // Tuning parameters - tweak these if results look off
    double m_clipLimit;  // How much contrast boost (higher = more aggressive)
    cv::Size m_tileGridSize;  // Grid size for adaptive processing
    double m_gammaValue;  // Brightness adjustment (1.5 = brighter)
};

#endif // LIGHTING_CORRECTOR_H
