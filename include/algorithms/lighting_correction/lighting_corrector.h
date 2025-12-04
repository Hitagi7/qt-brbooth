#ifndef LIGHTING_CORRECTOR_H
#define LIGHTING_CORRECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>  // OpenCL support
#include <QString>
#include <QDebug>
#include <vector>

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
     * @brief Apply lighting correction to entire image using histogram matching
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
    // Apply histogram matching to match input histogram to reference histogram
    cv::Mat applyHistogramMatching(const cv::Mat &input, const cv::Mat &reference);
    
    // Compute cumulative distribution function from histogram
    std::vector<float> computeCDF(const cv::Mat &hist);
    
    // Create lookup table for histogram matching
    std::vector<uchar> createHistogramMapping(const std::vector<float> &sourceCDF, 
                                               const std::vector<float> &referenceCDF);
    
    // Member variables
    bool m_gpuAvailable;
    bool m_initialized;
    cv::Mat m_referenceTemplate;  // Background image we use as lighting reference
};

#endif // LIGHTING_CORRECTOR_H
