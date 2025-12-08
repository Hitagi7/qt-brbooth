// Video Lighting Correction Implementation
// Video-optimized lighting correction algorithms for dynamic video processing

#include "core/capture.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include "core/lighting_corrector.h"

// Thread-safe wrapper for applyLightingToRawPersonRegion
// This function accesses member variables but is called from main thread context via lambda
// The lambda captures 'this' pointer, so we need to ensure thread safety
static cv::Mat applyLightingToRawPersonRegionSafe(const cv::Mat &personRegion, 
                                                   const cv::Mat &personMask,
                                                   LightingCorrector* lightingCorrector)
{
    // CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs for lighting correction - returning empty mat";
        return cv::Mat();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch between person region and mask - returning original";
        return personRegion.clone();
    }
    
    if (!lightingCorrector) {
        qWarning() << "No lighting corrector provided - returning original";
        return personRegion.clone();
    }
    
    try {
        // Apply global lighting correction as a simplified approach
        // This avoids accessing member variables that might not be thread-safe
        return lightingCorrector->applyGlobalLightingCorrection(personRegion);
    } catch (const std::exception& e) {
        qWarning() << "Lighting correction failed:" << e.what() << "- returning original";
        return personRegion.clone();
    }
}

// ============================================================================
//  VIDEO-OPTIMIZED LIGHTING CORRECTION (Based on Static Mode Algorithm)
// ============================================================================

cv::Mat Capture::applyVideoOptimizedLighting(const cv::Mat &personRegion, 
                                              const cv::Mat &personMask,
                                              LightingCorrector* lightingCorrector)
{
    // VIDEO-SPECIFIC: Follow static mode algorithm but optimized for video processing
    // This uses the same LAB color space matching as static mode but with video-optimized parameters
    
    // CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.type() != CV_8UC3) {
        qWarning() << "Invalid person region format for video lighting - returning original";
        return personRegion.clone();
    }
    
    if (personMask.type() != CV_8UC1) {
        qWarning() << "Invalid mask format for video lighting - returning original";
        return personRegion.clone();
    }
    
    // Start with exact copy of person region
    cv::Mat result = personRegion.clone();
    
    // If no lighting corrector, return original
    if (!lightingCorrector) {
        return result;
    }
    
    try {
        // Get template reference for color matching (same as static mode)
        cv::Mat templateRef = lightingCorrector->getReferenceTemplate();
        
        if (templateRef.empty()) {
            // No template: Apply very subtle brightness adjustment for video (even more subtle than static)
            // VIDEO-OPTIMIZED: More conservative adjustments for video frames
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    if (y < personMask.rows && x < personMask.cols && 
                        personMask.at<uchar>(y, x) > 0) {  // Person pixel
                        cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                        // VIDEO-SPECIFIC: Very subtle changes (less aggressive than static mode)
                        pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 1.05);  // Very slight blue boost
                        pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 1.02);  // Very slight green boost
                        pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 1.04);  // Very slight red boost
                    }
                }
            }
        } else {
            // VIDEO-OPTIMIZED: Use same LAB color space matching as static mode but with video-specific parameters
            cv::resize(templateRef, templateRef, personRegion.size());
            
            // Convert to LAB for color matching (same as static mode)
            cv::Mat personLab, templateLab;
            cv::cvtColor(personRegion, personLab, cv::COLOR_BGR2Lab);
            cv::cvtColor(templateRef, templateLab, cv::COLOR_BGR2Lab);
            
            // Calculate template statistics (same as static mode)
            cv::Scalar templateMean, templateStd;
            cv::meanStdDev(templateLab, templateMean, templateStd);
            
            // Apply color matching to person region (same algorithm as static mode)
            cv::Mat resultLab = personLab.clone();
            std::vector<cv::Mat> channels;
            cv::split(resultLab, channels);
            
            // Calculate person statistics for comparison
            cv::Scalar personMean, personStd;
            cv::meanStdDev(personLab, personMean, personStd);
            
            // VIDEO-SPECIFIC: More conservative adjustments than static mode
            // Static mode uses 15% adjustment, video uses 10% for smoother frame-to-frame transitions
            for (int c = 0; c < 3; c++) {
                // Calculate the difference between template and person
                double lightingDiff = templateMean[c] - personMean[c];
                
                // VIDEO-OPTIMIZED: Apply more conservative adjustment (10% vs 15% in static)
                // This prevents jarring changes between video frames
                channels[c] = channels[c] + lightingDiff * 0.10;
            }
            
            // VIDEO-SPECIFIC: More conservative brightness adjustment
            // Static mode uses 10%, video uses 5% for smoother transitions
            double brightnessDiff = templateMean[0] - personMean[0]; // L channel
            if (brightnessDiff > 0) {
                channels[0] = channels[0] + brightnessDiff * 0.05; // Very slight brightness boost
            }
            
            cv::merge(channels, resultLab);
            cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
            
            // Apply mask to ensure only person pixels are affected
            cv::Mat maskedResult;
            result.copyTo(maskedResult, personMask);
            personRegion.copyTo(maskedResult, ~personMask);
            result = maskedResult;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        qWarning() << "Video lighting correction exception:" << e.what() << "- returning original";
        return personRegion.clone();
    }
}
