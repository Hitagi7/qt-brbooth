// Dynamic Video Edge Blending Implementation
// Thread-safe edge blending for dynamic video frames

#include "core/capture.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include "core/lighting_corrector.h"

// Forward declarations for helper functions defined in capture_edge_blending.cpp
extern cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                                 GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream);
extern cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

// Forward declaration - this is a member function of Capture class
// We'll call it through the Capture instance

// THREAD-SAFE WRAPPER: Takes all parameters instead of accessing member variables
cv::Mat Capture::applyDynamicFrameEdgeBlendingSafe(const cv::Mat &composedFrame,
                                                   const cv::Mat &rawPersonRegion,
                                                   const cv::Mat &rawPersonMask,
                                                   const cv::Mat &backgroundFrame,
                                                   LightingCorrector* lightingCorrector,
                                                   double personScaleFactor,
                                                   const cv::Mat &lastTemplateBackground,
                                                   bool useCUDA,
                                                   GPUMemoryPool* gpuMemoryPool)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        qWarning() << "Invalid input data for edge blending, using global correction";
        if (lightingCorrector) {
            return lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    }
    
    try {
        // Start with clean background or use provided background frame
        cv::Mat result;
        cv::Mat cleanBackground;
        
        if (!backgroundFrame.empty()) {
            cv::resize(backgroundFrame, cleanBackground, composedFrame.size());
        } else {
            // Extract background from dynamic template or use clean template
            if (!lastTemplateBackground.empty()) {
                cv::resize(lastTemplateBackground, cleanBackground, composedFrame.size());
            } else {
                // Fallback to zero background
                cleanBackground = cv::Mat::zeros(composedFrame.size(), composedFrame.type());
            }
        }
        result = cleanBackground.clone();
        
        // VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        // Follows the same algorithm as static mode but with video-optimized parameters
        // Note: This is a member function, but we're in a member function context, so we can call it directly
        // However, since this is a static-safe wrapper, we need to call it through a helper
        // For now, we'll use a local implementation or call through the class
        cv::Mat lightingCorrectedPerson;
        // We need to call the member function - but since we're in a member function, we can use 'this'
        // However, to maintain thread safety, we'll use a helper function
        // Actually, we can call it directly since we're in Capture::applyDynamicFrameEdgeBlendingSafe
        lightingCorrectedPerson = this->applyVideoOptimizedLighting(rawPersonRegion, rawPersonMask, lightingCorrector);
        
        // SCALING PRESERVATION: Scale the lighting-corrected person using the recorded scaling factor
        cv::Mat scaledPerson, scaledMask;
        
        // Calculate the scaled size using the recorded scaling factor
        cv::Size backgroundSize = result.size();
        cv::Size scaledPersonSize;
        
        if (qAbs(personScaleFactor - 1.0) > 0.01) {
            int scaledWidth = static_cast<int>(backgroundSize.width * personScaleFactor + 0.5);
            int scaledHeight = static_cast<int>(backgroundSize.height * personScaleFactor + 0.5);
            
            //  CRASH PREVENTION: Ensure scaled size is always valid (at least 1x1)
            scaledWidth = qMax(1, scaledWidth);
            scaledHeight = qMax(1, scaledHeight);
            
            scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
        } else {
            scaledPersonSize = backgroundSize;
        }
        
        // Scale person and mask to the calculated size
        cv::resize(lightingCorrectedPerson, scaledPerson, scaledPersonSize);
        cv::resize(rawPersonMask, scaledMask, scaledPersonSize);
        
        // Calculate centered offset for placing the scaled person
        cv::Size actualScaledSize(scaledPerson.cols, scaledPerson.rows);
        int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
        int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;
        
        // If person is scaled down, place it on a full-size canvas at the centered position
        cv::Mat fullSizePerson, fullSizeMask;
        if (actualScaledSize != backgroundSize) {
            // Create full-size images initialized to zeros
            fullSizePerson = cv::Mat::zeros(backgroundSize, scaledPerson.type());
            fullSizeMask = cv::Mat::zeros(backgroundSize, CV_8UC1);
            
            // Ensure offsets are valid
            if (xOffset >= 0 && yOffset >= 0 &&
                xOffset + actualScaledSize.width <= backgroundSize.width &&
                yOffset + actualScaledSize.height <= backgroundSize.height) {
                
                // Place scaled person at centered position
                cv::Rect roi(xOffset, yOffset, actualScaledSize.width, actualScaledSize.height);
                scaledPerson.copyTo(fullSizePerson(roi));
                
                // Convert mask to grayscale if needed, then copy to ROI
                if (scaledMask.type() != CV_8UC1) {
                    cv::Mat grayMask;
                    cv::cvtColor(scaledMask, grayMask, cv::COLOR_BGR2GRAY);
                    grayMask.copyTo(fullSizeMask(roi));
                } else {
                    scaledMask.copyTo(fullSizeMask(roi));
                }
            } else {
                qWarning() << "Invalid offset, using direct copy";
                cv::resize(scaledPerson, fullSizePerson, backgroundSize);
                cv::resize(scaledMask, fullSizeMask, backgroundSize);
            }
        } else {
            // Person is full size, use as is
            fullSizePerson = scaledPerson;
            if (scaledMask.type() != CV_8UC1) {
                cv::cvtColor(scaledMask, fullSizeMask, cv::COLOR_BGR2GRAY);
            } else {
                fullSizeMask = scaledMask;
            }
        }
        
        // Now use fullSizePerson and fullSizeMask for blending
        scaledPerson = fullSizePerson;
        scaledMask = fullSizeMask;
        
        // Apply guided filter edge blending
        cv::Mat binMask;
        cv::threshold(scaledMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // First: shrink mask slightly to avoid fringe, then hard-copy interior
        cv::Mat interiorMask;
        cv::erode(binMask, interiorMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1)));
        scaledPerson.copyTo(result, interiorMask);

        //  ENHANCED: Pre-smooth the mask before guided filtering for better edge quality
        cv::Mat smoothedBinMask;
        cv::GaussianBlur(binMask, smoothedBinMask, cv::Size(9, 9), 2.0); // Smooth mask edges first
        
        //  CUDA-Accelerated Guided image filtering
        const int gfRadius = 12; // Increased window size for smoother edges
        const float gfEps = 5e-3f; // Reduced regularization for better edge preservation
        
        // CRASH PREVENTION: Validate GPU memory pool before use
        cv::Mat alphaFloat;
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& guidedFilterStream = gpuMemoryPool->getCompositionStream();
                alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, smoothedBinMask, gfRadius, gfEps, *gpuMemoryPool, guidedFilterStream);
            } catch (const cv::Exception& e) {
                qWarning() << "GPU guided filter failed:" << e.what() << "- using CPU fallback";
                // CPU fallback would go here if needed
                alphaFloat = cv::Mat::ones(result.size(), CV_32F);
            }
        } else {
            // CPU fallback
            alphaFloat = cv::Mat::ones(result.size(), CV_32F);
        }
        
        //  ENHANCED: Apply edge blurring
        const float edgeBlurRadius = 5.0f; // Increased blur radius for smoother transitions
        cv::Mat edgeBlurredPerson;
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& guidedFilterStream = gpuMemoryPool->getCompositionStream();
                edgeBlurredPerson = applyEdgeBlurringCUDA(scaledPerson, binMask, cleanBackground, edgeBlurRadius, *gpuMemoryPool, guidedFilterStream);
                if (!edgeBlurredPerson.empty()) {
                    scaledPerson = edgeBlurredPerson;
                }
            } catch (const cv::Exception& e) {
                qWarning() << "GPU edge blurring failed:" << e.what();
            }
        }
        
        if (edgeBlurredPerson.empty()) {
            edgeBlurredPerson = applyEdgeBlurringAlternative(scaledPerson, binMask, edgeBlurRadius);
            if (!edgeBlurredPerson.empty()) {
                scaledPerson = edgeBlurredPerson;
            }
        }
        
        // Build thin inner/outer rings around the boundary
        cv::Mat inner, outer, ringInner, ringOuter;
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // Wider transition
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*6+1, 2*6+1))); // Wider feather
        cv::subtract(binMask, inner, ringInner);
        cv::subtract(outer, binMask, ringOuter);
        
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);
        alphaFloat.setTo(0.0f, outer == 0);
        alphaFloat = alphaFloat * 0.6f; // Less aggressive reduction for smoother blending

        // Composite
        cv::Mat personF, bgF; 
        scaledPerson.convertTo(personF, CV_32F); 
        cleanBackground.convertTo(bgF, CV_32F);
        std::vector<cv::Mat> a3 = {alphaFloat, alphaFloat, alphaFloat};
        cv::Mat alpha3; 
        cv::merge(a3, alpha3);
        
        cv::Mat alphaSafe;
        cv::max(alpha3, 0.05f, alphaSafe);
        cv::Mat Fclean = (personF - bgF.mul(1.0f - alpha3)).mul(1.0f / alphaSafe);
        cv::Mat compF = Fclean.mul(alpha3) + bgF.mul(1.0f - alpha3);
        cv::Mat out8u; 
        compF.convertTo(out8u, CV_8U);
        out8u.copyTo(result, ringInner);

        cleanBackground.copyTo(result, ringOuter);
        
        //  FINAL EDGE BLURRING
        const float finalEdgeBlurRadius = 6.0f; // Increased for smoother final edges
        if (gpuMemoryPool && useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::Stream& finalStream = gpuMemoryPool->getCompositionStream();
                cv::Mat finalEdgeBlurred = applyEdgeBlurringCUDA(result, binMask, cleanBackground, finalEdgeBlurRadius, *gpuMemoryPool, finalStream);
                if (!finalEdgeBlurred.empty()) {
                    result = finalEdgeBlurred;
                }
            } catch (const cv::Exception& e) {
                qWarning() << "Final GPU edge blurring failed:" << e.what();
            }
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "DYNAMIC EDGE BLENDING: Edge blending failed:" << e.what() << "- using global correction";
        if (lightingCorrector) {
            return lightingCorrector->applyGlobalLightingCorrection(composedFrame);
        }
        return composedFrame;
    } catch (const std::exception& e) {
        qWarning() << "DYNAMIC EDGE BLENDING: Exception:" << e.what() << "- using original";
        return composedFrame;
    }
}
