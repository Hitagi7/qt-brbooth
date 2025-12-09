// Lighting Correction Implementation
// Extracted from capture.cpp for better code organization

#include "core/capture.h"
#include "core/capture_edge_blending.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

cv::Mat Capture::applyPostProcessingLighting()
{
    qDebug() << "POST-PROCESSING: Apply lighting to raw person data and re-composite";
    
    // Check if we have raw person data
    if (m_lastRawPersonRegion.empty() || m_lastRawPersonMask.empty()) {
        qWarning() << "No raw person data available, returning original segmented frame";
        return m_lastSegmentedFrame.clone();
    }
    
    // Start from a clean background template/dynamic video frame (no person composited yet)
    cv::Mat result;
    cv::Mat cleanBackground;
    // Only use cached template background if we're actually using background templates
    // This prevents using stale cached backgrounds from previous template selections
    if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty() && !m_lastTemplateBackground.empty()) {
        cleanBackground = m_lastTemplateBackground.clone();
        qDebug() << "POST-PROCESSING: Using cached template background";
    } else if (m_useBackgroundTemplate && !m_selectedBackgroundTemplate.isEmpty()) {
        // Check if this is bg6.png (white background special case)
        if (m_selectedBackgroundTemplate.contains("bg6.png")) {
            // Create white background instead of loading a file
            cleanBackground = cv::Mat(m_lastSegmentedFrame.size(), m_lastSegmentedFrame.type(), cv::Scalar(255, 255, 255));
            qDebug() << "POST-PROCESSING: Created white background for bg6.png";
        } else {
            QString resolvedPath = resolveTemplatePath(m_selectedBackgroundTemplate);
            cv::Mat bg = cv::imread(resolvedPath.toStdString());
            if (!bg.empty()) {
                cv::resize(bg, cleanBackground, m_lastSegmentedFrame.size());
                qDebug() << "POST-PROCESSING: Loaded background template from" << resolvedPath;
            } else {
                qWarning() << "POST-PROCESSING: Failed to load background from" << resolvedPath;
            }
        }
    }
    if (cleanBackground.empty()) {
        // Fallback to a blank frame matching the output size if no cached template available
        cleanBackground = cv::Mat::zeros(m_lastSegmentedFrame.size(), m_lastSegmentedFrame.type());
        qDebug() << "POST-PROCESSING: Using black background (fallback)";
    }
    result = cleanBackground.clone();
    
    // Apply lighting to the raw person region (post-processing as in original)
    cv::Mat lightingCorrectedPerson = applyLightingToRawPersonRegion(m_lastRawPersonRegion, m_lastRawPersonMask);
    
    // Scale the lighting-corrected person respecting the person scale factor (same as original segmentation)
    cv::Mat scaledPerson, scaledMask;
    cv::Size backgroundSize = result.size();
    cv::Size scaledPersonSize;
    
    if (qAbs(m_personScaleFactor - 1.0) > 0.01) {
        int scaledWidth = static_cast<int>(backgroundSize.width * m_personScaleFactor + 0.5);
        int scaledHeight = static_cast<int>(backgroundSize.height * m_personScaleFactor + 0.5);
        scaledWidth = qMax(1, scaledWidth);
        scaledHeight = qMax(1, scaledHeight);
        scaledPersonSize = cv::Size(scaledWidth, scaledHeight);
        qDebug() << "POST-PROCESSING: Scaling person to" << scaledWidth << "x" << scaledHeight << "with factor" << m_personScaleFactor;
    } else {
        scaledPersonSize = backgroundSize;
    }
    
    cv::resize(lightingCorrectedPerson, scaledPerson, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
    cv::resize(m_lastRawPersonMask, scaledMask, scaledPersonSize, 0, 0, cv::INTER_LINEAR);
    
    // Calculate centered offset for placing the scaled person
    cv::Size actualScaledSize(scaledPerson.cols, scaledPerson.rows);
    int xOffset = (backgroundSize.width - actualScaledSize.width) / 2;
    int yOffset = (backgroundSize.height - actualScaledSize.height) / 2;
    
    // If person is scaled down, we need to place it on a full-size canvas at the centered position
    cv::Mat fullSizePerson, fullSizeMask;
    if (actualScaledSize != backgroundSize) {
        // Create full-size images initialized to zeros
        fullSizePerson = cv::Mat::zeros(backgroundSize, scaledPerson.type());
        fullSizeMask = cv::Mat::zeros(backgroundSize, CV_8UC1);
        
        // Ensure offsets are valid
        if (xOffset >= 0 && yOffset >= 0 &&
            xOffset + actualScaledSize.width <= backgroundSize.width &&
            xOffset + actualScaledSize.height <= backgroundSize.height) {
            
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
            
            qDebug() << "POST-PROCESSING: Placed scaled person at offset" << xOffset << "," << yOffset;
        } else {
            qWarning() << "POST-PROCESSING: Invalid offset, using direct copy";
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
    
    // Soft-edge alpha blend only around the person (robust feather, background untouched)
    try {
        // Ensure binary mask 0/255
        cv::Mat binMask;
        cv::threshold(scaledMask, binMask, 127, 255, cv::THRESH_BINARY);

        // First: shrink mask slightly to avoid fringe, then hard-copy interior
        cv::Mat interiorMask;
        cv::erode(binMask, interiorMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*2+1, 2*2+1))); // ~2px shrink
        scaledPerson.copyTo(result, interiorMask);

        // Use clean template/dynamic background for edge blending
        cv::Mat backgroundFrame = cleanBackground;

        //  CUDA-Accelerated Guided image filtering to refine a soft alpha only on a thin edge ring
        // Guidance is the current output (result) which already has person hard-copied
        const int gfRadius = 8; // window size (reduced for better performance)
        const float gfEps = 1e-2f; // regularization (increased for better performance)
        
        // Use GPU memory pool stream and buffers for optimized guided filtering
        cv::cuda::Stream& guidedFilterStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat alphaFloat = guidedFilterGrayAlphaCUDAOptimized(result, binMask, gfRadius, gfEps, m_gpuMemoryPool, guidedFilterStream);
        
        //  ENHANCED: Apply edge blurring to create smooth transitions between background and segmented object
        const float edgeBlurRadius = 3.0f; // Increased blur radius for better background-object transition
        cv::Mat edgeBlurredPerson = applyEdgeBlurringCUDA(scaledPerson, binMask, backgroundFrame, edgeBlurRadius, m_gpuMemoryPool, guidedFilterStream);
        if (!edgeBlurredPerson.empty()) {
            scaledPerson = edgeBlurredPerson;
            qDebug() << "STATIC MODE: Applied CUDA edge blurring with radius" << edgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
            edgeBlurredPerson = applyEdgeBlurringAlternative(scaledPerson, binMask, edgeBlurRadius);
            if (!edgeBlurredPerson.empty()) {
                scaledPerson = edgeBlurredPerson;
                qDebug() << "STATIC MODE: Applied alternative edge blurring with radius" << edgeBlurRadius;
            }
        }
        
        // Build thin inner/outer rings around the boundary for localized updates only
        cv::Mat inner, outer, ringInner, ringOuter;
        cv::erode(binMask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*1+1, 2*1+1))); // shrink by ~1px for inner ring
        cv::dilate(binMask, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*4+1, 2*4+1))); // expand by ~4px for outer ring
        cv::subtract(binMask, inner, ringInner);   // just inside the boundary
        cv::subtract(outer, binMask, ringOuter);   // just outside the boundary
        // Clamp strictly
        alphaFloat.setTo(1.0f, interiorMask > 0);  // full person interior remains 1
        alphaFloat.setTo(0.0f, outer == 0); // outside remains 0
        // Strongly bias ring blend toward template to eliminate colored outlines
        alphaFloat = alphaFloat * 0.3f;

        // Composite only where outer>0 to avoid touching background (use original colors)
        cv::Mat personF, bgF; scaledPerson.convertTo(personF, CV_32F); backgroundFrame.convertTo(bgF, CV_32F);
        std::vector<cv::Mat> a3 = {alphaFloat, alphaFloat, alphaFloat};
        cv::Mat alpha3; cv::merge(a3, alpha3);
        // Inner ring: solve for decontaminated foreground using matting equation, then composite
        // F_clean = (I - (1 - alpha) * B) / max(alpha, eps)
        cv::Mat alphaSafe;
        cv::max(alpha3, 0.05f, alphaSafe); // avoid division by very small alpha
        cv::Mat Fclean = (personF - bgF.mul(1.0f - alpha3)).mul(1.0f / alphaSafe);
        cv::Mat compF = Fclean.mul(alpha3) + bgF.mul(1.0f - alpha3);
        cv::Mat out8u; compF.convertTo(out8u, CV_8U);
        out8u.copyTo(result, ringInner);

        // Outer ring: copy template directly to eliminate any colored outline
        backgroundFrame.copyTo(result, ringOuter);
        
        //  FINAL EDGE BLURRING: Apply edge blurring to the final composite result
        const float finalEdgeBlurRadius = 4.0f; // Stronger blur for final result
        cv::cuda::Stream& finalStream = m_gpuMemoryPool.getCompositionStream();
        cv::Mat finalEdgeBlurred = applyEdgeBlurringCUDA(result, binMask, cleanBackground, finalEdgeBlurRadius, m_gpuMemoryPool, finalStream);
        if (!finalEdgeBlurred.empty()) {
            result = finalEdgeBlurred;
            qDebug() << "STATIC MODE: Applied final CUDA edge blurring to composite result with radius" << finalEdgeBlurRadius;
        } else {
            // Fallback to alternative method if CUDA fails
            finalEdgeBlurred = applyEdgeBlurringAlternative(result, binMask, finalEdgeBlurRadius);
            if (!finalEdgeBlurred.empty()) {
                result = finalEdgeBlurred;
                qDebug() << "STATIC MODE: Applied final alternative edge blurring to composite result with radius" << finalEdgeBlurRadius;
            }
        }
    } catch (const cv::Exception &e) {
        qWarning() << "Soft-edge blend failed:" << e.what();
        scaledPerson.copyTo(result, scaledMask);
    }
    
    // Save debug images
    cv::imwrite("debug_post_original_segmented.png", m_lastSegmentedFrame);
    cv::imwrite("debug_post_lighting_corrected_person.png", lightingCorrectedPerson);
    cv::imwrite("debug_post_final_result.png", result);
    qDebug() << "POST-PROCESSING: Applied lighting to person and re-composited";
    qDebug() << "Debug images saved: post_original_segmented, post_lighting_corrected_person, post_final_result";
    
    return result;
}

cv::Mat Capture::applyLightingToRawPersonRegion(const cv::Mat &personRegion, const cv::Mat &personMask)
{
    qDebug() << "RAW PERSON APPROACH: Apply lighting to extracted person region only";
    
    //  CRASH PREVENTION: Validate inputs
    if (personRegion.empty() || personMask.empty()) {
        qWarning() << "Invalid inputs - returning empty mat";
        return cv::Mat();
    }
    
    if (personRegion.size() != personMask.size()) {
        qWarning() << "Size mismatch between person region and mask - returning original";
        return personRegion.clone();
    }
    
    if (personRegion.type() != CV_8UC3) {
        qWarning() << "Invalid person region format - returning original";
        return personRegion.clone();
    }
    
    if (personMask.type() != CV_8UC1) {
        qWarning() << "Invalid mask format - returning original";
        return personRegion.clone();
    }
    
    // Start with exact copy of person region
    cv::Mat result;
    try {
        result = personRegion.clone();
    } catch (const std::exception& e) {
        qWarning() << "Failed to clone person region:" << e.what();
        return cv::Mat();
    }
    
    //  CRASH PREVENTION: Check lighting corrector availability
    if (!m_lightingCorrector) {
        qWarning() << "No lighting corrector available - returning original";
        return result;
    }
    
    try {
        // Get template reference for color matching
        cv::Mat templateRef = m_lightingCorrector->getReferenceTemplate();
        if (templateRef.empty()) {
            qWarning() << "No template reference, applying subtle lighting correction";
            // Apply subtle lighting correction to make person blend better
            for (int y = 0; y < result.rows; y++) {
                for (int x = 0; x < result.cols; x++) {
                    if (y < personMask.rows && x < personMask.cols && 
                        personMask.at<uchar>(y, x) > 0) {  // Person pixel
                        cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                        // SUBTLE CHANGES FOR NATURAL BLENDING:
                        pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 1.1);  // Slightly brighter blue
                        pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 1.05); // Slightly brighter green
                        pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 1.08); // Slightly brighter red
                    }
                }
            }
        } else {
            // Apply template-based color matching
            cv::resize(templateRef, templateRef, personRegion.size());
            
            // Convert to LAB for color matching
            cv::Mat personLab, templateLab;
            cv::cvtColor(personRegion, personLab, cv::COLOR_BGR2Lab);
            cv::cvtColor(templateRef, templateLab, cv::COLOR_BGR2Lab);
            
            // Calculate template statistics
            cv::Scalar templateMean, templateStd;
            cv::meanStdDev(templateLab, templateMean, templateStd);
            
            // Apply color matching to person region
            cv::Mat resultLab = personLab.clone();
            std::vector<cv::Mat> channels;
            cv::split(resultLab, channels);
            
            // Apply template color matching for natural blending
            // Calculate person statistics for comparison
            cv::Scalar personMean, personStd;
            cv::meanStdDev(personLab, personMean, personStd);
            
            // Adjust person lighting to match template characteristics
            for (int c = 0; c < 3; c++) {
                // Calculate the difference between template and person
                double lightingDiff = templateMean[c] - personMean[c];
                
                // Apply subtle adjustment (only 15% of the difference for natural blending)
                channels[c] = channels[c] + lightingDiff * 0.15;
            }
            
            // Additional brightness adjustment for better blending
            // If template is brighter, slightly brighten the person
            double brightnessDiff = templateMean[0] - personMean[0]; // L channel
            if (brightnessDiff > 0) {
                channels[0] = channels[0] + brightnessDiff * 0.1; // Slight brightness boost
            }
            
            cv::merge(channels, resultLab);
            cv::cvtColor(resultLab, result, cv::COLOR_Lab2BGR);
            
            // Apply mask to ensure only person pixels are affected
            cv::Mat maskedResult;
            result.copyTo(maskedResult, personMask);
            personRegion.copyTo(maskedResult, ~personMask);
            result = maskedResult;
        }
                
        // Save debug images (safely)
        try {
            cv::imwrite("debug_raw_person_original.png", personRegion);
            cv::imwrite("debug_raw_person_mask.png", personMask);
            cv::imwrite("debug_raw_person_result.png", result);
            qDebug() << "RAW PERSON APPROACH: Applied lighting to person region only";
            qDebug() << "Debug images saved: raw_person_original, raw_person_mask, raw_person_result";
        } catch (const std::exception& e) {
            qWarning() << "Failed to save debug images:" << e.what();
        }
                
    } catch (const std::exception& e) {
        qWarning() << "Exception in lighting correction:" << e.what() << "- returning original";
        return personRegion.clone();
    }
    
    return result;
}

