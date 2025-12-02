// Edge Blending Implementation
// Extracted from capture.cpp for better code organization

#include "core/capture.h"
#include <QDebug>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

// Forward declaration for CPU fallback
static cv::Mat guidedFilterGrayAlphaCPU(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps);
static cv::Mat applyEdgeBlurringCPU(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius);

// CUDA-Accelerated Guided Filter for Alpha Matting
cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                             GPUMemoryPool &memoryPool, cv::cuda::Stream &stream)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    // Check CUDA availability
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        qWarning() << "CUDA not available, falling back to CPU guided filter";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }

    try {
        //  Performance monitoring for guided filtering
        QElapsedTimer guidedFilterTimer;
        guidedFilterTimer.start();
        
        // Get pre-allocated GPU buffers from memory pool (optimized buffer usage)
        cv::cuda::GpuMat& gpuGuide = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMask = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuP = memoryPool.getNextGuidedFilterBuffer();
        
        // Upload to GPU with stream
        gpuGuide.upload(guideBGR, stream);
        gpuMask.upload(hardMask, stream);
        
        // Convert guide to grayscale on GPU if needed
        if (guideBGR.channels() == 3) {
            cv::cuda::cvtColor(gpuGuide, gpuI, cv::COLOR_BGR2GRAY, 0, stream);
        } else {
            gpuI = gpuGuide;
        }
        
        // Convert to float32 on GPU
        gpuI.convertTo(gpuI, CV_32F, 1.0f / 255.0f, stream);
        if (hardMask.type() != CV_32F) {
            gpuMask.convertTo(gpuP, CV_32F, 1.0f / 255.0f, stream);
        } else {
            gpuP = gpuMask;
        }
        
        // Get additional buffers from memory pool
        cv::cuda::GpuMat& gpuMeanI = memoryPool.getNextBoxFilterBuffer();
        cv::cuda::GpuMat& gpuMeanP = memoryPool.getNextBoxFilterBuffer();
        cv::cuda::GpuMat& gpuCorrI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuCorrIp = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuVarI = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuCovIp = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuA = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuB = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMeanA = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuMeanB = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuQ = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuAlpha = memoryPool.getNextGuidedFilterBuffer();
        
        // Create box filter for GPU (reuse existing filter if available)
        cv::Ptr<cv::cuda::Filter> boxFilter = cv::cuda::createBoxFilter(CV_32F, CV_32F, cv::Size(radius, radius));
        
        // Step 1: Compute means and correlations on GPU
        boxFilter->apply(gpuI, gpuMeanI, stream);
        boxFilter->apply(gpuP, gpuMeanP, stream);
        
        // Compute I*I and I*P on GPU
        cv::cuda::GpuMat& gpuISquared = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::GpuMat& gpuIP = memoryPool.getNextGuidedFilterBuffer();
        cv::cuda::multiply(gpuI, gpuI, gpuISquared, 1.0, -1, stream);
        cv::cuda::multiply(gpuI, gpuP, gpuIP, 1.0, -1, stream);
        
        boxFilter->apply(gpuISquared, gpuCorrI, stream);
        boxFilter->apply(gpuIP, gpuCorrIp, stream);
        
        // Step 2: Compute variance and covariance on GPU
        cv::cuda::multiply(gpuMeanI, gpuMeanI, gpuVarI, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrI, gpuVarI, gpuVarI, cv::noArray(), -1, stream);
        
        cv::cuda::multiply(gpuMeanI, gpuMeanP, gpuCovIp, 1.0, -1, stream);
        cv::cuda::subtract(gpuCorrIp, gpuCovIp, gpuCovIp, cv::noArray(), -1, stream);
        
        // Step 3: Compute coefficients a and b on GPU
        cv::cuda::GpuMat& gpuEps = memoryPool.getNextGuidedFilterBuffer();
        gpuEps.upload(cv::Mat::ones(gpuVarI.size(), CV_32F) * eps, stream);
        cv::cuda::add(gpuVarI, gpuEps, gpuVarI, cv::noArray(), -1, stream);
        cv::cuda::divide(gpuCovIp, gpuVarI, gpuA, 1.0, -1, stream);
        
        cv::cuda::multiply(gpuA, gpuMeanI, gpuB, 1.0, -1, stream);
        cv::cuda::subtract(gpuMeanP, gpuB, gpuB, cv::noArray(), -1, stream);
        
        // Step 4: Compute mean of coefficients on GPU
        boxFilter->apply(gpuA, gpuMeanA, stream);
        boxFilter->apply(gpuB, gpuMeanB, stream);
        
        // Step 5: Compute final result on GPU
        cv::cuda::multiply(gpuMeanA, gpuI, gpuQ, 1.0, -1, stream);
        cv::cuda::add(gpuQ, gpuMeanB, gpuQ, cv::noArray(), -1, stream);
        
        // Clamp result to [0,1] on GPU
        cv::cuda::threshold(gpuQ, gpuAlpha, 0.0f, 0.0f, cv::THRESH_TOZERO, stream);
        cv::cuda::threshold(gpuAlpha, gpuAlpha, 1.0f, 1.0f, cv::THRESH_TRUNC, stream);
        
        // Download result back to CPU
        cv::Mat result;
        gpuAlpha.download(result, stream);
        stream.waitForCompletion();
        
        //  Performance monitoring - log guided filtering time
        qint64 guidedFilterTime = guidedFilterTimer.elapsed();
        if (guidedFilterTime > 5) { // Only log if it takes more than 5ms
            qDebug() << "CUDA Guided Filter Performance:" << guidedFilterTime << "ms for" 
                     << guideBGR.cols << "x" << guideBGR.rows << "image";
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "CUDA guided filter failed:" << e.what() << "- falling back to CPU";
        return guidedFilterGrayAlphaCPU(guideBGR, hardMask, radius, eps);
    }
}

// CPU fallback for guided filtering (original implementation)
static cv::Mat guidedFilterGrayAlphaCPU(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps)
{
    CV_Assert(!guideBGR.empty());
    CV_Assert(!hardMask.empty());

    cv::Mat I8, I, p;
    if (guideBGR.channels() == 3) {
        cv::cvtColor(guideBGR, I8, cv::COLOR_BGR2GRAY);
    } else {
        I8 = guideBGR;
    }
    I8.convertTo(I, CV_32F, 1.0f / 255.0f);
    if (hardMask.type() != CV_32F) {
        hardMask.convertTo(p, CV_32F, 1.0f / 255.0f);
    } else {
        p = hardMask;
    }

    cv::Mat mean_I, mean_p, corr_I, corr_Ip;
    cv::boxFilter(I, mean_I, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(p, mean_p, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(I), corr_I, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(I.mul(p), corr_Ip, CV_32F, cv::Size(radius, radius));

    cv::Mat var_I = corr_I - mean_I.mul(mean_I);
    cv::Mat cov_Ip = corr_Ip - mean_I.mul(mean_p);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_32F, cv::Size(radius, radius));
    cv::boxFilter(b, mean_b, CV_32F, cv::Size(radius, radius));

    cv::Mat q = mean_a.mul(I) + mean_b;
    cv::Mat alpha; cv::min(cv::max(q, 0.0f), 1.0f, alpha);
    return alpha;
}

// CUDA-Accelerated Edge Blurring for Enhanced Edge-Blending
// GPU-optimized edge blurring that mixes background template with segmented object edges
cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    // Check CUDA availability
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        qWarning() << "CUDA not available for edge blurring, falling back to CPU";
        return applyEdgeBlurringCPU(segmentedObject, objectMask, backgroundTemplate, blurRadius);
    }

    try {
        //  Performance monitoring for edge blurring
        QElapsedTimer edgeBlurTimer;
        edgeBlurTimer.start();

        // Get pre-allocated GPU buffers from memory pool
        cv::cuda::GpuMat& gpuObject = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuMask = memoryPool.getNextEdgeDetectionBuffer();
        cv::cuda::GpuMat& gpuBackground = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuBlurred = memoryPool.getNextEdgeBlurBuffer();
        cv::cuda::GpuMat& gpuResult = memoryPool.getNextEdgeBlurBuffer();

        // Upload to GPU with stream
        gpuObject.upload(segmentedObject, stream);
        gpuMask.upload(objectMask, stream);
        gpuBackground.upload(backgroundTemplate, stream);

        // Convert mask to grayscale if needed
        if (objectMask.channels() == 3) {
            cv::cuda::cvtColor(gpuMask, gpuMask, cv::COLOR_BGR2GRAY, 0, stream);
        }

        // Step 1: Create transition zone by dilating the mask outward
        cv::cuda::GpuMat gpuDilatedMask;
        cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, CV_8UC1, 
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*static_cast<int>(blurRadius)+1, 2*static_cast<int>(blurRadius)+1))
        );
        dilateFilter->apply(gpuMask, gpuDilatedMask, stream);

        // Step 2: Create transition zone by subtracting original mask from dilated mask
        cv::cuda::GpuMat gpuTransitionZone;
        cv::cuda::subtract(gpuDilatedMask, gpuMask, gpuTransitionZone, cv::noArray(), -1, stream);

        // Step 3: Create inner edge zone by eroding the mask
        cv::cuda::GpuMat gpuErodedMask;
        cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(
            cv::MORPH_ERODE, CV_8UC1, 
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))
        );
        erodeFilter->apply(gpuMask, gpuErodedMask, stream);

        // Step 4: Create inner edge zone by subtracting eroded mask from original mask
        cv::cuda::GpuMat gpuInnerEdgeZone;
        cv::cuda::subtract(gpuMask, gpuErodedMask, gpuInnerEdgeZone, cv::noArray(), -1, stream);

        // Step 5: Combine transition zone and inner edge zone for comprehensive edge blurring
        cv::cuda::GpuMat gpuCombinedEdgeZone;
        cv::cuda::bitwise_or(gpuTransitionZone, gpuInnerEdgeZone, gpuCombinedEdgeZone, cv::noArray(), stream);

        // Step 6: Apply Gaussian blur to both object and background
        cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(
            CV_8UC3, CV_8UC3, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f
        );
        gaussianFilter->apply(gpuObject, gpuBlurred, stream);
        
        cv::cuda::GpuMat gpuBlurredBackground;
        gaussianFilter->apply(gpuBackground, gpuBlurredBackground, stream);

        // Step 7: Create mixed background-object blend for edge zones
        cv::cuda::GpuMat gpuMixedBlend;
        cv::cuda::addWeighted(gpuBlurred, 0.6f, gpuBlurredBackground, 0.4f, 0, gpuMixedBlend, -1, stream);

        // Step 8: Apply smooth blending using the combined edge zone
        // Copy original object to result
        gpuObject.copyTo(gpuResult);
        
        // Apply mixed background-object blend in the combined edge zone
        gpuMixedBlend.copyTo(gpuResult, gpuCombinedEdgeZone);

        // Download result back to CPU
        cv::Mat result;
        gpuResult.download(result, stream);
        stream.waitForCompletion();

        //  Performance monitoring - log edge blurring time
        qint64 edgeBlurTime = edgeBlurTimer.elapsed();
        if (edgeBlurTime > 3) { // Only log if it takes more than 3ms
            qDebug() << "CUDA Edge Blur Performance:" << edgeBlurTime << "ms for" 
                     << segmentedObject.cols << "x" << segmentedObject.rows << "image, radius:" << blurRadius;
        }

        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "CUDA edge blurring failed:" << e.what() << "- falling back to CPU";
        return applyEdgeBlurringCPU(segmentedObject, objectMask, backgroundTemplate, blurRadius);
    }
}

// CPU fallback for edge blurring
static cv::Mat applyEdgeBlurringCPU(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    try {
        // Convert mask to grayscale if needed
        cv::Mat mask;
        if (objectMask.channels() == 3) {
            cv::cvtColor(objectMask, mask, cv::COLOR_BGR2GRAY);
        } else {
            mask = objectMask.clone();
        }

        // Step 1: Create transition zone by dilating the mask outward
        cv::Mat dilatedMask;
        cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
            cv::Size(2*static_cast<int>(blurRadius)+1, 2*static_cast<int>(blurRadius)+1));
        cv::dilate(mask, dilatedMask, dilateKernel);

        // Step 2: Create transition zone by subtracting original mask from dilated mask
        cv::Mat transitionZone;
        cv::subtract(dilatedMask, mask, transitionZone);

        // Step 3: Create inner edge zone by eroding the mask
        cv::Mat erodedMask;
        cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mask, erodedMask, erodeKernel);

        // Step 4: Create inner edge zone by subtracting eroded mask from original mask
        cv::Mat innerEdgeZone;
        cv::subtract(mask, erodedMask, innerEdgeZone);

        // Step 5: Combine transition zone and inner edge zone for comprehensive edge blurring
        cv::Mat combinedEdgeZone;
        cv::bitwise_or(transitionZone, innerEdgeZone, combinedEdgeZone);

        // Step 6: Apply Gaussian blur to both object and background
        cv::Mat blurred;
        cv::GaussianBlur(segmentedObject, blurred, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);
        
        cv::Mat blurredBackground;
        cv::GaussianBlur(backgroundTemplate, blurredBackground, cv::Size(0, 0), blurRadius * 1.5f, blurRadius * 1.5f);

        // Step 7: Create mixed background-object blend for edge zones
        cv::Mat mixedBlend;
        cv::addWeighted(blurred, 0.6f, blurredBackground, 0.4f, 0, mixedBlend);

        // Step 8: Apply smooth blending using the combined edge zone
        cv::Mat result = segmentedObject.clone();
        
        // Apply mixed background-object blend in the combined edge zone
        mixedBlend.copyTo(result, combinedEdgeZone);

        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "CPU edge blurring failed:" << e.what() << "- returning original";
        return segmentedObject.clone();
    }
}

// Alternative Edge Blurring Method using Distance Transform
// This method uses distance transform to create smooth edge transitions
cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius)
{
    CV_Assert(!segmentedObject.empty());
    CV_Assert(!objectMask.empty());

    try {
        // Convert mask to grayscale if needed
        cv::Mat mask;
        if (objectMask.channels() == 3) {
            cv::cvtColor(objectMask, mask, cv::COLOR_BGR2GRAY);
        } else {
            mask = objectMask.clone();
        }

        // Step 1: Create distance transform from mask boundary
        cv::Mat distTransform;
        cv::distanceTransform(mask, distTransform, cv::DIST_L2, 5);
        
        // Step 2: Normalize distance transform to [0, 1] range
        cv::Mat normalizedDist;
        cv::normalize(distTransform, normalizedDist, 0, 1.0, cv::NORM_MINMAX, CV_32F);
        
        // Step 3: Create edge mask by thresholding distance transform
        cv::Mat edgeMask;
        float threshold = blurRadius / 10.0f; // Adjust threshold based on blur radius
        cv::threshold(normalizedDist, edgeMask, threshold, 1.0, cv::THRESH_BINARY);
        edgeMask.convertTo(edgeMask, CV_8U, 255.0f);
        
        // Step 4: Apply Gaussian blur to the entire object
        cv::Mat blurred;
        cv::GaussianBlur(segmentedObject, blurred, cv::Size(0, 0), blurRadius, blurRadius);
        
        // Step 5: Blend using distance-based alpha
        cv::Mat result = segmentedObject.clone();
        
        // Create alpha mask from distance transform
        cv::Mat alphaMask;
        normalizedDist.convertTo(alphaMask, CV_8U, 255.0f);
        
        // Apply blending only in edge regions
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (edgeMask.at<uchar>(y, x) > 0) {
                    float alpha = normalizedDist.at<float>(y, x);
                    cv::Vec3b original = result.at<cv::Vec3b>(y, x);
                    cv::Vec3b blurred_pixel = blurred.at<cv::Vec3b>(y, x);
                    
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uchar>(original[0] * (1.0f - alpha) + blurred_pixel[0] * alpha),
                        static_cast<uchar>(original[1] * (1.0f - alpha) + blurred_pixel[1] * alpha),
                        static_cast<uchar>(original[2] * (1.0f - alpha) + blurred_pixel[2] * alpha)
                    );
                }
            }
        }
        
        return result;
        
    } catch (const cv::Exception &e) {
        qWarning() << "Alternative edge blurring failed:" << e.what() << "- returning original";
        return segmentedObject.clone();
    }
}

