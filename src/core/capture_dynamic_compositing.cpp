// Dynamic Video Compositing Implementation
// Simple compositing algorithm for dynamic video frames (fallback method)

#include "core/capture.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include "core/lighting_corrector.h"

// Forward declaration - this is a member function of Capture class
// We'll call it through the Capture instance (this)

// THREAD-SAFE WRAPPER: Takes all parameters instead of accessing member variables
cv::Mat Capture::applySimpleDynamicCompositingSafe(const cv::Mat &composedFrame,
                                                    const cv::Mat &rawPersonRegion,
                                                    const cv::Mat &rawPersonMask,
                                                    const cv::Mat &backgroundFrame,
                                                    LightingCorrector* lightingCorrector,
                                                    double /*personScaleFactor*/,
                                                    bool useCUDA)
{
    // Validate inputs
    if (composedFrame.empty() || rawPersonRegion.empty() || rawPersonMask.empty()) {
        return composedFrame;
    }
    
    try {
        // Get background - use provided frame or fallback
        cv::Mat bg = backgroundFrame.empty() ? cv::Mat::zeros(composedFrame.size(), composedFrame.type()) : backgroundFrame.clone();
        if (bg.size() != composedFrame.size()) {
            cv::resize(bg, bg, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // Scale person region to match frame size if needed
        cv::Mat personRegion = rawPersonRegion;
        cv::Mat personMask = rawPersonMask;
        if (personRegion.size() != composedFrame.size()) {
            cv::resize(personRegion, personRegion, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(personMask, personMask, composedFrame.size(), 0, 0, cv::INTER_LINEAR);
        }
        
        // STEP 1: VIDEO-SPECIFIC LIGHTING: Use static mode approach but optimized for video
        cv::Mat lightingCorrectedPerson = this->applyVideoOptimizedLighting(personRegion, personMask, lightingCorrector);
        
        // STEP 2: Create smoothed mask for edge blending
        cv::Mat binMask;
        if (personMask.channels() == 3) {
            cv::cvtColor(personMask, binMask, cv::COLOR_BGR2GRAY);
        } else {
            binMask = personMask.clone();
        }
        cv::threshold(binMask, binMask, 127, 255, cv::THRESH_BINARY);
        
        // Enhanced edge smoothing for better quality
        cv::Mat smoothedMask;
        cv::GaussianBlur(binMask, smoothedMask, cv::Size(11, 11), 2.5); // Larger blur for smoother edges
        
        // Normalize to [0, 1] for alpha blending
        cv::Mat alphaMask;
        smoothedMask.convertTo(alphaMask, CV_32F, 1.0/255.0);
        
        // STEP 3: Fast GPU-accelerated alpha blending
        if (useCUDA && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                cv::cuda::GpuMat gpuBg, gpuPerson, gpuAlpha, gpuResult;
                gpuBg.upload(bg);
                gpuPerson.upload(lightingCorrectedPerson);
                gpuAlpha.upload(alphaMask);
                
                cv::cuda::GpuMat gpuBgF, gpuPersonF;
                gpuBg.convertTo(gpuBgF, CV_32F);
                gpuPerson.convertTo(gpuPersonF, CV_32F);
                
                std::vector<cv::cuda::GpuMat> alphaChannels = {gpuAlpha, gpuAlpha, gpuAlpha};
                cv::cuda::GpuMat gpuAlpha3;
                cv::cuda::merge(alphaChannels, gpuAlpha3);
                
                cv::cuda::GpuMat gpuResultF, gpuPersonBlended, gpuBgBlended;
                cv::cuda::multiply(gpuPersonF, gpuAlpha3, gpuPersonBlended);
                
                cv::cuda::GpuMat gpuOnes;
                gpuOnes.create(gpuAlpha3.size(), gpuAlpha3.type());
                gpuOnes.setTo(cv::Scalar(1.0, 1.0, 1.0));
                cv::cuda::GpuMat gpuOneMinusAlpha;
                cv::cuda::subtract(gpuOnes, gpuAlpha3, gpuOneMinusAlpha);
                cv::cuda::multiply(gpuBgF, gpuOneMinusAlpha, gpuBgBlended);
                cv::cuda::add(gpuPersonBlended, gpuBgBlended, gpuResultF);
                gpuResultF.convertTo(gpuResult, CV_8U);
                
                cv::Mat result;
                gpuResult.download(result);
                return result;
            } catch (const cv::Exception& e) {
                qWarning() << "GPU compositing failed:" << e.what() << "- using CPU";
            }
        }
        
        // CPU fallback
        cv::Mat personF, bgF;
        lightingCorrectedPerson.convertTo(personF, CV_32F);
        bg.convertTo(bgF, CV_32F);
        
        std::vector<cv::Mat> alphaChannels = {alphaMask, alphaMask, alphaMask};
        cv::Mat alpha3;
        cv::merge(alphaChannels, alpha3);
        
        cv::Mat resultF = personF.mul(alpha3) + bgF.mul(cv::Scalar(1.0, 1.0, 1.0) - alpha3);
        cv::Mat result;
        resultF.convertTo(result, CV_8U);
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Dynamic compositing failed:" << e.what();
        return composedFrame;
    } catch (const std::exception& e) {
        qWarning() << "Dynamic compositing exception:" << e.what();
        return composedFrame;
    }
}
