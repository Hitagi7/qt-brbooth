#ifndef CAPTURE_EDGE_BLENDING_H
#define CAPTURE_EDGE_BLENDING_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

class GPUMemoryPool;

// Forward declarations for edge blending functions
cv::Mat guidedFilterGrayAlphaCUDAOptimized(const cv::Mat &guideBGR, const cv::Mat &hardMask, int radius, float eps, 
                                             GPUMemoryPool &memoryPool, cv::cuda::Stream &stream = cv::cuda::Stream::Null());

cv::Mat applyEdgeBlurringCUDA(const cv::Mat &segmentedObject, const cv::Mat &objectMask, const cv::Mat &backgroundTemplate, float blurRadius, 
                                    GPUMemoryPool &memoryPool, cv::cuda::Stream &stream = cv::cuda::Stream::Null());

cv::Mat applyEdgeBlurringAlternative(const cv::Mat &segmentedObject, const cv::Mat &objectMask, float blurRadius);

#endif // CAPTURE_EDGE_BLENDING_H

