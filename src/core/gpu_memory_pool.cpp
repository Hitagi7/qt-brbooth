// GPU Memory Pool Implementation
// Extracted from capture.cpp for better code organization

#include "core/capture.h"
#include <QDebug>

// GPU Memory Pool Implementation

GPUMemoryPool::GPUMemoryPool()
    : detectionStream()
    , segmentationStream()
    , compositionStream()
    , currentFrameBuffer(0)
    , currentSegBuffer(0)
    , currentDetBuffer(0)
    , currentTempBuffer(0)
    , currentGuidedFilterBuffer(0)
    , currentBoxFilterBuffer(0)
    , currentEdgeBlurBuffer(0)
    , currentEdgeDetectionBuffer(0)
    , initialized(false)
    , poolWidth(0)
    , poolHeight(0)
{
    // Initialize cv::Ptr members to nullptr (cannot be default-constructed)
    morphCloseFilter = nullptr;
    morphOpenFilter = nullptr;
    morphDilateFilter = nullptr;
    cannyDetector = nullptr;
    qDebug() << " GPU Memory Pool: Constructor called";
}

GPUMemoryPool::~GPUMemoryPool()
{
    qDebug() << " GPU Memory Pool: Destructor called";
    release();
}

void GPUMemoryPool::initialize(int width, int height)
{
    if (initialized && poolWidth == width && poolHeight == height) {
        qDebug() << " GPU Memory Pool: Already initialized with correct dimensions";
        return;
    }

    qDebug() << " GPU Memory Pool: Initializing with dimensions" << width << "x" << height;

    try {
        // Release existing resources
        release();

        // Initialize frame buffers (triple buffering)
        for (int i = 0; i < 3; ++i) {
            gpuFrameBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Frame buffer" << i << "allocated";
        }

        // Initialize segmentation buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuSegmentationBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Segmentation buffer" << i << "allocated";
        }

        // Initialize detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuDetectionBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Detection buffer" << i << "allocated";
        }

        // Initialize temporary buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuTempBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Temp buffer" << i << "allocated";
        }

        //  Initialize guided filtering buffers (quad buffering for complex operations)
        for (int i = 0; i < 4; ++i) {
            gpuGuidedFilterBuffers[i] = cv::cuda::GpuMat(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Guided filter buffer" << i << "allocated";
        }

        // Initialize box filter buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuBoxFilterBuffers[i] = cv::cuda::GpuMat(height, width, CV_32F);
            qDebug() << " GPU Memory Pool: Box filter buffer" << i << "allocated";
        }

        //  Initialize edge blurring buffers (triple buffering for complex operations)
        for (int i = 0; i < 3; ++i) {
            gpuEdgeBlurBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC3);
            qDebug() << " GPU Memory Pool: Edge blur buffer" << i << "allocated";
        }

        // Initialize edge detection buffers (double buffering)
        for (int i = 0; i < 2; ++i) {
            gpuEdgeDetectionBuffers[i] = cv::cuda::GpuMat(height, width, CV_8UC1);
            qDebug() << " GPU Memory Pool: Edge detection buffer" << i << "allocated";
        }

        // Create reusable CUDA filters (create once, use many times)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        morphCloseFilter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel);
        morphOpenFilter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel);
        morphDilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
        cannyDetector = cv::cuda::createCannyEdgeDetector(50, 150);

        qDebug() << " GPU Memory Pool: CUDA filters created successfully";

        // Initialize CUDA streams for parallel processing
        detectionStream = cv::cuda::Stream();
        segmentationStream = cv::cuda::Stream();
        compositionStream = cv::cuda::Stream();

        qDebug() << " GPU Memory Pool: CUDA streams initialized";

        // Update state
        poolWidth = width;
        poolHeight = height;
        initialized = true;

        qDebug() << " GPU Memory Pool: Initialization completed successfully";

    } catch (const cv::Exception& e) {
        qWarning() << " GPU Memory Pool: Initialization failed:" << e.what();
        release();
    }
}

cv::cuda::GpuMat& GPUMemoryPool::getNextFrameBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuFrameBuffers[currentFrameBuffer];
    currentFrameBuffer = (currentFrameBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextSegmentationBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuSegmentationBuffers[currentSegBuffer];
    currentSegBuffer = (currentSegBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuDetectionBuffers[currentDetBuffer];
    currentDetBuffer = (currentDetBuffer + 1) % 2; // Double buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextTempBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuTempBuffers[currentTempBuffer];
    currentTempBuffer = (currentTempBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Guided Filtering buffer access methods
cv::cuda::GpuMat& GPUMemoryPool::getNextGuidedFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuGuidedFilterBuffers[currentGuidedFilterBuffer];
    currentGuidedFilterBuffer = (currentGuidedFilterBuffer + 1) % 4; // Quad buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextBoxFilterBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuBoxFilterBuffers[currentBoxFilterBuffer];
    currentBoxFilterBuffer = (currentBoxFilterBuffer + 1) % 2; // Double buffering
    return buffer;
}

//  Edge Blurring buffer access methods
cv::cuda::GpuMat& GPUMemoryPool::getNextEdgeBlurBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuEdgeBlurBuffers[currentEdgeBlurBuffer];
    currentEdgeBlurBuffer = (currentEdgeBlurBuffer + 1) % 3; // Triple buffering
    return buffer;
}

cv::cuda::GpuMat& GPUMemoryPool::getNextEdgeDetectionBuffer()
{
    if (!initialized) {
        qWarning() << " GPU Memory Pool: Not initialized, returning empty buffer";
        static cv::cuda::GpuMat emptyBuffer;
        return emptyBuffer;
    }

    cv::cuda::GpuMat& buffer = gpuEdgeDetectionBuffers[currentEdgeDetectionBuffer];
    currentEdgeDetectionBuffer = (currentEdgeDetectionBuffer + 1) % 2; // Double buffering
    return buffer;
}

void GPUMemoryPool::release()
{
    if (!initialized) {
        return;
    }

    qDebug() << " GPU Memory Pool: Releasing resources";

    // Release GPU buffers
    for (int i = 0; i < 3; ++i) {
        gpuFrameBuffers[i].release();
    }

    for (int i = 0; i < 2; ++i) {
        gpuSegmentationBuffers[i].release();
        gpuDetectionBuffers[i].release();
        gpuTempBuffers[i].release();
        gpuBoxFilterBuffers[i].release();
    }

    // Release guided filtering buffers
    for (int i = 0; i < 4; ++i) {
        gpuGuidedFilterBuffers[i].release();
    }

    // Release edge blurring buffers
    for (int i = 0; i < 3; ++i) {
        gpuEdgeBlurBuffers[i].release();
    }

    for (int i = 0; i < 2; ++i) {
        gpuEdgeDetectionBuffers[i].release();
    }

    // Release CUDA filters
    morphCloseFilter.release();
    morphOpenFilter.release();
    morphDilateFilter.release();
    cannyDetector.release();

    // Reset state
    initialized = false;
    poolWidth = 0;
    poolHeight = 0;
    currentFrameBuffer = 0;
    currentSegBuffer = 0;
    currentDetBuffer = 0;
    currentTempBuffer = 0;

    qDebug() << " GPU Memory Pool: Resources released";
}

void GPUMemoryPool::resetBuffers()
{
    if (!initialized) {
        return;
    }

    qDebug() << " GPU Memory Pool: Resetting buffer indices";

    currentFrameBuffer = 0;
    currentSegBuffer = 0;
    currentDetBuffer = 0;
    currentTempBuffer = 0;
    currentGuidedFilterBuffer = 0;
    currentBoxFilterBuffer = 0;
    currentEdgeBlurBuffer = 0;
    currentEdgeDetectionBuffer = 0;
}

