#ifndef TRT_HAND_LANDMARKER_H
#define TRT_HAND_LANDMARKER_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>

enum class HandGestureState {
    Unknown = 0,
    Opening,
    Open,
    Closing,
    Closed
};

struct HandLandmarkResult {
    bool valid{false};
    float score{0.0f};
    std::array<cv::Point2f, 21> landmarks{};
    HandGestureState gesture{HandGestureState::Unknown};
};

class TrtHandLandmarker
{
public:
    TrtHandLandmarker();
    ~TrtHandLandmarker();

    bool loadEngine(const std::string &enginePath);
    bool isLoaded() const { return m_loaded; }

    bool infer(const cv::cuda::GpuMat &frameBgr, HandLandmarkResult &outResult);

    void setSmoothingFactor(float alpha) { m_smoothingFactor = alpha; }
    void setStateWindow(int frames) { m_stateWindow = std::max(1, frames); }

private:
    bool allocateBuffers();
    void releaseBuffers();
    bool preprocess(const cv::cuda::GpuMat &frameBgr);
    void postprocess(HandLandmarkResult &result);
    HandGestureState classifyState(const std::array<cv::Point2f, 21> &landmarks);

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void *m_deviceBindings[8]{};
    int m_bindingInput{-1};
    int m_bindingLandmarks{-1};
    int m_bindingHandedness{-1};
    int m_bindingPresence{-1};
    std::vector<float> m_outputLandmarks;
    std::vector<float> m_outputHandedness;
    std::vector<float> m_outputPresence;

    cv::cuda::GpuMat m_gpuInput;
    cv::cuda::GpuMat m_gpuRgb;
    cv::cuda::GpuMat m_gpuResized;
    cv::cuda::GpuMat m_gpuFloat;

    std::array<cv::Point2f, 21> m_prevLandmarks{};
    bool m_hasPrev{false};
    float m_smoothingFactor{0.4f};
    int m_stateWindow{5};
    std::vector<HandGestureState> m_recentStates;
    cudaStream_t m_stream{nullptr};
    bool m_loaded{false};
};

#endif // TRT_HAND_LANDMARKER_H

