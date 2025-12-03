#include "algorithms/hand_detection/trt_hand_landmarker.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

namespace {

constexpr int kInputWidth = 224;
constexpr int kInputHeight = 224;
constexpr int kInputChannels = 3;
constexpr int kLandmarkCount = 21;
constexpr int kLandmarkDims = 3;
constexpr int kLandmarkOutputSize = kLandmarkCount * kLandmarkDims;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            qWarning() << "TensorRT:" << msg;
        }
    }
};

// TensorRT 10.1 uses automatic memory management, no destroy() needed

float distance(const cv::Point2f &a, const cv::Point2f &b)
{
    const cv::Point2f diff = a - b;
    return std::sqrt(diff.dot(diff));
}

} // namespace

static Logger gLogger;

TrtHandLandmarker::TrtHandLandmarker()
    : m_runtime(nullptr)
    , m_engine(nullptr)
    , m_context(nullptr)
{
}

TrtHandLandmarker::~TrtHandLandmarker()
{
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    releaseBuffers();
}

bool TrtHandLandmarker::loadEngine(const std::string &enginePath)
{
    releaseBuffers();
    m_loaded = false;

    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        qWarning() << "TrtHandLandmarker: failed to open engine" << QString::fromStdString(enginePath);
        return false;
    }

    engineFile.seekg(0, std::ifstream::end);
    const size_t size = static_cast<size_t>(engineFile.tellg());
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> serialized(size);
    engineFile.read(serialized.data(), size);

    m_runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!m_runtime) {
        qWarning() << "TrtHandLandmarker: failed to create runtime";
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(serialized.data(), size));
    if (!m_engine) {
        qWarning() << "TrtHandLandmarker: failed to deserialize engine";
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        qWarning() << "TrtHandLandmarker: failed to create execution context";
        return false;
    }

    if (!allocateBuffers()) {
        qWarning() << "TrtHandLandmarker: failed to allocate buffers";
        return false;
    }

    if (!m_stream) {
        cudaStreamCreate(&m_stream);
    }

    m_recentStates.clear();
    m_recentStates.reserve(16);
    m_hasPrev = false;
    m_loaded = true;
    qDebug() << "TrtHandLandmarker: engine loaded" << QString::fromStdString(enginePath);
    return true;
}

bool TrtHandLandmarker::allocateBuffers()
{
    for (void *&binding : m_deviceBindings) {
        if (binding) {
            cudaFree(binding);
            binding = nullptr;
        }
    }

    const int nBindings = m_engine->getNbBindings();
    m_bindingInput = m_engine->getBindingIndex("input_1");
    m_bindingLandmarks = m_engine->getBindingIndex("Identity");
    m_bindingHandedness = m_engine->getBindingIndex("Identity_2");
    m_bindingPresence = m_engine->getBindingIndex("Identity_1");

    if (m_bindingInput < 0 || m_bindingLandmarks < 0 || m_bindingHandedness < 0 || m_bindingPresence < 0) {
        qWarning() << "TrtHandLandmarker: failed to resolve binding indices";
        return false;
    }

    for (int i = 0; i < nBindings; ++i) {
        const nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        size_t count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            const int dim = dims.d[j];
            count *= dim > 0 ? dim : 1;
        }

        const size_t bytes = count * sizeof(float);
        if (cudaMalloc(&m_deviceBindings[i], bytes) != cudaSuccess) {
            qWarning() << "TrtHandLandmarker: cudaMalloc failed for binding" << i;
            return false;
        }

        if (i == m_bindingLandmarks) {
            m_outputLandmarks.resize(count);
        } else if (i == m_bindingHandedness) {
            m_outputHandedness.resize(count);
        } else if (i == m_bindingPresence) {
            m_outputPresence.resize(count);
        }
    }

    return true;
}

void TrtHandLandmarker::releaseBuffers()
{
    for (void *&binding : m_deviceBindings) {
        if (binding) {
            cudaFree(binding);
            binding = nullptr;
        }
    }
    m_outputLandmarks.clear();
    m_outputHandedness.clear();
    m_outputPresence.clear();
}

bool TrtHandLandmarker::preprocess(const cv::cuda::GpuMat &frameBgr)
{
    if (frameBgr.empty()) {
        return false;
    }

    if (m_gpuInput.size() != frameBgr.size()) {
        m_gpuInput = cv::cuda::GpuMat(frameBgr.size(), frameBgr.type());
        m_gpuRgb = cv::cuda::GpuMat(frameBgr.size(), frameBgr.type());
    }

    frameBgr.copyTo(m_gpuInput);
    cv::cuda::cvtColor(m_gpuInput, m_gpuRgb, cv::COLOR_BGR2RGB);

    if (m_gpuResized.size() != cv::Size(kInputWidth, kInputHeight)) {
        m_gpuResized = cv::cuda::GpuMat(cv::Size(kInputWidth, kInputHeight), CV_8UC3);
    }
    cv::cuda::resize(m_gpuRgb, m_gpuResized, cv::Size(kInputWidth, kInputHeight));

    m_gpuResized.convertTo(m_gpuFloat, CV_32FC3, 1.0 / 255.0);

    const size_t bytes = kInputWidth * kInputHeight * kInputChannels * sizeof(float);
    if (cudaMemcpyAsync(m_deviceBindings[m_bindingInput], m_gpuFloat.ptr<float>(), bytes, cudaMemcpyDeviceToDevice, m_stream) != cudaSuccess) {
        qWarning() << "TrtHandLandmarker: cudaMemcpy to binding 0 failed";
        return false;
    }

    return true;
}

bool TrtHandLandmarker::infer(const cv::cuda::GpuMat &frameBgr, HandLandmarkResult &outResult)
{
    if (!m_loaded) {
        return false;
    }

    if (!preprocess(frameBgr)) {
        return false;
    }

    if (!m_context->enqueueV2(m_deviceBindings, m_stream, nullptr)) {
        qWarning() << "TrtHandLandmarker: enqueue failed";
        return false;
    }

    cudaStreamSynchronize(m_stream);

    if (cudaMemcpy(m_outputLandmarks.data(), m_deviceBindings[m_bindingLandmarks], m_outputLandmarks.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        qWarning() << "TrtHandLandmarker: cudaMemcpy landmarks failed";
        return false;
    }
    cudaMemcpy(m_outputHandedness.data(), m_deviceBindings[m_bindingHandedness], m_outputHandedness.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_outputPresence.data(), m_deviceBindings[m_bindingPresence], m_outputPresence.size() * sizeof(float), cudaMemcpyDeviceToHost);

    postprocess(outResult);
    return outResult.valid;
}

void TrtHandLandmarker::postprocess(HandLandmarkResult &result)
{
    result.valid = m_outputPresence[0] > 0.5f;
    result.score = m_outputHandedness[0];

    if (!result.valid) {
        m_recentStates.clear();
        m_hasPrev = false;
        result.gesture = HandGestureState::Unknown;
        return;
    }

    const float width = static_cast<float>(m_gpuInput.cols);
    const float height = static_cast<float>(m_gpuInput.rows);

    for (int i = 0; i < kLandmarkCount; ++i) {
        const float x = m_outputLandmarks[i * kLandmarkDims + 0];
        const float y = m_outputLandmarks[i * kLandmarkDims + 1];

        cv::Point2f pt{x * width, y * height};

        if (m_hasPrev) {
            const cv::Point2f prev = m_prevLandmarks[i];
            pt = prev * (1.0f - m_smoothingFactor) + pt * m_smoothingFactor;
        }

        result.landmarks[i] = pt;
    }

    m_prevLandmarks = result.landmarks;
    m_hasPrev = true;

    const HandGestureState state = classifyState(result.landmarks);
    m_recentStates.push_back(state);
    if (static_cast<int>(m_recentStates.size()) > m_stateWindow) {
        m_recentStates.erase(m_recentStates.begin());
    }

    std::array<int, 5> counts{};
    for (HandGestureState s : m_recentStates) {
        counts[static_cast<int>(s)]++;
    }

    int bestIndex = 0;
    for (int i = 1; i < static_cast<int>(counts.size()); ++i) {
        if (counts[i] > counts[bestIndex]) {
            bestIndex = i;
        }
    }

    result.gesture = static_cast<HandGestureState>(bestIndex);
}

HandGestureState TrtHandLandmarker::classifyState(const std::array<cv::Point2f, 21> &landmarks)
{
    const cv::Point2f wrist = landmarks[0];
    const cv::Point2f indexMCP = landmarks[5];
    const cv::Point2f pinkyMCP = landmarks[17];

    const float palmWidth = distance(indexMCP, pinkyMCP);
    if (palmWidth <= 1e-3f) {
        return HandGestureState::Unknown;
    }

    auto curlRatio = [&](int tip, int mcp) {
        const float tipDist = distance(landmarks[tip], wrist);
        const float mcpDist = distance(landmarks[mcp], wrist);
        return tipDist / std::max(mcpDist, 1e-3f);
    };

    const float indexCurl = curlRatio(8, 5);
    const float middleCurl = curlRatio(12, 9);
    const float ringCurl = curlRatio(16, 13);
    const float pinkyCurl = curlRatio(20, 17);
    const float thumbCurl = curlRatio(4, 1);

    const float avgCurl = (indexCurl + middleCurl + ringCurl + pinkyCurl) * 0.25f;

    if (avgCurl > 0.8f && thumbCurl > 0.7f) {
        return HandGestureState::Open;
    }
    if (avgCurl < 0.45f) {
        return HandGestureState::Closed;
    }

    if (!m_recentStates.empty()) {
        const HandGestureState prev = m_recentStates.back();
        if (avgCurl > 0.6f && prev == HandGestureState::Closed) {
            return HandGestureState::Opening;
        }
        if (avgCurl < 0.6f && prev == HandGestureState::Open) {
            return HandGestureState::Closing;
        }
    }

    return HandGestureState::Unknown;
}

