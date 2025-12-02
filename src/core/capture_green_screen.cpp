// Green Screen Processing Implementation
// Extracted from capture.cpp for better code organization

#include "core/capture.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

void Capture::setGreenScreenEnabled(bool enabled)
{
    m_greenScreenEnabled = enabled;
}

bool Capture::isGreenScreenEnabled() const
{
    return m_greenScreenEnabled;
}

void Capture::setGreenHueRange(int hueMin, int hueMax)
{
    m_greenHueMin = std::max(0, std::min(179, hueMin));
    m_greenHueMax = std::max(0, std::min(179, hueMax));
}

void Capture::setGreenSaturationMin(int sMin)
{
    m_greenSatMin = std::max(0, std::min(255, sMin));
}

void Capture::setGreenValueMin(int vMin)
{
    m_greenValMin = std::max(0, std::min(255, vMin));
}

void Capture::updateGreenBackgroundModel(const cv::Mat &frame) const
{
    if (frame.empty() || frame.channels() != 3) {
        return;
    }

    const int rawBorderX = std::max(6, frame.cols / 24);
    const int rawBorderY = std::max(6, frame.rows / 24);
    const int borderX = std::min(rawBorderX, frame.cols);
    const int borderY = std::min(rawBorderY, frame.rows);

    if (borderX <= 0 || borderY <= 0) {
        return;
    }

    cv::Mat sampleMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    const cv::Rect topRect(0, 0, frame.cols, borderY);
    const cv::Rect bottomRect(0, std::max(0, frame.rows - borderY), frame.cols, borderY);
    const cv::Rect leftRect(0, 0, borderX, frame.rows);
    const cv::Rect rightRect(std::max(0, frame.cols - borderX), 0, borderX, frame.rows);

    cv::rectangle(sampleMask, topRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, bottomRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, leftRect, cv::Scalar(255), cv::FILLED);
    cv::rectangle(sampleMask, rightRect, cv::Scalar(255), cv::FILLED);

    const int samplePixels = cv::countNonZero(sampleMask);
    if (samplePixels < (frame.rows + frame.cols)) {
        return;
    }

    cv::Mat hsv, ycrcb;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);

    cv::Scalar hsvMean, hsvStd;
    cv::Scalar ycrcbMean, ycrcbStd;
    cv::Scalar bgrMean, bgrStd;

    cv::meanStdDev(hsv, hsvMean, hsvStd, sampleMask);
    cv::meanStdDev(ycrcb, ycrcbMean, ycrcbStd, sampleMask);
    cv::meanStdDev(frame, bgrMean, bgrStd, sampleMask);

    cv::Mat sampleData(samplePixels, 3, CV_32F);
    int idx = 0;
    for (int y = 0; y < frame.rows; ++y) {
        const uchar *maskPtr = sampleMask.ptr<uchar>(y);
        const cv::Vec3b *pixPtr = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            if (!maskPtr[x]) continue;
            float *row = sampleData.ptr<float>(idx++);
            row[0] = static_cast<float>(pixPtr[x][0]);
            row[1] = static_cast<float>(pixPtr[x][1]);
            row[2] = static_cast<float>(pixPtr[x][2]);
        }
    }
    if (idx > 3) {
        cv::Mat cropped = sampleData.rowRange(0, idx);
        cv::Mat cov, meanRow;
        cv::calcCovarMatrix(cropped, cov, meanRow,
                            cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
        cv::Mat covDouble;
        cov.convertTo(covDouble, CV_64F);
        covDouble += cv::Mat::eye(3, 3, CV_64F) * 1e-3;
        cv::Mat invCov;
        if (cv::invert(covDouble, invCov, cv::DECOMP_SVD)) {
            m_bgColorInvCov(0, 0) = invCov.at<double>(0, 0);
            m_bgColorInvCov(0, 1) = invCov.at<double>(0, 1);
            m_bgColorInvCov(0, 2) = invCov.at<double>(0, 2);
            m_bgColorInvCov(1, 0) = invCov.at<double>(1, 0);
            m_bgColorInvCov(1, 1) = invCov.at<double>(1, 1);
            m_bgColorInvCov(1, 2) = invCov.at<double>(1, 2);
            m_bgColorInvCov(2, 0) = invCov.at<double>(2, 0);
            m_bgColorInvCov(2, 1) = invCov.at<double>(2, 1);
            m_bgColorInvCov(2, 2) = invCov.at<double>(2, 2);
            m_bgColorInvCovReady = true;
        }
    }

    const bool alreadyInitialized = m_bgModelInitialized;
    auto blendValue = [alreadyInitialized](double current, double measurement) {
        if (!alreadyInitialized) return measurement;
        return 0.85 * current + 0.15 * measurement;
    };

    m_bgHueMean = blendValue(m_bgHueMean, hsvMean[0]);
    m_bgHueStd = blendValue(m_bgHueStd, std::max(1.0, hsvStd[0]));
    m_bgSatMean = blendValue(m_bgSatMean, hsvMean[1]);
    m_bgSatStd = blendValue(m_bgSatStd, std::max(1.0, hsvStd[1]));
    m_bgValMean = blendValue(m_bgValMean, hsvMean[2]);
    m_bgValStd = blendValue(m_bgValStd, std::max(1.0, hsvStd[2]));

    m_bgCbMean = blendValue(m_bgCbMean, ycrcbMean[2]);
    m_bgCbStd = blendValue(m_bgCbStd, std::max(1.0, ycrcbStd[2]));
    m_bgCrMean = blendValue(m_bgCrMean, ycrcbMean[1]);
    m_bgCrStd = blendValue(m_bgCrStd, std::max(1.0, ycrcbStd[1]));

    m_bgBlueMean = blendValue(m_bgBlueMean, bgrMean[0]);
    m_bgGreenMean = blendValue(m_bgGreenMean, bgrMean[1]);
    m_bgRedMean = blendValue(m_bgRedMean, bgrMean[2]);
    m_bgBlueStd = blendValue(m_bgBlueStd, std::max(4.0, bgrStd[0]));
    m_bgGreenStd = blendValue(m_bgGreenStd, std::max(4.0, bgrStd[1]));
    m_bgRedStd = blendValue(m_bgRedStd, std::max(4.0, bgrStd[2]));

    m_bgModelInitialized = true;
}

Capture::AdaptiveGreenThresholds Capture::computeAdaptiveGreenThresholds() const
{
    AdaptiveGreenThresholds thresholds;

    auto clampHue = [](int value) {
        return std::max(0, std::min(179, value));
    };
    auto clampByte = [](int value) {
        return std::max(0, std::min(255, value));
    };

    if (!m_bgModelInitialized) {
        thresholds.hueMin = clampHue(m_greenHueMin);
        thresholds.hueMax = clampHue(m_greenHueMax);
        thresholds.strictSatMin = clampByte(m_greenSatMin);
        thresholds.relaxedSatMin = clampByte(std::max(10, m_greenSatMin - 10));
        thresholds.strictValMin = clampByte(m_greenValMin);
        thresholds.relaxedValMin = clampByte(std::max(10, m_greenValMin - 10));
        thresholds.darkSatMin = clampByte(std::max(5, m_greenSatMin - 10));
        thresholds.darkValMax = clampByte(m_greenValMin + 50);
        thresholds.cbMin = 50.0;
        thresholds.cbMax = 150.0;
        thresholds.crMax = 150.0;
        thresholds.greenDelta = 8.0;
        thresholds.greenRatioMin = 0.45;
        thresholds.lumaMin = 45.0;
        thresholds.probabilityThreshold = 0.55;
        thresholds.guardValueMax = 150;
        thresholds.guardSatMax = 90;
        thresholds.edgeGuardMin = 45.0;
        thresholds.hueGuardPadding = 6;
        return thresholds;
    }

    const double hueStd = std::max(4.0, m_bgHueStd);
    const double satStd = std::max(4.0, m_bgSatStd);
    const double valStd = std::max(4.0, m_bgValStd);
    const double cbStd = std::max(2.5, m_bgCbStd);
    const double crStd = std::max(2.5, m_bgCrStd);

    const int huePadding = static_cast<int>(std::round(2.5 * hueStd)) + 4;
    const int relaxedSatAmount = static_cast<int>(std::round(1.9 * satStd)) + 5;
    const int relaxedValAmount = static_cast<int>(std::round(1.6 * valStd)) + 5;

    thresholds.hueMin = clampHue(static_cast<int>(std::round(m_bgHueMean)) - huePadding);
    thresholds.hueMax = clampHue(static_cast<int>(std::round(m_bgHueMean)) + huePadding);
    thresholds.strictSatMin = clampByte(static_cast<int>(std::round(m_bgSatMean - 0.6 * satStd)));
    thresholds.relaxedSatMin = clampByte(std::max(18, static_cast<int>(std::round(m_bgSatMean - relaxedSatAmount))));
    thresholds.strictValMin = clampByte(static_cast<int>(std::round(m_bgValMean - 0.6 * valStd)));
    thresholds.relaxedValMin = clampByte(std::max(18, static_cast<int>(std::round(m_bgValMean - relaxedValAmount))));
    thresholds.darkSatMin = clampByte(std::max(5, static_cast<int>(std::round(m_bgSatMean - 0.8 * satStd))));
    thresholds.darkValMax = clampByte(static_cast<int>(std::round(m_bgValMean + 2.2 * valStd)));

    const double cbRange = 2.2 * cbStd + 6.0;
    thresholds.cbMin = std::max(0.0, m_bgCbMean - cbRange);
    thresholds.cbMax = std::min(255.0, m_bgCbMean + cbRange);
    thresholds.crMax = std::min(255.0, m_bgCrMean + 2.4 * crStd + 6.0);

    const double greenDominance = m_bgGreenMean - std::max(m_bgRedMean, m_bgBlueMean);
    thresholds.greenDelta = std::max(4.0, greenDominance * 0.35 + 6.0);
    const double sumRGB = std::max(1.0, m_bgRedMean + m_bgGreenMean + m_bgBlueMean);
    const double bgRatio = m_bgGreenMean / sumRGB;
    thresholds.greenRatioMin = std::clamp(bgRatio - 0.05, 0.35, 0.8);
    thresholds.lumaMin = std::max(25.0, m_bgValMean - 1.2 * valStd);
    thresholds.probabilityThreshold = 0.58;
    thresholds.guardValueMax = clampByte(static_cast<int>(std::round(std::min(200.0, thresholds.lumaMin + 70.0))));
    thresholds.guardSatMax = clampByte(std::max(25, static_cast<int>(std::round(m_bgSatMean - 0.3 * satStd))));
    thresholds.edgeGuardMin = std::max(35.0, 55.0 - 0.25 * valStd);
    thresholds.hueGuardPadding = 8;
    auto invVariance = [](double stdVal) {
        const double boundedStd = std::max(5.0, stdVal);
        return 1.0 / (boundedStd * boundedStd + 50.0);
    };
    thresholds.invVarB = invVariance(m_bgBlueStd);
    thresholds.invVarG = invVariance(m_bgGreenStd);
    thresholds.invVarR = invVariance(m_bgRedStd);
    const double avgStd = (m_bgBlueStd + m_bgGreenStd + m_bgRedStd) / 3.0;
    thresholds.colorDistanceThreshold = std::max(2.5, std::min(4.5, 1.2 + avgStd * 0.08));
    thresholds.colorGuardThreshold = thresholds.colorDistanceThreshold + 1.6;

    return thresholds;
}

// AGGRESSIVE GREEN REMOVAL: Remove all green pixels including boundary pixels
// FAST & ACCURATE: Works for both green AND teal/cyan backdrops
cv::Mat Capture::createGreenScreenPersonMask(const cv::Mat &frame) const
{
    if (frame.empty()) return cv::Mat();

    try {
        // Split BGR channels
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        cv::Mat green = channels[1];
        cv::Mat red = channels[2];

        // FIXED: Properly detect green/cyan/teal while preserving blue colors
        // Green: G > R, G > B  
        // Cyan/Teal: G > R, B > R, G ≈ B
        // Blue: B > R, B > G (should NOT be removed)
        cv::Mat blue = channels[0];
        cv::Mat greenDominantR;
        cv::subtract(green, red, greenDominantR);  // G - R
        
        cv::Mat greenDominantB;
        cv::subtract(green, blue, greenDominantB);  // G - B
        
        // Green/cyan/teal: G > R AND (G > B OR G ≈ B)
        // This excludes pure blue colors where B > G
        cv::Mat greenOrTeal;
        cv::threshold(greenDominantR, greenOrTeal, 15, 255, cv::THRESH_BINARY);  // G > R
        
        cv::Mat greenDominatesBlue;
        cv::threshold(greenDominantB, greenDominatesBlue, -10, 255, cv::THRESH_BINARY);  // G > B - 10 (allows G ≈ B)
        
        // Combine: green/teal = (G > R) AND (G > B - 10)
        cv::Mat greenMask;
        cv::bitwise_and(greenOrTeal, greenDominatesBlue, greenMask);

        // Invert: NOT green/teal = person (preserves blue colors)
        cv::Mat personMask;
        cv::bitwise_not(greenMask, personMask);

        return personMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "Error in createGreenScreenPersonMask:" << e.what();
        return cv::Mat();
    }
}

//  GPU-ACCELERATED GREEN SCREEN MASKING with Optimized Memory Management
cv::cuda::GpuMat Capture::createGreenScreenPersonMaskGPU(const cv::cuda::GpuMat &gpuFrame) const
{
    cv::cuda::GpuMat emptyMask;
    if (gpuFrame.empty()) {
        qWarning() << "GPU frame is empty, cannot create green screen mask";
        return emptyMask;
    }

    if (!m_bgModelInitialized) {
        try {
            cv::Mat fallbackFrame;
            gpuFrame.download(fallbackFrame);
            updateGreenBackgroundModel(fallbackFrame);
        } catch (const cv::Exception &) {
            // Ignore download errors; thresholds fallback to defaults
        }
    }

    try {
        //  INITIALIZE CACHED FILTERS (once only, reused for all frames)
        static bool filtersInitialized = false;
        if (!filtersInitialized && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            try {
                // Cache filters to avoid recreating on every frame
                const_cast<Capture*>(this)->m_greenScreenCannyDetector = cv::cuda::createCannyEdgeDetector(30, 90);
                const_cast<Capture*>(this)->m_greenScreenGaussianBlur = cv::cuda::createGaussianFilter(CV_8U, CV_8U, cv::Size(5, 5), 1.0);
                filtersInitialized = true;
                qDebug() << "GPU green screen filters initialized and cached";
            } catch (const cv::Exception &e) {
                qWarning() << "Failed to initialize GPU green screen filters:" << e.what();
                filtersInitialized = false;
            }
        }

        // GPU: FAST & ACCURATE - Works for both green AND teal/cyan backdrops
        // Split BGR channels
        std::vector<cv::cuda::GpuMat> channels(3);
        cv::cuda::split(gpuFrame, channels);
        cv::cuda::GpuMat green = channels[1];
        cv::cuda::GpuMat red = channels[2];

        // FIXED: Properly detect green/cyan/teal while preserving blue colors
        // Green: G > R, G > B  |  Cyan/Teal: G > R, B > R, G ≈ B
        // Blue: B > R, B > G (should NOT be removed)
        cv::cuda::GpuMat blue = channels[0];
        cv::cuda::GpuMat greenDominantR;
        cv::cuda::subtract(green, red, greenDominantR);  // G - R
        
        cv::cuda::GpuMat greenDominantB;
        cv::cuda::subtract(green, blue, greenDominantB);  // G - B
        
        // Green/cyan/teal: G > R AND (G > B OR G ≈ B)
        // This excludes pure blue colors where B > G
        cv::cuda::GpuMat greenOrTeal;
        cv::cuda::threshold(greenDominantR, greenOrTeal, 15, 255, cv::THRESH_BINARY);  // G > R
        
        cv::cuda::GpuMat greenDominatesBlue;
        cv::cuda::threshold(greenDominantB, greenDominatesBlue, -10, 255, cv::THRESH_BINARY);  // G > B - 10 (allows G ≈ B)
        
        // Combine: green/teal = (G > R) AND (G > B - 10)
        cv::cuda::GpuMat greenMask;
        cv::cuda::bitwise_and(greenOrTeal, greenDominatesBlue, greenMask);

        // Invert: NOT green/teal = person (preserves blue colors)
        cv::cuda::GpuMat gpuPersonMask;
        cv::cuda::bitwise_not(greenMask, gpuPersonMask);

        return gpuPersonMask;

    } catch (const cv::Exception &e) {
        qWarning() << "GPU green screen masking failed:" << e.what() << "- returning empty mask";
        return emptyMask;
    } catch (const std::exception &e) {
        qWarning() << "Exception in GPU green screen masking:" << e.what();
        return emptyMask;
    } catch (...) {
        qWarning() << "Unknown exception in GPU green screen masking";
        return emptyMask;
    }
}

// GPU-ACCELERATED GREEN SPILL REMOVAL - Remove green tint from person pixels
cv::cuda::GpuMat Capture::removeGreenSpillGPU(const cv::cuda::GpuMat &gpuFrame, const cv::cuda::GpuMat &gpuMask) const
{
    cv::cuda::GpuMat result;
    if (gpuFrame.empty() || gpuMask.empty()) {
        return gpuFrame.clone();
    }

    try {
        // Convert to HSV for color correction
        cv::cuda::GpuMat gpuHSV;
        cv::cuda::cvtColor(gpuFrame, gpuHSV, cv::COLOR_BGR2HSV);

        // Split HSV channels
        std::vector<cv::cuda::GpuMat> hsvChannels(3);
        cv::cuda::split(gpuHSV, hsvChannels);

        // Create a desaturation map based on green hue proximity
        // Pixels closer to green hue will be more desaturated
        cv::cuda::GpuMat hueChannel = hsvChannels[0].clone();
        
        // Create desaturation mask for greenish pixels (narrower range to preserve person's colors)
        cv::cuda::GpuMat greenishMask1, greenishMask2;
        cv::cuda::threshold(hueChannel, greenishMask1, m_greenHueMin, 255, cv::THRESH_BINARY);  // No extra margin
        cv::cuda::threshold(hueChannel, greenishMask2, m_greenHueMax, 255, cv::THRESH_BINARY_INV);  // No extra margin
        cv::cuda::GpuMat greenishRange;
        cv::cuda::bitwise_and(greenishMask1, greenishMask2, greenishRange);
        
        // Desaturate pixels in green hue range
        cv::cuda::GpuMat satChannel = hsvChannels[1].clone();
        cv::cuda::GpuMat desaturated;
        cv::cuda::multiply(satChannel, cv::Scalar(0.3), desaturated, 1.0, satChannel.type()); // Reduce saturation to 30%
        
        // Apply desaturation only to greenish pixels within person mask
        cv::cuda::GpuMat spillMask;
        cv::cuda::bitwise_and(greenishRange, gpuMask, spillMask);
        desaturated.copyTo(satChannel, spillMask);
        
        // Merge back
        hsvChannels[1] = satChannel;
        cv::cuda::merge(hsvChannels, gpuHSV);
        
        // Convert back to BGR
        cv::cuda::cvtColor(gpuHSV, result, cv::COLOR_HSV2BGR);
        
        // Also apply color correction in BGR space to remove green channel dominance
        std::vector<cv::cuda::GpuMat> bgrChannels(3);
        cv::cuda::split(result, bgrChannels);
        
        // Reduce green channel where spill is detected
        cv::cuda::GpuMat reducedGreen;
        cv::cuda::multiply(bgrChannels[1], cv::Scalar(0.85), reducedGreen, 1.0, bgrChannels[1].type());
        reducedGreen.copyTo(bgrChannels[1], spillMask);
        
        // Slightly boost blue and red to compensate
        cv::cuda::GpuMat boostedBlue, boostedRed;
        cv::cuda::multiply(bgrChannels[0], cv::Scalar(1.08), boostedBlue, 1.0, bgrChannels[0].type());
        cv::cuda::multiply(bgrChannels[2], cv::Scalar(1.08), boostedRed, 1.0, bgrChannels[2].type());
        boostedBlue.copyTo(bgrChannels[0], spillMask);
        boostedRed.copyTo(bgrChannels[2], spillMask);
        
        cv::cuda::merge(bgrChannels, result);
        
        // Synchronize
        cv::cuda::Stream::Null().waitForCompletion();
        
        return result;

    } catch (const cv::Exception &e) {
        qWarning() << "GPU green spill removal failed:" << e.what();
        return gpuFrame.clone();
    } catch (const std::exception &e) {
        qWarning() << "Exception in GPU green spill removal:" << e.what();
        return gpuFrame.clone();
    }
}

// Derive bounding boxes from a binary person mask
std::vector<cv::Rect> Capture::deriveDetectionsFromMask(const cv::Mat &mask) const
{
    std::vector<cv::Rect> detections;
    if (mask.empty()) return detections;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // GPU MEMORY PROTECTION: Maximum detection size to prevent GPU memory overflow
    const int MAX_DETECTION_WIDTH = std::min(1920, mask.cols);
    const int MAX_DETECTION_HEIGHT = std::min(1080, mask.rows);
    const int MIN_DETECTION_AREA = 1000; // Minimum area to filter noise

    for (const auto &c : contours) {
        cv::Rect r = cv::boundingRect(c);
        
        // Filter by minimum area
        if (r.area() < MIN_DETECTION_AREA) continue;
        
        // CLAMP detection rectangle to safe bounds
        r.x = std::max(0, r.x);
        r.y = std::max(0, r.y);
        r.width = std::min(r.width, std::min(MAX_DETECTION_WIDTH, mask.cols - r.x));
        r.height = std::min(r.height, std::min(MAX_DETECTION_HEIGHT, mask.rows - r.y));
        
        // Validate final rectangle
        if (r.width > 0 && r.height > 0 && r.area() >= MIN_DETECTION_AREA) {
            detections.push_back(r);
        }
    }

    // Prefer the largest contours (likely persons)
    std::sort(detections.begin(), detections.end(), [](const cv::Rect &a, const cv::Rect &b){ return a.area() > b.area(); });
    if (detections.size() > 3) detections.resize(3);
    
    qDebug() << "Derived" << detections.size() << "valid detections from mask";
    return detections;
}

