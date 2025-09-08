#include "algorithms/lighting_correction/lighting_corrector.h"
#include <QFileInfo>
#include <cmath>

LightingCorrector::LightingCorrector()
    : m_enabled(true)
    , m_gpuAvailable(false)
    , m_initialized(false)
    , m_clipLimit(8.0)  // More aggressive CLAHE for visible effect
    , m_tileGridSize(8, 8)
    , m_gammaValue(1.5)  // More aggressive gamma correction
    , m_colorBalanceStrength(1.2)  // More aggressive color balance
{
    qDebug() << "🌟 LightingCorrector: Constructor called";
}

LightingCorrector::~LightingCorrector()
{
    cleanup();
    qDebug() << "🌟 LightingCorrector: Destructor called";
}

bool LightingCorrector::initialize()
{
    qDebug() << "🌟 LightingCorrector: Initializing lighting correction system";
    
    try {
        // Check GPU availability
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            qDebug() << "🌟 LightingCorrector: CUDA available, initializing GPU resources";
            
            // Initialize GPU CLAHE
            m_gpuCLAHE = cv::cuda::createCLAHE();
            m_gpuCLAHE->setClipLimit(m_clipLimit);
            m_gpuCLAHE->setTilesGridSize(m_tileGridSize);
            
            // Initialize GPU buffers
            m_gpuInputBuffer = cv::cuda::GpuMat();
            m_gpuMaskBuffer = cv::cuda::GpuMat();
            m_gpuTemplateBuffer = cv::cuda::GpuMat();
            m_gpuOutputBuffer = cv::cuda::GpuMat();
            m_gpuTempBuffer1 = cv::cuda::GpuMat();
            m_gpuTempBuffer2 = cv::cuda::GpuMat();
            
            m_gpuAvailable = true;
            qDebug() << "🌟 LightingCorrector: GPU resources initialized successfully";
        } else {
            qDebug() << "🌟 LightingCorrector: CUDA not available, using CPU processing";
            m_gpuAvailable = false;
        }
        
        // Initialize CPU CLAHE
        m_cpuCLAHE = cv::createCLAHE();
        m_cpuCLAHE->setClipLimit(m_clipLimit);
        m_cpuCLAHE->setTilesGridSize(m_tileGridSize);
        
        m_initialized = true;
        qDebug() << "🌟 LightingCorrector: Initialization completed successfully";
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Initialization failed:" << e.what();
        m_initialized = false;
        m_gpuAvailable = false;
        return false;
    }
}

bool LightingCorrector::setReferenceTemplate(const QString &templatePath)
{
    if (!m_initialized) {
        qWarning() << "🌟 LightingCorrector: Not initialized, cannot set reference template";
        return false;
    }
    
    if (templatePath.isEmpty()) {
        qWarning() << "🌟 LightingCorrector: Empty template path provided";
        return false;
    }
    
    QFileInfo fileInfo(templatePath);
    if (!fileInfo.exists()) {
        qWarning() << "🌟 LightingCorrector: Template file does not exist:" << templatePath;
        return false;
    }
    
    try {
        // Load template image
        m_referenceTemplate = cv::imread(templatePath.toStdString());
        if (m_referenceTemplate.empty()) {
            qWarning() << "🌟 LightingCorrector: Failed to load template image:" << templatePath;
            return false;
        }
        
        // Analyze template lighting characteristics
        m_templateLightingProfile = analyzeTemplateLighting(m_referenceTemplate);
        
        m_templatePath = templatePath;
        qDebug() << "🌟 LightingCorrector: Reference template set successfully:" << templatePath;
        qDebug() << "🌟 LightingCorrector: Template size:" << m_referenceTemplate.cols << "x" << m_referenceTemplate.rows;
        
        return true;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Failed to set reference template:" << e.what();
        return false;
    }
}

// Helper: Match histogram of source to reference
static cv::Mat matchHistogram(const cv::Mat& src, const cv::Mat& ref, const cv::Mat& mask = cv::Mat())
{
    CV_Assert(src.type() == CV_8UC3 && ref.type() == CV_8UC3);

    cv::Mat srcYCrCb, refYCrCb;
    cv::cvtColor(src, srcYCrCb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(ref, refYCrCb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> srcChannels, refChannels;
    cv::split(srcYCrCb, srcChannels);
    cv::split(refYCrCb, refChannels);

    for (int i = 0; i < 3; i++) {
        // Compute histogram
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};

        cv::Mat srcHist, refHist;
        cv::calcHist(&srcChannels[i], 1, 0, mask, srcHist, 1, &histSize, &histRange, true, false);
        cv::calcHist(&refChannels[i], 1, 0, cv::Mat(), refHist, 1, &histSize, &histRange, true, false);

        // Normalize histograms
        cv::normalize(srcHist, srcHist, 0, 1, cv::NORM_MINMAX);
        cv::normalize(refHist, refHist, 0, 1, cv::NORM_MINMAX);

        // Build LUT for mapping
        std::vector<int> lut(256, 0);
        int ri = 0;
        float srcCDF = 0, refCDF = 0;

        for (int v = 0; v < 256; v++) {
            srcCDF += srcHist.at<float>(v);
            while (ri < 256 && refCDF < srcCDF) {
                refCDF += refHist.at<float>(ri++);
            }
            lut[v] = std::min(ri, 255);
        }

        // Apply LUT
        cv::Mat lutMat(1, 256, CV_8U);
        for (int v = 0; v < 256; v++) lutMat.at<uchar>(v) = static_cast<uchar>(lut[v]);
        cv::LUT(srcChannels[i], lutMat, srcChannels[i]);
    }

    cv::merge(srcChannels, srcYCrCb);
    cv::Mat matched;
    cv::cvtColor(srcYCrCb, matched, cv::COLOR_YCrCb2BGR);
    return matched;
}

cv::Mat LightingCorrector::applyPersonLightingCorrection(
    const cv::Mat &inputImage,
    const cv::Mat &personMaskIn,
    const cv::Mat &referenceTemplate)
{
    if (!m_enabled || !m_initialized) return inputImage.clone();
    if (inputImage.empty() || personMaskIn.empty() || referenceTemplate.empty())
        return inputImage.clone();

    try {
        // ----- TUNABLE STRENGTH (increase if still subtle) -----
        const float AB_AMPLIFY   = 3.0f;   // amplifies a/b (makes hue change very visible)
        const float BGR_TINT_STR = 0.9f;   // additive tint strength toward template (0..1)
        const float BGR_SCALE_STR= 1.1f;   // multiply person channels slightly toward template
        // ------------------------------------------------------

        // --- 0) Prepare copies and canonical formats ---
        cv::Mat frame;
        if (inputImage.type() != CV_8UC3) inputImage.convertTo(frame, CV_8UC3);
        else frame = inputImage.clone();

        cv::Mat mask;
        if (personMaskIn.channels() == 3) cv::cvtColor(personMaskIn, mask, cv::COLOR_BGR2GRAY);
        else mask = personMaskIn.clone();
        if (mask.type() != CV_8U) mask.convertTo(mask, CV_8U);
        double mn, mx; cv::minMaxLoc(mask, &mn, &mx);
        if (mx <= 1.0) mask *= 255;

        // --- 1) Template resize ---
        cv::Mat tmpl;
        if (referenceTemplate.size() != frame.size())
            cv::resize(referenceTemplate, tmpl, frame.size(), 0, 0, cv::INTER_AREA);
        else
            tmpl = referenceTemplate.clone();

        // --- 2) Mask cleanup: open/close, keep largest CC, fill holes ---
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15)));

        // Keep largest component
        {
            cv::Mat labels, stats, centroids;
            int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
            if (n > 1) {
                int best = 1; int bestA = stats.at<int>(1, cv::CC_STAT_AREA);
                for (int i = 2; i < n; ++i) {
                    int a = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (a > bestA) { bestA = a; best = i; }
                }
                mask = (labels == best);
                mask.convertTo(mask, CV_8U, 255);
            }
        }

        // Fill holes with flood fill technique
        {
            cv::Mat inv; cv::bitwise_not(mask, inv);
            cv::Mat flood = inv.clone();
            cv::floodFill(flood, cv::Point(0,0), cv::Scalar(0));
            cv::bitwise_not(flood, flood);
            cv::bitwise_or(mask, flood, mask);
        }

        // Final small close to ensure solidity
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));

        // --- 3) Build feather/alpha from distance transform ---
        cv::Mat dist; cv::distanceTransform(mask, dist, cv::DIST_L2, 5);
        double maxDist; cv::minMaxLoc(dist, nullptr, &maxDist);
        cv::Mat feather;
        if (maxDist < 1e-3) {
            feather = mask.clone(); feather.convertTo(feather, CV_32F, 1.0/255.0);
        } else {
            dist.convertTo(feather, CV_32F, 1.0 / (maxDist + 1e-6));
            cv::pow(feather, 0.45f, feather); // makes edges soft
        }

        // --- 4) Extract personOnly and inpaint holes (ensures no black pixels) ---
        cv::Mat personOnly = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(personOnly, mask);

        // Identify small holes inside bounding box (where mask==0 but inside bbox)
        cv::Rect bbox = cv::boundingRect(mask);
        if (bbox.width == 0 || bbox.height == 0) return inputImage.clone();
        cv::Mat maskROI = mask(bbox);
        cv::Mat personROI = personOnly(bbox);

        // Create hole mask in ROI (0 where hole)
        cv::Mat holeMask; cv::bitwise_not(maskROI, holeMask); // holes are 255 in holeMask
        // Only inpaint if there are holes
        if (cv::countNonZero(holeMask) > 0) {
            // Expand hole mask slightly to allow inpaint context
            cv::Mat holeDil; cv::dilate(holeMask, holeDil, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
            // Inpaint personROI on a copy
            cv::Mat personROIcopy = personROI.clone();
            // inpaint needs 8-bit 1-channel mask where non-zero indicates inpainting regions
            cv::inpaint(personROIcopy, holeDil, personROIcopy, 3.0, cv::INPAINT_TELEA);
            personROIcopy.copyTo(personROI, cv::Mat()); // replace ROI
            // put back to personOnly
            personROI.copyTo(personOnly(bbox));
        }

        // --- 5) Compute color stats on template ROI and person (for strong mapping) ---
        cv::Mat tmplROI = tmpl(bbox);
        // Convert to Lab for a/b amplification
        cv::Mat personLab, tmplLab;
        cv::cvtColor(personOnly, personLab, cv::COLOR_BGR2Lab);
        cv::cvtColor(tmpl, tmplLab, cv::COLOR_BGR2Lab);

        // Compute masked mean/std for person
        cv::Scalar meanP, stdP;
        {
            cv::Mat maskFloat; mask.convertTo(maskFloat, CV_8U);
            cv::meanStdDev(personLab, meanP, stdP, maskFloat);
            for (int i=0;i<3;++i) if (stdP[i] < 1.0) stdP[i] = 1.0; // avoid zero
        }

        // Compute mean/std for template ROI (unmasked)
        cv::Scalar meanT, stdT;
        {
            cv::Mat roiLab = tmplLab(bbox);
            cv::meanStdDev(roiLab, meanT, stdT);
            for (int i=0;i<3;++i) if (stdT[i] < 2.0) stdT[i] = 12.0;
        }

        // --- 6) Strong (full) Reinhard transfer on Lab but with AB amplification ---
        std::vector<cv::Mat> ch;
        cv::split(personLab, ch);
        for (int i=0;i<3;++i) {
            ch[i].convertTo(ch[i], CV_32F);
            double muP = meanP[i], sdP = stdP[i];
            double muT = meanT[i], sdT = stdT[i];
            ch[i] = ((ch[i] - float(muP)) / float(sdP)) * float(sdT) + float(muT);
            // Amplify a/b channels strongly to make hue change obvious
            if (i == 1 || i == 2) {
                // shift values away from original mean by factor AB_AMPLIFY
                ch[i] = (ch[i] - float(muT)) * AB_AMPLIFY + float(muT);
            }
            cv::min(ch[i], 255.0f, ch[i]);
            cv::max(ch[i], 0.0f, ch[i]);
            ch[i].convertTo(ch[i], CV_8U);
        }
        cv::merge(ch, personLab);

        cv::Mat relitLab2BGR;
        cv::cvtColor(personLab, relitLab2BGR, cv::COLOR_Lab2BGR);

        // --- 7) Strong per-channel BGR tint/scale toward template mean (visible cast) ---
        cv::Scalar avgT_BGR = cv::mean(tmpl(bbox));
        cv::Scalar avgP_BGR = cv::mean(relitLab2BGR, mask);
        cv::Mat relitFloat; relitLab2BGR.convertTo(relitFloat, CV_32F);
        std::vector<cv::Mat> rc(3); cv::split(relitFloat, rc);
        for (int c=0;c<3;++c) {
            float target = float(avgT_BGR[c]);
            float source = float(avgP_BGR[c]) + 1e-6f;
            float scale = (target / source) * BGR_SCALE_STR;
            // apply scaling then additive tint toward template color
            rc[c] = rc[c] * scale + (target - source) * BGR_TINT_STR;
        }
        cv::merge(rc, relitFloat);

        // convert back to 8U - ensure no zeros
        cv::Mat relitBGR; relitFloat.convertTo(relitBGR, CV_8U);
        // if any zero pixels still exist inside bbox, run a local inpaint
        {
            cv::Mat zeroMask;
            cv::inRange(relitBGR(bbox), cv::Scalar(0,0,0), cv::Scalar(0,0,0), zeroMask);
            if (cv::countNonZero(zeroMask) > 0) {
                cv::Mat tmp = relitBGR(bbox).clone();
                cv::inpaint(tmp, zeroMask, tmp, 3.0, cv::INPAINT_TELEA);
                tmp.copyTo(relitBGR(bbox));
            }
        }

        // --- 8) Edge de-spill: desaturate thin ring to avoid fringe color -- optional ---
        {
            cv::Mat eroded; cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
            cv::Mat ring; cv::subtract(mask, eroded, ring);
            if (cv::countNonZero(ring) > 0) {
                cv::Mat hsv; cv::cvtColor(relitBGR, hsv, cv::COLOR_BGR2HSV);
                std::vector<cv::Mat> hsvC; cv::split(hsv, hsvC);
                // reduce saturation on ring
                hsvC[1].setTo(cv::Scalar((int)(0.6*255)), ring);
                cv::merge(hsvC, hsv);
                cv::cvtColor(hsv, relitBGR, cv::COLOR_HSV2BGR);
            }
        }

        // --- 9) Composite using feather alpha (float) ---
        cv::Mat relF, tmplF, alpha3;
        relitBGR.convertTo(relF, CV_32F);
        tmpl.convertTo(tmplF, CV_32F);
        std::vector<cv::Mat> fch = {feather, feather, feather};
        cv::merge(fch, alpha3);

        cv::Mat compositeF = relF.mul(alpha3) + tmplF.mul(1.0f - alpha3);
        cv::Mat out; compositeF.convertTo(out, CV_8U);

        return out;
    }
    catch (const cv::Exception &e) {
        qWarning() << "applyPersonLightingCorrection exception: " << e.what();
        return inputImage.clone();
    }
}

cv::Mat LightingCorrector::applyGlobalLightingCorrection(const cv::Mat &inputImage)
{
    if (!m_enabled || !m_initialized) {
        qDebug() << "🌟 LightingCorrector: Not enabled or not initialized, returning original";
        return inputImage.clone();
    }
    
    if (inputImage.empty()) {
        qWarning() << "🌟 LightingCorrector: Empty input image for global correction";
        return cv::Mat();
    }
    
    qDebug() << "🌟 LightingCorrector: Applying global lighting correction to image size:" << inputImage.cols << "x" << inputImage.rows;
    
    try {
        cv::Mat result;
        
        // Convert to LAB color space
        cv::Mat labImage;
        cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);
        
        // Split channels
        std::vector<cv::Mat> labChannels;
        cv::split(labImage, labChannels);
        
        // Apply CLAHE to L channel
        m_cpuCLAHE->apply(labChannels[0], labChannels[0]);
        
        // Merge channels
        cv::merge(labChannels, result);
        
        // Convert back to BGR
        cv::cvtColor(result, result, cv::COLOR_Lab2BGR);
        
        // Apply gamma correction
        result = applyGammaCorrection(result, m_gammaValue);
        
        qDebug() << "🌟 LightingCorrector: Global lighting correction completed successfully";
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Global correction failed:" << e.what();
        return inputImage.clone();
    }
}


cv::Mat LightingCorrector::analyzeTemplateLighting(const cv::Mat &templateImage)
{
    try {
        // Convert to LAB and analyze L channel statistics
        cv::Mat labTemplate;
        cv::cvtColor(templateImage, labTemplate, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> channels;
        cv::split(labTemplate, channels);
        
        // Calculate lighting profile (mean, std dev, etc.)
        cv::Scalar mean, stddev;
        cv::meanStdDev(channels[0], mean, stddev);
        
        // Create a simple lighting profile
        cv::Mat profile = cv::Mat::zeros(1, 3, CV_64F);
        profile.at<double>(0) = mean[0];  // Mean lightness
        profile.at<double>(1) = stddev[0]; // Standard deviation
        profile.at<double>(2) = cv::mean(channels[0])[0]; // Overall brightness
        
        return profile;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Template analysis failed:" << e.what();
        return cv::Mat::zeros(1, 3, CV_64F);
    }
}

cv::Mat LightingCorrector::applyColorBalanceWithReference(const cv::Mat &input, 
                                                         const cv::Mat &reference, 
                                                         const cv::Mat &mask)
{
    try {
        cv::Mat result = input.clone();
        
        // Calculate mean values for person region only
        cv::Scalar personMean = cv::mean(input, mask);
        cv::Scalar referenceMean = cv::mean(reference);
        
        // Calculate correction factors
        double factorB = referenceMean[0] / (personMean[0] + 1e-6);
        double factorG = referenceMean[1] / (personMean[1] + 1e-6);
        double factorR = referenceMean[2] / (personMean[2] + 1e-6);
        
        // Limit correction factors
        factorB = std::min(std::max(factorB, 0.7), 1.3);
        factorG = std::min(std::max(factorG, 0.7), 1.3);
        factorR = std::min(std::max(factorR, 0.7), 1.3);
        
        // Apply correction only to person region
        std::vector<cv::Mat> channels;
        cv::split(result, channels);
        
        cv::Mat maskFloat;
        mask.convertTo(maskFloat, CV_32F, 1.0/255.0);
        
        // Apply color balance with mask
        cv::Mat correctedB, correctedG, correctedR;
        cv::multiply(channels[0], factorB, correctedB);
        cv::multiply(channels[1], factorG, correctedG);
        cv::multiply(channels[2], factorR, correctedR);
        
        // Blend with original using mask
        cv::Mat finalB, finalG, finalR;
        cv::multiply(channels[0], (1.0 - maskFloat), finalB);
        cv::multiply(correctedB, maskFloat, correctedB);
        cv::add(finalB, correctedB, finalB);
        
        cv::multiply(channels[1], (1.0 - maskFloat), finalG);
        cv::multiply(correctedG, maskFloat, correctedG);
        cv::add(finalG, correctedG, finalG);
        
        cv::multiply(channels[2], (1.0 - maskFloat), finalR);
        cv::multiply(correctedR, maskFloat, finalR);
        cv::add(finalR, finalR, finalR);
        
        std::vector<cv::Mat> finalChannels = {finalB, finalG, finalR};
        cv::merge(finalChannels, result);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Color balance failed:" << e.what();
        return input.clone();
    }
}

cv::Mat LightingCorrector::applyGammaCorrection(const cv::Mat &input, double gamma)
{
    try {
        cv::Mat result;
        
        // Create lookup table
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar* p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, 1.0 / gamma) * 255.0);
        }
        
        // Apply gamma correction
        cv::LUT(input, lookupTable, result);
        
        return result;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Gamma correction failed:" << e.what();
        return input.clone();
    }
}

cv::Mat LightingCorrector::createSmoothMask(const cv::Mat &binaryMask)
{
    try {
        cv::Mat smoothMask;
        
        // Apply Gaussian blur for smooth edges
        cv::GaussianBlur(binaryMask, smoothMask, cv::Size(15, 15), 0);
        
        // Apply morphological operations for better mask quality
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(smoothMask, smoothMask, cv::MORPH_CLOSE, kernel);
        
        return smoothMask;
        
    } catch (const cv::Exception& e) {
        qWarning() << "🌟 LightingCorrector: Smooth mask creation failed:" << e.what();
        return binaryMask.clone();
    }
}

void LightingCorrector::setEnabled(bool enabled)
{
    m_enabled = enabled;
    qDebug() << "🌟 LightingCorrector: Lighting correction" << (enabled ? "enabled" : "disabled");
}

bool LightingCorrector::isEnabled() const
{
    return m_enabled;
}

bool LightingCorrector::isGPUAvailable() const
{
    return m_gpuAvailable;
}

cv::Mat LightingCorrector::getReferenceTemplate() const
{
    return m_referenceTemplate.clone();
}

void LightingCorrector::cleanup()
{
    qDebug() << "🌟 LightingCorrector: Cleaning up resources";
    
    // Release GPU resources
    m_gpuCLAHE.release();
    m_gpuInputBuffer.release();
    m_gpuMaskBuffer.release();
    m_gpuTemplateBuffer.release();
    m_gpuOutputBuffer.release();
    m_gpuTempBuffer1.release();
    m_gpuTempBuffer2.release();
    
    // Release CPU resources
    m_cpuCLAHE.release();
    
    // Release template data
    m_referenceTemplate.release();
    m_templateLightingProfile.release();
    
    m_initialized = false;
    m_gpuAvailable = false;
}
