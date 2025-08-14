#include "mediapipe_like_hand_tracker.h"

using std::vector;

namespace {
static inline cv::Rect clampRect(const cv::Rect& r, int w, int h) {
	int x = std::max(0, r.x);
	int y = std::max(0, r.y);
	int rw = std::min(r.width, w - x);
	int rh = std::min(r.height, h - y);
	return cv::Rect(x, y, std::max(0, rw), std::max(0, rh));
}
}

HandTrackerMP::HandTrackerMP()
	: m_wasOpen(false)
	, m_wasClosed(false)
	, m_stableFrames(0)
	, m_triggered(false)
	, m_hasLock(false)
	, m_bgInit(false)
	, m_frameWidth(0)
	, m_frameHeight(0)
	, m_frameCount(0)
    , m_motionThreshold(14)
    , m_minMotionArea(250)
    , m_redetectInterval(1)
	, m_minRoiSize(48)
	, m_maxRoiSize(360)
    , m_requiredStableFrames(1) {
	m_timer.start();
}

void HandTrackerMP::reset() {
	m_wasOpen = false;
	m_wasClosed = false;
	m_stableFrames = 0;
	m_triggered = false;
	m_hasLock = false;
	m_prevPts.clear();
    m_prevGray.release();
    m_bgFloat.release();
	m_bgInit = false;
	m_roi = cv::Rect();
}

void HandTrackerMP::initialize(int frameWidth, int frameHeight) {
	m_frameWidth = frameWidth;
	m_frameHeight = frameHeight;
	reset();
}

bool HandTrackerMP::detectMotion(const cv::Mat& gray, cv::Mat& motionMask) {
    // Downscale for faster motion estimation
    cv::Mat small;
    const int targetW = std::max(160, gray.cols / 2);
    const int targetH = std::max(120, gray.rows / 2);
    cv::resize(gray, small, cv::Size(targetW, targetH), 0, 0, cv::INTER_AREA);

    if (!m_bgInit) {
        small.convertTo(m_bgFloat, CV_32F);
        m_bgInit = true;
        return false;
    }
    cv::Mat bg8;
    m_bgFloat.convertTo(bg8, CV_8U);
    cv::Mat diffSmall;
    cv::absdiff(small, bg8, diffSmall);
    cv::Mat maskSmall;
    cv::threshold(diffSmall, maskSmall, m_motionThreshold, 255, cv::THRESH_BINARY);
    cv::morphologyEx(maskSmall, maskSmall, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    // Upscale mask to original size
    cv::resize(maskSmall, motionMask, gray.size(), 0, 0, cv::INTER_NEAREST);
    int area = cv::countNonZero(maskSmall);
    // Slowly update background (EMA) in small scale
    cv::Mat smallF; small.convertTo(smallF, CV_32F);
    cv::accumulateWeighted(smallF, m_bgFloat, 0.02);
    return area > m_minMotionArea;
}

bool HandTrackerMP::acquireRoiFromMotion(const cv::Mat& gray, const cv::Mat& motionMask) {
	vector<vector<cv::Point>> contours;
	cv::findContours(motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	int bestIdx = -1; double bestArea = 0;
	for (int i = 0; i < (int)contours.size(); ++i) {
		double a = cv::contourArea(contours[i]);
		if (a > bestArea) { bestArea = a; bestIdx = i; }
	}
	if (bestIdx < 0) return false;
	cv::Rect r = cv::boundingRect(contours[bestIdx]);
	r = clampRect(r, gray.cols, gray.rows);
	if (r.width < m_minRoiSize || r.height < m_minRoiSize) return false;
	if (r.width > m_maxRoiSize || r.height > m_maxRoiSize) return false;
	m_roi = r;
	m_hasLock = true;
	// initialize track points inside ROI
	vector<cv::Point2f> pts;
	cv::goodFeaturesToTrack(gray(m_roi), pts, 60, 0.01, 3);
	for (auto& p : pts) p += cv::Point2f((float)m_roi.x, (float)m_roi.y);
	m_prevPts = std::move(pts);
	return !m_prevPts.empty();
}

void HandTrackerMP::trackRoiLK(const cv::Mat& grayPrev, const cv::Mat& grayCurr) {
	if (m_prevPts.empty()) { m_hasLock = false; return; }
	vector<cv::Point2f> nextPts; vector<unsigned char> status; vector<float> err;
	cv::calcOpticalFlowPyrLK(grayPrev, grayCurr, m_prevPts, nextPts, status, err,
		cv::Size(15,15), 2);
	vector<cv::Point2f> kept; kept.reserve(nextPts.size());
	for (size_t i = 0; i < nextPts.size(); ++i) if (status[i]) kept.push_back(nextPts[i]);
	if (kept.size() < 8) { m_hasLock = false; return; }
	// Fit new ROI to tracked points
	cv::Rect r = cv::boundingRect(kept);
	r = clampRect(r, grayCurr.cols, grayCurr.rows);
	// EMA smoothing
	if (m_roi.area() > 0) {
		r.x = (int)(0.7 * m_roi.x + 0.3 * r.x);
		r.y = (int)(0.7 * m_roi.y + 0.3 * r.y);
		r.width = (int)(0.7 * m_roi.width + 0.3 * r.width);
		r.height = (int)(0.7 * m_roi.height + 0.3 * r.height);
	}
	m_roi = r;
	m_prevPts = std::move(kept);
}

bool HandTrackerMP::analyzeGestureClosed(const cv::Mat& gray, const cv::Rect& roi) const {
    cv::Rect r = clampRect(roi, gray.cols, gray.rows);
    if (r.area() <= 0) return false;
    cv::Mat patch = gray(r);
    // Light denoise for robust contours
    cv::GaussianBlur(patch, patch, cv::Size(3,3), 0);
    // Binary inverse: hand darker becomes white
    cv::Mat bin; cv::threshold(patch, bin, 0, 255, cv::THRESH_BINARY_INV|cv::THRESH_OTSU);
    vector<vector<cv::Point>> cs; cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty()) return false;
    int k = -1; double best = 0; for (int i=0;i<(int)cs.size();++i){ double a=cv::contourArea(cs[i]); if (a>best){best=a;k=i;} }
    if (k<0) return false;
    vector<int> hullIdx; cv::convexHull(cs[k], hullIdx);
    vector<cv::Point> hull; hull.reserve(hullIdx.size()); for (int idx: hullIdx) hull.push_back(cs[k][idx]);
    vector<cv::Vec4i> defects; if (hullIdx.size() >= 3) cv::convexityDefects(cs[k], hullIdx, defects);
    int significantDef = 0; for (auto &d : defects) { if (d[3] / 256.0 > 10) significantDef++; }
    double area = cv::contourArea(cs[k]); double harea = std::max(1.0, cv::contourArea(hull));
    double ratio = area / harea; // closed -> ratio high (compact)
    // More permissive: allow side-on fists; also use few defects as closed cue
    return (ratio > 0.62) || (significantDef <= 1 && ratio > 0.52);
}

bool HandTrackerMP::analyzeGestureOpen(const cv::Mat& gray, const cv::Rect& roi) const {
    cv::Rect r = clampRect(roi, gray.cols, gray.rows);
    if (r.area() <= 0) return false;
    cv::Mat patch = gray(r);
    cv::GaussianBlur(patch, patch, cv::Size(3,3), 0);
    cv::Mat bin; cv::threshold(patch, bin, 0, 255, cv::THRESH_BINARY_INV|cv::THRESH_OTSU);
    vector<vector<cv::Point>> cs; cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty()) return false;
    int k = -1; double best = 0; for (int i=0;i<(int)cs.size();++i){ double a=cv::contourArea(cs[i]); if (a>best){best=a;k=i;} }
    if (k<0) return false;
    vector<int> hullIdx; cv::convexHull(cs[k], hullIdx);
    vector<cv::Point> hull; hull.reserve(hullIdx.size()); for (int idx: hullIdx) hull.push_back(cs[k][idx]);
    vector<cv::Vec4i> defects; if (hullIdx.size() >= 3) cv::convexityDefects(cs[k], hullIdx, defects);
    int significantDef = 0; for (auto &d : defects) { if (d[3] / 256.0 > 10) significantDef++; }
    double area = cv::contourArea(cs[k]); double harea = std::max(1.0, cv::contourArea(hull));
    double ratio = area / harea; // open -> ratio lower (fingers spread)
    return (ratio < 0.68) || (significantDef >= 2);
}

void HandTrackerMP::update(const cv::Mat& frameBgr) {
	m_frameCount++;
	if (frameBgr.empty()) return;
	cv::Mat gray; cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);

	// Try to keep/refresh lock
	if (!m_hasLock) {
		cv::Mat motionMask;
        bool hasMotion = detectMotion(gray, motionMask);
		if (hasMotion && (m_frameCount % m_redetectInterval == 0)) {
			m_hasLock = acquireRoiFromMotion(gray, motionMask);
		}
	} else {
        if (!m_prevGray.empty()) trackRoiLK(m_prevGray, gray);
		// refresh features if needed
		if (m_prevPts.size() < 12) {
			vector<cv::Point2f> pts; cv::goodFeaturesToTrack(gray(m_roi), pts, 60, 0.01, 3);
			for (auto& p : pts) p += cv::Point2f((float)m_roi.x, (float)m_roi.y);
			m_prevPts = std::move(pts);
		}
	}

    // Gesture logic only when we have a stable ROI
    if (m_hasLock) {
        bool isOpen = analyzeGestureOpen(gray, m_roi);
        bool isClosed = analyzeGestureClosed(gray, m_roi);
        if (isOpen) {
            m_wasOpen = true; m_wasClosed = false; m_stableFrames = 0; m_triggered = false;
        } else if (isClosed) {
            // Trigger if we previously saw OPEN, or if closed is stable by itself
            m_stableFrames++;
            if ((m_wasOpen && m_stableFrames >= m_requiredStableFrames) || m_stableFrames >= 2) {
                m_wasClosed = true; m_triggered = true; m_wasOpen = false; m_stableFrames = 0;
            }
        } else {
            m_stableFrames = 0;
        }
    }

	m_prevGray = gray.clone();
}

bool HandTrackerMP::shouldTriggerCapture() {
	if (m_triggered) {
		m_triggered = false; // one-shot
		return true;
	}
	return false;
}


