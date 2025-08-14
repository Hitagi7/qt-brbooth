#ifndef MEDIAPIPE_LIKE_HAND_TRACKER_H
#define MEDIAPIPE_LIKE_HAND_TRACKER_H

#include <opencv2/opencv.hpp>
#include <QElapsedTimer>

// Lightweight MediaPipe-like hand tracker (palm/ROI tracking + gesture state)
// - Finds a moving hand ROI using motion gating
// - Tracks the ROI over time using feature points and EMA smoothing
// - Computes open/closed via hull area ratio within ROI
// - Triggers only on open -> closed transition while locked

class HandTrackerMP {
public:
	HandTrackerMP();
	~HandTrackerMP() = default;

	void reset();
	void initialize(int frameWidth, int frameHeight);
	void update(const cv::Mat& frameBgr);

	bool hasLock() const { return m_hasLock; }
	cv::Rect getRoi() const { return m_roi; }
	bool shouldTriggerCapture();

private:
	// Internal helpers
	bool detectMotion(const cv::Mat& gray, cv::Mat& motionMask);
	bool acquireRoiFromMotion(const cv::Mat& gray, const cv::Mat& motionMask);
	void trackRoiLK(const cv::Mat& grayPrev, const cv::Mat& grayCurr);
	bool analyzeGestureClosed(const cv::Mat& gray, const cv::Rect& roi) const;
	bool analyzeGestureOpen(const cv::Mat& gray, const cv::Rect& roi) const;

	// Gesture state
	bool m_wasOpen;
	bool m_wasClosed;
	int m_stableFrames;
	bool m_triggered;

	// Tracking state
	bool m_hasLock;
	cv::Rect m_roi;
	cv::Mat m_prevGray;
	std::vector<cv::Point2f> m_prevPts;
	QElapsedTimer m_timer;

    // Background model (float for accumulateWeighted)
    cv::Mat m_bgFloat;
	bool m_bgInit;

	// Params
	int m_frameWidth;
	int m_frameHeight;
	int m_frameCount;
	// thresholds
	int m_motionThreshold;       // absdiff threshold
	int m_minMotionArea;         // min moving area to consider
	int m_redetectInterval;      // frames between acquisitions when unlocked
	int m_minRoiSize;            // minimum hand roi size
	int m_maxRoiSize;            // maximum hand roi size
	int m_requiredStableFrames;  // gesture stability frames
};

#endif // MEDIAPIPE_LIKE_HAND_TRACKER_H


