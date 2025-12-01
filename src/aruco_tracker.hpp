#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>

struct BoardPose {
    bool   valid = false;
    cv::Mat rvec;   // 3x1, CV_64F
    cv::Mat tvec;   // 3x1, CV_64F  (anchored at the chosen board corner)
    double reproj_err = 0.0;
    int    inliers = 0;
};

class ArucoTracking {
public:
    ArucoTracking();

    void setCameraIntrinsics(const cv::Mat& K, const cv::Mat& D);

    void setGridBoard(int markersX, int markersY, float markerLength, float markerGap,
        int dictId = cv::aruco::DICT_6X6_250);

    // corner_idx: 0=TL, 1=TR, 2=BR, 3=BL
    void setAnchorMarkerCorner(int marker_id, int corner_idx);

    void setLockId(int marker_id);
    void setTemporalSmoothing(double alpha, double gate_unused = 0.0);

    void update(const cv::Mat& frame_bgr_or_gray);

    BoardPose latest() const { return last_; }

    const std::vector<int>& ids() const { return ids_; }
    const std::vector<std::vector<cv::Point2f>>& corners() const { return corners_; }
    const std::vector<std::vector<cv::Point2f>>& rejected() const { return rejected_; }

private:
    // Camera
    cv::Mat K_, D_; // CV_64F

    // Dictionary (value type for old API)
    cv::aruco::Dictionary dict_;

    // Board
    cv::Ptr<cv::aruco::GridBoard> board_;
    int markersX_ = 3, markersY_ = 2;
    float markerLength_ = 0.04f;
    float markerGap_ = 0.01f;

    // Anchor
    int anchor_id_ = 0;
    int anchor_corner_idx_ = 0; // TL

    // Detection
    cv::Ptr<cv::aruco::DetectorParameters> params_;
    std::vector<int> ids_;
    std::vector<std::vector<cv::Point2f>> corners_, rejected_;

    // Locking and smoothing
    int lock_id_ = -1;
    bool have_prev_ = false;
    double ema_alpha_ = 0.;
    cv::Mat prev_rvec_, prev_tvec_;

    BoardPose last_;

    // Helpers
    static cv::Mat to64(const cv::Mat& m);
    static cv::Mat rvecSlerp(const cv::Mat& r0, const cv::Mat& r1, double a);

    double reprojErrBoard(const cv::Mat& rvec_board, const cv::Mat& tvec_board) const;

    // API adapters that only use old getters
    static const std::vector<int>& boardIds(const cv::Ptr<cv::aruco::GridBoard>& b);
    static const std::vector<std::vector<cv::Point3f>>& boardObj(const cv::Ptr<cv::aruco::GridBoard>& b);
};
