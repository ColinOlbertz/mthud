#include "aruco_tracker.hpp"
#include <numeric>
#include <unordered_map>

// ---------- utils ----------
cv::Mat ArucoTracking::to64(const cv::Mat& m) { cv::Mat o; m.convertTo(o, CV_64F); return o; }

static inline cv::Mat quatFromR(const cv::Mat& R) {
    cv::Mat q(4, 1, CV_64F);
    double tr = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    if (tr > 0.0) {
        double S = std::sqrt(tr + 1.0) * 2.0;
        q.at<double>(0) = 0.25 * S;
        q.at<double>(1) = (R.at<double>(2, 1) - R.at<double>(1, 2)) / S;
        q.at<double>(2) = (R.at<double>(0, 2) - R.at<double>(2, 0)) / S;
        q.at<double>(3) = (R.at<double>(1, 0) - R.at<double>(0, 1)) / S;
    }
    else if (R.at<double>(0, 0) > R.at<double>(1, 1) && R.at<double>(0, 0) > R.at<double>(2, 2)) {
        double S = std::sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2)) * 2.0;
        q.at<double>(0) = (R.at<double>(2, 1) - R.at<double>(1, 2)) / S;
        q.at<double>(1) = 0.25 * S;
        q.at<double>(2) = (R.at<double>(0, 1) + R.at<double>(1, 0)) / S;
        q.at<double>(3) = (R.at<double>(0, 2) + R.at<double>(2, 0)) / S;
    }
    else if (R.at<double>(1, 1) > R.at<double>(2, 2)) {
        double S = std::sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) - R.at<double>(2, 2)) * 2.0;
        q.at<double>(0) = (R.at<double>(0, 2) - R.at<double>(2, 0)) / S;
        q.at<double>(1) = (R.at<double>(0, 1) + R.at<double>(1, 0)) / S;
        q.at<double>(2) = 0.25 * S;
        q.at<double>(3) = (R.at<double>(1, 2) + R.at<double>(2, 1)) / S;
    }
    else {
        double S = std::sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) - R.at<double>(1, 1)) * 2.0;
        q.at<double>(0) = (R.at<double>(1, 0) - R.at<double>(0, 1)) / S;
        q.at<double>(1) = (R.at<double>(0, 2) + R.at<double>(2, 0)) / S;
        q.at<double>(2) = (R.at<double>(1, 2) + R.at<double>(2, 1)) / S;
        q.at<double>(3) = 0.25 * S;
    }
    q /= std::sqrt(q.dot(q));
    return q;
}

static inline cv::Mat quatSlerp(const cv::Mat& qa, const cv::Mat& qb, double t) {
    double dot = qa.dot(qb);
    cv::Mat q1 = qb.clone();
    if (dot < 0.0) { q1 = -q1; dot = -dot; }
    const double TH = 0.9995;
    if (dot > TH) { cv::Mat r = qa * (1.0 - t) + q1 * t; r /= std::sqrt(r.dot(r)); return r; }
    double th0 = std::acos(dot), sth0 = std::sin(th0);
    double th = th0 * t, sth = std::sin(th);
    double s0 = std::cos(th) - dot * sth / sth0;
    double s1 = sth / sth0;
    return qa * s0 + q1 * s1;
}

cv::Mat ArucoTracking::rvecSlerp(const cv::Mat& r0, const cv::Mat& r1, double a) {
    cv::Mat R0, R1; cv::Rodrigues(r0, R0); cv::Rodrigues(r1, R1);
    cv::Mat q0 = quatFromR(R0), q1 = quatFromR(R1);
    cv::Mat qs = quatSlerp(q0, q1, a);
    double w = qs.at<double>(0), x = qs.at<double>(1), y = qs.at<double>(2), z = qs.at<double>(3);
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y));
    cv::Mat r; cv::Rodrigues(R, r); return r;
}

// ---------- API adapters: always use old getters ----------
const std::vector<int>& ArucoTracking::boardIds(const cv::Ptr<cv::aruco::GridBoard>& b) {
    return b->getIds();
}
const std::vector<std::vector<cv::Point3f>>& ArucoTracking::boardObj(const cv::Ptr<cv::aruco::GridBoard>& b) {
    return b->getObjPoints();
}

// ---------- class ----------
ArucoTracking::ArucoTracking() {
    // Old API: construct params with new
    params_ = cv::Ptr<cv::aruco::DetectorParameters>(new cv::aruco::DetectorParameters());
    // If available, refine corners
#ifdef cv_aruco_CornerRefineMethod
    params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif
}

void ArucoTracking::setCameraIntrinsics(const cv::Mat& K, const cv::Mat& D) {
    K_ = to64(K); D_ = to64(D);
}

void ArucoTracking::setGridBoard(int mx, int my, float markerLen, float gap, int dictId) {
    markersX_ = mx; markersY_ = my; markerLength_ = markerLen; markerGap_ = gap;
    // Old API: getPredefinedDictionary returns Dictionary by value
    dict_ = cv::aruco::getPredefinedDictionary(dictId);
    // Old API: no GridBoard::create()
    board_ = cv::Ptr<cv::aruco::GridBoard>(new cv::aruco::GridBoard(cv::Size(mx, my), markerLen, gap, dict_));
}

void ArucoTracking::setAnchorMarkerCorner(int marker_id, int corner_idx) {
    anchor_id_ = marker_id;
    anchor_corner_idx_ = std::max(0, std::min(3, corner_idx));
}

void ArucoTracking::setLockId(int marker_id) { lock_id_ = marker_id; }

void ArucoTracking::setTemporalSmoothing(double a, double) {
    ema_alpha_ = std::max(0.0, std::min(1.0, a));
}

double ArucoTracking::reprojErrBoard(const cv::Mat& rvec_board, const cv::Mat& tvec_board) const {
    if (ids_.empty() || !board_) return 0.0;
    std::unordered_map<int, int> id2idx;
    const auto& bIds = boardIds(board_);
    for (size_t i = 0; i < bIds.size(); ++i) id2idx[bIds[i]] = (int)i;

    double sum = 0.0; size_t cnt = 0;
    std::vector<cv::Point2f> proj;
    const auto& bObj = boardObj(board_);
    for (size_t d = 0; d < ids_.size(); ++d) {
        auto it = id2idx.find(ids_[d]);
        if (it == id2idx.end()) continue;
        const auto& obj = bObj[it->second];
        proj.clear();
        cv::projectPoints(obj, rvec_board, tvec_board, K_, D_, proj);
        const auto& img = corners_[d];
        for (int k = 0; k < 4; ++k) { cv::Point2f e = proj[k] - img[k]; sum += std::sqrt(e.dot(e)); ++cnt; }
    }
    return (cnt > 0) ? float(sum / cnt) : 0.0;
}

void ArucoTracking::update(const cv::Mat& frame_bgr_or_gray) {
    last_ = BoardPose{};

    cv::Mat gray;
    if (frame_bgr_or_gray.channels() == 1) gray = frame_bgr_or_gray;
    else cv::cvtColor(frame_bgr_or_gray, gray, cv::COLOR_BGR2GRAY);

    ids_.clear(); corners_.clear(); rejected_.clear();

    cv::Ptr<cv::aruco::Dictionary> dictPtr = cv::makePtr<cv::aruco::Dictionary>();
    *dictPtr = dict_;  // copy fields

    cv::aruco::detectMarkers(gray, dictPtr, corners_, ids_, params_, rejected_);

    if (ids_.empty() || !board_) return;

    if (lock_id_ >= 0) {
        if (std::find(ids_.begin(), ids_.end(), lock_id_) == ids_.end()) return;
    }

    cv::Mat rvec_board, tvec_board;
    int inliers = cv::aruco::estimatePoseBoard(corners_, ids_, board_, K_, D_, rvec_board, tvec_board);
    if (inliers <= 0) return;

    // Locate anchor marker on the board
    int anchor_board_idx = -1;
    const auto& bIds = boardIds(board_);
    for (size_t i = 0; i < bIds.size(); ++i) if (bIds[i] == anchor_id_) { anchor_board_idx = (int)i; break; }
    if (anchor_board_idx < 0) return;

    const auto& bObj = boardObj(board_);
    const cv::Point3f anchor_board = bObj[anchor_board_idx][anchor_corner_idx_];

    cv::Mat Rb; cv::Rodrigues(rvec_board, Rb);
    cv::Mat t_anchor = tvec_board + Rb * (cv::Mat_<double>(3, 1) << anchor_board.x, anchor_board.y, anchor_board.z);

    if (!have_prev_) { prev_rvec_ = rvec_board.clone(); prev_tvec_ = t_anchor.clone(); have_prev_ = true; }
    else {
        prev_rvec_ = rvecSlerp(prev_rvec_, rvec_board, ema_alpha_);
        prev_tvec_ = prev_tvec_ * (1.0 - ema_alpha_) + t_anchor * ema_alpha_;
    }

    last_.valid = true;
    last_.rvec = prev_rvec_.clone();
    last_.tvec = prev_tvec_.clone();
    last_.reproj_err = reprojErrBoard(rvec_board, tvec_board);
    last_.inliers = inliers;
}
