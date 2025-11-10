// src/tools/calibrate_aruco.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

// ---------- Camera + board config ----------
static const int   CAM_ID = 0;
static const int   CAM_W = 1280;
static const int   CAM_H = 720;
static const int   CAM_FPS = 30;

static const int   DICT_ID = cv::aruco::DICT_6X6_250;
static const int   BOARD_MX = 3;          // cols
static const int   BOARD_MY = 2;          // rows
static const double MARKER_LEN_M = 0.050;      // exact marker side in meters
static const double GAP_M = 0.0125;     // exact gap in meters
static const int   FIRST_ID = 0;          // first marker id on the board

static const char* OUT_PATH = "calibration.json";

// ---------- MSMF opener (Windows) ----------
static bool probeFrame(cv::VideoCapture& cap) {
    cv::Mat f; try { if (!cap.read(f) || f.empty()) return false; }
    catch (...) { return false; }
    return true;
}
static bool openMSMF(cv::VideoCapture& cap, int index, int w, int h, int fps) {
#ifdef _WIN32
    _putenv_s("OPENCV_VIDEOIO_PRIORITY_MSMF", "1");
    _putenv_s("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0");
    _putenv_s("OPENCV_VIDEOIO_MSMF_ENABLE_GDI", "1");
#endif
    cap.release();
    if (!cap.open(index, cv::CAP_MSMF)) return false;
    cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    const int YUY2 = cv::VideoWriter::fourcc('Y', 'U', 'Y', '2');
    const int MJPG = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cap.set(cv::CAP_PROP_FOURCC, YUY2);
    if (!probeFrame(cap)) { cap.set(cv::CAP_PROP_FOURCC, MJPG); if (!probeFrame(cap)) if (!probeFrame(cap)) return false; }
    return true;
}

// ---------- Board geometry: id -> 4 object-space corners (TL,TR,BR,BL) ----------
static void markerObjectCorners(int id, std::vector<cv::Point3f>& obj) {
    int idx = id - FIRST_ID;
    if (idx < 0) { obj.clear(); return; }
    int col = idx % BOARD_MX;
    int row = idx / BOARD_MX;
    if (row < 0 || row >= BOARD_MY) { obj.clear(); return; }

    double cellX = col * (MARKER_LEN_M + GAP_M);
    double cellY = row * (MARKER_LEN_M + GAP_M);

    double x0 = cellX;
    double y0 = cellY;
    double L = MARKER_LEN_M;

    // TL, TR, BR, BL in the board XY plane (Z=0)
    obj.resize(4);
    obj[0] = cv::Point3f((float)x0, (float)(y0 + L), 0.0f); // TL
    obj[1] = cv::Point3f((float)(x0 + L), (float)(y0 + L), 0.0f); // TR
    obj[2] = cv::Point3f((float)(x0 + L), (float)(y0), 0.0f); // BR
    obj[3] = cv::Point3f((float)x0, (float)(y0), 0.0f); // BL
}

static void putInfo(cv::Mat& img, const std::string& s, int y, cv::Scalar col = { 255,255,255 }) {
    cv::putText(img, s, { 10,y }, cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv::LINE_AA);
}

int main() {
    // Camera
    cv::VideoCapture cap;
    if (!openMSMF(cap, CAM_ID, CAM_W, CAM_H, CAM_FPS)) {
        std::cerr << "Failed to open camera\n"; return -1;
    }
    std::cout << "Video backend: " << cap.getBackendName() << "\n";

    // ArUco detector configured for your API (Dictionary by value)
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(DICT_ID);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.cornerRefinementWinSize = 5;
    params.cornerRefinementMaxIterations = 20;
    params.cornerRefinementMinAccuracy = 0.01;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>=7)
    cv::aruco::ArucoDetector detector(dict, params);
#endif

    // Samples
    std::vector<std::vector<cv::Point3f>> objectPointsPerView;
    std::vector<std::vector<cv::Point2f>> imagePointsPerView;

    bool calibrated = false;
    cv::Mat K, D;
    double rms = 0.0;

    std::cout <<
        "Keys: S=sample  U=undo  C=calibrate  W=write calibration.json  Q=quit\n"
        "Collect 15–30 diverse views. Vary distance, tilt, and position.\n";

    cv::Mat frame, gray, vis;
    for (;;) {
        if (!cap.read(frame) || frame.empty()) continue;
        if (frame.type() == CV_8UC4) cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        vis = frame.clone();

        // Detect markers
        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>=7)
        detector.detectMarkers(gray, corners, ids, rejected);
#else
        cv::aruco::detectMarkers(gray, dict, corners, ids, cv::aruco::DetectorParameters(), rejected);
#endif
        if (!ids.empty()) cv::aruco::drawDetectedMarkers(vis, corners, ids);

        putInfo(vis, "S=sample  U=undo  C=calibrate  W=write  Q=quit", 30);
        putInfo(vis, "Views: " + std::to_string((int)imagePointsPerView.size()), 60);
        if (!ids.empty()) putInfo(vis, "Detected markers: " + std::to_string((int)ids.size()), 90);
        if (calibrated) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "RMS=%.3f px  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  k1=%.4f k2=%.4f",
                rms,
                K.empty() ? 0.0 : K.at<double>(0, 0), K.empty() ? 0.0 : K.at<double>(1, 1),
                K.empty() ? 0.0 : K.at<double>(0, 2), K.empty() ? 0.0 : K.at<double>(1, 2),
                D.empty() ? 0.0 : D.at<double>(0, 0), D.empty() ? 0.0 : D.at<double>(0, 1));
            putInfo(vis, buf, 120, { 0,255,0 });
        }
        cv::imshow("Calibrate ArUco", vis);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        if (key == 's' || key == 'S') {
            if (ids.size() >= 1) {
                std::vector<cv::Point3f> objAll;
                std::vector<cv::Point2f> imgAll;
                for (size_t i = 0; i < ids.size(); ++i) {
                    std::vector<cv::Point3f> obj;
                    markerObjectCorners(ids[i], obj);
                    if (obj.size() != 4 || corners[i].size() != 4) continue;
                    // Append 4 corners from this marker
                    objAll.insert(objAll.end(), obj.begin(), obj.end());
                    imgAll.insert(imgAll.end(), corners[i].begin(), corners[i].end());
                }
                if (imgAll.size() >= 12) { // at least 3 markers in view helps
                    objectPointsPerView.push_back(std::move(objAll));
                    imagePointsPerView.push_back(std::move(imgAll));
                    std::cout << "Saved view #" << imagePointsPerView.size() << "\n";
                }
                else {
                    std::cout << "Need more markers in view; got " << (imgAll.size() / 4) << "\n";
                }
            }
            else {
                std::cout << "No markers detected.\n";
            }
        }

        if (key == 'u' || key == 'U') {
            if (!imagePointsPerView.empty()) {
                imagePointsPerView.pop_back();
                objectPointsPerView.pop_back();
                std::cout << "Undid last sample. Views=" << imagePointsPerView.size() << "\n";
            }
        }

        if (key == 'c' || key == 'C') {
            if (imagePointsPerView.size() < 6) {
                std::cout << "Need more views (>=6). Current=" << imagePointsPerView.size() << "\n";
                continue;
            }
            K = cv::Mat::eye(3, 3, CV_64F);
            D = cv::Mat::zeros(1, 8, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;
            int flags = 0; // optionally: cv::CALIB_RATIONAL_MODEL
            cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-7);
            rms = cv::calibrateCamera(objectPointsPerView, imagePointsPerView,
                cv::Size(CAM_W, CAM_H), K, D, rvecs, tvecs, flags, tc);
            calibrated = true;
            std::cout << "Calibration RMS = " << rms << " px\nK=\n" << K << "\nD=\n" << D << "\n";
        }

        if (key == 'w' || key == 'W') {
            if (!calibrated) { std::cout << "Calibrate first (C).\n"; continue; }
            try {
                cv::FileStorage fs(OUT_PATH, cv::FileStorage::WRITE);
                fs << "image_width" << CAM_W;
                fs << "image_height" << CAM_H;
                fs << "camera_matrix" << K;
                // strip trailing zeros
                int used = D.cols;
                while (used > 4 && std::abs(D.at<double>(0, used - 1)) < 1e-12) used--;
                cv::Mat Dused = D.colRange(0, used).clone();
                fs << "dist_coeffs" << Dused;
                fs << "marker_length_m" << MARKER_LEN_M;
                fs << "dictionary" << DICT_ID;
                fs << "board_mx" << BOARD_MX;
                fs << "board_my" << BOARD_MY;
                fs << "gap_m" << GAP_M;
                fs << "rms_px" << rms;
                fs.release();
                std::cout << "Wrote " << OUT_PATH << "\n";
            }
            catch (const std::exception& e) {
                std::cerr << "Write failed: " << e.what() << "\n";
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
