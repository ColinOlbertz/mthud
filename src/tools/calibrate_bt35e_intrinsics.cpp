#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <mfapi.h>
  #include <mfidl.h>
#endif

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

namespace {

struct Options {
    int camIndex = 0;
    int width = 1280;
    int height = 720;
    int fps = 30;
    int boardMx = 3;
    int boardMy = 2;
    int firstId = 0;
    int dictId = cv::aruco::DICT_6X6_250;
    double markerLenM = 0.050;
    double gapM = 0.012;
    std::string cameraName;
    std::string outPath = "bt35e_intrinsics.json";
};

static std::optional<std::string> getEnvString(const char* key) {
#ifdef _WIN32
    size_t len = 0;
    char* buf = nullptr;
    if (_dupenv_s(&buf, &len, key) != 0 || !buf) return std::nullopt;
    std::string val(buf, len ? len - 1 : 0);
    std::free(buf);
    return val;
#else
    const char* v = std::getenv(key);
    if (!v) return std::nullopt;
    return std::string(v);
#endif
}

static bool argValue(int argc, char** argv, int& i, std::string& out) {
    if (i + 1 >= argc) return false;
    out = argv[++i];
    return true;
}

static void printUsage() {
    std::cout <<
        "BT-35E intrinsics calibration\n\n"
        "Usage:\n"
        "  calibrate_bt35e_intrinsics [options]\n\n"
        "Options:\n"
        "  --name BT-35E           Camera friendly name. Default: CAM_NAME or BT-35E when present.\n"
        "  --cam 1                 Camera index fallback.\n"
        "  --out file.json         Output JSON. Default: bt35e_intrinsics.json\n"
        "  --width 1280 --height 720 --fps 30\n"
        "  --mx 3 --my 2           ArUco grid cols/rows.\n"
        "  --marker 0.050          Marker side length in meters.\n"
        "  --gap 0.012             Gap between markers in meters.\n"
        "  --first-id 0\n\n"
        "Keys:\n"
        "  S sample current view, U undo, C calibrate, W write JSON, Q/Esc quit\n";
}

static Options parseOptions(int argc, char** argv) {
    Options o;
    if (auto e = getEnvString("CAM_INDEX"); e && !e->empty()) o.camIndex = std::atoi(e->c_str());
    if (auto e = getEnvString("CAM_NAME"); e && !e->empty()) o.cameraName = *e;
    if (auto e = getEnvString("CAM_RES"); e && !e->empty()) {
        auto x = e->find('x');
        if (x != std::string::npos) {
            o.width = std::atoi(e->substr(0, x).c_str());
            o.height = std::atoi(e->substr(x + 1).c_str());
        }
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i], v;
        if (a == "-h" || a == "--help") { printUsage(); std::exit(0); }
        if ((a == "--name") && argValue(argc, argv, i, v)) o.cameraName = v;
        else if ((a == "--cam" || a == "-c") && argValue(argc, argv, i, v)) o.camIndex = std::atoi(v.c_str());
        else if (a == "--out" && argValue(argc, argv, i, v)) o.outPath = v;
        else if (a == "--width" && argValue(argc, argv, i, v)) o.width = std::atoi(v.c_str());
        else if (a == "--height" && argValue(argc, argv, i, v)) o.height = std::atoi(v.c_str());
        else if (a == "--fps" && argValue(argc, argv, i, v)) o.fps = std::atoi(v.c_str());
        else if (a == "--mx" && argValue(argc, argv, i, v)) o.boardMx = std::atoi(v.c_str());
        else if (a == "--my" && argValue(argc, argv, i, v)) o.boardMy = std::atoi(v.c_str());
        else if (a == "--marker" && argValue(argc, argv, i, v)) o.markerLenM = std::atof(v.c_str());
        else if (a == "--gap" && argValue(argc, argv, i, v)) o.gapM = std::atof(v.c_str());
        else if (a == "--first-id" && argValue(argc, argv, i, v)) o.firstId = std::atoi(v.c_str());
    }
    return o;
}

#ifdef _WIN32
static std::wstring widenForWindows(const std::string& s) {
    if (s.empty()) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    UINT cp = CP_UTF8;
    if (len <= 0) {
        cp = CP_ACP;
        len = MultiByteToWideChar(cp, 0, s.c_str(), -1, nullptr, 0);
    }
    if (len <= 0) return {};
    std::wstring out(size_t(len - 1), L'\0');
    MultiByteToWideChar(cp, 0, s.c_str(), -1, out.data(), len);
    return out;
}

static std::wstring lowerWide(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return wchar_t(std::towlower(c)); });
    return s;
}

static std::optional<int> mfCameraIndexByName(const std::string& nameNeedle) {
    std::wstring needle = lowerWide(widenForWindows(nameNeedle));
    if (needle.empty()) return std::nullopt;

    HRESULT coHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool didCoInit = SUCCEEDED(coHr);
    if (FAILED(coHr) && coHr != RPC_E_CHANGED_MODE) return std::nullopt;
    if (FAILED(MFStartup(MF_VERSION))) {
        if (didCoInit) CoUninitialize();
        return std::nullopt;
    }

    IMFAttributes* attrs = nullptr;
    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    std::optional<int> found;

    if (SUCCEEDED(MFCreateAttributes(&attrs, 1)) &&
        SUCCEEDED(attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                 MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)) &&
        SUCCEEDED(MFEnumDeviceSources(attrs, &devices, &count))) {
        for (UINT32 i = 0; i < count; ++i) {
            WCHAR* friendlyName = nullptr;
            UINT32 friendlyNameLen = 0;
            if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                                         &friendlyName, &friendlyNameLen))) {
                std::wstring candidate = lowerWide(std::wstring(friendlyName, friendlyNameLen));
                CoTaskMemFree(friendlyName);
                if (candidate.find(needle) != std::wstring::npos) {
                    found = static_cast<int>(i);
                    break;
                }
            }
        }
    }

    if (devices) {
        for (UINT32 i = 0; i < count; ++i) if (devices[i]) devices[i]->Release();
        CoTaskMemFree(devices);
    }
    if (attrs) attrs->Release();
    MFShutdown();
    if (didCoInit) CoUninitialize();
    return found;
}
#endif

static bool probe(cv::VideoCapture& cap) {
    cv::Mat frame;
    for (int i = 0; i < 8; ++i) {
        if (cap.grab() && cap.retrieve(frame) && !frame.empty()) return true;
    }
    return false;
}

static bool openCamera(cv::VideoCapture& cap, const Options& options, int& openedIndex) {
#ifdef _WIN32
    _putenv_s("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0");
    _putenv_s("OPENCV_VIDEOIO_PRIORITY_MSMF", "1");
    std::string desiredName = options.cameraName.empty() ? "BT-35E" : options.cameraName;
    if (auto idx = mfCameraIndexByName(desiredName)) {
        openedIndex = *idx;
        std::cerr << "Resolved camera \"" << desiredName << "\" to Media Foundation index " << openedIndex << "\n";
    } else {
        openedIndex = options.camIndex;
        std::cerr << "Using camera index " << openedIndex << "\n";
    }
    const int backends[] = { cv::CAP_MSMF, cv::CAP_DSHOW, cv::CAP_ANY };
    const int fourccs[] = {
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'),
        0
    };
#else
    openedIndex = options.camIndex;
    const int backends[] = { cv::CAP_ANY };
    const int fourccs[] = { 0, cv::VideoWriter::fourcc('M', 'J', 'P', 'G') };
#endif

    for (int backend : backends) {
        for (int fourcc : fourccs) {
            cap.release();
            if (!cap.open(openedIndex, backend)) continue;
            if (fourcc != 0) cap.set(cv::CAP_PROP_FOURCC, fourcc);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
            cap.set(cv::CAP_PROP_FPS, options.fps);
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
#ifdef _WIN32
            cap.set(cv::CAP_PROP_HW_ACCELERATION, (double)cv::VIDEO_ACCELERATION_NONE);
#endif
            if (probe(cap)) {
                std::cerr << "Opened camera index " << openedIndex << " backend=" << cap.getBackendName()
                          << " size=" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                          << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
                return true;
            }
        }
    }
    cap.release();
    return false;
}

static void markerObjectCorners(int id, const Options& o, std::vector<cv::Point3f>& obj) {
    int idx = id - o.firstId;
    if (idx < 0) { obj.clear(); return; }
    int col = idx % o.boardMx;
    int row = idx / o.boardMx;
    if (row < 0 || row >= o.boardMy) { obj.clear(); return; }

    const double x0 = col * (o.markerLenM + o.gapM);
    const double y0 = row * (o.markerLenM + o.gapM);
    const double L = o.markerLenM;
    obj = {
        cv::Point3f(float(x0),     float(y0 + L), 0.0f),
        cv::Point3f(float(x0 + L), float(y0 + L), 0.0f),
        cv::Point3f(float(x0 + L), float(y0),     0.0f),
        cv::Point3f(float(x0),     float(y0),     0.0f)
    };
}

static void putInfo(cv::Mat& img, const std::string& text, int y, cv::Scalar color = {255, 255, 255}) {
    cv::putText(img, text, {10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

static bool collectView(const std::vector<int>& ids,
                        const std::vector<std::vector<cv::Point2f>>& corners,
                        const Options& options,
                        std::vector<cv::Point3f>& objAll,
                        std::vector<cv::Point2f>& imgAll) {
    objAll.clear();
    imgAll.clear();
    for (size_t i = 0; i < ids.size(); ++i) {
        std::vector<cv::Point3f> obj;
        markerObjectCorners(ids[i], options, obj);
        if (obj.size() != 4 || corners[i].size() != 4) continue;
        objAll.insert(objAll.end(), obj.begin(), obj.end());
        imgAll.insert(imgAll.end(), corners[i].begin(), corners[i].end());
    }
    return imgAll.size() >= 12;
}

static void writeJsonArray(std::ostream& os, const cv::Mat& m) {
    cv::Mat flat = m.reshape(1, 1);
    os << "[";
    for (int i = 0; i < flat.cols; ++i) {
        if (i) os << ", ";
        os << std::setprecision(12) << flat.at<double>(0, i);
    }
    os << "]";
}

static bool writeCalibration(const std::string& path, const Options& options, int openedIndex,
                             const cv::Size& imageSize, const cv::Mat& K, const cv::Mat& D,
                             double rms, size_t views) {
    std::ofstream out(path);
    if (!out) return false;

    cv::Mat K64, D64;
    K.convertTo(K64, CV_64F);
    D.convertTo(D64, CV_64F);
    cv::Mat Drow = D64.reshape(1, 1);
    int used = Drow.cols;
    while (used > 5 && std::abs(Drow.at<double>(0, used - 1)) < 1e-12) --used;
    Drow = Drow.colRange(0, used).clone();

    out << "{\n";
    out << "  \"image_width\": " << imageSize.width << ",\n";
    out << "  \"image_height\": " << imageSize.height << ",\n";
    out << "  \"camera_index\": " << openedIndex << ",\n";
    out << "  \"camera_name\": \"" << (options.cameraName.empty() ? "BT-35E" : options.cameraName) << "\",\n";
    out << "  \"camera_matrix\": ";
    writeJsonArray(out, K64);
    out << ",\n  \"distortion_coefficients\": ";
    writeJsonArray(out, Drow);
    out << ",\n  \"rms_px\": " << std::setprecision(8) << rms << ",\n";
    out << "  \"sample_count\": " << views << ",\n";
    out << "  \"board_mx\": " << options.boardMx << ",\n";
    out << "  \"board_my\": " << options.boardMy << ",\n";
    out << "  \"marker_length_m\": " << std::setprecision(8) << options.markerLenM << ",\n";
    out << "  \"gap_m\": " << std::setprecision(8) << options.gapM << ",\n";
    out << "  \"dictionary\": " << options.dictId << "\n";
    out << "}\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options options = parseOptions(argc, argv);

    cv::VideoCapture cap;
    int openedIndex = options.camIndex;
    if (!openCamera(cap, options, openedIndex)) {
        std::cerr << "Failed to open camera. Try --cam 0/1 or --name BT-35E.\n";
        return 1;
    }

    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(options.dictId);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.cornerRefinementWinSize = 5;
    params.cornerRefinementMaxIterations = 40;
    params.cornerRefinementMinAccuracy = 0.01;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::aruco::ArucoDetector detector(dict, params);
#endif

    std::vector<std::vector<cv::Point3f>> objectPointsPerView;
    std::vector<std::vector<cv::Point2f>> imagePointsPerView;
    cv::Mat K, D;
    double rms = 0.0;
    bool calibrated = false;

    std::cout << "Collect 15-30 views. Fill corners/edges of the image, vary distance and tilt.\n"
              << "Keys: S=sample  U=undo  C=calibrate  W=write " << options.outPath
              << "  Q/Esc=quit\n";

    cv::Mat frame, gray, vis;
    cv::Size imageSize(options.width, options.height);
    for (;;) {
        if (!cap.read(frame) || frame.empty()) continue;
        if (frame.channels() == 4) cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        imageSize = frame.size();
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        vis = frame.clone();

        std::vector<std::vector<cv::Point2f>> corners, rejected;
        std::vector<int> ids;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
        detector.detectMarkers(gray, corners, ids, rejected);
#else
        cv::Ptr<cv::aruco::DetectorParameters> paramsPtr(new cv::aruco::DetectorParameters(params));
        cv::aruco::detectMarkers(gray, dict, corners, ids, paramsPtr, rejected);
#endif
        if (!ids.empty()) cv::aruco::drawDetectedMarkers(vis, corners, ids);

        std::vector<cv::Point3f> objCandidate;
        std::vector<cv::Point2f> imgCandidate;
        bool goodView = collectView(ids, corners, options, objCandidate, imgCandidate);

        putInfo(vis, "S sample  U undo  C calibrate  W write  Q quit", 28);
        putInfo(vis, "Views: " + std::to_string(imagePointsPerView.size()) +
                     "  markers: " + std::to_string(ids.size()) +
                     (goodView ? "  sample-ready" : "  need >=3 visible board markers"),
                58, goodView ? cv::Scalar(0,255,0) : cv::Scalar(0,180,255));
        if (calibrated) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3)
               << "RMS " << rms << " px  fx " << K.at<double>(0,0)
               << " fy " << K.at<double>(1,1)
               << " cx " << K.at<double>(0,2)
               << " cy " << K.at<double>(1,2);
            putInfo(vis, ss.str(), 88, {0,255,0});
        }

        cv::imshow("BT-35E Intrinsics Calibration", vis);
        int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == 's' || key == 'S') {
            if (!goodView) {
                std::cout << "Not enough markers from the configured board are visible.\n";
                continue;
            }
            objectPointsPerView.push_back(std::move(objCandidate));
            imagePointsPerView.push_back(std::move(imgCandidate));
            calibrated = false;
            std::cout << "Saved view #" << imagePointsPerView.size() << "\n";
        } else if (key == 'u' || key == 'U') {
            if (!imagePointsPerView.empty()) {
                imagePointsPerView.pop_back();
                objectPointsPerView.pop_back();
                calibrated = false;
                std::cout << "Undid last view. Views=" << imagePointsPerView.size() << "\n";
            }
        } else if (key == 'c' || key == 'C') {
            if (imagePointsPerView.size() < 8) {
                std::cout << "Need at least 8 views; 15-30 is better. Current=" << imagePointsPerView.size() << "\n";
                continue;
            }
            K = cv::Mat::eye(3, 3, CV_64F);
            D = cv::Mat::zeros(1, 8, CV_64F);
            std::vector<cv::Mat> rvecs, tvecs;
            int flags = cv::CALIB_RATIONAL_MODEL;
            cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-8);
            rms = cv::calibrateCamera(objectPointsPerView, imagePointsPerView, imageSize,
                                      K, D, rvecs, tvecs, flags, tc);
            calibrated = true;
            std::cout << "Calibration RMS=" << rms << " px\nK=\n" << K << "\nD=\n" << D << "\n";
        } else if (key == 'w' || key == 'W') {
            if (!calibrated) {
                std::cout << "Calibrate first with C.\n";
                continue;
            }
            if (!writeCalibration(options.outPath, options, openedIndex, imageSize, K, D,
                                  rms, imagePointsPerView.size())) {
                std::cerr << "Failed to write " << options.outPath << "\n";
                continue;
            }
            std::cout << "Wrote " << options.outPath << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
