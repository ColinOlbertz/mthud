#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include "aruco_tracker.hpp"
#include "hud_renderer.hpp"
#include "sensor.hpp"

using nlohmann::json;

static const float PI = 3.14159265358979323846f;

//--------------------- linux vs windows cam setup --------------------------------
#ifdef __linux__
  #include <fcntl.h>
  #include <sys/ioctl.h>
  #include <linux/videodev2.h>
  #include <unistd.h>
#endif

static int preferredBackend() {
#if defined(_WIN32)
    return cv::CAP_DSHOW;
#elif defined(__linux__)
    return cv::CAP_V4L2;
#else
    return cv::CAP_ANY;
#endif
}

#ifdef __linux__
static std::string v4l2NameFor(int idx) {
    char dev[64]; std::snprintf(dev, sizeof(dev), "/dev/video%d", idx);
    int fd = ::open(dev, O_RDONLY | O_NONBLOCK);
    if (fd < 0) return {};
    v4l2_capability cap{};
    if (::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        ::close(fd);
        return std::string(reinterpret_cast<char*>(cap.card));
    }
    ::close(fd);
    return {};
}
#endif

static std::vector<int> scanCameras(int maxIdx = 12) {
    std::vector<int> ok;
    for (int i = 0; i <= maxIdx; ++i) {
        cv::VideoCapture t(i, preferredBackend());
        if (t.isOpened()) ok.push_back(i);
    }
    return ok;
}

static bool openCapture(cv::VideoCapture& cap, int index,
                        int w, int h, double fps) {
    cap.release();
    if (!cap.open(index, preferredBackend())) return false;
    if (w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  w);
    if (h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    if (fps > 0) cap.set(cv::CAP_PROP_FPS, fps);
    cv::Mat probe;
    if (!cap.read(probe) || probe.empty()) return false;
    return true;
}

static int parseCamIndex(int argc, char** argv, int fallback = 2) {
    if (const char* e = std::getenv("CAM_INDEX"); e && *e) return std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-c" || a == "--cam") && i + 1 < argc) return std::atoi(argv[i + 1]);
        // allow first bare integer
        char* end = nullptr; long v = std::strtol(argv[i], &end, 10);
        if (end && *end == '\0') return int(v);
    }
    return fallback;
}

// -------- file-scope constants (ONE definition) --------
namespace {
    constexpr int HUD_TEX_W = 1024;
    constexpr int HUD_TEX_H = 768;
    constexpr int CANVAS_W = 1920;   // design canvas (4:3)
    constexpr int CANVAS_H = 1440;
}

// ---------- helpers ----------
static bool loadCalibrationJSON(const std::string& path, cv::Mat& K, cv::Mat& D, cv::Size& size) {
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) return false;
        fs["K"] >> K; fs["D"] >> D;
        int w = 0, h = 0; fs["image_width"] >> w; fs["image_height"] >> h;
        if (w > 0 && h > 0) size = cv::Size(w, h);
        if (K.empty() || D.empty()) return false;
        K.convertTo(K, CV_64F); D.convertTo(D, CV_64F);
        return true;
    }
    catch (...) { return false; }
}
static void approxFOVIntrinsics(cv::Size sz, double hfov_deg, cv::Mat& K, cv::Mat& D) {
    double hfov = hfov_deg * CV_PI / 180.0;
    double fx = sz.width / (2.0 * std::tan(hfov * 0.5));
    double fy = fx, cx = sz.width * 0.5, cy = sz.height * 0.5;
    K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    D = cv::Mat::zeros(1, 5, CV_64F);
}
static void buildBillboardPts4x3(const cv::Point3f& origin_TR, float width_m, std::vector<cv::Point3f>& out) {
    const float w = width_m, h = w * 0.75f, depth = 0.0f;
    out.resize(4);
    out[0] = { origin_TR.x,     origin_TR.y,     origin_TR.z + depth }; // TR
    out[1] = { origin_TR.x - w, origin_TR.y,     origin_TR.z + depth }; // TL
    out[2] = { origin_TR.x - w, origin_TR.y - h, origin_TR.z + depth }; // BL
    out[3] = { origin_TR.x,     origin_TR.y - h, origin_TR.z + depth }; // BR
}

// ---------- controls + persistence ----------
static const char* WIN_CTRL = "HUD Controls"; static const char* PERSIST = "hud_layout_controls.json";
static int TB_pitch_scale_x1000 = 20;   // NDC/deg *1000 → converted to px/deg
static int TB_pitch_trim_x1000 = 0;    // NDC trim *1000 → converted to px
static int TB_bank_trim_deg = 0;    // deg
static int TB_bank_top_px = 60;   // distance from top to arc top, canvas px
static int TB_bank_rad_px = 220;  // arc radius in canvas px
static int TB_ladder_half_px = 360;  // ladder half-width in canvas px
static int TB_ladder_xoff_px = 0;    // ladder lateral offset in canvas px
static int TB_text_scale_pct = 120;  // text size percent (50..400)
static int TB_text_flip_x = 1;    // 0/1
static int TB_text_flip_y = 0;    // 0/1

static void save_controls() {
    json j{
        {"pitch_scale_x1000",TB_pitch_scale_x1000},
        {"pitch_trim_x1000", TB_pitch_trim_x1000},
        {"bank_trim_deg",    TB_bank_trim_deg},
        {"bank_top_px",      TB_bank_top_px},
        {"bank_radius_px",   TB_bank_rad_px},
        {"ladder_half_px",   TB_ladder_half_px},
        {"ladder_xoff_px",   TB_ladder_xoff_px},
        {"text_scale_pct",   TB_text_scale_pct},
        {"text_flip_x",      TB_text_flip_x},
        {"text_flip_y",      TB_text_flip_y}
    };
    std::ofstream(PERSIST) << j.dump(2);
}
static void load_controls() {
    std::ifstream f(PERSIST); if (!f) return;
    json j; f >> j;
    auto get = [&](const char* k, int& v) { if (j.contains(k)) v = j[k].get<int>(); };
    get("pitch_scale_x1000", TB_pitch_scale_x1000);
    get("pitch_trim_x1000", TB_pitch_trim_x1000);
    get("bank_trim_deg", TB_bank_trim_deg);
    get("bank_top_px", TB_bank_top_px);
    get("bank_radius_px", TB_bank_rad_px);
    get("ladder_half_px", TB_ladder_half_px);
    get("ladder_xoff_px", TB_ladder_xoff_px);
    get("text_scale_pct", TB_text_scale_pct);
    get("text_flip_x", TB_text_flip_x);
    get("text_flip_y", TB_text_flip_y);
    if (j.contains("pitch_ndc_x1000") && !j.contains("pitch_trim_x1000"))
        TB_pitch_trim_x1000 = j["pitch_ndc_x1000"].get<int>();
}

int main(int argc, char** argv) {
    // --- Camera
    /*int camIndex = 0; if (argc > 1) camIndex = std::atoi(argv[1]);
    cv::VideoCapture cap(camIndex, cv::CAP_ANY); if (!cap.isOpened()) { std::cerr << "Camera open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);*/
    int camIndex = parseCamIndex(argc, argv, /*fallback*/0);

    cv::VideoCapture cap;
    if (!openCapture(cap, camIndex, 1280, 720, 30)) {
        auto avail = scanCameras(12);
        if (avail.empty()) {
            std::cerr << "No camera could be opened.\n";
            return 1;
        }
        camIndex = avail.front();
        if (!openCapture(cap, camIndex, 1280, 720, 30)) {
            std::cerr << "Failed to open first available camera.\n";
            return 1;
        }
    }

    #ifdef __linux__
    std::cerr << "Active camera index: " << camIndex
            << " name: " << v4l2NameFor(camIndex) << "\n";
    #else
    std::cerr << "Active camera index: " << camIndex << "\n";
    #endif

    // Optional MJPG request (some UVC cams ignore it). Keep if it helped on Windows.
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));


    cv::Mat frame; if (!cap.read(frame) || frame.empty()) { std::cerr << "No frames\n"; return 1; }
    cv::Size imgSize = frame.size();

    // --- Calibration
    cv::Mat K, D;
    if (!loadCalibrationJSON("calibration.json", K, D, imgSize)) {
        approxFOVIntrinsics(imgSize, 70.0, K, D);
        std::cout << "No calibration.json. Using FOV approximation.\n";
    }

    // --- ArUco tracker
    ArucoTracking tracker;
    const int MX = 3, MY = 2;
    const float MARKER_LEN = 0.050f, GAP = 0.012f;
    tracker.setCameraIntrinsics(K, D);
    tracker.setGridBoard(MX, MY, MARKER_LEN, GAP);
    tracker.setAnchorMarkerCorner(0, 0);
    tracker.setTemporalSmoothing(0.25);

    cv::namedWindow("camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("camera", imgSize.width, imgSize.height);

    // --- Sensor
    auto sensor = makeSensor(); sensor->start();

    // --- GLFW init
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1024, 768, "HUD", nullptr, nullptr);
    if (!win) { std::cerr << "glfwCreateWindow failed\n"; glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "glad load failed\n"; return 1; }
    glfwSwapInterval(1);

    // --- Offscreen HUD render target (FBO + texture)
    GLuint hudFBO = 0, hudTex = 0;
    glGenTextures(1, &hudTex);
    glBindTexture(GL_TEXTURE_2D, hudTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, HUD_TEX_W, HUD_TEX_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenFramebuffers(1, &hudFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hudFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hudTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) std::cerr << "HUD FBO incomplete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // --- Renderer
    HudRenderer hud; if (!hud.init()) { std::cerr << "hud.init failed\n"; return 1; }

    // --- Controls
    cv::namedWindow(WIN_CTRL, cv::WINDOW_NORMAL); cv::resizeWindow(WIN_CTRL, 520, 360);
    load_controls();
    auto tb = [&](const char* name, int* var, int maxv) { cv::createTrackbar(name, WIN_CTRL, var, maxv, nullptr); };
    tb("Pitch scale NDC/deg x1000", &TB_pitch_scale_x1000, 200);
    tb("Pitch trim NDC x1000", &TB_pitch_trim_x1000, 1000);
    tb("Bank trim deg", &TB_bank_trim_deg, 60);
    tb("BankTop px", &TB_bank_top_px, 400);
    tb("BankRadius px", &TB_bank_rad_px, 800);
    tb("Ladder half width px", &TB_ladder_half_px, 800);
    tb("Ladder X offset px", &TB_ladder_xoff_px, 600);
    tb("Text scale %", &TB_text_scale_pct, 400);
    tb("Flip text X (0/1)", &TB_text_flip_x, 1);
    tb("Flip text Y (0/1)", &TB_text_flip_y, 1);

    // --- CPU buffers reused every frame
    cv::Mat hudBGRA(HUD_TEX_H, HUD_TEX_W, CV_8UC4);                   // FBO readback
    cv::Mat hudWarpBGRA(imgSize, CV_8UC4, cv::Scalar(0, 0, 0, 0));       // warped into camera space
    cv::Mat hudBGR(imgSize, CV_8UC3);
    cv::Mat alpha(imgSize, CV_32F), a3(imgSize, CV_32FC3);
    cv::Mat camF(imgSize, CV_32FC3), hudF(imgSize, CV_32FC3);

    // --- smoothing state
    double spd_kt_smooth = 0.0, alt_ft_smooth = 0.0;
    bool   first_samples = true;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;
        tracker.update(frame);
        BoardPose pose = tracker.latest();

        // build camera viz base
        cv::Mat camera = frame.clone();
        if (!tracker.ids().empty())
            cv::aruco::drawDetectedMarkers(camera, tracker.corners(), tracker.ids());

        // draw axis (optional)
        if (pose.valid) {
            float A = MARKER_LEN * 0.7f;
            std::vector<cv::Point3f> axisPts = { {0,0,0},{A,0,0},{0,A,0},{0,0,A} };
            std::vector<cv::Point2f> ip;
            cv::projectPoints(axisPts, pose.rvec, pose.tvec, K, D, ip);
            auto lineAA = [&](int a, int b, cv::Scalar c) { cv::line(camera, ip[a], ip[b], c, 2, cv::LINE_AA); };
            lineAA(0, 1, { 0,  0,255 }); // X red
            lineAA(0, 2, { 0,255,  0 }); // Y green
            lineAA(0, 3, { 255,  0,  0 }); // Z blue
        }

        // --- Sensor → derived readouts
        SensorSample s = sensor->latest();

        // velocity as NED (adapt if your build is ENU)
        double Vn = s.vel_x_ms;
        double Ve = s.vel_y_ms;
        double Vd = s.vel_z_ms;

        double Vh = std::hypot(Vn, Ve);
        double V = std::sqrt(Vh * Vh + Vd * Vd);

        const double MS_TO_KT = 1.9438444924406;
        double spd_kt_raw = V * MS_TO_KT;

        bool fpm_ok = std::isfinite(V) && Vh > 0.5; // ~1 kt threshold
        double chi_rad = std::atan2(Ve, Vn);
        double gamma_rad = std::atan2(-Vd, std::max(1e-3, Vh));

        double yaw_rad = s.yaw_deg * PI / 180.0;
        double pitch_rad = s.pitch_deg * PI / 180.0;

        auto wrapPi = [](double a) { while (a > PI) a -= 2 * PI; while (a < -PI) a += 2 * PI; return a; };
        double dx_rad = wrapPi(chi_rad - yaw_rad);
        double dy_rad = gamma_rad + pitch_rad;

        double alt_ft_raw = 0.0;
        bool alt_is_gps = false;
		const double QNH_HPA = 1013.25;
        if (std::isfinite(s.alt_msl_m) && std::abs(s.alt_msl_m) > 1e-3) {
            alt_is_gps = true;
            alt_ft_raw = s.alt_msl_m * 3.280839895;
        }
        else if (std::isfinite(s.baro_hpa) && s.baro_hpa > 1.0) {
            alt_is_gps = false;
            alt_ft_raw = (1.0 - std::pow(std::max(1e-3, s.baro_hpa) / QNH_HPA, 0.190284)) * 145366.45;
        }

        const double A_SPD = 0.2, A_ALT = 0.2;
        if (first_samples) { spd_kt_smooth = spd_kt_raw; alt_ft_smooth = alt_ft_raw; first_samples = false; }
        else {
            spd_kt_smooth = (1.0 - A_SPD) * spd_kt_smooth + A_SPD * spd_kt_raw;
            alt_ft_smooth = (1.0 - A_ALT) * alt_ft_smooth + A_ALT * alt_ft_raw;
        }

        // --- Build HudState (one time, used for both draws)
        HudState hs; hs.canvas_w = CANVAS_W; hs.canvas_h = CANVAS_H;

        float ndc_per_deg = std::max(0.001f, TB_pitch_scale_x1000 / 1000.0f);
        float trim_ndc = TB_pitch_trim_x1000 / 1000.0f;
        hs.pitch_px_per_deg = ndc_per_deg * (CANVAS_H * 0.5f);
        float pitch_deg = float(-s.pitch_deg);
        hs.pitch_px = (-pitch_deg) * hs.pitch_px_per_deg + trim_ndc * (CANVAS_H * 0.5f);

        hs.bank_rad = float((s.bank_deg + TB_bank_trim_deg) * (CV_PI / 180.0));
        hs.hdg_deg = float(s.yaw_deg);
        hs.speed_kt = float(std::max(0.0, spd_kt_smooth));
        hs.alt_ft = float(alt_ft_smooth);
        hs.alt_is_gps = alt_is_gps;
        hs.qnh_hpa = float(QNH_HPA);

        hs.ladder_half_px = float(TB_ladder_half_px);
        hs.ladder_xoff_px = float(TB_ladder_xoff_px);

        hs.text_scale = std::max(0.2f, TB_text_scale_pct / 100.0f);
        hs.flip_text_x = TB_text_flip_x;
        hs.flip_text_y = TB_text_flip_y;

        hs.fpm_dx_deg = float(dx_rad * 180.0 / PI);
        hs.fpm_dy_deg = float(dy_rad * 180.0 / PI);
        hs.fpm_valid = fpm_ok;

        float Rpx = float(TB_bank_rad_px);
        float centerY_px = (CANVAS_H * 0.5f) - (float(TB_bank_top_px) + Rpx);
        hud.setBankArcPx(centerY_px, Rpx);

        // OFFSCREEN PASS: draw HUD into FBO (transparent clear)
        glfwMakeContextCurrent(win); glfwPollEvents();

        glBindFramebuffer(GL_FRAMEBUFFER, hudFBO);
        glViewport(0, 0, HUD_TEX_W, HUD_TEX_H);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        hud.draw(hs);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Read back + flip for OpenCV
        glBindFramebuffer(GL_FRAMEBUFFER, hudFBO);
        glReadPixels(0, 0, HUD_TEX_W, HUD_TEX_H, GL_BGRA, GL_UNSIGNED_BYTE, hudBGRA.data);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        //cv::flip(hudBGRA, hudBGRA, 0);

        // CAMERA COMPOSITE: warp FBO image onto 4:3 board
        if (pose.valid) {
            std::vector<cv::Point3f> bb_obj;
            buildBillboardPts4x3({ 0,0,0 }, 0.24f, bb_obj); // width in meters
            std::vector<cv::Point2f> bb_img;
            cv::projectPoints(bb_obj, pose.rvec, pose.tvec, K, D, bb_img);

            // src: TL,TR,BR,BL   dst: map TR,TL,BL,BR → TL,TR,BR,BL
            std::vector<cv::Point2f> srcHUD = {
                {0.f, 0.f},
                {float(HUD_TEX_W - 1), 0.f},
                {float(HUD_TEX_W - 1), float(HUD_TEX_H - 1)},
                {0.f, float(HUD_TEX_H - 1)}
            };
            std::vector<cv::Point2f> dstIMG = { bb_img[1], bb_img[0], bb_img[3], bb_img[2] };

            cv::Mat Hh = cv::getPerspectiveTransform(srcHUD, dstIMG);

            hudWarpBGRA.setTo(cv::Scalar(0, 0, 0, 0));
            cv::warpPerspective(hudBGRA, hudWarpBGRA, Hh, camera.size(),
                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

            // alpha blend
            std::vector<cv::Mat> ch; cv::split(hudWarpBGRA, ch);
            cv::merge(std::vector<cv::Mat>{ch[0], ch[1], ch[2]}, hudBGR);
            ch[3].convertTo(alpha, CV_32F, 1.0 / 255.0);
            cv::merge(std::vector<cv::Mat>{alpha, alpha, alpha}, a3);

            camera.convertTo(camF, CV_32F);
            hudBGR.convertTo(hudF, CV_32F);
            cv::multiply(hudF, a3, hudF);
            cv::multiply(camF, cv::Scalar(1, 1, 1) - a3, camF);
            cv::Mat outF = hudF + camF; outF.convertTo(camera, CV_8U);
        }

        // show camera (with mini-HUD)
        cv::imshow("camera", camera);
        int k = cv::waitKey(1);
        if (k == 27) break;

        if (k == 'l') {                      // list available cameras
            auto avail = scanCameras(12);
            std::cerr << "Cameras found:";
            if (avail.empty()) std::cerr << " none\n";
            else {
                std::cerr << "\n";
                for (int idx : avail) {
                #ifdef __linux__
                            std::cerr << "  [" << idx << "] " << v4l2NameFor(idx) << "\n";
                #else
                            std::cerr << "  [" << idx << "]\n";
                #endif
                }
            }
        }

        if (k == 'n') {                      // next camera
            auto avail = scanCameras(12);
            if (!avail.empty()) {
                auto it = std::find(avail.begin(), avail.end(), camIndex);
                if (it == avail.end() || ++it == avail.end()) it = avail.begin();
                int next = *it;
                if (openCapture(cap, next, 1280, 720, 30)) {
                    camIndex = next;
        #ifdef __linux__
                    std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
        #else
                    std::cerr << "Switched to [" << camIndex << "]\n";
        #endif
                }
            }
        }

        if (k == 'p') {                      // previous camera
            auto avail = scanCameras(12);
            if (!avail.empty()) {
                auto it = std::find(avail.begin(), avail.end(), camIndex);
                if (it == avail.begin() || it == avail.end()) it = avail.end();
                --it;
                int prev = *it;
                if (openCapture(cap, prev, 1280, 720, 30)) {
                    camIndex = prev;
        #ifdef __linux__
                    std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
        #else
                    std::cerr << "Switched to [" << camIndex << "]\n";
        #endif
                }
            }
        }

        // direct select with 0..9
        if (k >= '0' && k <= '9') {
            int idx = k - '0';
            if (openCapture(cap, idx, 1280, 720, 30)) {
                camIndex = idx;
        #ifdef __linux__
                std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
        #else
                std::cerr << "Switched to [" << camIndex << "]\n";
        #endif
            } else {
                std::cerr << "Failed to open camera " << idx << "\n";
            }
        }
        // WINDOW PASS: draw HUD to the GLFW window (letterboxed)

        int W = 0, H = 0;
        glfwGetFramebufferSize(win, &W, &H);
        if (W == 0 || H == 0) { glfwSwapBuffers(win); continue; }

        float sx = W / float(CANVAS_W), sy = H / float(CANVAS_H);
        float scale = std::min(sx, sy);
        int VW = int(CANVAS_W * scale), VH = int(CANVAS_H * scale);
        int VX = (W - VW) / 2, VY = (H - VH) / 2;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(VX, VY, VW, VH);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        hud.draw(hs);
        glfwSwapBuffers(win);

        if (glfwWindowShouldClose(win)) break;
    }

    save_controls();
    hud.shutdown();
    glfwDestroyWindow(win); glfwTerminate();
    sensor->stop();
    return 0;
}
