#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <chrono>

#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include "app.hpp"
#include "aruco_tracker.hpp"
#include "hud_renderer.hpp"
#include "sensor.hpp"

using nlohmann::json;

static const float PI = 3.14159265358979323846f;

static const float HINV_ID[9] = { 1,0,0,  0,1,0,  0,0,1 };

// ---- detector ring buffer + latest pose (definitions matching app.hpp)
std::array<cv::Mat,3> g_grayBuf;
std::atomic<int>      g_wr{-1};
std::atomic<bool>     g_detRun{true};
std::mutex            g_poseMtx;
BoardPose             g_pose;   // starts invalid; tracker will fill

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
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cv::Mat probe;
    if (!cap.read(probe) || probe.empty()) return false;
    return true;
}

static int parseCamIndex(int argc, char** argv, int fallback = 0) {
    if (const char* e = std::getenv("CAM_INDEX"); e && *e) return std::atoi(e);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-c" || a == "--cam") && i + 1 < argc) return std::atoi(argv[i + 1]);
        char* end = nullptr; long v = std::strtol(argv[i], &end, 10);
        if (end && *end == '\0') return int(v);
    }
    return fallback;
}

// ---------- Shared state between render thread (main) and UI thread ----------

static std::atomic<bool> g_running{true};
static std::mutex        g_camMutex;
static cv::Mat           g_camPreview;  // last composited camera+HUD frame shown in UI thread

// -------- file-scope constants --------
namespace {
    constexpr int HUD_TEX_W = 1024;
    constexpr int HUD_TEX_H = 768;
    constexpr int CANVAS_W = 1920;   // design canvas (4:3)
    constexpr int CANVAS_H = 1440;
}

FrameClock clk;

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log);
        std::cerr << "Shader compile error: " << log << "\n"; glDeleteShader(s); return 0; }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram(); glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    GLint ok = GL_FALSE; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log);
        std::cerr << "Program link error: " << log << "\n"; glDeleteProgram(p); return 0; }
    glDeleteShader(vs); glDeleteShader(fs); return p;
}
struct OverlayTexQuad {
    GLuint prog=0, vao=0, vbo=0;
    GLint  uTex=-1, uScaleX=-1, uScaleY=-1, uOffX=-1, uOffY=-1, uRot=-1, uPivotX=-1, uPivotY=-1;

    bool init() {
        const char* vs = R"(#version 330 core
            layout(location=0) in vec2 aPos;
            layout(location=1) in vec2 aUV;
            out vec2 vUV;
            void main(){ vUV=aUV; gl_Position = vec4(aPos,0.0,1.0); })";

        // SRT on UVs around a pivot. Note: scaling here is “display zoom”,
        // so we divide by scale to zoom the content in.
        const char* fs = R"(#version 330 core
            in vec2 vUV; out vec4 FragColor;
            uniform sampler2D uTex;
            uniform float uScaleX;  // >= 0.01
            uniform float uScaleY;
            uniform float uOffX;    // UV units
            uniform float uOffY;
            uniform float uRot;     // radians
            uniform float uPivotX;  // 0..1
            uniform float uPivotY;
            void main(){
                vec2 pivot = vec2(uPivotX, uPivotY);
                vec2 p = vUV - pivot;
                float c = cos(uRot), s = sin(uRot);
                mat2 R = mat2(c, -s, s, c);
                p = R * (p / vec2(max(uScaleX,0.01), max(uScaleY,0.01))) + pivot + vec2(uOffX, uOffY);
                FragColor = texture(uTex, p);
            })";

        GLuint v = compileShader(GL_VERTEX_SHADER, vs);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
        prog = (v && f) ? linkProgram(v, f) : 0;
        if (!prog) return false;

        uTex    = glGetUniformLocation(prog, "uTex");
        uScaleX = glGetUniformLocation(prog, "uScaleX");
        uScaleY = glGetUniformLocation(prog, "uScaleY");
        uOffX   = glGetUniformLocation(prog, "uOffX");
        uOffY   = glGetUniformLocation(prog, "uOffY");
        uRot    = glGetUniformLocation(prog, "uRot");
        uPivotX = glGetUniformLocation(prog, "uPivotX");
        uPivotY = glGetUniformLocation(prog, "uPivotY");

        const float verts[] = {
            //  pos.xy   uv.xy
            -1.f,-1.f,  0.f,1.f,
             1.f,-1.f,  1.f,1.f,
             1.f, 1.f,  1.f,0.f,
            -1.f,-1.f,  0.f,1.f,
             1.f, 1.f,  1.f,0.f,
            -1.f, 1.f,  0.f,0.f
        };
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
        glBindVertexArray(0);

        glUseProgram(prog);
        glUniform1i(uTex, 0);
        glUseProgram(0);
        return true;
    }

    void setSRT(float scaleX, float scaleY, float offUVx, float offUVy, float rotRad, float pivotX, float pivotY) {
        glUseProgram(prog);
        glUniform1f(uScaleX, scaleX);
        glUniform1f(uScaleY, scaleY);
        glUniform1f(uOffX,   offUVx);
        glUniform1f(uOffY,   offUVy);
        glUniform1f(uRot,    rotRad);
        glUniform1f(uPivotX, pivotX);
        glUniform1f(uPivotY, pivotY);
        glUseProgram(0);
    }

    void draw(GLuint tex) {
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glUseProgram(0);
    }
    void shutdown() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (prog) glDeleteProgram(prog);
        prog = vao = vbo = 0;
    }
};

// ---------- controls + persistence ----------
//static const char* WIN_CTRL = "HUD Controls"; 
static const char* PERSIST = "hud_layout_controls.json";

// HUD placement controls
static int TB_pitch_scale_x1000 = 20;   // manual NDC/deg *1000
static int TB_pitch_trim_x1000  = 0;    // NDC trim *1000
static int TB_bank_trim_deg     = 0;    // deg
static int TB_bank_top_px       = 60;   // px
static int TB_bank_rad_px       = 220;  // px
static int TB_ladder_half_px    = 360;  // px
static int TB_ladder_xoff_px    = 0;    // px
static int TB_text_scale_pct    = 120;  // %
static int TB_text_flip_x       = 1;    // 0/1
static int TB_text_flip_y       = 0;    // 0/1

// Auto pitch scale from camera intrinsics+homography
static int  TB_auto_pitch_from_cam = 1;   // 0/1
static int  TB_auto_center_y_px    = -1;  // -1=mid
static int  TB_probe_dY_canvas     = 100; // px

// ---- Goggles overlay controls (2D SRT applied after HUD warping) ----
static int TB_gog_show_markers = 1;   // 0/1: draw detected markers in goggles
static int TB_gog_show_axes    = 1;   // 0/1: draw axes in goggles

// Offsets in pixels mapped from 0..2000 => -1000..+1000
static int TB_ov_off_x_px = 1000;     // centered() => -1000..+1000
static int TB_ov_off_y_px = 1000;     // centered() => -1000..+1000
// Zoom range 1..400 % (allows much smaller than before)
static int TB_ov_scale_pct = 100;     // 1..400
// Rotation around pivot, 0..360 -> -180..+180
static int TB_ov_rot_deg   = 180;     // centered() => -180..+180
static int TB_ov_pitch_deg = 180;
static int TB_ov_yaw_deg   = 180;
// Pivot: 0=center (0.5,0.5), 1=top-left (0,0)
static int TB_ov_pivot_tl  = 0;       // 0/1


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
        {"text_flip_y",      TB_text_flip_y},
        {"auto_pitch_from_cam", TB_auto_pitch_from_cam},
        {"auto_center_y_px", TB_auto_center_y_px},
        {"probe_dY_canvas",  TB_probe_dY_canvas},
        {"gog_show_markers", TB_gog_show_markers},
        {"gog_show_axes",    TB_gog_show_axes},
        {"ov_off_x_px",    TB_ov_off_x_px},
        {"ov_off_y_px",    TB_ov_off_y_px},
        {"ov_scale_pct",    TB_ov_scale_pct},
        {"ov_rot_deg",    TB_ov_rot_deg},
        {"ov_pitch_deg",    TB_ov_pitch_deg},
        {"ov_yaw_deg",    TB_ov_yaw_deg},
        {"ov_pivot_tl",    TB_ov_pivot_tl}
        

    };
    std::ofstream(PERSIST) << j.dump(2);
}
static void load_controls() {
    std::ifstream f(PERSIST); if (!f) return;
    json j; f >> j;
    auto get = [&](const char* k, int& v) { if (j.contains(k)) v = j[k].get<int>(); };
    get("pitch_scale_x1000", TB_pitch_scale_x1000);
    get("pitch_trim_x1000",  TB_pitch_trim_x1000);
    get("bank_trim_deg",     TB_bank_trim_deg);
    get("bank_top_px",       TB_bank_top_px);
    get("bank_radius_px",    TB_bank_rad_px);
    get("ladder_half_px",    TB_ladder_half_px);
    get("ladder_xoff_px",    TB_ladder_xoff_px);
    get("text_scale_pct",    TB_text_scale_pct);
    get("text_flip_x",       TB_text_flip_x);
    get("text_flip_y",       TB_text_flip_y);
    get("auto_pitch_from_cam", TB_auto_pitch_from_cam);
    get("auto_center_y_px",    TB_auto_center_y_px);
    get("probe_dY_canvas",     TB_probe_dY_canvas);
    get("gog_show_markers",    TB_gog_show_markers);
    get("gog_show_axes",       TB_gog_show_axes);
    get("ov_off_x_px",    TB_ov_off_x_px);
    get("ov_off_y_px",    TB_ov_off_y_px);
    get("ov_scale_pct",    TB_ov_scale_pct);
    get("ov_rot_deg",    TB_ov_rot_deg);
    get("ov_pitch_deg",    TB_ov_pitch_deg);
    get("ov_yaw_deg",    TB_ov_yaw_deg);
    get("ov_pivot_tl",    TB_ov_pivot_tl);
        
    if (j.contains("pitch_ndc_x1000") && !j.contains("pitch_trim_x1000"))
        TB_pitch_trim_x1000 = j["pitch_ndc_x1000"].get<int>();
}

// ---------- helpers ----------
// ---------- UI: split controls into 2 windows ----------

static void createControlsGroup1()
{
    cv::namedWindow("HUD Controls 1", cv::WINDOW_NORMAL);
    cv::resizeWindow("HUD Controls 1", 420, 600);

    auto tb = [&](const char* name, int* var, int maxv) {
        cv::createTrackbar(name, "HUD Controls 1", var, maxv, nullptr);
    };

    // Core HUD geometry + text
    tb("PitchScale",      &TB_pitch_scale_x1000, 200);
    tb("PitchTrim",       &TB_pitch_trim_x1000, 1000);
    tb("BankTrim",        &TB_bank_trim_deg, 60);
    tb("BankTop_px",      &TB_bank_top_px, 400);
    tb("BankRadius_px",   &TB_bank_rad_px, 800);
    tb("LadderHalf_px",   &TB_ladder_half_px, 800);
    tb("LadderXOffset_px",&TB_ladder_xoff_px, 600);
    tb("TextScale_pct",   &TB_text_scale_pct, 400);
    tb("FlipTextX",       &TB_text_flip_x, 1);
    tb("FlipTextY",       &TB_text_flip_y, 1);
}

static void createControlsGroup2(int canvasH)
{
    cv::namedWindow("HUD Controls 2", cv::WINDOW_NORMAL);
    cv::resizeWindow("HUD Controls 2", 420, 600);

    auto tb = [&](const char* name, int* var, int maxv) {
        cv::createTrackbar(name, "HUD Controls 2", var, maxv, nullptr);
    };

    // Auto pitch calibration + overlay/goggles controls
    tb("AutoPitch",       &TB_auto_pitch_from_cam, 1);
    tb("AutoCenterY_px",  &TB_auto_center_y_px, canvasH);
    tb("AutoProbe_dY",    &TB_probe_dY_canvas, 400);

    tb("ShowMarkers",     &TB_gog_show_markers, 1);
    tb("ShowAxes",        &TB_gog_show_axes, 1);

    tb("OV_Scale_pct",    &TB_ov_scale_pct, 400);
    tb("OV_OffX",         &TB_ov_off_x_px, 2000);
    tb("OV_OffY",         &TB_ov_off_y_px, 2000);
    tb("OV_Rot_deg",      &TB_ov_rot_deg, 360);
    tb("OV_Pitch_deg",    &TB_ov_pitch_deg, 360);
    tb("OV_Yaw_deg",      &TB_ov_yaw_deg, 360);
    tb("OV_PivotTL",      &TB_ov_pivot_tl, 1);
}
// ---------- UI Thread: OpenCV windows + trackbars ----------

static void uiThreadFunc(cv::Size imgSize)
{
    // Load persisted control values BEFORE creating sliders
    load_controls();

    // Camera preview window
    cv::namedWindow("camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("camera", imgSize.width, imgSize.height);

    // Two control windows
    createControlsGroup1();
    createControlsGroup2(CANVAS_H);

    while (g_running) {
        cv::Mat preview;
        {
            std::lock_guard<std::mutex> lock(g_camMutex);
            if (!g_camPreview.empty())
                g_camPreview.copyTo(preview);
        }

        if (!preview.empty())
            cv::imshow("camera", preview);

        // process UI events, keep it short to reduce latency
        int key = cv::waitKey(1);
        if (key == 27) { // ESC in UI also stops everything
            g_running = false;
        }
    }

    // Save control state on exit
    save_controls();
    cv::destroyAllWindows();
}

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

// apply homography (64F)
static inline cv::Point2f applyH64(const cv::Mat& H64, float x, float y){
    const double a = H64.at<double>(0,0)*x + H64.at<double>(0,1)*y + H64.at<double>(0,2);
    const double b = H64.at<double>(1,0)*x + H64.at<double>(1,1)*y + H64.at<double>(1,2);
    const double c = H64.at<double>(2,0)*x + H64.at<double>(2,1)*y + H64.at<double>(2,2);
    const float inv = (c != 0.0) ? float(1.0/c) : 1.0f;
    return { float(a)*inv, float(b)*inv };
}

// canvas px/deg from intrinsics + local homography sampling
static bool canvasPxPerDegree_fromIntrinsics(const cv::Mat& Hh64,
                                             const cv::Mat& K,
                                             const cv::Mat& D,
                                             float canvasY,
                                             float probe_dY_canvas,
                                             float& out_canvas_px_per_deg)
{
    if (Hh64.empty() || K.empty()) return false;

    const float x_fbo   = HUD_TEX_W * 0.5f;
    const float fbo_per_canvas = HUD_TEX_H / float(CANVAS_H);
    const float y0_fbo  = canvasY * fbo_per_canvas;
    const float y1_fbo  = (canvasY + probe_dY_canvas) * fbo_per_canvas;

    const cv::Point2f p0_img = applyH64(Hh64, x_fbo, y0_fbo);
    const cv::Point2f p1_img = applyH64(Hh64, x_fbo, y1_fbo);

    std::vector<cv::Point2f> pts{p0_img, p1_img}, und;
    cv::undistortPoints(pts, und, K, D);
    cv::Vec3d r0(und[0].x, und[0].y, 1.0); r0 /= cv::norm(r0);
    cv::Vec3d r1(und[1].x, und[1].y, 1.0); r1 /= cv::norm(r1);

    double cosang = r0.dot(r1);
    cosang = std::clamp(cosang, -1.0, 1.0);
    const double dtheta_rad = std::acos(cosang);
    if (dtheta_rad < 1e-9) return false;

    out_canvas_px_per_deg = probe_dY_canvas / float(dtheta_rad * 180.0 / CV_PI);
    return true;
}

static GLFWwindow* createOverlayWindow(GLFWwindow* shareWith, int desiredMonitor = -1,
                                       bool transparent = true, bool borderless = true) {
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, transparent ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, borderless ? GLFW_FALSE : GLFW_TRUE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);

    int monitorCount = 0;
    GLFWmonitor** mons = glfwGetMonitors(&monitorCount);
    GLFWmonitor* target = nullptr;

    if (monitorCount > 0) {
        if (desiredMonitor >= 0 && desiredMonitor < monitorCount) {
            target = mons[desiredMonitor];
        } else if (monitorCount >= 2) {
            target = mons[1];
        } else {
            target = mons[0];
        }
    }

    const GLFWvidmode* mode = glfwGetVideoMode(target);
    int W = mode ? mode->width : 2560;
    int H = mode ? mode->height : 1920;

    GLFWwindow* w = glfwCreateWindow(W, H, "HUD Overlay", target, shareWith);
    if (!w) return nullptr;

    glfwMakeContextCurrent(w);
    glfwSwapInterval(0);       // vsync on the glasses
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0.f, 0.f, 0.f, transparent ? 0.f : 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(w);

    return w;
}

int main(int argc, char** argv) {
    // --- Sensor first (claim the serial port before hammering USB cameras)
    struct SensorRunner {
        std::unique_ptr<ISensorSource> ptr;
        SensorRunner() : ptr(makeSensor()) { if (ptr) ptr->start(); }
        ~SensorRunner() { if (ptr) ptr->stop(); }
        ISensorSource* operator->() { return ptr.get(); }
        const ISensorSource* operator->() const { return ptr.get(); }
    } sensor;

    // --- Camera
    int camIndex = parseCamIndex(argc, argv, /*fallback*/0);

    cv::setUseOptimized(true);
    cv::setUseOpenVX(false);
    cv::setNumThreads(6);      // often best for ArUco; test 2..6


    cv::VideoCapture cap;
    if (!openCapture(cap, camIndex, 1280, 720, 60)) {
        auto avail = scanCameras(12);
        if (avail.empty()) {
            std::cerr << "No camera could be opened.\n";
            return 1;
        }
        camIndex = avail.front();
        if (!openCapture(cap, camIndex, 1280, 720, 60)) {
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

    // Optional MJPG request
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

    cv::Mat frame;
    if (!cap.grab() || !cap.retrieve(frame) || frame.empty()) {
        std::cerr << "Initial frame empty\n";
        return 1;
    }
    cv::Size imgSize = frame.size();

    for (auto& b : g_grayBuf)
    b.create(imgSize.height, imgSize.width, CV_8UC1);

    // --- Start UI thread (OpenCV camera + trackbars)
    std::thread uiThread(uiThreadFunc, imgSize);

    // --- Calibration
    cv::Mat K, D;
    if (!loadCalibrationJSON("calibration.json", K, D, imgSize)) {
        approxFOVIntrinsics(imgSize, 70.0, K, D);
        std::cout << "No calibration.json. Using FOV approximation.\n";
    }
    // Ensure 64F
    K.convertTo(K, CV_64F); D.convertTo(D, CV_64F);

    // --- ArUco tracker
    ArucoTracking tracker;
    const int MX = 3, MY = 2;
    const float MARKER_LEN = 0.050f, GAP = 0.012f;
    std::thread detThread([&]{
        cv::Mat localGray;
        int last = -1;

        // Optional: detector params / downscale tuned for speed
        tracker.setCameraIntrinsics(K, D);
        tracker.setGridBoard(MX, MY, MARKER_LEN, GAP);
        tracker.setAnchorMarkerCorner(0, 0);
        tracker.setTemporalSmoothing(0.9);

        // tracker.setDownscale(cfg.downscale_pct); // e.g. 0.75
        // cv::aruco::DetectorParameters p = ...; tracker.setParams(p);

        while (g_detRun.load(std::memory_order_acquire)) {
            // Spin/wait for a new published index
            int idx = g_wr.load(std::memory_order_acquire);
            if (idx == last) { 
                std::this_thread::yield(); 
                continue; 
            }
            last = idx;

            // Copy the published gray frame into a private Mat
            g_grayBuf[idx].copyTo(localGray);

            // Run detection (heavy)
            tracker.update(localGray);

            // Publish latest pose
            BoardPose p = tracker.latest();
            {
                std::lock_guard<std::mutex> lk(g_poseMtx);
                g_pose = p;
            }
        }
    });

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
    glfwSwapInterval(0);

    // --- Dedicated CAMERA window (GPU composite target)
    // GLFWwindow* camWin = glfwCreateWindow(imgSize.width, imgSize.height, "Camera", nullptr, win /*share ctx*/);
    // if (!camWin) {
    //     std::cerr << "glfwCreateWindow(Camera) failed\n";
    // } else {
    //     glfwMakeContextCurrent(camWin);
    //     glfwSwapInterval(0);   // unsynced: lower latency
    //     glClearColor(0,0,0,1);
    //     glClear(GL_COLOR_BUFFER_BIT);
    //     glfwSwapBuffers(camWin);
    //     glfwMakeContextCurrent(win); // back to HUD
    // }



    // --- Offscreen HUD render target (FBO + texture)
    GLuint hudFBO = 0, hudTex = 0;
    glGenTextures(1, &hudTex);
    glBindTexture(GL_TEXTURE_2D, hudTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, HUD_TEX_W, HUD_TEX_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &hudFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hudFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hudTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) std::cerr << "HUD FBO incomplete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // GPU warp & composite programs
    GpuWarp warp;
    GpuComposite comp;
    if (!warp.init()) std::cerr << "GpuWarp init failed\n";
    if (!comp.init()) std::cerr << "GpuComposite init failed\n";

    // Camera texture (BGR upload)
    GLuint camTex = 0;
    glGenTextures(1, &camTex);
    glBindTexture(GL_TEXTURE_2D, camTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, imgSize.width, imgSize.height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Readback buffer for FBO
    cv::Mat hudBGRA(HUD_TEX_H, HUD_TEX_W, CV_8UC4);

    // --- Renderer
    HudRenderer hud; if (!hud.init()) { std::cerr << "hud.init failed\n"; return 1; }

    // --- Overlay window for goggles + textured quad
    GLFWwindow* hudOverlay = nullptr;
    int currentOverlayIdx = -1;
    OverlayTexQuad ovQuad;

    auto recreateOverlayAt = [&](int idx) {
        if (hudOverlay) {
            glfwMakeContextCurrent(hudOverlay);
            //glFinish();
            ovQuad.shutdown();
            glfwDestroyWindow(hudOverlay);
            hudOverlay = nullptr;
        }
        int mc = 0; glfwGetMonitors(&mc);
        if (mc <= 0 || idx < 0 || idx >= mc) {
            std::cerr << "Overlay monitor " << idx << " invalid; overlay disabled\n";
            currentOverlayIdx = -1;
            glfwMakeContextCurrent(win);
            return;
        }
        std::cerr << "Creating overlay on Monitor[" << idx << "]\n";
        hudOverlay = createOverlayWindow(win, idx, /*transparent*/false, /*borderless*/true);
        if (!hudOverlay) {
            std::cerr << "createOverlayWindow failed\n";
            currentOverlayIdx = -1;
            glfwMakeContextCurrent(win);
            return;
        }
        glfwMakeContextCurrent(hudOverlay);
        glfwSwapInterval(0);
        if (!ovQuad.init()) {
            std::cerr << "Overlay TexQuad init failed; overlay disabled\n";
            glfwMakeContextCurrent(win);
            glfwDestroyWindow(hudOverlay);
            hudOverlay = nullptr;
            currentOverlayIdx = -1;
            return;
        }
        glfwMakeContextCurrent(win);
        glfwSwapInterval(0); // dev window free runs; reduces stutter
        currentOverlayIdx = idx;
    };

    {
        int mc = 0; glfwGetMonitors(&mc);
        recreateOverlayAt(mc > 1 ? 1 : 0);
    }

    // --- Reused CPU mats
    cv::Mat hudWarpBGRA(imgSize, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::Mat hudBGR(imgSize, CV_8UC3);
    cv::Mat alpha(imgSize, CV_32F), a3(imgSize, CV_32FC3);
    cv::Mat camF(imgSize, CV_32FC3), hudF(imgSize, CV_32FC3);

    // --- smoothing state
    double spd_kt_smooth = 0.0, alt_ft_smooth = 0.0;
    bool   first_samples = true;

    while (g_running) {
        clk.begin();

        if (!cap.grab()) {
            std::cerr << "cap.grab() failed\n";
            break;
        }
        if (!cap.retrieve(frame) || frame.empty()) {
            std::cerr << "cap.retrieve() empty\n";
            break;
        }

        clk.cap();

        // Publish gray to the detector ring buffer
        int next = (g_wr.load(std::memory_order_relaxed) + 1) % 3;
        frame.copyTo(g_grayBuf[next]);                          // reuse preallocated slot
        g_wr.store(next, std::memory_order_release);
        //tracker.update(gray);
        BoardPose pose;
        {
            std::lock_guard<std::mutex> lk(g_poseMtx);
            pose = g_pose;     // cheap copy
        }

        clk.det();

        //BoardPose pose = tracker.latest();

        // --- Compute homography early (so scale applies to BOTH camera and goggles)
        cv::Mat Hh64; bool haveH = false;
        std::vector<cv::Point2f> bb_img;

        if (pose.valid) {
            // 3D billboard in board coordinates (a 4:3 rectangle)
            std::vector<cv::Point3f> bb_obj;
            const float BOARD_W_M = 0.24f; // must match warping board width
            buildBillboardPts4x3({0,0,0}, BOARD_W_M, bb_obj);

            // --- Apply manual yaw / pitch offsets to the HUD plane (board-local)
            // Trackbar values are [0..360]; map them to [-180..+180] degrees
            const float yaw_deg   = float(centered(TB_ov_yaw_deg,   360));
            const float pitch_deg = float(centered(TB_ov_pitch_deg, 360));

            if (std::fabs(yaw_deg) > 0.001f || std::fabs(pitch_deg) > 0.001f) {
                const float yaw   = yaw_deg   * float(CV_PI / 180.0);
                const float pitch = pitch_deg * float(CV_PI / 180.0);

                const float cy = std::cos(yaw),   sy = std::sin(yaw);
                const float cp = std::cos(pitch), sp = std::sin(pitch);

                // Yaw around board Y axis, pitch around board X axis
                cv::Matx33f R_y(  cy, 0.0f,  sy,
                                0.0f, 1.0f, 0.0f,
                                -sy, 0.0f,  cy );

                cv::Matx33f R_x( 1.0f, 0.0f, 0.0f,
                                0.0f,  cp, -sp,
                                0.0f,  sp,  cp );

                cv::Matx33f R_off = R_y * R_x;

                // Rotate billboard around its centre, not the corner
                cv::Point3f c(0,0,0);
                for (const auto& p : bb_obj) {
                    c.x += p.x; c.y += p.y; c.z += p.z;
                }
                c.x /= static_cast<float>(bb_obj.size());
                c.y /= static_cast<float>(bb_obj.size());
                c.z /= static_cast<float>(bb_obj.size());

                for (auto& p : bb_obj) {
                    cv::Vec3f v(p.x - c.x, p.y - c.y, p.z - c.z);
                    cv::Vec3f v2 = R_off * v;
                    p.x = c.x + v2[0];
                    p.y = c.y + v2[1];
                    p.z = c.z + v2[2];
                }
            }

            // Project the (optionally rotated) HUD rectangle into the camera
            cv::projectPoints(bb_obj, pose.rvec, pose.tvec, K, D, bb_img);

            // src HUD quad (FBO) → dst image quad (TR,TL,BL,BR → TL,TR,BR,BL)
            std::vector<cv::Point2f> srcHUD = {
                {0.f, 0.f},
                {float(HUD_TEX_W - 1), 0.f},
                {float(HUD_TEX_W - 1), float(HUD_TEX_H - 1)},
                {0.f, float(HUD_TEX_H - 1)}
            };
            std::vector<cv::Point2f> dstIMG = { bb_img[1], bb_img[0], bb_img[3], bb_img[2] };

            cv::Mat Hh = cv::getPerspectiveTransform(srcHUD, dstIMG);
            Hh.convertTo(Hh64, CV_64F);
            haveH = true;
        }

        cv::Mat Hinv64;
        float HinvGL[9] = {0};
        if (haveH) {
            cv::invert(Hh64, Hinv64);
            HinvGL[0] = (float)Hinv64.at<double>(0,0);
            HinvGL[1] = (float)Hinv64.at<double>(1,0);
            HinvGL[2] = (float)Hinv64.at<double>(2,0);
            HinvGL[3] = (float)Hinv64.at<double>(0,1);
            HinvGL[4] = (float)Hinv64.at<double>(1,1);
            HinvGL[5] = (float)Hinv64.at<double>(2,1);
            HinvGL[6] = (float)Hinv64.at<double>(0,2);
            HinvGL[7] = (float)Hinv64.at<double>(1,2);
            HinvGL[8] = (float)Hinv64.at<double>(2,2);
        }

        float SRTGL[9];
        buildSRT_px(SRTGL,
                    TB_ov_off_x_px, TB_ov_off_y_px,
                    TB_ov_scale_pct, TB_ov_rot_deg,
                    TB_ov_pivot_tl,
                    HUD_TEX_W, HUD_TEX_H);


        clk.hom();

        // --- Build camera viz base (for preview only)
        cv::Mat camera = frame.clone();
        if (!tracker.ids().empty())
            cv::aruco::drawDetectedMarkers(camera, tracker.corners(), tracker.ids());
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

        double Vn = s.vel_x_ms;
        double Ve = s.vel_y_ms;
        double Vd = s.vel_z_ms;

        double Vh = std::hypot(Vn, Ve);
        double V  = std::sqrt(Vh * Vh + Vd * Vd);

        const double MS_TO_KT = 1.9438444924406;
        double spd_kt_raw = V * MS_TO_KT;

        bool   fpm_ok   = std::isfinite(V) && Vh > 0.5; // ~1 kt threshold
        double chi_rad  = std::atan2(Ve, Vn);
        double gamma_rad= std::atan2(-Vd, std::max(1e-3, Vh));

        double yaw_rad  = s.yaw_deg   * PI / 180.0;
        double pitch_rad= s.pitch_deg * PI / 180.0;

        auto wrapPi = [](double a) { while (a > PI) a -= 2 * PI; while (a < -PI) a += 2 * PI; return a; };
        double dx_rad = wrapPi(chi_rad - yaw_rad);
        double dy_rad = gamma_rad + pitch_rad;

        double alt_ft_raw = 0.0; bool alt_is_gps = false;
        const double QNH_HPA = 1013.25;
        if (std::isfinite(s.alt_msl_m) && std::abs(s.alt_msl_m) > 1e-3) {
            alt_is_gps = true;
            alt_ft_raw = s.alt_msl_m * 3.280839895;
        } else if (std::isfinite(s.baro_hpa) && s.baro_hpa > 1.0) {
            alt_is_gps = false;
            alt_ft_raw = (1.0 - std::pow(std::max(1e-3, s.baro_hpa) / QNH_HPA, 0.190284)) * 145366.45;
        }

        const double A_SPD = 0.2, A_ALT = 0.2;
        if (first_samples) { spd_kt_smooth = spd_kt_raw; alt_ft_smooth = alt_ft_raw; first_samples = false; }
        else {
            spd_kt_smooth = (1.0 - A_SPD) * spd_kt_smooth + A_SPD * spd_kt_raw;
            alt_ft_smooth = (1.0 - A_ALT) * alt_ft_smooth + A_ALT * alt_ft_raw;
        }

        // --- Build HudState
        HudState hs; hs.canvas_w = CANVAS_W; hs.canvas_h = CANVAS_H;

        // start with manual scaling (as before)
        float ndc_per_deg = std::max(0.001f, TB_pitch_scale_x1000 / 1000.0f);
        float trim_ndc    = TB_pitch_trim_x1000 / 1000.0f;
        hs.pitch_px_per_deg = ndc_per_deg * (CANVAS_H * 0.5f);
        float pitch_deg = float(-s.pitch_deg);
        hs.pitch_px = (-pitch_deg) * hs.pitch_px_per_deg + trim_ndc * (CANVAS_H * 0.5f);

        hs.bank_rad = float((s.bank_deg + TB_bank_trim_deg) * (CV_PI / 180.0));
        hs.hdg_deg  = float(s.yaw_deg);
        hs.speed_kt = float(std::max(0.0, spd_kt_smooth));
        hs.alt_ft   = float(alt_ft_smooth);
        hs.alt_is_gps = alt_is_gps;
        hs.qnh_hpa  = float(QNH_HPA);

        hs.ladder_half_px = float(TB_ladder_half_px);
        hs.ladder_xoff_px = float(TB_ladder_xoff_px);

        hs.text_scale = std::max(0.2f, TB_text_scale_pct / 100.0f);
        hs.flip_text_x = TB_text_flip_x;
        hs.flip_text_y = TB_text_flip_y;

        hs.fpm_dx_deg = float(dx_rad * 180.0 / PI);
        hs.fpm_dy_deg = float(dy_rad * 180.0 / PI);
        hs.fpm_valid  = fpm_ok;

        // Bank arc placement for both
        float Rpx = float(TB_bank_rad_px);
        float centerY_px = (CANVAS_H * 0.5f) - (float(TB_bank_top_px) + Rpx);
        hud.setBankArcPx(centerY_px, Rpx);

        // --- AUTO pitch scale from intrinsics + homography (BEFORE FBO draw)
        if (haveH && TB_auto_pitch_from_cam) {
            const float cy = (TB_auto_center_y_px < 0 || TB_auto_center_y_px > CANVAS_H)
                           ? float(CANVAS_H * 0.5f)
                           : float(TB_auto_center_y_px);
            float px_per_deg_canvas = 0.f;
            if (canvasPxPerDegree_fromIntrinsics(Hh64, K, D,
                                                 cy, std::max(10, TB_probe_dY_canvas),
                                                 px_per_deg_canvas)) {
                px_per_deg_canvas = std::clamp(px_per_deg_canvas, 10.0f, 4000.0f);
                hs.pitch_px_per_deg = px_per_deg_canvas;
            }
            // Recompute offset with current trim
            pitch_deg = float(-s.pitch_deg);
            hs.pitch_px = (-pitch_deg) * hs.pitch_px_per_deg + trim_ndc * (CANVAS_H * 0.5f);
        }

        // --- OFFSCREEN PASS: draw HUD into FBO
        glfwMakeContextCurrent(win); glfwPollEvents();

        // Cycle overlay across monitors with F9
        static bool f9Latch = false;
        int f9 = glfwGetKey(win, GLFW_KEY_F9);
        if (f9 == GLFW_PRESS && !f9Latch) {
            int mc = 0; glfwGetMonitors(&mc);
            if (mc > 0) {
                int next = (currentOverlayIdx < 0) ? 0 : (currentOverlayIdx + 1) % mc;
                std::cerr << "Cycling overlay to Monitor[" << next << "]\n";
                recreateOverlayAt(next);
            }
            f9Latch = true;
        }
        if (f9 == GLFW_RELEASE) f9Latch = false;

        glBindFramebuffer(GL_FRAMEBUFFER, hudFBO);
        glActiveTexture(GL_TEXTURE0);
        glViewport(0, 0, HUD_TEX_W, HUD_TEX_H);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        hud.draw(hs);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        clk.hud();

        // if (camWin) {
        //     glfwMakeContextCurrent(camWin);

        //     int cw=0, ch=0; glfwGetFramebufferSize(camWin, &cw, &ch);
        //     Viewport cvp = letterbox(cw, ch, imgSize.width, imgSize.height);
        //     glViewport(cvp.x, cvp.y, cvp.w, cvp.h);

        //     glClearColor(0,0,0,1);
        //     glClear(GL_COLOR_BUFFER_BIT);

        //     glEnable(GL_BLEND);
        //     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        //     static const float HINV_ID[9] = {1,0,0,  0,1,0,  0,0,1};
        //     const float* Hx = haveH ? HinvGL : HINV_ID;

        //     // We'll pass SRT below in Part 2; for now pass identitySRT
        //     //static const float SRT_ID[9] = {1,0,0,  0,1,0,  0,0,1};

        //     comp.draw(
        //         /*camTex=*/camTex,
        //         /*hudTex=*/hudTex,
        //         /*Hinv=*/Hx,
        //         /*SRT =*/SRTGL,
        //         /*imgW=*/imgSize.width, /*imgH=*/imgSize.height,
        //         /*hudW=*/HUD_TEX_W,     /*hudH=*/HUD_TEX_H);

        //     glfwSwapBuffers(camWin);

        //     glActiveTexture(GL_TEXTURE0);
        //     glBindTexture(GL_TEXTURE_2D, 0);

        //     glfwMakeContextCurrent(win);
        // }

        // ---- Show camera preview (with composite)
        // cv::imshow("camera", camera);
        int k = cv::waitKey(1);
        if (k == 27) break;

        if (k == 'l') {                      // list cameras
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
                if (openCapture(cap, next, 1280, 720, 60)) {
                    camIndex = next;
                #ifdef __linux__
                    std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
                #else
                    std::cerr << "Switched to [" << camIndex << "]\n";
                #endif
                }
            }
        }
        if (k == 'p') {                      // prev camera
            auto avail = scanCameras(12);
            if (!avail.empty()) {
                auto it = std::find(avail.begin(), avail.end(), camIndex);
                if (it == avail.begin() || it == avail.end()) it = avail.end();
                --it;
                int prev = *it;
                if (openCapture(cap, prev, 1280, 720, 60)) {
                    camIndex = prev;
                #ifdef __linux__
                    std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
                #else
                    std::cerr << "Switched to [" << camIndex << "]\n";
                #endif
                }
            }
        }
        if (k >= '0' && k <= '9') {          // direct select 0..9
            int idx = k - '0';
            if (openCapture(cap, idx, 1280, 720, 60)) {
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

        // ---- Draw goggles (textured quad of hudWarpTex, letterboxed to camera aspect)
        if (hudOverlay){
            glfwMakeContextCurrent(hudOverlay);
            int fbw=0, fbh=0; glfwGetFramebufferSize(hudOverlay,&fbw,&fbh);
            // glBindFramebuffer(GL_FRAMEBUFFER,0);
            // glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST);
            // glViewport(0,0,fbw,fbh);
            // glClearColor(0.f,0.f,0.f,1.f);
            // glClear(GL_COLOR_BUFFER_BIT);

            Viewport vp = letterbox(fbw, fbh, imgSize.width, imgSize.height);

            // Draw HUD into the overlay using inverse homography on the GPU
            glViewport(vp.x, vp.y, vp.w, vp.h);

            // Transparent clear is harmless here; leave as-is if you prefer black
            glClearColor(0,0,0,0);
            glClear(GL_COLOR_BUFFER_BIT);

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // If no pose this frame, show nothing (alpha=0). With a valid pose draw HUD at full alpha.
            if (haveH) {
                warp.draw(
                    /*hudTex=*/hudTex,
                    /*Hinv=*/HinvGL,         /*sliders*/SRTGL,
                    /*imgW=*/imgSize.width,  /*imgH=*/imgSize.height,
                    /*hudW=*/HUD_TEX_W,      /*hudH=*/HUD_TEX_H,
                    /*alpha=*/1.0f
                );

            } else {
                // Optional: leave cleared (transparent) when pose is invalid
            }

            // Present the overlay
            glfwSwapBuffers(hudOverlay);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);

            glfwMakeContextCurrent(win);
        }

        // ---- Dev window: draw HUD directly (letterboxed on canvas aspect)
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

        {
            // hand the composited frame to the UI thread for display
            std::lock_guard<std::mutex> lock(g_camMutex);
            g_camPreview = camera.clone();
        }

        // Allow GL window close to terminate the app
        if (glfwWindowShouldClose(win)) {
            g_running = false;
            break;
        }

        clk.end();
    }

    save_controls();
    g_running = false;
    if (uiThread.joinable()) uiThread.join();

    if (hudOverlay) {
        glfwMakeContextCurrent(hudOverlay);
        ovQuad.shutdown();
        glfwMakeContextCurrent(win);
        glfwDestroyWindow(hudOverlay);
    }
    if (camTex) glDeleteTextures(1,&camTex);
    if (hudTex)     glDeleteTextures(1,&hudTex);
    if (hudFBO)     glDeleteFramebuffers(1,&hudFBO);

    g_detRun.store(false, std::memory_order_release);
    if (detThread.joinable()) detThread.join();


    hud.shutdown();
    glfwDestroyWindow(win); glfwTerminate();
    return 0;
}
