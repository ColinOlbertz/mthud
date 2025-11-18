#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <optional>

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
    constexpr int CANVAS_W = 1600;   // design canvas (4:3)
    constexpr int CANVAS_H = 1200;
}

// ---------- helpers ----------
// ---- Minimal textured-quad for overlay (draws a GL texture fullscreen) ----
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
    bool init() {
        const char* vs = R"(#version 330 core
            layout(location=0) in vec2 aPos;
            layout(location=1) in vec2 aUV;
            out vec2 vUV;
            void main(){ vUV=aUV; gl_Position = vec4(aPos,0.0,1.0); })";
        const char* fs = R"(#version 330 core
            in vec2 vUV; out vec4 FragColor;
            uniform sampler2D uTex;
            uniform vec2 uScale;         // scale around center
            uniform vec2 uOffset;        // additive UV offset
            void main(){
                vec2 uv = (vUV - vec2(0.5)) * uScale + vec2(0.5) + uOffset;
                uv = clamp(uv, vec2(0.0), vec2(1.0));
                FragColor = texture(uTex, uv);
            })";
        GLuint v = compileShader(GL_VERTEX_SHADER, vs);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
        prog = (v && f) ? linkProgram(v, f) : 0;
        if (!prog) return false;

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
        glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
        GLint locScale  = glGetUniformLocation(prog, "uScale");
        GLint locOffset = glGetUniformLocation(prog, "uOffset");
        glUniform2f(locScale,  1.0f, 1.0f);
        glUniform2f(locOffset, 0.0f, 0.0f);
        glUseProgram(0);
        return true;
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
    void setUV(float sx, float sy, float ox, float oy){
        glUseProgram(prog);
        GLint locScale  = glGetUniformLocation(prog, "uScale");
        GLint locOffset = glGetUniformLocation(prog, "uOffset");
        glUniform2f(locScale,  sx, sy);
        glUniform2f(locOffset, ox, oy);
        glUseProgram(0);
    }
    void shutdown() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (prog) glDeleteProgram(prog);
        prog = vao = vbo = 0;
    }
};

struct Viewport { int x, y, w, h; };
static Viewport letterbox(int fbw, int fbh, int srcw, int srch) {
    if (fbw <= 0 || fbh <= 0 || srcw <= 0 || srch <= 0) return {0,0,fbw,fbh};
    const double wndA = double(fbw) / double(fbh);
    const double srcA = double(srcw) / double(srch);
    Viewport v{};
    if (wndA >= srcA) { // pillarbox
        v.h = fbh;
        v.w = int(fbh * srcA + 0.5);
        v.x = (fbw - v.w) / 2; v.y = 0;
    } else {            // letterbox
        v.w = fbw;
        v.h = int(fbw / srcA + 0.5);
        v.x = 0; v.y = (fbh - v.h) / 2;
    }
    return v;
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

static GLFWwindow* createOverlayWindow(GLFWwindow* shareWith, int desiredMonitor = -1,
                                       bool transparent = true, bool borderless = true) {
    // transparency hint (needs a compositor on Linux)
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, transparent ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, borderless ? GLFW_FALSE : GLFW_TRUE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);     // keep on top in windowed mode
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);

    int monitorCount = 0;
    GLFWmonitor** mons = glfwGetMonitors(&monitorCount);
    GLFWmonitor* target = nullptr;

    if (monitorCount > 0) {
        if (desiredMonitor >= 0 && desiredMonitor < monitorCount) {
            target = mons[desiredMonitor];
        } else if (monitorCount >= 2) {
            // default to the second monitor if present (likely the BT-35E HDMI)
            target = mons[1];
        } else {
            target = mons[0];
        }
    }

    const GLFWvidmode* mode = glfwGetVideoMode(target);
    int W = mode ? mode->width : 1280;
    int H = mode ? mode->height : 720;

    // Fullscreen on target monitor for “pure overlay” feel
    GLFWwindow* w = glfwCreateWindow(W, H, "HUD Overlay", target, shareWith);
    if (!w) return nullptr;

    glfwMakeContextCurrent(w);
    glfwSwapInterval(0);       // reduce tearing on the glasses
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Clear to fully transparent if compositor + transparent FB available
    glClearColor(0.f, 0.f, 0.f, transparent ? 0.f : 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(w);

    return w;
}

static bool validateIntrinsics(cv::Mat& K, cv::Mat& D){
    if (K.empty() || K.rows != 3 || K.cols != 3) return false;
    if (K.type() != CV_64F) K.convertTo(K, CV_64F);
    if (D.empty()) { D = cv::Mat::zeros(1, 5, CV_64F); return true; }
    if (D.type() != CV_64F) D.convertTo(D, CV_64F);
    return true;
}

static void refreshAfterCameraChange(cv::VideoCapture& cap, cv::Size& imgSize, cv::Mat& K, cv::Mat& D, GLuint hudWarpTex, double fallback_hfov_deg = 70.0) {
    cv::Mat probe;
    if (!cap.read(probe) || probe.empty()) return;
    imgSize = probe.size();

    // Try to (re)load calibration; fall back to FOV approximation
    if (!loadCalibrationJSON("calibration.json", K, D, imgSize)) {
        approxFOVIntrinsics(imgSize, fallback_hfov_deg, K, D);
    }
    if (!validateIntrinsics(K, D)) {
        // last resort
        approxFOVIntrinsics(imgSize, fallback_hfov_deg, K, D);
        validateIntrinsics(K, D);
    }

    // Resize the overlay texture to match the new camera resolution
    glBindTexture(GL_TEXTURE_2D, hudWarpTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 imgSize.width, imgSize.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

static inline cv::Point2f applyH64(const cv::Mat& H64, float x, float y){
    const double a = H64.at<double>(0,0)*x + H64.at<double>(0,1)*y + H64.at<double>(0,2);
    const double b = H64.at<double>(1,0)*x + H64.at<double>(1,1)*y + H64.at<double>(1,2);
    const double c = H64.at<double>(2,0)*x + H64.at<double>(2,1)*y + H64.at<double>(2,2);
    const float inv = (c != 0.0) ? float(1.0/c) : 1.0f;
    return { float(a)*inv, float(b)*inv };
}

static bool canvasPxPerDegree_fromIntrinsics(const cv::Mat& Hh,
                                             const cv::Mat& K,
                                             const cv::Mat& D,
                                             float canvasY,
                                             float probe_dY_canvas,
                                             float& out_canvas_px_per_deg)
{
    if (Hh.empty() || K.empty()) return false;

    cv::Mat H64; Hh.convertTo(H64, CV_64F);

    // sample two canvas points separated by probe_dY_canvas
    const float x_fbo   = HUD_TEX_W * 0.5f;
    const float fbo_per_canvas = HUD_TEX_H / float(CANVAS_H);
    const float y0_fbo  = canvasY * fbo_per_canvas;
    const float y1_fbo  = (canvasY + probe_dY_canvas) * fbo_per_canvas;

    const cv::Point2f p0_img = applyH64(H64, x_fbo, y0_fbo);
    const cv::Point2f p1_img = applyH64(H64, x_fbo, y1_fbo);

    // undistort → normalized image plane, build unit rays
    std::vector<cv::Point2f> pts{p0_img, p1_img}, und;
    cv::undistortPoints(pts, und, K, D);
    cv::Vec3d r0(und[0].x, und[0].y, 1.0); r0 /= cv::norm(r0);
    cv::Vec3d r1(und[1].x, und[1].y, 1.0); r1 /= cv::norm(r1);

    double cosang = r0.dot(r1);
    cosang = std::clamp(cosang, -1.0, 1.0);
    const double dtheta_rad = std::acos(cosang);
    if (dtheta_rad < 1e-9) return false;

    // canvas pixels per degree at this location
    out_canvas_px_per_deg = probe_dY_canvas / float(dtheta_rad * 180.0 / CV_PI);
    return true;
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
static int TB_board_width_cm = 24;   // physical width of the HUD billboard in centimeters
static int TB_ov_scale_pct = 100;   // 50..200 %
static int TB_ov_off_x_px  = 1000;     // -500..+500 px
static int TB_ov_off_y_px  = 1000;
static int TB_ov_show_markers = 1;        // Aruco marker visibility toggle
static int TB_ov_show_axes = 1;        // Aruco axes visibility toggle
static int TB_ov_thicken_px = 0;   // 0..10 recommended; number of dilation pixels
static int  TB_auto_pitch_from_cam = 1;   // 0/1: auto px/deg from intrinsics
static int  TB_auto_center_y_px    = -1;  // -1 = canvas mid; else y on canvas to sample
static int  TB_probe_dY_canvas     = 100; // canvas pixels used for local derivative

static bool overlayReady = false;
static int  framesToSkip = 0;   // let compositor settle after (re)create


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
        {"board_width_cm",   TB_board_width_cm},
        {"ov_scale_pct",     TB_ov_scale_pct},
        {"ov_off_x_px ",     TB_ov_off_x_px},
        {"ov_off_y_px ",     TB_ov_off_y_px},
        {"Line thickness",   TB_ov_thicken_px}
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
    get("board_width_cm",TB_board_width_cm);
    get("ov_scale_pct",TB_ov_scale_pct);
    get("ov_off_x_px ",TB_ov_off_x_px);
    get("ov_off_y_px ",TB_ov_off_y_px);
    get("Line thickness",TB_ov_thicken_px);
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
    glfwSwapInterval(0);

    // --- Offscreen HUD render target (FBO + texture)
    GLuint hudFBO = 0, hudTex = 0;
    // Texture that will hold the ArUco-warped HUD (camera-size, updated each frame)
    GLuint hudWarpTex = 0;
    glGenTextures(1, &hudWarpTex);
    glBindTexture(GL_TEXTURE_2D, hudWarpTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgSize.width, imgSize.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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
    
    // --- Overlay window for goggles + its textured quad
    GLFWwindow* hudOverlay = nullptr;    // single overlay window
    int currentOverlayIdx = -1;

    hudOverlay = createOverlayWindow(
        /*shareWith*/ win,
        /*desiredMonitor*/ 1,   // change at runtime later if needed
        /*transparent*/ false,
        /*borderless*/  true
    );
    OverlayTexQuad ovQuad;
    if (hudOverlay){
        glfwMakeContextCurrent(hudOverlay);
        glfwSwapInterval(0);          // vsync on goggles
        if(!ovQuad.init()){
            std::cerr<<"Overlay TexQuad init failed; overlay disabled\n";
            glfwMakeContextCurrent(win);
            glfwDestroyWindow(hudOverlay);
            hudOverlay=nullptr;
        }else{
            glfwMakeContextCurrent(win);
            glfwSwapInterval(0);      // dev window: no vsync
        }
    }

    // Recreate overlay safely (handles missing/changed monitors, settles compositor)
    auto recreateOverlayAt = [&](int idx) {
        overlayReady = false;

        // Cleanly destroy previous overlay and its GL objects
        if (hudOverlay) {
            glfwMakeContextCurrent(hudOverlay);
            glFinish();
            ovQuad.shutdown();                 // <- quad belongs to overlay context
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
        hudOverlay = createOverlayWindow(
            /*shareWith*/ win,
            /*desiredMonitor*/ idx,
            /*transparent*/ false,
            /*borderless*/  true
        );
        if (!hudOverlay) {
            std::cerr << "createOverlayWindow failed\n";
            currentOverlayIdx = -1;
            glfwMakeContextCurrent(win);
            return;
        }

        // Init quad VAO/program in the OVERLAY context
        glfwMakeContextCurrent(hudOverlay);
        glfwSwapInterval(0);                   // vsync on goggles
        if (!ovQuad.init()) {
            std::cerr << "Overlay TexQuad init failed; overlay disabled\n";
            glfwMakeContextCurrent(win);
            glfwDestroyWindow(hudOverlay);
            hudOverlay = nullptr;
            currentOverlayIdx = -1;
            return;
        }

        // Dev window should not compete for vsync (reduces stutter)
        glfwMakeContextCurrent(win);
        glfwSwapInterval(0);

        currentOverlayIdx = idx;
        framesToSkip = 2;                       // let the compositor settle
        overlayReady = true;
    };


    // Choose initial monitor: prefer index 1 if it exists
    {
        int mc = 0; glfwGetMonitors(&mc);
        recreateOverlayAt(mc > 1 ? 1 : 0);
    }

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
    tb("Board width (cm)", &TB_board_width_cm, 200);  // 5..200 cm practical
    tb("OV scale %",   &TB_ov_scale_pct, 400);
    tb("OV off X (px)",&TB_ov_off_x_px,  200);
    tb("OV off Y (px)",&TB_ov_off_y_px,  200);
    tb("Aruco marker visibility (0/1)", &TB_ov_show_markers, 1);
    tb("Aruco axes visibility (0/1)", &TB_ov_show_axes, 1);
    tb("OV thicken (px)", &TB_ov_thicken_px, 10);
    tb("AUTO pitch from camera (0/1)", &TB_auto_pitch_from_cam, 1);
    tb("AUTO centerY px (-1=mid)",     &TB_auto_center_y_px,    CANVAS_H);
    tb("AUTO probe dY (px)",           &TB_probe_dY_canvas,     400);



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
            float A = MARKER_LEN * 0.057f;
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

        static bool lat_M=false, lat_X=false;
        int mkey = glfwGetKey(win, GLFW_KEY_M);
        if (mkey == GLFW_PRESS && !lat_M){
            TB_ov_show_markers = 1 - TB_ov_show_markers;
            std::cerr << "Goggles markers: " << (TB_ov_show_markers ? "ON\n":"OFF\n");
        }
        lat_M = (mkey == GLFW_PRESS);

        int xkey = glfwGetKey(win, GLFW_KEY_X);
        if (xkey == GLFW_PRESS && !lat_X){
            TB_ov_show_axes = 1 - TB_ov_show_axes;
            std::cerr << "Goggles axes: " << (TB_ov_show_axes ? "ON\n":"OFF\n");
        }
        lat_X = (xkey == GLFW_PRESS);



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
            buildBillboardPts4x3({ 0,0,0 }, TB_board_width_cm * 0.01f, bb_obj); // width in meters
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

            if (TB_auto_pitch_from_cam) {
                // where on the canvas to measure the local scale
                const float cy = (TB_auto_center_y_px < 0 || TB_auto_center_y_px > CANVAS_H)
                            ? float(CANVAS_H * 0.5f)
                            : float(TB_auto_center_y_px);

                float px_per_deg_canvas = 0.f;
                if (canvasPxPerDegree_fromIntrinsics(Hh, K, D,
                                                    cy,
                                                    std::max(10, TB_probe_dY_canvas),
                                                    px_per_deg_canvas))
                {
                    // clamp for sanity
                    px_per_deg_canvas = std::clamp(px_per_deg_canvas, 10.0f, 4000.0f);
                    hs.pitch_px_per_deg = px_per_deg_canvas;

                    // keep your existing trim/sign convention
                    const float trim_ndc   = TB_pitch_trim_x1000 / 1000.0f;
                    const float pitch_deg  = float(-s.pitch_deg);
                    hs.pitch_px = (-pitch_deg) * hs.pitch_px_per_deg + trim_ndc * (CANVAS_H * 0.5f);
                }
            }


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
            // Upload warped HUD (BGRA) to overlay texture
            // ---- Build overlay texture: warped HUD + ArUco debug (markers + axes), no camera ----

            // Start from the warped HUD (BGRA: HUD pixels have alpha, background is 0)
            cv::Mat overlayBGRA = hudWarpBGRA.clone();

            // Split to get B,G,R and A separately
            std::vector<cv::Mat> ch4; cv::split(overlayBGRA, ch4);  // ch4[0]=B, [1]=G, [2]=R, [3]=A
            cv::Mat overlayBGR; cv::merge(std::vector<cv::Mat>{ch4[0], ch4[1], ch4[2]}, overlayBGR);

        
            // 0) Helper: mark alpha where we draw
            auto stampAlpha = [&](const cv::Mat& bgrBefore, const cv::Mat& bgrAfter){
                cv::Mat diff; cv::absdiff(bgrAfter, bgrBefore, diff);
                std::vector<cv::Mat> dCh; cv::split(diff, dCh);
                cv::Mat any = dCh[0] | dCh[1] | dCh[2];
                ch4[3].setTo(255, any);
            };

            // 1) Draw detected markers manually (thicker than drawDetectedMarkers)
            if (TB_ov_show_markers && !tracker.corners().empty()) {
                cv::Mat before = overlayBGR.clone();

                const auto& cornersVec = tracker.corners();
                const auto& idsVec     = tracker.ids();
                const int thickness = 1;
                const int radius = 4;
                const cv::Scalar GREEN(0,255,0), BLACK(0,0,0), WHITE(255,255,255);

                for (size_t i=0; i<cornersVec.size(); ++i) {
                    const auto& c = cornersVec[i];  // vector<cv::Point2f> size 4
                    cv::Point p0 = c[0], p1 = c[1], p2 = c[2], p3 = c[3];

                    // Thicker green border with a thin black “shadow” to stand out
                    for (int k=0;k<4;++k) {
                        const cv::Point a = c[k], b = c[(k+1)&3];
                        cv::line(overlayBGR, a, b, BLACK, thickness+2, cv::LINE_AA);
                        cv::line(overlayBGR, a, b, GREEN, thickness,   cv::LINE_AA);
                    }
                    // Corner dots
                    cv::circle(overlayBGR, p0, radius+1, BLACK, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p1, radius+1, BLACK, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p2, radius+1, BLACK, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p3, radius+1, BLACK, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p0, radius,   WHITE, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p1, radius,   WHITE, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p2, radius,   WHITE, -1, cv::LINE_AA);
                    cv::circle(overlayBGR, p3, radius,   WHITE, -1, cv::LINE_AA);

                    // Label with ID near p0
                    if (i < idsVec.size()) {
                        std::string s = std::to_string(idsVec[i]);
                        cv::putText(overlayBGR, s, p0 + cv::Point(6,-6),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 3, cv::LINE_AA);
                        cv::putText(overlayBGR, s, p0 + cv::Point(6,-6),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv::LINE_AA);
                    }
                }
                stampAlpha(before, overlayBGR);
                

                // 2) Draw pose axes, larger and thicker
                {
                    // Scale by board width so it’s always visible
                    // (We’ll add TB_board_width_cm below; fallback to 24 cm if you haven’t wired it yet)
                    float boardW_m = std::max(0.05f, TB_board_width_cm * 0.01f);
                    std::vector<cv::Point3f> bb_obj;
                    buildBillboardPts4x3({ 0,0,0 }, boardW_m, bb_obj);
                    const float AXIS_FRAC = 0.40f;        // 40% of board width
                    float A_dbg = AXIS_FRAC * boardW_m;

                    std::vector<cv::Point3f> axisPts = { {0,0,0}, {A_dbg,0,0}, {0,A_dbg,0}, {0,0,A_dbg} };
                    std::vector<cv::Point2f> ip_dbg;
                    cv::projectPoints(axisPts, pose.rvec, pose.tvec, K, D, ip_dbg);

                    cv::Mat before = overlayBGR.clone();
                    auto lineAA = [&](int a, int b, const cv::Scalar& c) {
                        cv::line(overlayBGR, ip_dbg[a], ip_dbg[b], cv::Scalar(0,0,0), 3, cv::LINE_AA);
                        cv::line(overlayBGR, ip_dbg[a], ip_dbg[b], c,                2, cv::LINE_AA);
                    };
                    lineAA(0, 1, cv::Scalar(  0,   0, 255));  // X red (BGR)
                    lineAA(0, 2, cv::Scalar(  0, 255,   0));  // Y green
                    lineAA(0, 3, cv::Scalar(255,   0,   0));  // Z blue
                    stampAlpha(before, overlayBGR);
                }
            }
            // 3) Write updated BGR back into BGRA
            {
                std::vector<cv::Mat> bgrCh; cv::split(overlayBGR, bgrCh);
                ch4[0] = bgrCh[0]; ch4[1] = bgrCh[1]; ch4[2] = bgrCh[2];
                cv::merge(ch4, overlayBGRA);
            }
            // --- Thicken pass (goggles only) ---
            int th = std::max(0, TB_ov_thicken_px);
            cv::Mat uploadBGRA = overlayBGRA;               // or hudWarpBGRA if you don't build overlayBGRA
            if (th > 0) {
                // Use th iterations of a fast 3x3 dilation (good performance, predictable growth)
                // This fattens RGB and Alpha together so colors/AA expand consistently.
                cv::dilate(uploadBGRA, uploadBGRA, cv::Mat(), cv::Point(-1,-1), th);
            }

            // Upload combined overlay (BGRA) to the shared GL texture
            glBindTexture(GL_TEXTURE_2D, hudWarpTex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            imgSize.width, imgSize.height,
                            GL_BGRA, GL_UNSIGNED_BYTE, overlayBGRA.data);
            glFlush();

        }
        else {
            static std::vector<uint8_t> zero(imgSize.width*imgSize.height*4, 0);
            glBindTexture(GL_TEXTURE_2D, hudWarpTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgSize.width, imgSize.height, GL_RGBA, GL_UNSIGNED_BYTE, zero.data());
            glFlush();
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
                    glfwMakeContextCurrent(win);
                    refreshAfterCameraChange(cap, imgSize, K, D, hudWarpTex);   // updates imgSize, K/D, GL tex

                    // update ArUco tracker intrinsics as well
                    tracker.setCameraIntrinsics(K, D);

                    // re-init CPU mats to the new size (prevents stale sizes on blends)
                    hudWarpBGRA = cv::Mat(imgSize, CV_8UC4, cv::Scalar(0,0,0,0));
                    hudBGR      = cv::Mat(imgSize, CV_8UC3);
                    alpha       = cv::Mat(imgSize, CV_32F);
                    a3          = cv::Mat(imgSize, CV_32FC3);
                    camF        = cv::Mat(imgSize, CV_32FC3);
                    hudF        = cv::Mat(imgSize, CV_32FC3);

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
                    glfwMakeContextCurrent(win);
                    refreshAfterCameraChange(cap, imgSize, K, D, hudWarpTex);   // updates imgSize, K/D, GL tex

                    // update ArUco tracker intrinsics as well
                    tracker.setCameraIntrinsics(K, D);

                    // re-init CPU mats to the new size (prevents stale sizes on blends)
                    hudWarpBGRA = cv::Mat(imgSize, CV_8UC4, cv::Scalar(0,0,0,0));
                    hudBGR      = cv::Mat(imgSize, CV_8UC3);
                    alpha       = cv::Mat(imgSize, CV_32F);
                    a3          = cv::Mat(imgSize, CV_32FC3);
                    camF        = cv::Mat(imgSize, CV_32FC3);
                    hudF        = cv::Mat(imgSize, CV_32FC3);

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
                glfwMakeContextCurrent(win);
                refreshAfterCameraChange(cap, imgSize, K, D, hudWarpTex);   // updates imgSize, K/D, GL tex

                // update ArUco tracker intrinsics as well
                tracker.setCameraIntrinsics(K, D);

                // re-init CPU mats to the new size (prevents stale sizes on blends)
                hudWarpBGRA = cv::Mat(imgSize, CV_8UC4, cv::Scalar(0,0,0,0));
                hudBGR      = cv::Mat(imgSize, CV_8UC3);
                alpha       = cv::Mat(imgSize, CV_32F);
                a3          = cv::Mat(imgSize, CV_32FC3);
                camF        = cv::Mat(imgSize, CV_32FC3);
                hudF        = cv::Mat(imgSize, CV_32FC3);

        #ifdef __linux__
                std::cerr << "Switched to [" << camIndex << "] " << v4l2NameFor(camIndex) << "\n";
        #else
                std::cerr << "Switched to [" << camIndex << "]\n";
        #endif
            } else {
                std::cerr << "Failed to open camera " << idx << "\n";
            }
        }

        // ---- Overlay PASS: draw only the ArUco-warped HUD (no camera)
        if (hudOverlay){
            glfwMakeContextCurrent(hudOverlay);

            int fbw=0, fbh=0; glfwGetFramebufferSize(hudOverlay,&fbw,&fbh);
            glBindFramebuffer(GL_FRAMEBUFFER,0);
            glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST);
            glViewport(0,0,fbw,fbh);
            glClearColor(0.f,0.f,0.f,1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Match camera framing on the goggles
            Viewport vp = letterbox(fbw, fbh, imgSize.width, imgSize.height);
            glViewport(vp.x, vp.y, vp.w, vp.h);

            // Convert pixel offsets to normalized UV
            float s = std::max(0.05f, TB_ov_scale_pct / 100.0f);
            float ox = (TB_ov_off_x_px - 100) / float(std::max(1, imgSize.width));
            float oy = (TB_ov_off_y_px - 100) / float(std::max(1, imgSize.height));
            ovQuad.setUV(s, s, ox, oy);

            // Draw the warped HUD texture only
            ovQuad.draw(hudWarpTex);

            glfwSwapBuffers(hudOverlay);
            glfwMakeContextCurrent(win);   // back to dev window for the next pass
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
    if (hudOverlay) {
        glfwMakeContextCurrent(hudOverlay);
        ovQuad.shutdown();
        glfwMakeContextCurrent(win);
        glfwDestroyWindow(hudOverlay);
    }
    if (hudWarpTex) glDeleteTextures(1,&hudWarpTex);
    hud.shutdown();
    glfwDestroyWindow(win); glfwTerminate();
    sensor->stop();
    return 0;
}
