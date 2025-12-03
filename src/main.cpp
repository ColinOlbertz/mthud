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

#include "aruco_tracker.hpp"
#include "hud_renderer.hpp"
#include "sensor.hpp"

using nlohmann::json;

static const float PI = 3.14159265358979323846f;

static const float HINV_ID[9] = { 1,0,0,  0,1,0,  0,0,1 };

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

// ===== GPU warp: HUD-only -> image space via inverse homography =====
struct GpuWarp {
    GLuint prog=0, vao=0, vbo=0;
    GLint uHud=-1, uHinv=-1, uHudSize=-1, uImgSize=-1, uAlpha=-1, uSRT = -1;

    static GLuint sh(GLenum t, const char* s){
        GLuint x=glCreateShader(t); glShaderSource(x,1,&s,nullptr); glCompileShader(x);
        GLint ok=0; glGetShaderiv(x,GL_COMPILE_STATUS,&ok);
        if(!ok){ char log[2048]; glGetShaderInfoLog(x,2048,nullptr,log); std::cerr<<"shader: "<<log<<"\n"; }
        return x;
    }
    static GLuint link(GLuint vs, GLuint fs){
        GLuint p=glCreateProgram(); glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
        GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
        if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::cerr<<"link: "<<log<<"\n"; }
        glDeleteShader(vs); glDeleteShader(fs); return p;
    }
    bool init(){
        const char* vs = R"(#version 330 core
            layout(location=0) in vec2 aPos; out vec2 vPos;
            void main(){ vPos = aPos*0.5 + 0.5; gl_Position = vec4(aPos,0,1); })";
        const char* fs = R"(#version 330 core
            in vec2 vPos; out vec4 FragColor;
            uniform sampler2D uHud;
            uniform mat3  uHinv;     // image px -> hud px
            uniform mat3  uSRT;      // HUD-pixel SRT from sliders
            uniform vec2  uHudSize;
            uniform vec2  uImgSize;
            uniform float uAlpha;
            void main(){
                // vPos is 0..1 in the current viewport
                // Flip Y: OpenCV image uses top-left origin, GL uses bottom-left
                vec2 p_img = vec2(vPos.x * uImgSize.x, (1.0 - vPos.y) * uImgSize.y);

                // Map image px -> HUD px, then apply slider SRT in HUD-pixel space
                vec3 q = uSRT * (uHinv * vec3(p_img, 1.0));

                // PERSPECTIVE DIVIDE (this was missing)
                float w = (q.z != 0.0) ? q.z : 1e-6;
                vec2 uv = (q.xy / w) / uHudSize;

                // Clip outside HUD canvas
                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                    FragColor = vec4(0.0);
                    return;
                }

                vec4 c = texture(uHud, uv);
                FragColor = vec4(c.rgb, c.a * uAlpha);
        })";
        GLuint v=sh(GL_VERTEX_SHADER,vs), f=sh(GL_FRAGMENT_SHADER,fs);
        prog = link(v,f); if(!prog) return false;
        uHud     = glGetUniformLocation(prog,"uHud");
        uHinv    = glGetUniformLocation(prog,"uHinv");
        uHudSize = glGetUniformLocation(prog,"uHudSize");
        uImgSize = glGetUniformLocation(prog,"uImgSize");
        uAlpha   = glGetUniformLocation(prog,"uAlpha");
        uSRT    = glGetUniformLocation(prog,"uSRT");

        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        const float tri[12] = { -1,-1,  1,-1,  1, 1,   -1,-1,  1,1,  -1,1 };
        glBufferData(GL_ARRAY_BUFFER,sizeof(tri),tri,GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        return true;
    }
    void draw(GLuint hudTex, const float Hinv[9], const float SRT[9], int imgW,int imgH, int hudW,int hudH, float alpha){
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, hudTex);
        glUniform1i(uHud,0);
        glUniformMatrix3fv(uHinv,1,GL_FALSE,Hinv);
        glUniformMatrix3fv(uSRT, 1,GL_FALSE,SRT);
        glUniform2f(uHudSize,(float)hudW,(float)hudH);
        glUniform2f(uImgSize,(float)imgW,(float)imgH);
        glUniform1f(uAlpha,alpha);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES,0,6);
        glBindVertexArray(0);
        glUseProgram(0);
    }
};

// ===== GPU composite: camera + HUD (HUD sampled via inverse homography) =====
struct GpuComposite {
    GLuint prog=0, vao=0, vbo=0;
    GLint uCam=-1, uHud=-1, uHinv=-1, uHudSize=-1, uImgSize=-1, uSRT = -1;

    static GLuint sh(GLenum t, const char* s){
        GLuint x=glCreateShader(t); glShaderSource(x,1,&s,nullptr); glCompileShader(x);
        GLint ok=0; glGetShaderiv(x,GL_COMPILE_STATUS,&ok);
        if(!ok){ char log[2048]; glGetShaderInfoLog(x,2048,nullptr,log); std::cerr<<"shader: "<<log<<"\n"; }
        return x;
    }
    static GLuint link(GLuint vs, GLuint fs){
        GLuint p=glCreateProgram(); glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
        GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
        if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::cerr<<"link: "<<log<<"\n"; }
        glDeleteShader(vs); glDeleteShader(fs); return p;
    }
    bool init(){
        const char* vs = R"(#version 330 core
            layout(location=0) in vec2 aPos; out vec2 vPos;
            void main(){ vPos = aPos*0.5 + 0.5; gl_Position = vec4(aPos,0,1); })";
        const char* fs = R"(#version 330 core
            in vec2 vPos; out vec4 FragColor;
            uniform sampler2D uCam;
            uniform sampler2D uHud;
            uniform mat3  uHinv;
            uniform mat3  uSRT;
            uniform vec2  uHudSize;
            uniform vec2  uImgSize;
            void main(){
                // Camera sample (flip Y so it’s upright)
                vec3 cam = texture(uCam, vec2(vPos.x, 1.0 - vPos.y)).rgb;

                // Map viewport sample -> image px (Y flipped)
                vec2 p_img = vec2(vPos.x * uImgSize.x, (1.0 - vPos.y) * uImgSize.y);

                // image px -> HUD px, then slider SRT
                vec3 q = uSRT * (uHinv * vec3(p_img, 1.0));

                // PERSPECTIVE DIVIDE (this was missing)
                float w = (q.z != 0.0) ? q.z : 1e-6;
                vec2 uv = (q.xy / w) / uHudSize;

                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                    FragColor = vec4(cam, 1.0);  // out of HUD canvas => camera only
                    return;
                }

                vec4 hud = texture(uHud, uv);
                FragColor = vec4(mix(cam, hud.rgb, hud.a), 1.0);
            })";
        GLuint v=sh(GL_VERTEX_SHADER,vs), f=sh(GL_FRAGMENT_SHADER,fs);
        prog = link(v,f); if(!prog) return false;

        uCam     = glGetUniformLocation(prog,"uCam");
        uHud     = glGetUniformLocation(prog,"uHud");
        uHinv    = glGetUniformLocation(prog,"uHinv");
        uHudSize = glGetUniformLocation(prog,"uHudSize");
        uImgSize = glGetUniformLocation(prog,"uImgSize");
        uSRT    = glGetUniformLocation(prog,"uSRT");

        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        const float tri[12] = { -1,-1,  1,-1,  1, 1,   -1,-1,  1,1,  -1,1 };
        glBufferData(GL_ARRAY_BUFFER,sizeof(tri),tri,GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        return true;
    }
    void draw(GLuint camTex, GLuint hudTex, const float Hinv[9], const float SRT[9],
              int imgW,int imgH, int hudW,int hudH){
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, camTex); glUniform1i(uCam,0);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, hudTex); glUniform1i(uHud,1);
        glUniformMatrix3fv(uHinv,1,GL_FALSE,Hinv);
        glUniformMatrix3fv(uSRT, 1,GL_FALSE,SRT);
        glUniform2f(uHudSize,(float)hudW,(float)hudH);
        glUniform2f(uImgSize,(float)imgW,(float)imgH);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES,0,6);
        glBindVertexArray(0);
        glUseProgram(0);
    }
};

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

// ---------- lightweight frame profiler ----------
struct FrameClock {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0, t1, t2, t3, t4, t5;

    // exponential moving averages (ms)
    double ema_cap   = 0.0;  // capture: grab+retrieve
    double ema_det   = 0.0;  // detection: aruco update
    double ema_hom   = 0.0;  // homography math
    double ema_hud   = 0.0;  // HUD FBO render (+ any readback you still have)
    double ema_ovl   = 0.0;  // overlay draw + swap
    double ema_frame = 0.0;  // whole frame

    int frames = 0;

    static double ms(clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    static void ema_update(double& acc, double v) {
        constexpr double A = 0.20; // smoothing
        if (acc == 0.0) acc = v;
        else acc = (1.0 - A) * acc + A * v;
    }

    inline void begin() { t0 = clock::now(); }
    inline void stamp_cap_done() { t1 = clock::now(); }
    inline void stamp_det_done() { t2 = clock::now(); }
    inline void stamp_hom_done() { t3 = clock::now(); }
    inline void stamp_hud_done() { t4 = clock::now(); }
    inline void end_and_log() {
        t5 = clock::now();

        const double cap   = ms(t0, t1);
        const double det   = ms(t1, t2);
        const double hom   = ms(t2, t3);
        const double hud   = ms(t3, t4);
        const double ovl   = ms(t4, t5);
        const double frame = ms(t0, t5);

        ema_update(ema_cap,   cap);
        ema_update(ema_det,   det);
        ema_update(ema_hom,   hom);
        ema_update(ema_hud,   hud);
        ema_update(ema_ovl,   ovl);
        ema_update(ema_frame, frame);

        if (++frames % 60 == 0) {
            const double fps = (ema_frame > 0.0) ? 1000.0 / ema_frame : 0.0;
            std::fprintf(stderr,
                "[%5d] FPS %5.1f | cap %6.2f ms  det %6.2f  hom %6.2f  hud %6.2f  ovl %6.2f | frame %6.2f ms\n",
                frames, fps, ema_cap, ema_det, ema_hom, ema_hud, ema_ovl, ema_frame);
        }
    }
};
static FrameClock g_clk;

// ---------- GL helpers: minimal textured quad (draws a GL texture fullscreen) ----------

// Trackbars can’t be negative; map [0..MAX] -> [-HALF..+HALF]
static inline int centered(int v, int max) { return v - max/2; }

static void buildSRT_px(float SRT_out[9],
                        int TB_off_x_px, int TB_off_y_px,
                        int TB_scale_pct, int TB_rot_deg,
                        int TB_pivot_tl,  // 1=top-left, 0=center
                        int hudW, int hudH)
{
    // offsets in pixels: 0..2000 -> [-1000..+1000]
    const double dx = (double)centered(TB_off_x_px, 2000);
    const double dy = (double)centered(TB_off_y_px, 2000);

    // scale: 1..400 % -> 0.01..4.00
    const double s  = std::max(0.01, TB_scale_pct / 100.0);

    // rotation in radians: 0..360 -> [-180..+180]
    const double rot = centered(TB_rot_deg, 360) * (CV_PI/180.0);

    const double cx = (TB_pivot_tl ? 0.0 : 0.5 * hudW);
    const double cy = (TB_pivot_tl ? 0.0 : 0.5 * hudH);

    const double c = std::cos(rot), sn = std::sin(rot);

    // Column-major 3x3 (to match GLSL when GL_FALSE for transpose):
    // SRT = T(cx+dx, cy+dy) * R(rot) * S(s) * T(-cx, -cy)
    const double T1[9] = {1,0,0,  0,1,0,  -cx,-cy,1};
    const double  R[9] = {c,sn,0, -sn,c,0,  0,0,1};
    const double  S[9] = {s,0,0,  0,s,0,  0,0,1};
    const double T2[9] = {1,0,0,  0,1,0,  cx+dx,cy+dy,1};

    auto mul = [](const double A[9], const double B[9], double C[9]){
        // column-major mat mul: C = A * B
        for(int j=0;j<3;++j) for(int i=0;i<3;++i){
            C[j*3+i] = A[0*3+i]*B[j*3+0] + A[1*3+i]*B[j*3+1] + A[2*3+i]*B[j*3+2];
        }
    };
    double RS[9], RS_T1[9], T2_RS_T1[9];
    mul(R, S, RS);
    mul(RS, T1, RS_T1);
    mul(T2, RS_T1, T2_RS_T1);

    for(int k=0;k<9;++k) SRT_out[k] = (float)T2_RS_T1[k];
}

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
            -1.f,-1.f,  0.f,0.f,
             1.f,-1.f,  1.f,0.f,
             1.f, 1.f,  1.f,1.f,
            -1.f,-1.f,  0.f,0.f,
             1.f, 1.f,  1.f,1.f,
            -1.f, 1.f,  0.f,1.f
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

// Simple 2D line renderer in image pixel space (for marker overlays)
struct LineOverlay {
    GLuint prog = 0, vao = 0, vbo = 0;
    GLint  uImgSize = -1, uColor = -1;
    GLFWwindow* vaoCtx = nullptr; // VAOs are not shared across contexts

    bool init() {
        const char* vs = R"(#version 330 core
            layout(location=0) in vec2 aPos; // image pixels (origin top-left)
            uniform vec2 uImgSize;
            void main(){
                vec2 uv = aPos / uImgSize;
                float x = uv.x * 2.0 - 1.0;
                float y = (1.0 - uv.y) * 2.0 - 1.0; // flip Y to GL NDC
                gl_Position = vec4(x, y, 0.0, 1.0);
            })";
        const char* fs = R"(#version 330 core
            out vec4 FragColor;
            uniform vec4 uColor;
            void main(){ FragColor = uColor; })";

        GLuint v = compileShader(GL_VERTEX_SHADER, vs);
        GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
        prog = (v && f) ? linkProgram(v, f) : 0;
        if (!prog) return false;

        uImgSize = glGetUniformLocation(prog, "uImgSize");
        uColor   = glGetUniformLocation(prog, "uColor");

        glGenBuffers(1, &vbo);
        vaoCtx = nullptr; // force creation in first draw per-context
        return true;
    }

    void draw(const std::vector<float>& xy, int imgW, int imgH, float r, float g, float b, float a, float lineWidth = 2.0f) {
        if (xy.empty() || !prog) return;

        // Recreate VAO when context changes (VAOs are not shared)
        GLFWwindow* curCtx = glfwGetCurrentContext();
        if (!vao || vaoCtx != curCtx) {
            if (vao) glDeleteVertexArrays(1, &vao);
            glGenVertexArrays(1, &vao);
            vaoCtx = curCtx;
        }

        glUseProgram(prog);
        glUniform2f(uImgSize, float(imgW), float(imgH));
        glUniform4f(uColor, r, g, b, a);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, xy.size() * sizeof(float), xy.data(), GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glLineWidth(lineWidth);
        glDrawArrays(GL_LINES, 0, (GLsizei)(xy.size() / 2));
        glBindVertexArray(0);
        glUseProgram(0);
    }

    void shutdown() {
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (prog) glDeleteProgram(prog);
        vbo = vao = prog = 0;
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

// Display / view calibration
static int  TB_view_hfov_deg       = 23;  // deg; 0 = use camera intrinsics

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
        {"view_hfov_deg",    TB_view_hfov_deg},
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
    get("view_hfov_deg",      TB_view_hfov_deg);
    get("gog_show_markers",    TB_gog_show_markers);
    get("gog_show_axes",       TB_gog_show_axes);
    get("ov_off_x_px",    TB_ov_off_x_px);
    get("ov_off_y_px",    TB_ov_off_y_px);
    get("ov_scale_pct",    TB_ov_scale_pct);
    get("ov_rot_deg",    TB_ov_rot_deg);
    get("ov_pitch_deg",    TB_ov_pitch_deg);
    get("ov_yaw_deg",    TB_ov_yaw_deg);
    get("ov_pivot_tl",    TB_ov_pivot_tl);
        
    if (const char* e = std::getenv("VIEW_HFOV_DEG")) {
        int v = std::atoi(e);
        if (v >= 0 && v <= 170) TB_view_hfov_deg = v;
    }

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
    tb("ViewHFOV_deg",    &TB_view_hfov_deg, 160);

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
    // --- Camera
    int camIndex = parseCamIndex(argc, argv, /*fallback*/0);

    auto envInt = [](const char* k, int def)->int {
        if (const char* e = std::getenv(k)) {
            int v = std::atoi(e);
            if (v > 0) return v;
        }
        return def;
    };
    auto envDouble = [](const char* k, double def)->double {
        if (const char* e = std::getenv(k)) {
            double v = std::atof(e);
            if (v > 0.0) return v;
        }
        return def;
    };

    int camW = 1280, camH = 720;
    double camFPS = 30.0;
    if (const char* res = std::getenv("CAM_RES")) {
        int w = 0, h = 0;
        if (std::sscanf(res, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) { camW = w; camH = h; }
    }
    camW  = envInt("CAM_WIDTH",  camW);
    camH  = envInt("CAM_HEIGHT", camH);
    camFPS= envDouble("CAM_FPS", camFPS);

    cv::setUseOptimized(true);
    cv::setNumThreads(6);      // often best for ArUco; test 2..6


    cv::VideoCapture cap;
    if (!openCapture(cap, camIndex, camW, camH, camFPS)) {
        auto avail = scanCameras(12);
        if (avail.empty()) {
            std::cerr << "No camera could be opened.\n";
            return 1;
        }
        camIndex = avail.front();
        if (!openCapture(cap, camIndex, camW, camH, camFPS)) {
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
    // --- Start UI thread (OpenCV camera + trackbars)
    std::thread uiThread(uiThreadFunc, imgSize);

    // --- Calibration
    cv::Mat K_cam, D_cam;
    double cam_fallback_hfov_deg = 70.0;
    if (const char* e = std::getenv("CAM_HFOV_DEG")) {
        double v = std::atof(e);
        if (v > 0.0 && v < 180.0) cam_fallback_hfov_deg = v;
    }
    if (!loadCalibrationJSON("calibration.json", K_cam, D_cam, imgSize)) {
        approxFOVIntrinsics(imgSize, cam_fallback_hfov_deg, K_cam, D_cam);
        std::cout << "No calibration.json. Using FOV approximation (" << cam_fallback_hfov_deg << " deg hfov).\n";
    }
    // Ensure 64F
    K_cam.convertTo(K_cam, CV_64F); D_cam.convertTo(D_cam, CV_64F);

    // Separate intrinsics for how the HUD is rendered in the glasses.
    cv::Mat K_view = K_cam.clone(), D_view = D_cam.clone();
    auto refreshViewIntrinsics = [&](int hfov_deg) {
        if (hfov_deg > 0) approxFOVIntrinsics(imgSize, double(hfov_deg), K_view, D_view);
        else { K_view = K_cam.clone(); D_view = D_cam.clone(); }
    };
    refreshViewIntrinsics(TB_view_hfov_deg);
    int prev_view_hfov_deg = TB_view_hfov_deg;

    // --- ArUco tracker
    ArucoTracking tracker;
    const int MX = 3, MY = 2;
    const float MARKER_LEN = 0.050f, GAP = 0.012f;
    tracker.setCameraIntrinsics(K_cam, D_cam);
    tracker.setGridBoard(MX, MY, MARKER_LEN, GAP);
    tracker.setAnchorMarkerCorner(0, 0);
    //tracker.setTemporalSmoothing(0.5);

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
    if (!warp.init()) std::cerr << "GpuWarp init failed\n";

    LineOverlay lineOverlay;
    if (!lineOverlay.init()) {
        std::cerr << "Line overlay init failed\n";
    }

    // --- Renderer
    HudRenderer hud; if (!hud.init()) { std::cerr << "hud.init failed\n"; return 1; }

    // Use the FBO texture for the dev preview instead of redrawing the HUD a second time
    OverlayTexQuad previewQuad;
    if (!previewQuad.init()) {
        std::cerr << "Preview quad init failed\n";
    } else {
        previewQuad.setSRT(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f);
    }

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

    // --- smoothing state
    double spd_kt_smooth = 0.0, alt_ft_smooth = 0.0;
    bool   first_samples = true;

    while (g_running) {
        g_clk.begin();

        if (!cap.grab()) {
            std::cerr << "cap.grab() failed\n";
            break;
        }
        if (!cap.retrieve(frame) || frame.empty()) {
            std::cerr << "cap.retrieve() empty\n";
            break;
        }

        g_clk.stamp_cap_done();

        tracker.update(frame);

        g_clk.stamp_det_done();

        // Live-adjust view intrinsics if the HFOV slider changed
        if (TB_view_hfov_deg != prev_view_hfov_deg) {
            prev_view_hfov_deg = TB_view_hfov_deg;
            refreshViewIntrinsics(prev_view_hfov_deg);
        }

        BoardPose pose = tracker.latest();

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

            // Project the (optionally rotated) HUD rectangle using the view intrinsics
            cv::projectPoints(bb_obj, pose.rvec, pose.tvec, K_view, D_view, bb_img);

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


        g_clk.stamp_hom_done();

        // --- Build camera viz base (for preview only)
        cv::Mat camera = frame; // reuse captured buffer to avoid an extra clone
        if (!tracker.ids().empty())
            cv::aruco::drawDetectedMarkers(camera, tracker.corners(), tracker.ids());

        std::vector<cv::Point2f> axis_ip_cam, axis_ip_view;
        if (pose.valid) {
            float A = MARKER_LEN * 0.7f;
            std::vector<cv::Point3f> axisPts = { {0,0,0},{A,0,0},{0,A,0},{0,0,A} };
            cv::projectPoints(axisPts, pose.rvec, pose.tvec, K_cam, D_cam, axis_ip_cam);
            cv::projectPoints(axisPts, pose.rvec, pose.tvec, K_view, D_view, axis_ip_view);
            auto lineAA = [&](int a, int b, cv::Scalar c) { cv::line(camera, axis_ip_cam[a], axis_ip_cam[b], c, 2, cv::LINE_AA); };
            lineAA(0, 1, { 0,  0,255 }); // X red
            lineAA(0, 2, { 0,255,  0 }); // Y green
            lineAA(0, 3, { 255,  0,  0 }); // Z blue
        }

        // Pre-build ArUco overlays for the glasses (image space)
        std::vector<float> gogMarkerLines, gogAxisX, gogAxisY, gogAxisZ;
        if (TB_gog_show_markers && pose.valid) {
            // Reproject marker corners using the board pose so they stay registered in the glasses
            const float len = MARKER_LEN;
            const float gap = GAP;
            auto addMarker = [&](int id) {
                if (id < 0 || id >= MX * MY) return;
                int row = id / MX, col = id % MX;
                float x0 = col * (len + gap);
                float y0 = row * (len + gap);
                std::vector<cv::Point3f> obj = {
                    {x0,      y0,      0},
                    {x0+len,  y0,      0},
                    {x0+len,  y0+len,  0},
                    {x0,      y0+len,  0}
                };
                std::vector<cv::Point2f> img;
                cv::projectPoints(obj, pose.rvec, pose.tvec, K_view, D_view, img);
                auto addEdge = [&](const cv::Point2f& a, const cv::Point2f& b) {
                    gogMarkerLines.push_back(a.x); gogMarkerLines.push_back(a.y);
                    gogMarkerLines.push_back(b.x); gogMarkerLines.push_back(b.y);
                };
                if (img.size() == 4) {
                    addEdge(img[0], img[1]);
                    addEdge(img[1], img[2]);
                    addEdge(img[2], img[3]);
                    addEdge(img[3], img[0]);
                }
            };
            for (int id : tracker.ids()) addMarker(id);
        }
        if (TB_gog_show_axes && axis_ip_view.size() == 4) {
            auto pushLine = [](std::vector<float>& v, const cv::Point2f& a, const cv::Point2f& b) {
                v.push_back(a.x); v.push_back(a.y);
                v.push_back(b.x); v.push_back(b.y);
            };
            pushLine(gogAxisX, axis_ip_view[0], axis_ip_view[1]); // X
            pushLine(gogAxisY, axis_ip_view[0], axis_ip_view[2]); // Y
            pushLine(gogAxisZ, axis_ip_view[0], axis_ip_view[3]); // Z
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
            if (canvasPxPerDegree_fromIntrinsics(Hh64, K_view, D_view,
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

        g_clk.stamp_hud_done();

        // GPU composite window removed; camera preview stays in the OpenCV UI

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
                if (openCapture(cap, next, camW, camH, camFPS)) {
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
                if (openCapture(cap, prev, camW, camH, camFPS)) {
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
            if (openCapture(cap, idx, camW, camH, camFPS)) {
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
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glDisable(GL_STENCIL_TEST);

            // If no pose this frame, show nothing (alpha=0). With a valid pose draw HUD at full alpha.
            if (haveH) {
                warp.draw(
                    /*hudTex=*/hudTex,
                    /*Hinv=*/HinvGL,         /*sliders*/SRTGL,
                    /*imgW=*/imgSize.width,  /*imgH=*/imgSize.height,
                    /*hudW=*/HUD_TEX_W,      /*hudH=*/HUD_TEX_H,
                    /*alpha=*/1.0f
                );
            }

            // Optional ArUco overlays in the glasses (image space)
            if (lineOverlay.prog) {
                if (!gogMarkerLines.empty())
                    lineOverlay.draw(gogMarkerLines, imgSize.width, imgSize.height, 0.1f, 1.0f, 0.1f, 0.9f, 2.0f);
                if (!gogAxisX.empty())
                    lineOverlay.draw(gogAxisX, imgSize.width, imgSize.height, 0.0f, 0.4f, 1.0f, 0.9f, 3.0f);
                if (!gogAxisY.empty())
                    lineOverlay.draw(gogAxisY, imgSize.width, imgSize.height, 0.0f, 1.0f, 0.0f, 0.9f, 3.0f);
                if (!gogAxisZ.empty())
                    lineOverlay.draw(gogAxisZ, imgSize.width, imgSize.height, 1.0f, 0.0f, 0.0f, 0.9f, 3.0f);
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
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        if (previewQuad.prog) {
            previewQuad.draw(hudTex);
        } else {
            hud.draw(hs); // fallback if the preview shader failed
        }
        glfwSwapBuffers(win);

        {
            // hand the composited frame to the UI thread for display (skip if UI is busy)
            std::unique_lock<std::mutex> lock(g_camMutex, std::try_to_lock);
            if (lock.owns_lock()) {
                g_camPreview = camera.clone();
            }
        }

        // Allow GL window close to terminate the app
        if (glfwWindowShouldClose(win)) {
            g_running = false;
            break;
        }

        g_clk.end_and_log();
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
    previewQuad.shutdown();
    lineOverlay.shutdown();
    if (hudTex)     glDeleteTextures(1,&hudTex);
    if (hudFBO)     glDeleteFramebuffers(1,&hudFBO);

    hud.shutdown();
    glfwDestroyWindow(win); glfwTerminate();
    sensor->stop();
    return 0;
}
