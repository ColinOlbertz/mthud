#pragma once
#include <opencv2/opencv.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <array>
#include <atomic>
#include <mutex>
#include <chrono>
#include <glad/glad.h>
#include "aruco_tracker.hpp"

struct FrameClock {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0,t1,t2,t3,t4,t5;
  double ema_cap=0, ema_det=0, ema_hom=0, ema_hud=0, ema_ovl=0, ema_frame=0;
  int frames=0;
  static double ms(clock::time_point a, clock::time_point b){
    return std::chrono::duration<double, std::milli>(b-a).count();
  }
  static void ema(double& acc,double v){ constexpr double A=0.2; acc=(acc==0)?v:((1-A)*acc + A*v); }
  void begin(){ t0=clock::now(); } void cap(){ t1=clock::now(); } void det(){ t2=clock::now(); }
  void hom(){ t3=clock::now(); } void hud(){ t4=clock::now(); }
  void end(){
    t5=clock::now();
    double C=ms(t0,t1), D=ms(t1,t2), H=ms(t2,t3), U=ms(t3,t4), O=ms(t4,t5), F=ms(t0,t5);
    ema(ema_cap,C); ema(ema_det,D); ema(ema_hom,H); ema(ema_hud,U); ema(ema_ovl,O); ema(ema_frame,F);
    if(++frames%60==0){
      double fps=(ema_frame>0)?1000.0/ema_frame:0;
      std::fprintf(stderr,"[%5d] FPS %5.1f | cap %6.2f  det %6.2f  hom %6.2f  hud %6.2f  ovl %6.2f | frame %6.2f\n",
        frames,fps,ema_cap,ema_det,ema_hom,ema_hud,ema_ovl,ema_frame);
    }
  }
};

struct Viewport{ int x=0,y=0,w=0,h=0; };
inline Viewport letterbox(int W,int H,int srcW,int srcH){
  double sx = double(W)/srcW, sy = double(H)/srcH;
  double s = (sx<sy)? sx : sy;
  int vw = int(srcW*s), vh = int(srcH*s);
  return { (W-vw)/2, (H-vh)/2, vw, vh };
}

// Slider helpers
inline int centered(int v,int max){ return v - max/2; }

inline void mul3(const double A[9], const double B[9], double C[9]){
  for(int j=0;j<3;++j) for(int i=0;i<3;++i)
    C[j*3+i] = A[0*3+i]*B[j*3+0] + A[1*3+i]*B[j*3+1] + A[2*3+i]*B[j*3+2];
}

// Build SRT in HUD-pixel space (column-major for GLSL with transpose=GL_FALSE)
inline void buildSRT_px(float SRT[9],
                        int off_x_px,int off_y_px,int scale_pct,int rot_deg,int pivot_tl,
                        int hudW,int hudH){
  const double dx = centered(off_x_px,2000);
  const double dy = centered(off_y_px,2000);
  const double s  = std::max(0.01, scale_pct/100.0);
  const double r  = centered(rot_deg,360) * (M_PI/180.0);
  const double cx = (pivot_tl? 0.0 : 0.5*hudW);
  const double cy = (pivot_tl? 0.0 : 0.5*hudH);
  const double c = std::cos(r), sn = std::sin(r);
  const double T1[9]={1,0,0, 0,1,0, -cx,-cy,1};
  const double  R[9]={c,sn,0, -sn,c,0, 0,0,1};
  const double  S[9]={s,0,0, 0,s,0, 0,0,1};
  const double T2[9]={1,0,0, 0,1,0, cx+dx,cy+dy,1};
  double RS[9], RS_T1[9], SRTd[9];
  mul3(R,S,RS); mul3(RS,T1,RS_T1); mul3(T2,RS_T1,SRTd);
  for(int k=0;k<9;++k) SRT[k] = (float)SRTd[k];
}

// Minimal quad program with GPU homography warp & composite.
// NOTE: We keep shaders small and self-contained. You must call gladLoadGL before init().
struct GpuBase {
  GLuint prog=0, vao=0, vbo=0;
  static GLuint sh(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){ char log[2048]; glGetShaderInfoLog(s,2048,nullptr,log); std::fprintf(stderr,"Shader err: %s\n",log); }
    return s;
  }
  static GLuint link(GLuint vs, GLuint fs){
    GLuint p=glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs);
    glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,nullptr,log); std::fprintf(stderr,"Link err: %s\n",log); }
    return p;
  }
  void initQuad(){
    if(!vao){ glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
      glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
      const float tri[12] = {-1,-1, 1,-1, 1,1,  -1,-1, 1,1, -1,1};
      glBufferData(GL_ARRAY_BUFFER,sizeof(tri),tri,GL_STATIC_DRAW);
      glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
      glEnableVertexAttribArray(0);
      glBindVertexArray(0);
    }
  }
};

struct GpuWarp : GpuBase {
  GLint uHud=-1, uHinv=-1, uSRT=-1, uHudSize=-1, uImgSize=-1, uAlpha=-1;
  bool init(){
    const char* VS = R"(#version 330 core
      layout(location=0) in vec2 aPos; out vec2 vPos;
      void main(){ vPos = aPos*0.5 + 0.5; gl_Position = vec4(aPos,0,1); })";
    const char* FS = R"(#version 330 core
      in vec2 vPos; out vec4 FragColor;
      uniform sampler2D uHud;
      uniform mat3  uHinv;
      uniform mat3  uSRT;
      uniform vec2  uHudSize;
      uniform vec2  uImgSize;
      uniform float uAlpha;
      void main(){
        vec2 p_img = vec2(vPos.x * uImgSize.x, (1.0 - vPos.y) * uImgSize.y);
        vec3 q = uSRT * (uHinv * vec3(p_img, 1.0));
        float w = (q.z != 0.0) ? q.z : 1e-6;
        vec2 uv = (q.xy / w) / uHudSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
          FragColor = vec4(0.0); return;
        }
        vec4 hud = texture(uHud, uv);
        FragColor = vec4(hud.rgb, hud.a * uAlpha);
      })";
    prog = link(sh(GL_VERTEX_SHADER,VS), sh(GL_FRAGMENT_SHADER,FS));
    initQuad();
    uHud = glGetUniformLocation(prog,"uHud");
    uHinv= glGetUniformLocation(prog,"uHinv");
    uSRT = glGetUniformLocation(prog,"uSRT");
    uHudSize=glGetUniformLocation(prog,"uHudSize");
    uImgSize=glGetUniformLocation(prog,"uImgSize");
    uAlpha =glGetUniformLocation(prog,"uAlpha");
    return prog!=0;
  }
  void draw(GLuint hudTex, const float Hinv[9], const float SRT[9],
            int imgW,int imgH, int hudW,int hudH, float alpha){
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

struct GpuComposite : GpuBase {
  GLint uCam=-1, uHud=-1, uHinv=-1, uSRT=-1, uHudSize=-1, uImgSize=-1;
  bool init(){
    const char* VS = R"(#version 330 core
      layout(location=0) in vec2 aPos; out vec2 vPos;
      void main(){ vPos = aPos*0.5 + 0.5; gl_Position = vec4(aPos,0,1); })";
    const char* FS = R"(#version 330 core
      in vec2 vPos; out vec4 FragColor;
      uniform sampler2D uCam;
      uniform sampler2D uHud;
      uniform mat3  uHinv;
      uniform mat3  uSRT;
      uniform vec2  uHudSize;
      uniform vec2  uImgSize;
      void main(){
        vec3 cam = texture(uCam, vec2(vPos.x, 1.0 - vPos.y)).rgb;
        vec2 p_img = vec2(vPos.x * uImgSize.x, (1.0 - vPos.y) * uImgSize.y);
        vec3 q = uSRT * (uHinv * vec3(p_img, 1.0));
        float w = (q.z != 0.0) ? q.z : 1e-6;
        vec2 uv = (q.xy / w) / uHudSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
          FragColor = vec4(cam, 1.0); return;
        }
        vec4 hud = texture(uHud, uv);
        FragColor = vec4(mix(cam, hud.rgb, hud.a), 1.0);
      })";
    prog = link(sh(GL_VERTEX_SHADER,VS), sh(GL_FRAGMENT_SHADER,FS));
    initQuad();
    uCam = glGetUniformLocation(prog,"uCam");
    uHud = glGetUniformLocation(prog,"uHud");
    uHinv= glGetUniformLocation(prog,"uHinv");
    uSRT = glGetUniformLocation(prog,"uSRT");
    uHudSize=glGetUniformLocation(prog,"uHudSize");
    uImgSize=glGetUniformLocation(prog,"uImgSize");
    return prog!=0;
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

// Externs you define in main.cpp (or another TU)
struct BoardPose; // forward from your aruco_tracker
extern std::array<cv::Mat,3> g_bufGray;
extern std::atomic<int> g_wr;
extern std::atomic<bool> g_capRun, g_detRun;
extern std::mutex g_poseMtx;
extern BoardPose g_pose;

// robust camera open (index OR name, DSHOW→MSMF→FFMPEG)
auto openAnyCamera = [](cv::VideoCapture& cap,
    int idx,
    const std::vector<std::string>& names,
    int w, int h) -> bool
    {
        // Try DSHOW by name(s)
        for (auto& n : names) {
            if (cap.open("video=" + n, cv::CAP_DSHOW)) goto cfg;
        }
        // Try DSHOW by index
        if (idx >= 0 && cap.open(idx, cv::CAP_DSHOW)) goto cfg;

        // Try MSMF by name(s)
        for (auto& n : names) {
            if (cap.open("video=" + n, cv::CAP_MSMF)) goto cfg;
        }
        // Try MSMF by index
        if (idx >= 0 && cap.open(idx, cv::CAP_MSMF)) goto cfg;

        // Try FFMPEG network stream (DroidCam HTTP) — replace IP if you want this fallback
        if (cap.open("http://192.168.178.66:4747/video", cv::CAP_FFMPEG)) goto cfg;

        return false;

    cfg:
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_HW_ACCELERATION, (double)cv::VIDEO_ACCELERATION_NONE);
        cap.set(cv::CAP_PROP_FPS, 60);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        return true;
    };


inline GLuint makeCameraProgram() {
    const char* VS = R"(#version 330 core
  layout (location = 0) in vec2 aPos;
  layout (location = 1) in vec2 aTex;
  out vec2 TexCoord;
  void main(){ TexCoord = vec2(aTex.x, 1.0 - aTex.y); gl_Position = vec4(aPos,0,1);} )";
    const char* FS = R"(#version 330 core
  in vec2 TexCoord; out vec4 FragColor; uniform sampler2D tex;
  void main(){ FragColor = texture(tex, TexCoord); })";
    auto compile = [&](GLenum t, const char* s) { GLuint id = glCreateShader(t); glShaderSource(id, 1, &s, nullptr); glCompileShader(id); return id; };
    GLuint vs = compile(GL_VERTEX_SHADER, VS), fs = compile(GL_FRAGMENT_SHADER, FS);
    GLuint p = glCreateProgram(); glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs); return p;
}

inline void makeFullScreenQuad(GLuint& vao, GLuint& vbo, GLuint& ebo) {
    float verts[] = { -1.f,-1.f, 0.f,0.f,  1.f,-1.f, 1.f,0.f,  1.f,1.f, 1.f,1.f, -1.f,1.f, 0.f,1.f };
    unsigned int idx[] = { 0,1,2, 2,3,0 };
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); glEnableVertexAttribArray(1);
}

// SPSC ring buffer: main thread writes gray frames, detector thread reads
extern std::array<cv::Mat,3> g_grayBuf;
extern std::atomic<int>      g_wr;        // last written index, -1 at start
extern std::atomic<bool>     g_detRun;    // worker loop flag

// Latest pose produced by the detector thread
extern std::mutex            g_poseMtx;
extern BoardPose             g_pose;      // copyable struct from your tracker