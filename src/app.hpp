// src/app.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

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
        cap.set(cv::CAP_PROP_FPS, 30);
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
