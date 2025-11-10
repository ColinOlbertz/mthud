#include "hud_renderer.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static const float PI = 3.14159265358979323846f;

static inline float wrap360(float d) { float x = fmod(d, 360.0f); if (x < 0) x += 360.0f; return x; }

// ===== GL helpers =====
GLuint HudRenderer::compile(GLenum t, const char* s) { 
    GLuint id = glCreateShader(t); 
    glShaderSource(id, 1, &s, nullptr); 
    glCompileShader(id); 
    GLint ok = 0; 
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok); 
    if (!ok) { 
        char log[2048]; 
        glGetShaderInfoLog(id, 2048, nullptr, log); 
        std::cerr << "shader: " << log << "\n"; 
    } 
    return id; 
}

GLuint HudRenderer::link(GLuint vs, GLuint fs) { 
    GLuint p = glCreateProgram(); 
    glAttachShader(p, vs); 
    glAttachShader(p, fs); 
    glLinkProgram(p); 
    GLint ok = 0; 
    glGetProgramiv(p, GL_LINK_STATUS, &ok); 
    if (!ok) { 
        char log[2048]; 
        glGetProgramInfoLog(p, 2048, nullptr, log); 
        std::cerr << "link: " << log << "\n"; 
    } 
    glDeleteShader(vs); 
    glDeleteShader(fs); 
    return p; 
}

// ===== Shaders (canvas-space, origin at screen center, +y up) =====
static const char* HUD_VERT = R"(#version 330 core
layout(location=0) in vec2 aPos;      // canvas pixels, centered
uniform mat2  uR;                      // rotation
uniform vec2  uT;                      // translation in canvas px
uniform vec2  uP;                      // pivot in canvas px
uniform vec2  uCanvasScale;            // (2/CW, 2/CH)
void main(){
  vec2 p = uT + uR * (aPos - uP) + uP;   // pure 2D affine in the plane
  vec2 clip = p * uCanvasScale;          // center-origin canvas → clip
  gl_Position = vec4(clip, 0.0, 1.0);
})";

static const char* HUD_FRAG = R"(#version 330 core
out vec4 FragColor; uniform vec3 uColor; uniform float uAlpha;
void main(){ FragColor = vec4(uColor, uAlpha); })";

static const char* TXT_VERT = R"(#version 330 core
layout(location=0) in vec2 aPos;      // canvas px
layout(location=1) in vec2 aUV;
out vec2 vUV;
uniform mat2  uR; uniform vec2 uT; uniform vec2 uP; uniform vec2 uCanvasScale; uniform vec2 uFlip;
// uFlip.x/uFlip.y are 0 or 1 to mirror horizontally/vertically
void main(){
  vUV = mix(aUV, vec2(1.0) - aUV, uFlip);
  vec2 p = uT + uR * (aPos - uP) + uP;
  vec2 clip = p * uCanvasScale;
  gl_Position=vec4(clip,0,1);
})";

static const char* TXT_FRAG = R"(#version 330 core
in vec2 vUV; out vec4 FragColor; uniform sampler2D tex; uniform float uAlpha;
void main(){ vec4 c = texture(tex, vUV); FragColor = vec4(c.rgb, c.a * uAlpha); })";

bool HudRenderer::init() {
    // line program
    {
        GLuint vs = compile(GL_VERTEX_SHADER, HUD_VERT);
        GLuint fs = compile(GL_FRAGMENT_SHADER, HUD_FRAG);
        prog_ = link(vs, fs);
        uR_ = glGetUniformLocation(prog_, "uR");
        uT_ = glGetUniformLocation(prog_, "uT");
        uP_ = glGetUniformLocation(prog_, "uP");
        uCanvasScale_ = glGetUniformLocation(prog_, "uCanvasScale");
        uColor_ = glGetUniformLocation(prog_, "uColor");
        uAlpha_ = glGetUniformLocation(prog_, "uAlpha");
    }

    // VAOs for lines and triangles
    glGenVertexArrays(1, &vaoLines_); glGenBuffers(1, &vboLines_);
    glBindVertexArray(vaoLines_); glBindBuffer(GL_ARRAY_BUFFER, vboLines_);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glGenVertexArrays(1, &vaoTri_); glGenBuffers(1, &vboTri_);
    glBindVertexArray(vaoTri_); glBindBuffer(GL_ARRAY_BUFFER, vboTri_);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // text program
    {
        GLuint v = compile(GL_VERTEX_SHADER, TXT_VERT);
        GLuint f = compile(GL_FRAGMENT_SHADER, TXT_FRAG);
        progTxt_ = link(v, f);
        uTxtR_ = glGetUniformLocation(progTxt_, "uR");
        uTxtT_ = glGetUniformLocation(progTxt_, "uT");
        uTxtP_ = glGetUniformLocation(progTxt_, "uP");
        uTxtCanvasScale_ = glGetUniformLocation(progTxt_, "uCanvasScale");
        uTxtAlpha_ = glGetUniformLocation(progTxt_, "uAlpha");
        uTxtFlip_ = glGetUniformLocation(progTxt_, "uFlip");

        glGenVertexArrays(1, &vaoTxt_); glGenBuffers(1, &vboTxt_);
        glBindVertexArray(vaoTxt_); glBindBuffer(GL_ARRAY_BUFFER, vboTxt_);
        glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); glEnableVertexAttribArray(1);
        glGenTextures(1, &texTxt_);
        glBindTexture(GL_TEXTURE_2D, texTxt_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // arcs
    glGenVertexArrays(1, &vaoTopArc_); glGenBuffers(1, &vboTopArc_);
    glGenVertexArrays(1, &vaoCompArc_); glGenBuffers(1, &vboCompArc_);

    return true;
}

void HudRenderer::shutdown() { /* optional */ }

void HudRenderer::setBankArcPx(float centerY_px, float radius_px) { arcCenterY_px_ = centerY_px; arcRadius_px_ = std::max(1.0f, radius_px); }

void HudRenderer::ensureTopArc_(int CW, int CH) {
    std::vector<float> v; v.reserve(2 * 121);
    for (int d = -60; d <= 60; ++d) { float a = d * PI / 180.f; float x = arcRadius_px_ * sinf(a); float y = arcRadius_px_ * cosf(a) + arcCenterY_px_; v.push_back(x); v.push_back(y); }
    glBindVertexArray(vaoTopArc_); glBindBuffer(GL_ARRAY_BUFFER, vboTopArc_);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_DYNAMIC_DRAW);
}
void HudRenderer::ensureCompassArc_(int CW, int CH) {
    float compCenterY_px = -0.90f * (CH * 0.5f);
    float compR_px = 0.85f * arcRadius_px_;
    std::vector<float> v; v.reserve(2 * 360);
    for (int d = -180; d < 180; ++d) { float a = d * PI / 180.f; float x = compR_px * sinf(a); float y = compR_px * cosf(a) + compCenterY_px; v.push_back(x); v.push_back(y); }
    glBindVertexArray(vaoCompArc_); glBindBuffer(GL_ARRAY_BUFFER, vboCompArc_);
    glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_DYNAMIC_DRAW);
}

void HudRenderer::updateLines_(const float* xy, int count_pairs) { glBindVertexArray(vaoLines_); glBindBuffer(GL_ARRAY_BUFFER, vboLines_); glBufferData(GL_ARRAY_BUFFER, count_pairs * 2 * sizeof(float), xy, GL_DYNAMIC_DRAW); }
void HudRenderer::updateTri_(const float* xy, int count_pairs) { glBindVertexArray(vaoTri_); glBindBuffer(GL_ARRAY_BUFFER, vboTri_); glBufferData(GL_ARRAY_BUFFER, count_pairs * 2 * sizeof(float), xy, GL_DYNAMIC_DRAW); }

void HudRenderer::setCommonUniforms_(int CW, int CH, float angle, float Tx, float Ty, float Px, float Py) {
    float c = cosf(angle), s = sinf(angle); float R[4] = { c,-s,s,c };
    glUseProgram(prog_);
    glUniformMatrix2fv(uR_, 1, GL_FALSE, R);
    glUniform2f(uT_, Tx, Ty);
    glUniform2f(uP_, Px, Py);
    glUniform2f(uCanvasScale_, 2.0f / float(CW), 2.0f / float(CH));
}
void HudRenderer::setTextUniforms_(int CW, int CH, float angle, float Tx, float Ty, float Px, float Py, float alpha) {
    float c = cosf(angle), s = sinf(angle); float R[4] = { c,-s,s,c };
    glUseProgram(progTxt_);
    glUniformMatrix2fv(uTxtR_, 1, GL_FALSE, R);
    glUniform2f(uTxtT_, Tx, Ty);
    glUniform2f(uTxtP_, Px, Py);
    glUniform2f(uTxtCanvasScale_, 2.0f / float(CW), 2.0f / float(CH));
    glUniform1f(uTxtAlpha_, alpha);
}

void HudRenderer::drawTextLabelPx_(const std::string& text,
    float x_px, float y_px, float h_px,
    float ang, float centerX, float centerY,
    float pivotX, float pivotY,
    int CW, int CH, float alpha,
    float flipX, float flipY) {
    if (text.empty() || h_px <= 0) return;

    // rasterize label
    int baseline = 0; const double fontScale = 1.0; const int thickness = 2; const int Htex = 64; // raster height
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    int Wpx = std::max(8, sz.width + 8), Hpx = Htex + 8;
    cv::Mat rgba(Hpx, Wpx, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::putText(rgba, text, { 4, Htex }, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0, 255), thickness, cv::LINE_AA);

    glBindTexture(GL_TEXTURE_2D, texTxt_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba.cols, rgba.rows, 0, GL_BGRA, GL_UNSIGNED_BYTE, rgba.data);

    // scale bitmap to requested height in canvas px
    float w_px = h_px * float(rgba.cols) / float(rgba.rows);
    float x0 = x_px, y0 = y_px;                // local quad origin relative to (centerX,centerY)
    float x1 = x_px + w_px, y1 = y_px + h_px;

    float quad[16] = { x0,y0, 0,0,  x1,y0, 1,0,  x0,y1, 0,1,  x1,y1, 1,1 };
    glBindVertexArray(vaoTxt_); glBindBuffer(GL_ARRAY_BUFFER, vboTxt_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(quad), quad);

    setTextUniforms_(CW, CH, ang, centerX, centerY, pivotX, pivotY, alpha);
    glUniform2f(uTxtFlip_, flipX, flipY);
    glBindTexture(GL_TEXTURE_2D, texTxt_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

void HudRenderer::fillRect_(int CW, int CH,
    float cx, float cy, float w, float h,
    float angle_rad,
    float r, float g, float b, float alpha)
{
    // Local quad centered at (0,0). We'll place/rotate it via uniforms.
    const float x0 = -0.5f * w, x1 = 0.5f * w;
    const float y0 = -0.5f * h, y1 = 0.5f * h;

    const float tri[12] = {
        x0,y0,  x1,y0,  x1,y1,
        x0,y0,  x1,y1,  x0,y1
    };

    updateTri_(tri, 6);
    // Pure 2D affine: p = T + R * aPos  (pivot = 0, so rotation is around rect center)
    setCommonUniforms_(CW, CH, angle_rad, cx, cy, 0.0f, 0.0f);

    glUseProgram(prog_);
    glUniform3f(uColor_, r, g, b);
    glUniform1f(uAlpha_, alpha);   // use 1.0 for fully opaque “mask”
    glBindVertexArray(vaoTri_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void HudRenderer::draw(const HudState& s) {
    const int CW = std::max(1, s.canvas_w);
    const int CH = std::max(1, s.canvas_h);

    glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float roll = -s.bank_rad;
    const float cr = cosf(roll), sr = sinf(roll);
    auto rot2 = [&](float x, float y) { 
        return std::pair<float, float>(cr * x - sr * y, sr * x + cr * y); 
    };

    // regen arcs if changed
    ensureTopArc_(CW, CH);
    ensureCompassArc_(CW, CH);

    // colors
    auto setColor = [&](float r, float g, float b, float a) { glUseProgram(prog_); glUniform3f(uColor_, r, g, b); glUniform1f(uAlpha_, a); };

    // ===== Horizon (rotating) =====
    {
        float line[4] = { -CW * 0.5f, 0.0f,  +CW * 0.5f, 0.0f };
        updateLines_(line, 2);
        auto T_h = rot2(0.0f, s.pitch_px);
        setCommonUniforms_(CW, CH, roll, -T_h.first, T_h.second, 0.0f, 0.0f);
        setColor(0.0f, 1.0f, 0.0f, 0.95f);
        glBindVertexArray(vaoLines_); glLineWidth(2.0f); glDrawArrays(GL_LINES, 0, 2);
    }

    // ===== Pitch ladder (rotating) =====
    {
        const float gap_major = 24.0f;   // px between center gap halves for major marks
        const float gap_minor = 18.0f;
        const float lip_len_major = 28.0f;
        const float lip_len_minor = 20.0f;

        for (int deg = -90; deg <= 90; deg += 5) {
            if (deg == 0) continue;
            float offY = s.pitch_px + (deg) * s.pitch_px_per_deg; // +up
            if (offY < -CH || offY > CH) continue;
            bool major = (deg % 10) == 0;
            float useHalf = major ? s.ladder_half_px : s.ladder_half_px * 0.875f;
            float gap = major ? gap_major : gap_minor;
            float dir = (deg > 0) ? -1.f : +1.f;
            float lip = major ? lip_len_major : lip_len_minor;
            float offX = s.ladder_xoff_px;
			auto T_rung = rot2(offX, offY);

            std::vector<float> v; v.reserve(256);
            if (deg > 0) { // solid bars
                if (useHalf > gap * 0.5f) { v.insert(v.end(), { +gap * 0.5f,0, +useHalf,0 }); }
                if (-gap * 0.5f > -useHalf) { v.insert(v.end(), { -useHalf,0, -gap * 0.5f,0 }); }
            }
            else { // dashed bars
                const float dash = 40.0f, gapd = 20.0f, lip_gap = 12.0f;
                float endR = +useHalf - lip_gap; for (float x = +gap * 0.5f; x < endR - 1e-3f; ) { float x1 = std::min(x + dash, endR); v.insert(v.end(), { x,0, x1,0 }); x = x1 + gapd; }
                float startL = -useHalf + lip_gap; for (float x = startL; x < -gap * 0.5f - 1e-3f; ) { float x1 = std::min(x + dash, -gap * 0.5f); v.insert(v.end(), { x,0, x1,0 }); x = x1 + gapd; }
            }
            // lips
            v.insert(v.end(), { +useHalf,0, +useHalf,dir * lip,  -useHalf,0, -useHalf,dir * lip });

            updateLines_(v.data(), int(v.size() / 2));
            setCommonUniforms_(CW, CH, roll, -T_rung.first, T_rung.second, 0.0f, 0.0f);
            setColor(0.0f, 0.9f, 0.0f, major ? 0.95f : 0.85f);
            glBindVertexArray(vaoLines_); glLineWidth(2.0f); glDrawArrays(GL_LINES, 0, GLsizei(v.size() / 2));

            // labels at ends, upright
            const float hLabel = 32.0f * std::max(0.2f, s.text_scale);
            const float margin = 16.0f;
            std::string txt = (deg < 0 ? "-" : "") + std::to_string(std::abs(deg));
            // right label
            drawTextLabelPx_(txt, +useHalf + margin, -hLabel * 0.5f, hLabel, roll,
                -T_rung.first, T_rung.second, 0.0f, 0.0f, CW, CH, 0.95f, float(s.flip_text_x), float(s.flip_text_y));
            // left label
            drawTextLabelPx_(txt, -(useHalf + margin + hLabel * 0.7f * (float)txt.size()), -hLabel * 0.5f, hLabel, roll,
                -T_rung.first, T_rung.second, 0.0f, 0.0f, CW, CH, 0.95f, float(s.flip_text_x), float(s.flip_text_y));
        }
    }

    // ===== Top bank arc (static) =====
    {
        glUseProgram(prog_);
        setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        setColor(0.0f, 0.85f, 0.0f, 0.95f);
        glBindVertexArray(vaoTopArc_);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINE_STRIP, 0, 121);
		ensureTopArc_(CW, CH);

        {
            const int marksDeg[] = {0, 5, 10, 15, 30, 45, 60 };
            const float lenPx[] = { 36.f,12.f,12.f,24.f,36.f,24.f,36.f }; // inward length per mark
            std::vector<float> t; t.reserve(2 * 2 * 5 * 2);      // pairs of (x,y)

            auto addTick = [&](float deg, float len) {
                float a = deg * PI / 180.f;
                float px = arcRadius_px_ * sinf(a);
                float py = arcRadius_px_ * cosf(a) + arcCenterY_px_;
                float urx = sinf(a), ury = cosf(a);              // outward unit radial
                // draw inward from the arc
                t.push_back(px); 
                t.push_back(py);
                t.push_back(px + urx * len); t.push_back(py + ury * len);
            };

            for (int i = 0; i < 7; ++i) {
                addTick(+marksDeg[i], lenPx[i]);
                addTick(-marksDeg[i], lenPx[i]);
            }

            updateLines_(t.data(), int(t.size() / 2));
            setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            setColor(0.0f, 0.85f, 0.0f, 0.95f);
            glBindVertexArray(vaoLines_);
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, GLsizei(t.size() / 2));
        }

        // bank pointer triangle at current bank
        float a = std::max(-PI / 3.f, std::min(PI / 3.f, s.bank_rad));
        float px = arcRadius_px_ * sinf(a); float py = arcRadius_px_ * cosf(a) + arcCenterY_px_;
        float urx = sinf(a), ury = cosf(a); float utx = cosf(a), uty = -sinf(a);
        float base_back = 20.0f, half_w = 12.0f;
        float tip_x = px; float tip_y = py;
        float base_cx = px - urx * base_back; float base_cy = py - ury * base_back;
        float left_x = base_cx + utx * half_w; float left_y = base_cy + uty * half_w;
        float right_x = base_cx - utx * half_w; float right_y = base_cy - uty * half_w;
        float tri[6] = { tip_x,tip_y, left_x,left_y, right_x,right_y };
        updateTri_(tri, 3);
        setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        setColor(0.0f, 1.0f, 0.0f, 0.95f);
        glBindVertexArray(vaoTri_); glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // ===== Waterline symbol at center =====
    {
        float half = 0.14f * (CW * 0.5f); float tooth = 0.02f * (CH * 0.5f);
        std::vector<float> v; v.reserve(32);
        v.insert(v.end(), { -half,0 });
        v.insert(v.end(), { -0.5f * half,0 });
        v.insert(v.end(), { -0.25f * half,-2 * tooth });
        v.insert(v.end(), { 0.0f, 0.02f * (CH * 0.5f) });
        v.insert(v.end(), { +0.25f * half,-2 * tooth });
        v.insert(v.end(), { +0.5f * half,0 });
        v.insert(v.end(), { +half,0 });
        updateLines_(v.data(), int(v.size() / 2));
        setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        setColor(0.0f, 1.0f, 0.0f, 0.95f);
        glBindVertexArray(vaoLines_); glLineWidth(2.0f); glDrawArrays(GL_LINE_STRIP, 0, GLsizei(v.size() / 2));
    }

    // ===== Compass (bottom arc rotates; labels upright) =====
    {
        float compCenterY_px = -1.15f * (CH * 0.5f);
        float compR_px = 0.85f * arcRadius_px_;
        float hdg_rad = s.hdg_deg * (PI / 180.0f);

        // rotating card: arc geometry already built at origin; rotate around (0,compCenterY_px)
        glBindVertexArray(vaoCompArc_);
        setCommonUniforms_(CW, CH, hdg_rad, 0.0f, 0.0f, 0.0f, compCenterY_px);
        setColor(0.8f, 0.8f, 0.0f, 0.95f);
        glLineWidth(2.0f); glDrawArrays(GL_LINE_STRIP, 0, 360);

        // ticks on the rotating card
        std::vector<float> ticks; ticks.reserve(2 * 200);
        auto push = [&](float x0, float y0, float x1, float y1) { 
            ticks.push_back(x0); 
            ticks.push_back(y0); 
            ticks.push_back(x1); 
            ticks.push_back(y1); 
        };
        for (int d = -180; d < 180; d += 5) { 
            bool major = (d % 10) == 0; 
            float a = d * PI / 180.f; 
            float px = compR_px * sinf(a); 
            float py = compR_px * cosf(a) + compCenterY_px; 
            float urx = sinf(a), ury = cosf(a); 
            float len = major ? 34.f : 22.f; 
            push(px, py, px - urx * len, py - ury * len); 
        }
        updateLines_(ticks.data(), int(ticks.size() / 2));
        setCommonUniforms_(CW, CH, hdg_rad, 0.0f, 0.0f, 0.0f, compCenterY_px);
        setColor(0.0f, 0.85f, 0.0f, 0.95f);
        glBindVertexArray(vaoLines_); glLineWidth(2.0f); glDrawArrays(GL_LINES, 0, GLsizei(ticks.size() / 2));

        // labels ride the arc but stay upright: rotate positions on CPU
        auto rotP = [&](float x, float y, float ang) { 
            float c = cosf(ang), s = sinf(ang);
            float pivx = 0.0f, pivy = compCenterY_px; 
            float rx = c * (x - pivx) - s * (y - pivy) + pivx;
            float ry = s * (x - pivx) + c * (y - pivy) + pivy;
            return std::pair<float, float>(rx, ry); 
        };
        const float tick_len = 34.f;
        const float label_out = 15.f;   // how far outside the arc to place labels (px)
        const float label_h_px = 28.f * std::max(0.2f, s.text_scale);

        for (int d = -180; d < 180; d += 30) { 
            float a = d * PI / 180.f; 
            float px = compR_px * sinf(a); 
            float py = compR_px * cosf(a) + compCenterY_px; 
            float urx = sinf(a), ury = cosf(a); 
            float lx = px + urx * (tick_len + label_out);
            float ly = py + ury * (tick_len + label_out);

            auto p = rotP(lx, ly, -hdg_rad);

            char buf[8];
            std::snprintf(buf, sizeof(buf), "%03d", (int)wrap360((float)d));

            // center small quad around the anchor; flip flags come from HudState
            drawTextLabelPx_(buf, -18.f, -0.5f * label_h_px, label_h_px, 0.0f,
                p.first, p.second, 0.0f, 0.0f, CW, CH,
                0.95f, float(s.flip_text_x), float(s.flip_text_y));
        }

        // fixed caret at top of arc
        float a0 = 0.0f; 
        float px0 = compR_px * sinf(a0); 
        float py0 = compR_px * cosf(a0) + compCenterY_px; 
        float urx = sinf(a0), ury = cosf(a0); 
        float utx = cosf(a0), uty = -sinf(a0);
        float base_back = 18.f, half_w = 10.f, drop = 40.f;
        float tip_x = px0 + urx * 20.f - urx * drop, tip_y = py0 + ury * 20.f - ury * drop;
        float base_cx = px0 - urx * base_back - urx * drop, base_cy = py0 - ury * base_back - ury * drop;
        float left_x = base_cx + utx * half_w, left_y = base_cy + uty * half_w; 
        float right_x = base_cx - utx * half_w, right_y = base_cy - uty * half_w;
        float tri2[6] = { tip_x,tip_y, left_x,left_y, right_x,right_y };
        updateTri_(tri2, 3);
        setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        setColor(0.0f, 1.0f, 0.0f, 1.0f);
        glBindVertexArray(vaoTri_); glDrawArrays(GL_TRIANGLES, 0, 3);

        // heading readout below the caret (3 digits)
        {
            int hdg = 360 - (int)std::lround(wrap360(s.hdg_deg));
            char hbuf[8];
            std::snprintf(hbuf, sizeof(hbuf), "%03d", hdg);

            float h = 36.f * std::max(0.2f, s.text_scale);  // text height in px
            float w_est = 0.6f * h * 3.0f;                  // rough width to center

            // Anchor: a bit below the caret base
            float anchor_x = base_cx + 24.f;
            float anchor_y = base_cy + 16.f;                 // shift down a tad from base

            drawTextLabelPx_(hbuf,
                -0.5f * w_est,   // center horizontally
                -(h + 8.f),      // put text under the base
                h, 0.0f,
                anchor_x, anchor_y,
                0.0f, 0.0f, CW, CH,
                0.95f, float(s.flip_text_x), float(s.flip_text_y));
        }

        // ===== Vertical tapes: left = airspeed (kt), right = altitude (ft) =====
        {
            const int   CW = std::max(1, s.canvas_w);
            const int   CH = std::max(1, s.canvas_h);
            const float halfW = CW * 0.5f;

            // Tape x-positions (push as needed)
            const float xSpeed = -halfW * 0.78f;  // left tape
            const float xAlt = halfW * 0.78f;  // right tape

            // Visible vertical “window” (centered)
            const float winHalf = CH * 0.35f;     // pixels above/below center

            // Scale density: pixels per unit
            const float pxPerKt = 20.0f;           // 10 kt → 200 px
            const float pxPerFt = 2.00f;          // 100 ft → 200 px

            // Tick granularity
            const int spdMinor = 5, spdMajor = 10;   // kt
            const int altMinor = 50, altMajor = 100;  // ft

            auto drawTape = [&](float xBase,
                float value, float pxPerUnit,
                int minorStep, int majorStep,
                bool labelsOnRight,
                const char* fmt,
                float minorLen, float majorLen)
                {
                    // numeric span in view
                    const float unitsHalf = winHalf / pxPerUnit;
                    const int vMin = int(std::floor((value - unitsHalf) / minorStep)) * minorStep;
                    const int vMax = int(std::ceil((value + unitsHalf) / minorStep)) * minorStep;

                    std::vector<float> segs; segs.reserve(4 * ((vMax - vMin) / minorStep + 8));
                    auto isMajor = [&](int v) { return (v % majorStep) == 0; };
                    auto pushSeg = [&](float y, float len) {
                        float x0 = xBase, x1 = xBase + (labelsOnRight ? -len : +len);
                        segs.push_back(x0); segs.push_back(y);
                        segs.push_back(x1); segs.push_back(y);
                        };

                    // ticks + labels (keep upright; angle = 0)
                    for (int v = vMin; v <= vMax; v += minorStep)
                    {
                        float y = (v - value) * pxPerUnit; // +up
                        if (y < -winHalf - 8 || y > winHalf + 8) continue;

                        bool maj = isMajor(v);
                        pushSeg(y, maj ? majorLen : minorLen);

                        if (maj) {
                            char buf[16];
                            std::snprintf(buf, sizeof(buf), fmt, v);
                            float h = 28.f * std::max(0.2f, s.text_scale);
                            float w_est = 0.6f * h * std::strlen(buf);
                            float lx = labelsOnRight ? (xBase - majorLen - w_est - 8.f)
                                : (xBase + majorLen + 8.f);
                            drawTextLabelPx_(buf,
                                0.0f, -0.5f * h, h, 0.0f,
                                lx, y, 0.0f, 0.0f, CW, CH,
                                0.95f, float(s.flip_text_x), float(s.flip_text_y));
                        }
                    }

                    // submit ticks
                    if (!segs.empty()) {
                        updateLines_(segs.data(), int(segs.size() / 2));
                        setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                        glUseProgram(prog_); glUniform3f(uColor_, 0.0f, 1.0f, 0.0f); glUniform1f(uAlpha_, 0.95f);
                        glBindVertexArray(vaoLines_); glLineWidth(2.0f);
                        glDrawArrays(GL_LINES, 0, GLsizei(segs.size() / 2));
                    }

                    // spine
                    float spine[4] = { xBase, -winHalf, xBase, +winHalf };
                    updateLines_(spine, 2);
                    setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                    glUseProgram(prog_); glUniform3f(uColor_, 0.0f, 0.8f, 0.0f); glUniform1f(uAlpha_, 0.95f);
                    glBindVertexArray(vaoLines_); glDrawArrays(GL_LINES, 0, 2);

                    // center “window” + live value
                    const float boxW = 128.f, boxH = 48.f;
                    float bx = xBase + (labelsOnRight ? -(majorLen + 8.f + boxW * 0.5f)
                        : +(majorLen + 8.f + boxW * 0.5f));
                    float by = 0.0f;

                    fillRect_(CW, CH, bx, by, boxW, boxH, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

                    // rectangle outline (as 4 line segments)
                    float r[8] = { bx - boxW * 0.5f, by - boxH * 0.5f,
                                   bx + boxW * 0.5f, by - boxH * 0.5f,
                                   bx + boxW * 0.5f, by + boxH * 0.5f,
                                   bx - boxW * 0.5f, by + boxH * 0.5f };
                    float rect[16] = { r[0],r[1], r[2],r[3],
                                       r[2],r[3], r[4],r[5],
                                       r[4],r[5], r[6],r[7],
                                       r[6],r[7], r[0],r[1] };
                    updateLines_(rect, 8);
                    setCommonUniforms_(CW, CH, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                    glUseProgram(prog_); glUniform3f(uColor_, 0.0f, 1.0f, 0.0f); glUniform1f(uAlpha_, 0.95f);
                    glBindVertexArray(vaoLines_); glDrawArrays(GL_LINES, 0, 8);

                    // numeric readout inside the window
                    char vbuf[16];
                    std::snprintf(vbuf, sizeof(vbuf), fmt, int(std::round(value)));
                    float h = 36.f * std::max(0.2f, s.text_scale);
                    float w_est = 0.6f * h * std::strlen(vbuf);
                    drawTextLabelPx_(vbuf, -0.2f * w_est, -0.2f * h, h, 0.0f,
                        bx, by, 0.0f, 0.0f, CW, CH,
                        1.0f, float(s.flip_text_x), float(s.flip_text_y));
                };

            // SPEED (left): labels on the right of spine; 3 digits
            drawTape(xSpeed, s.speed_kt, pxPerKt, spdMinor, spdMajor,
                /*labelsOnRight=*/true, "%03d",
                /*minorLen=*/12.f, /*majorLen=*/22.f);

            // ALTITUDE (right): labels on the left of spine; 5 digits
            const float altMinorLen = 10.f, altMajorLen = 20.f;
            drawTape(xAlt, s.alt_ft, pxPerFt, altMinor, altMajor,
                /*labelsOnRight=*/false, "%05d",
                altMinorLen, altMajorLen);

            // --- Altitude source tag under the value window ---
            {
                // Match the constants used inside drawTape’s window
                const float boxW = 128.f, boxH = 48.f;
                const float margin = 10.f;

                // Window center for the right tape when labelsOnRight == false
                float bx = xAlt + (altMajorLen + 8.f + boxW * 0.5f);
                float by = 0.0f;

                // Small text
                float h1 = 40.f * std::max(0.2f, s.text_scale);
                auto drawCentered = [&](const std::string& txt, float yTop) {
                    float w_est = 0.6f * h1 * float(txt.size());
                    drawTextLabelPx_(txt,
                        -0.5f * w_est,           // center horizontally
                        yTop,                    // top-left y (px), +up
                        h1, 0.0f,                // height, angle
                        bx, by,                  // anchor at window center
                        0.0f, 0.0f, CW, CH,
                        0.95f, float(s.flip_text_x), float(s.flip_text_y));
                    };

                // Place first line just below the box; second line below that if BARO
                float yTop1 = -(winHalf + margin + h1);
                if (s.alt_is_gps) {
                    drawCentered("GPS", yTop1);
                }
                else {
                    drawCentered("BARO", yTop1);
                    char qbuf[16];
                    std::snprintf(qbuf, sizeof(qbuf), "%.0f", s.qnh_hpa);
                    drawCentered(qbuf, yTop1 - h1 - 8.f);
                }
            }

        }

        // ===== Flight-Path Marker (circle with wings & tail) =====
        if (s.fpm_valid) {
            const int CW = std::max(1, s.canvas_w);
            const int CH = std::max(1, s.canvas_h);

            // Convert degrees to pixels using your existing vertical scale
            float k_px_per_deg = s.pitch_px_per_deg;

            // Offsets in the *local boresight frame* (+x right, +y up)
            float x_local = s.fpm_dx_deg * k_px_per_deg;
            float y_local = s.fpm_dy_deg * k_px_per_deg;

            // We want the FPM wings level with the horizon. Reuse your roll counter-rotation.
            // Same convention you used for the horizon/ladder translations:
            const float roll = -s.bank_rad;
            const float cr = cosf(roll), sr = sinf(roll);
            auto rot2 = [&](float x, float y) { return std::pair<float, float>(cr * x - sr * y, sr * x + cr * y); };

            auto T = rot2(x_local, y_local);

            // Build the symbol in local coords (origin at FPM center)
            // Circle
            std::vector<float> segs;
            segs.reserve(4 * 64 + 12);
            auto add = [&](float x0, float y0, float x1, float y1) {
                segs.push_back(x0); segs.push_back(y0);
                segs.push_back(x1); segs.push_back(y1);
                };

            const float r = 18.0f; // circle radius (px)
            const float wing = 26.0f; // wing length from circle edge
            const float tail = 34.0f; // tail length below circle

            // circle as line segments
            for (int i = 0; i < 64; i++) {
                float a0 = (2.0f * PI / 64.0f) * i;
                float a1 = (2.0f * PI / 64.0f) * (i + 1);
                float x0 = r * std::cos(a0), y0 = r * std::sin(a0);
                float x1 = r * std::cos(a1), y1 = r * std::sin(a1);
                add(x0, y0, x1, y1);
            }
            // wings (level in local frame; roll stabilizes them)
            add(-(r + wing), 0.0f, -r, 0.0f);
            add(+(r + wing), 0.0f, r, 0.0f);
            // tail (downward from circle)
            add(0.0f, -r, 0.0f, -(r + tail));

            updateLines_(segs.data(), int(segs.size() / 2));
            setCommonUniforms_(CW, CH, roll, -T.first, T.second, 0.0f, 0.0f);
            glUseProgram(prog_); glUniform3f(uColor_, 0.0f, 1.0f, 0.0f); glUniform1f(uAlpha_, 1.0f);
            glBindVertexArray(vaoLines_);
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, GLsizei(segs.size() / 2));
        }


    }
}