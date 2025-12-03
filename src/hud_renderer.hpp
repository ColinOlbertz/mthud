#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <unordered_map>
#include <list>
#include <limits>

struct HudState {
    // aircraft state
    float bank_rad{ 0.0f };
    float hdg_deg{ 0.0f };
    float speed_kt{ 0.0f };     //Groundspeed right now
    float alt_ft{ 0.0f };

    // canvas size (design units, not window pixels)
    int   canvas_w{ 1920 };
    int   canvas_h{ 1440 };

    // pitch ladder layout in canvas pixels
    float pitch_px{ 0.0f };              // vertical center offset of horizon (px, +up)
    float pitch_px_per_deg{ 10.0f };     // px/deg mapping for ladder spacing
    float ladder_half_px{ 500.0f };      // half width of ladder bars in px
    float ladder_xoff_px{ 0.0f };        // lateral offset (+right)

    // top bank arc (canvas px)
    float bank_arc_centerY_px{ 400.0f };
    float bank_arc_radius_px{ 300.0f };

    // flight-path marker (angles relative to boresight; +x right, +y up)
    float fpm_dx_deg{ 0.0f };   // horizontal FPM offset (deg) = track - heading
    float fpm_dy_deg{ 0.0f };   // vertical FPM offset (deg)   = flight-path angle - pitch
    bool  fpm_valid{ false };   // draw only when velocity is valid

    // altitude source indicator
    bool  alt_is_gps{ true };     // true → GPS MSL; false → BARO
    float qnh_hpa{ 1013.25f };    // shown when BARO

    // text controls
    float text_scale{ 2.0f };   // multiplier for label size (e.g., 1.2 = +20%)
    int   flip_text_x{ 1 };     // 0/1 mirror horizontally
    int   flip_text_y{ 0 };     // 0/1 mirror vertically

};

class HudRenderer {
public:
    bool init();
    void shutdown();

    // set top-arc geometry in canvas pixels
    void setBankArcPx(float centerY_px, float radius_px);

    // main draw
    void draw(const HudState& s);

private:
    // programs
    GLuint prog_{ 0 };
    GLint  uR_{ -1 }, uT_{ -1 }, uP_{ -1 }, uCanvasScale_{ -1 }, uColor_{ -1 }, uAlpha_{ -1 };

    GLuint progTxt_{ 0 };
    GLint  uTxtR_{ -1 }, uTxtT_{ -1 }, uTxtP_{ -1 }, uTxtCanvasScale_{ -1 }, uTxtAlpha_{ -1 }, uTxtFlip_{ -1 };

    // geometry
    GLuint vaoLines_{ 0 }, vboLines_{ 0 };     // dynamic line list (x0,y0,x1,y1,...)
    GLuint vaoTri_{ 0 }, vboTri_{ 0 };       // triangle list (x,y pairs)

    // text
    GLuint vaoTxt_{ 0 }, vboTxt_{ 0 };
    struct TextCacheEntry {
        GLuint tex{ 0 };
        int width{ 0 };
        int height{ 0 };
        std::list<std::string>::iterator lruIt;
    };
    std::unordered_map<std::string, TextCacheEntry> textCache_;
    std::list<std::string> textCacheLru_;

    // prebuilt arcs
    GLuint vaoTopArc_{ 0 }, vboTopArc_{ 0 };
    GLuint vaoCompArc_{ 0 }, vboCompArc_{ 0 };

    // cached arc params
    float arcCenterY_px_{ 400.0f };
    float arcRadius_px_{ 300.0f };
    float lastTopArcCenterY_{ std::numeric_limits<float>::quiet_NaN() };
    float lastTopArcRadius_{ std::numeric_limits<float>::quiet_NaN() };
    float lastCompCenterY_{ std::numeric_limits<float>::quiet_NaN() };
    float lastCompRadius_{ std::numeric_limits<float>::quiet_NaN() };
    int   lastCompCH_{ -1 };

    // helpers
    static GLuint compile(GLenum t, const char* s);
    static GLuint link(GLuint vs, GLuint fs);

    void ensureTopArc_(int CW, int CH);
    void ensureCompassArc_(int CW, int CH);

    void updateLines_(const float* xy, int count_pairs);
    void updateTri_(const float* xy, int count_pairs);

    void setCommonUniforms_(int CW, int CH, float angle_rad, float Tx, float Ty, float Px, float Py);
    void setTextUniforms_(int CW, int CH, float angle_rad, float Tx, float Ty, float Px, float Py, float alpha);

    const TextCacheEntry& getTextEntry_(const std::string& text);
    void clearTextCache_();

    // draw a label whose quad is defined in canvas pixels, local origin at (x_px,y_px) relative to (centerX,centerY)
    void drawTextLabelPx_(const std::string& text,
        float x_px, float y_px, float h_px,
        float ang, float centerX, float centerY,
        float pivotX, float pivotY,
        int CW, int CH, float alpha = 0.95f,
        float flipX = 0.0f, float flipY = 0.0f);

    void fillRect_(int CW, int CH,
        float cx, float cy, float w, float h,
        float angle_rad,
        float r, float g, float b, float alpha);

    static constexpr size_t kMaxTextCache_ = 512;
};
