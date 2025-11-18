// src/config.cpp
#include "config.hpp"
#include <fstream>
using json = nlohmann::json;

bool loadConfig(const std::string& path, AppConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    json j; f >> j;
    if (j.contains("cam_width"))  cfg.cam_width = j["cam_width"];
    if (j.contains("cam_height")) cfg.cam_height = j["cam_height"];
    if (j.contains("force_dshow")) cfg.force_dshow = j["force_dshow"];
    if (j.contains("hud_ndc_per_deg")) cfg.hud_ndc_per_deg = j["hud_ndc_per_deg"];
    return true;
}
