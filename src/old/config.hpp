// src/config.hpp
#pragma once
#include <string>
#include <nlohmann/json.hpp>

struct AppConfig {
	int cam_width = 1280;
	int cam_height = 720;
	bool force_dshow = true;
	float hud_ndc_per_deg = 0.02f;
};

bool loadConfig(const std::string& path, AppConfig& cfg);
