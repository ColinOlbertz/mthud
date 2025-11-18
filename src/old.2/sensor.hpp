#pragma once
#include <atomic>
#include <thread>
#include <memory>

struct SensorSample {
	double t_host{ 0 };

	// Attitude
	double bank_deg{ 0 };
	double pitch_deg{ 0 };
	double yaw_deg{ 0 };

	// Rates (optional)
	double gyro_z_dps{ 0 };

	// Kinematics
	double vel_x_ms{ 0 };
	double vel_y_ms{ 0 };
	double vel_z_ms{ 0 };

	// Navigation
	double lat_deg{ 0 };
	double lon_deg{ 0 };
	double alt_msl_m{ 0 };

	// Environment
	double baro_hpa{ 0 };

	// GNSS
	int    sats_used{ 0 };
};

struct ISensorSource {
	virtual ~ISensorSource() = default;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual SensorSample latest() const = 0;
};

// Demo sensor (still useful for testing)
struct DemoSensor : ISensorSource {
	std::atomic<bool> run{ false };
	std::atomic<SensorSample> cur{ SensorSample{} };
	std::thread th;
	void start() override;
	void stop() override;
	SensorSample latest() const override { return cur.load(std::memory_order_relaxed); }
	~DemoSensor() { stop(); }
};

// Factory: implemented in sensor_xsens.cpp. Returns native Xsens if USE_XSENS=1, else Demo.
std::unique_ptr<ISensorSource> makeSensor();
