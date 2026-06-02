#include "sensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <cctype>
#include <string>
#include <vector>

static double nowSeconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// ---------------- DemoSensor ----------------
void DemoSensor::start() {
    run = true;
    th = std::thread([this]() {
        using clock = std::chrono::high_resolution_clock;
        auto t0 = clock::now();
        while (run) {
            auto now = clock::now();
            double t = std::chrono::duration<double>(now - t0).count();
            SensorSample s;
            s.t_host = nowSeconds();
            s.bank_deg = 15.0 * std::sin(t * 0.9);
            s.pitch_deg = 7.0 * std::sin(t * 0.6 + 0.8);
            s.gyro_x_dps = 0.0;
            s.gyro_y_dps = 0.0;
            s.gyro_z_dps = 0.0;
            cur.store(s, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        });
}
void DemoSensor::stop() { run = false; if (th.joinable()) th.join(); }

#if USE_XSENS
#if defined(XSENS_USE_STATIC)
#include <xscontroller/xscontrol_def.h>
#include <xscontroller/xsdevice_def.h>
#include <xscontroller/xsscanner.h>
#include <xscommon/journaller.h>
#include <xstypes/xsbaud.h>
#include <xstypes/xsbaudrate.h>
#include <xstypes/xsdatapacket.h>
#include <xstypes/xseuler.h>
#include <xstypes/xsfilterprofilearray.h>
#include <xstypes/xsoutputconfigurationarray.h>
#include <xstypes/xsportinfoarray.h>
#include <xstypes/xsdeviceid.h>
#include <xstypes/xspressure.h>
#else
#include <xsensdeviceapi.h>
#include <xstypes/xsbaud.h>
#include <xstypes/xsbaudrate.h>
#include <xstypes/xsfilterprofilearray.h>
#endif

#if defined(XSENS_USE_STATIC)
// Required by Xsens journaller macros used inside xspublic
Journaller* gJournal = nullptr;
#endif

static bool icontains(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return true;
    return std::search(hay.begin(), hay.end(),
                       needle.begin(), needle.end(),
                       [](char a, char b) {
                           return std::tolower(static_cast<unsigned char>(a)) ==
                                  std::tolower(static_cast<unsigned char>(b));
                       }) != hay.end();
}

static int envInt(const char* key, int fallback, int lo, int hi) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end == raw) return fallback;
    return std::clamp(int(v), lo, hi);
}

static double envDouble(const char* key, double fallback, double lo, double hi) {
    const char* raw = std::getenv(key);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    double v = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(v)) return fallback;
    return std::clamp(v, lo, hi);
}

static bool looksMagneticProfile(const XsFilterProfile& p) {
    std::string label = p.label() ? p.label() : "";
    std::string kind = p.kind() ? p.kind() : "";
    return p.type() == XFPK_Heading ||
           icontains(kind, "heading") ||
           icontains(label, "heading") ||
           icontains(label, "north") ||
           icontains(label, "mag") ||
           icontains(label, "compass");
}

static bool looksSixAxisProfile(const XsFilterProfile& p) {
    std::string label = p.label() ? p.label() : "";
    std::string kind = p.kind() ? p.kind() : "";
    return icontains(kind, "vru") ||
           icontains(label, "vru") ||
           icontains(label, "6d") ||
           icontains(label, "6-axis") ||
           icontains(label, "6 axis") ||
           icontains(label, "nomag") ||
           icontains(label, "no mag") ||
           icontains(label, "inertial") ||
           icontains(label, "imu");
}

static const XsFilterProfile* pickSixAxisProfile(const XsFilterProfileArray& profiles) {
    XsSize count = profiles.size();
    if (count == 0) return nullptr;
    const XsFilterProfile* fallback = &profiles[0];
    for (const auto& p : profiles) {
        if (looksSixAxisProfile(p) && !looksMagneticProfile(p)) {
            return &p;
        }
    }
    for (const auto& p : profiles) {
        if (!looksMagneticProfile(p)) {
            return &p;
        }
    }
    return fallback;
}

static const XsFilterProfile* pickHeadingProfile(const XsFilterProfileArray& profiles) {
    XsSize count = profiles.size();
    if (count == 0) return nullptr;
    const XsFilterProfile* fallback = &profiles[0];
    for (const auto& p : profiles) {
        if (looksMagneticProfile(p)) return &p;
    }
    return fallback;
}

// Keep Xsens types private to this TU
class MyXsCallback : public XsCallback {
public:
    std::atomic<SensorSample> latest{ SensorSample{} };
    std::atomic<double> last_packet_time{ 0.0 };

private:
    struct StartupCalibration {
        double duration_s{ 2.0 };
        double start_s{ 0.0 };
        int gyro_count{ 0 };
        int accel_count{ 0 };
        double gyro_sum[3]{ 0.0, 0.0, 0.0 };
        double accel_sum[3]{ 0.0, 0.0, 0.0 };
        double gyro_bias[3]{ 0.0, 0.0, 0.0 };
        double accel_bias[3]{ 0.0, 0.0, 0.0 };
        bool active{ true };
        bool complete{ false };

        void reset(double duration) {
            *this = StartupCalibration{};
            duration_s = duration;
            active = duration_s > 0.0;
            complete = !active;
        }
    };

    SensorSample last_{ };
    StartupCalibration calibration_;

public:
    void setStartupCalibrationDuration(double seconds) {
        calibration_.reset(seconds);
    }

    void handlePacket(const XsDataPacket* packet) {
        SensorSample s = last_;
        s.t_host = nowSeconds();
        last_packet_time.store(s.t_host, std::memory_order_relaxed);
        bool packetHasGyro = false;
        bool packetHasAccel = false;

        // --- Orientation ---
        if (packet->containsOrientation()) {
            XsEuler e = packet->orientationEuler();
            s.bank_deg = e.roll();
            s.pitch_deg = e.pitch();
            s.yaw_deg = e.yaw();
        }

        // --- Raw inertial data for 6-axis comparison/fusion (no magnetometer) ---
        if (packet->containsCalibratedGyroscopeData()) {
            XsVector g = packet->calibratedGyroscopeData(); // rad/s
            if (g.size() >= 3) {
                constexpr double RAD_TO_DEG = 57.2957795130823208768;
                s.gyro_x_dps = g[0] * RAD_TO_DEG;
                s.gyro_y_dps = g[1] * RAD_TO_DEG;
                s.gyro_z_dps = g[2] * RAD_TO_DEG;
                s.has_gyro = true;
                packetHasGyro = true;
            }
        }

        if (packet->containsFreeAcceleration()) {
            XsVector a = packet->freeAcceleration(); // m/s^2, gravity compensated by Xsens
            if (a.size() >= 3) {
                s.acc_x_ms2 = a[0];
                s.acc_y_ms2 = a[1];
                s.acc_z_ms2 = a[2];
                s.has_accel = true;
                packetHasAccel = true;
            }
        }
        else if (packet->containsCalibratedAcceleration()) {
            XsVector a = packet->calibratedAcceleration(); // fallback: includes gravity
            if (a.size() >= 3) {
                s.acc_x_ms2 = a[0];
                s.acc_y_ms2 = a[1];
                s.acc_z_ms2 = a[2];
                s.has_accel = true;
                packetHasAccel = true;
            }
        }

        if (calibration_.active && !calibration_.complete) {
            if (calibration_.start_s <= 0.0) {
                calibration_.start_s = s.t_host;
                std::cerr << "Xsens startup calibration: keep the unit still for "
                          << calibration_.duration_s << " s\n";
            }
            const double rawElapsed = s.t_host - calibration_.start_s;
            const double elapsed = rawElapsed > 0.0 ? rawElapsed : 0.0;
            s.xsens_calibrating = elapsed < calibration_.duration_s;
            s.xsens_calibration_progress = std::clamp(elapsed / calibration_.duration_s, 0.0, 1.0);

            if (s.xsens_calibrating) {
                if (packetHasGyro) {
                    calibration_.gyro_sum[0] += s.gyro_x_dps;
                    calibration_.gyro_sum[1] += s.gyro_y_dps;
                    calibration_.gyro_sum[2] += s.gyro_z_dps;
                    ++calibration_.gyro_count;
                }
                if (packetHasAccel) {
                    calibration_.accel_sum[0] += s.acc_x_ms2;
                    calibration_.accel_sum[1] += s.acc_y_ms2;
                    calibration_.accel_sum[2] += s.acc_z_ms2;
                    ++calibration_.accel_count;
                }
            }
            else {
                if (calibration_.gyro_count > 0) {
                    calibration_.gyro_bias[0] = calibration_.gyro_sum[0] / calibration_.gyro_count;
                    calibration_.gyro_bias[1] = calibration_.gyro_sum[1] / calibration_.gyro_count;
                    calibration_.gyro_bias[2] = calibration_.gyro_sum[2] / calibration_.gyro_count;
                }
                if (calibration_.accel_count > 0) {
                    calibration_.accel_bias[0] = calibration_.accel_sum[0] / calibration_.accel_count;
                    calibration_.accel_bias[1] = calibration_.accel_sum[1] / calibration_.accel_count;
                    calibration_.accel_bias[2] = calibration_.accel_sum[2] / calibration_.accel_count;
                }
                calibration_.complete = true;
                std::cerr << "Xsens startup calibration complete: gyro bias dps ["
                          << calibration_.gyro_bias[0] << ", "
                          << calibration_.gyro_bias[1] << ", "
                          << calibration_.gyro_bias[2] << "], accel bias m/s2 ["
                          << calibration_.accel_bias[0] << ", "
                          << calibration_.accel_bias[1] << ", "
                          << calibration_.accel_bias[2] << "]\n";
            }
        }

        if (calibration_.complete) {
            if (s.has_gyro) {
                s.gyro_x_dps -= calibration_.gyro_bias[0];
                s.gyro_y_dps -= calibration_.gyro_bias[1];
                s.gyro_z_dps -= calibration_.gyro_bias[2];
            }
            if (s.has_accel) {
                s.acc_x_ms2 -= calibration_.accel_bias[0];
                s.acc_y_ms2 -= calibration_.accel_bias[1];
                s.acc_z_ms2 -= calibration_.accel_bias[2];
            }
            s.xsens_calibrating = false;
            s.xsens_calibration_progress = 1.0;
        }

        // --- Velocity ---
        if (packet->containsVelocity()) {
            XsVector v = packet->velocity(); 
            if (v.size() >= 3) {
                s.vel_x_ms = v[0];
                s.vel_y_ms = v[1];
                s.vel_z_ms = v[2];
            }
        }
        
        // --- Lat/Lon ---
        if (packet->containsLatitudeLongitude()) {
        XsVector ll = packet->latitudeLongitude();
        s.lat_deg = ll[0]; s.lon_deg = ll[1];
        }

        // --- Altitude (Msl/Baro) ---
        if (packet->containsAltitudeMsl()) {
        s.alt_msl_m = packet->altitudeMsl();
        }

        if (packet->containsPressure()) {
            XsPressure p = packet->pressure();  // struct, not double
            double valuePa = 0.0;
            // Many XsPressure structs have a .m_pressure field in Pascals
            try { valuePa = p.m_pressure; }
            catch (...) { valuePa = 0.0; }
            s.baro_hpa = valuePa / 100.0;  // convert to hPa
        }


        // --- GNSS Satellite Info ---
        auto setSatCountFromSatInfo = [&s](const auto& info) {
            // try container 'm_satellites' (vector/array-like)
            if constexpr (requires { info.m_satellites; info.m_satellites.size(); }) {
                s.sats_used = static_cast<int>(info.m_satellites.size());
            }
            else if constexpr (requires { info.satellites; info.satellites.size(); }) {
                s.sats_used = static_cast<int>(info.satellites.size());
            }
            };

        if (packet->containsRawGnssSatInfo()) {
            auto info = packet->rawGnssSatInfo();
            setSatCountFromSatInfo(info);
        }

        last_ = s;
        latest.store(s, std::memory_order_relaxed);
    }

    void onDataAvailable(XsDevice*, const XsDataPacket* packet) override { handlePacket(packet); }
    void onLiveDataAvailable(XsDevice*, const XsDataPacket* packet) override { handlePacket(packet); }
};

class XsensSensorImpl : public ISensorSource {
    std::atomic<bool> run{ false };
    std::thread th;

    XsControl* control = nullptr;
    XsDevice* device = nullptr;
    MyXsCallback cb;

public:
    void start() override {
        run = true;
        th = std::thread([this]() { this->threadMain(); });
    }
    void stop() override {
        run = false;
        if (th.joinable()) th.join();
        closeDevice_();
        if (control) {
            control->destruct(); control = nullptr;
        }
    }
    SensorSample latest() const override { return cb.latest.load(std::memory_order_relaxed); }
    ~XsensSensorImpl() { stop(); }

private:
    void closeDevice_() {
        if (device) {
            try { device->removeCallbackHandler(&cb); } catch (...) {}
            try { device->gotoConfig(); } catch (...) {}
        }
        if (control && device) {
            try { control->closePort(device->portInfo().portName().toStdString()); }
            catch (...) {}
        }
        device = nullptr;
    }

    void threadMain() {
        control = XsControl::construct();
        if (!control) { std::cerr << "XsControl::construct failed\n"; return; }

        auto setFilterProfile = [&]() -> bool {
            if (!device) return false;

            if (const char* envProf = std::getenv("XSENS_FILTER_PROFILE")) {
                if (*envProf) {
                    XsString prof(envProf);
                    if (device->setOnboardFilterProfile(prof)) {
                        std::cerr << "Using XSENS_FILTER_PROFILE=" << envProf << "\n";
                        return true;
                    }
                    std::cerr << "XSENS_FILTER_PROFILE=" << envProf << " not accepted, auto-selecting\n";
                }
            }

            XsFilterProfileArray profiles = device->availableOnboardFilterProfiles();
            const bool preferSixAxis = envInt("XSENS_USE_MAGNETOMETER", 1, 0, 1) == 0;
            const XsFilterProfile* hp = preferSixAxis ? pickSixAxisProfile(profiles)
                                                      : pickHeadingProfile(profiles);
            if (!hp) {
                std::cerr << "No filter profiles reported; keeping device default\n";
                return false;
            }

            if (device->setOnboardFilterProfile(int(hp->type()))) {
                std::cerr << "Selected " << (preferSixAxis ? "6-axis/non-magnetic" : "heading")
                          << " filter profile: " << (hp->label() ? hp->label() : "(unnamed)")
                          << " (type " << int(hp->type()) << ")\n";
                return true;
            }
            std::cerr << "setOnboardFilterProfile failed for type " << int(hp->type()) << "\n";
            return false;
        };

        auto openAndConfigure = [&]() -> bool {
            // Build port/baud selection close to the official example
            XsPortInfo chosen;
            XsBaudRate chosenBaud = XBR_Invalid;

            // Env override: XSENS_PORT + optional XSENS_BAUD
            if (const char* envPort = std::getenv("XSENS_PORT")) {
                if (*envPort) {
                    std::string portStr(envPort);
                    if (const char* envBaud = std::getenv("XSENS_BAUD")) {
                        long b = std::strtol(envBaud, nullptr, 10);
                        if (b > 0) chosenBaud = XsBaud::numericToRate(int(b));
                    }
                    chosen.setPortName(XsString(portStr.c_str()));
                    chosen.setBaudrate(chosenBaud);
                    std::cerr << "XSENS_PORT override: " << portStr;
                    if (chosenBaud != XBR_Invalid) std::cerr << " baud " << XsBaud::rateToNumeric(chosenBaud);
                    std::cerr << "\n";
                }
            }

            // If no override, use scanner result (first MTi)
            if (chosen.portName().empty()) {
                XsPortInfoArray ports = XsScanner::scanPorts();
                std::cerr << "Xsens scan found " << ports.size() << " ports\n";
                for (const auto& p : ports) {
                    std::cerr << "  " << p.portName().toStdString()
                              << " id=" << p.deviceId().toString().toStdString()
                              << " baud=" << (int)p.baudrate()
                              << (p.deviceId().isMti() || p.deviceId().isMtig() ? " [MTi]" : "")
                              << "\n";
                    if (chosen.portName().empty() && (p.deviceId().isMti() || p.deviceId().isMtig())) {
                        chosen = p;
                    }
                }
            }

            if (chosen.portName().empty()) { std::cerr << "No MTi device found\n"; return false; }
            if (chosenBaud == XBR_Invalid) {
                chosenBaud = chosen.baudrate();
                if (chosenBaud == XBR_Invalid) chosenBaud = XBR_115k2; // fallback
            }

            std::string portStr = chosen.portName().toStdString();
            std::cerr << "Opening " << portStr << " at " << XsBaud::rateToNumeric(chosenBaud) << "\n";
            if (!control->openPort(portStr, chosenBaud)) {
                std::cerr << "openPort failed: " << control->lastResultText().toStdString()
                          << " (" << (int)control->lastResult() << ")\n";
                return false;
            }

            device = control->device(chosen.deviceId());
            if (!device) { device = control->device(XsDeviceId()); }
            if (!device) { std::cerr << "Failed to get device handle\n"; return false; }

            if (!device->gotoConfig()) { std::cerr << "gotoConfig failed\n"; return false; }

            setFilterProfile();

            // === Output configuration (your SDK ids) ===
            XsOutputConfigurationArray cfgs;
            const int imuRate = envInt("XSENS_IMU_RATE", 200, 25, 400);
            const int eulerRate = envInt("XSENS_EULER_RATE", 400, 25, 400);
            const auto imuRateHz = static_cast<uint16_t>(imuRate);
            const auto eulerRateHz = static_cast<uint16_t>(eulerRate);
            auto fillOutputConfig = [&](XsDataIdentifier accelId) {
                cfgs.clear();
                cfgs.push_back(XsOutputConfiguration(XDI_EulerAngles, eulerRateHz));
                cfgs.push_back(XsOutputConfiguration(XDI_RateOfTurn, imuRateHz));
                cfgs.push_back(XsOutputConfiguration(accelId, imuRateHz));
                cfgs.push_back(XsOutputConfiguration(XDI_VelocityXYZ, 50));
                cfgs.push_back(XsOutputConfiguration(XDI_BaroPressure, 50));
                cfgs.push_back(XsOutputConfiguration(XDI_LatLon, 50));
                cfgs.push_back(XsOutputConfiguration(XDI_AltitudeMsl, 50));
                cfgs.push_back(XsOutputConfiguration(XDI_GnssSatInfo, 1));
            };

            fillOutputConfig(XDI_FreeAcceleration);
            if (!device->setOutputConfiguration(cfgs)) {
                std::cerr << "setOutputConfiguration with free acceleration failed; retrying calibrated acceleration\n";
                fillOutputConfig(XDI_Acceleration);
                if (!device->setOutputConfiguration(cfgs)) {
                    std::cerr << "setOutputConfiguration failed (try lower rates)\n";
                    return false;
                }
            }

            const double startupCalibrationSec = envDouble("XSENS_STARTUP_CALIBRATION_SEC", 2.0, 0.0, 10.0);
            if (startupCalibrationSec > 0.0) {
                const auto seconds = static_cast<uint16_t>(std::clamp<int>(
                    int(std::ceil(startupCalibrationSec)), 1, 65535));
                if (device->setNoRotation(seconds)) {
                    std::cerr << "Requested Xsens no-rotation gyro bias update for "
                              << seconds << " s\n";
                }
                else {
                    std::cerr << "Xsens no-rotation command was not accepted; using host-side bias calibration only\n";
                }
            }

            device->addCallbackHandler(&cb);
            cb.setStartupCalibrationDuration(startupCalibrationSec);
            if (!device->gotoMeasurement()) { std::cerr << "gotoMeasurement failed\n"; return false; }
            return true;
        };

        constexpr double kDataTimeoutSec = 1.0;
        constexpr double kInitialTimeoutSec = 2.0;

        while (run) {
            closeDevice_();
            cb.last_packet_time.store(0.0, std::memory_order_relaxed);
            if (!openAndConfigure()) {
                closeDevice_();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            const double start_s = nowSeconds();
            while (run) {
                const double last_s = cb.last_packet_time.load(std::memory_order_relaxed);
                const double now_s = nowSeconds();
                if ((last_s > 0.0 && (now_s - last_s) > kDataTimeoutSec) ||
                    (last_s <= 0.0 && (now_s - start_s) > kInitialTimeoutSec)) {
                    std::cerr << "Xsens data timeout; reconnecting\n";
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        closeDevice_();
    }
};
#endif // USE_XSENS

// ---------------- Factory ----------------
std::unique_ptr<ISensorSource> makeSensor() {
#if USE_XSENS
    return std::make_unique<XsensSensorImpl>();
#else
    return std::make_unique<DemoSensor>();
#endif
}
