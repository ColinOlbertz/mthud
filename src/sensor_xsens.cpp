#include "sensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
            s.t_host = t;
            s.bank_deg = 15.0 * std::sin(t * 0.9);
            s.pitch_deg = 7.0 * std::sin(t * 0.6 + 0.8);
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
#include <xstypes/xsoutputconfigurationarray.h>
#include <xstypes/xsportinfoarray.h>
#include <xstypes/xsdeviceid.h>
#include <xstypes/xspressure.h>
#else
#include <xsensdeviceapi.h>
#include <xstypes/xsbaud.h>
#include <xstypes/xsbaudrate.h>
#endif

#if defined(XSENS_USE_STATIC)
// Required by Xsens journaller macros used inside xspublic
Journaller* gJournal = nullptr;
#endif

// Keep Xsens types private to this TU
class MyXsCallback : public XsCallback {
public:
    std::atomic<SensorSample> latest{ SensorSample{} };

private:
    SensorSample last_{ };

public:
    void handlePacket(const XsDataPacket* packet) {
        SensorSample s = last_;
        s.t_host = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // --- Orientation ---
        if (packet->containsOrientation()) {
            XsEuler e = packet->orientationEuler();
            s.bank_deg = e.roll();
            s.pitch_deg = e.pitch();
            s.yaw_deg = e.yaw();
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
        if (device) { try { device->gotoConfig(); } catch (...) {} }
        if (control) {
            try { if (device) control->closePort(device->portInfo().portName().toStdString()); }
            catch (...) {}
            control->destruct(); control = nullptr; device = nullptr;
        }
    }
    SensorSample latest() const override { return cb.latest.load(std::memory_order_relaxed); }
    ~XsensSensorImpl() { stop(); }

private:
    void threadMain() {
        control = XsControl::construct();
        if (!control) { std::cerr << "XsControl::construct failed\n"; return; }

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

        if (chosen.portName().empty()) { std::cerr << "No MTi device found\n"; return; }
        if (chosenBaud == XBR_Invalid) {
            chosenBaud = chosen.baudrate();
            if (chosenBaud == XBR_Invalid) chosenBaud = XBR_115k2; // fallback
        }

        std::string portStr = chosen.portName().toStdString();
        std::cerr << "Opening " << portStr << " at " << XsBaud::rateToNumeric(chosenBaud) << "\n";
        if (!control->openPort(portStr, chosenBaud)) {
            std::cerr << "openPort failed: " << control->lastResultText().toStdString()
                      << " (" << (int)control->lastResult() << ")\n";
            return;
        }

        device = control->device(chosen.deviceId());
        if (!device) { device = control->device(XsDeviceId()); }
        if (!device) { std::cerr << "Failed to get device handle\n"; return; }

        if (!device->gotoConfig()) { std::cerr << "gotoConfig failed\n"; return; }

        // === Output configuration (your SDK ids) ===
        XsOutputConfigurationArray cfgs;
        // Use ids you listed: XDI_EulerAngles, XDI_VelocityXYZ, XDI_BaroPressure, XDI_LatLon, XDI_AltitudeMsl, XDI_GnssSatInfo
        // Start minimal: Euler at 200 Hz (tune if needed)
        cfgs.push_back(XsOutputConfiguration(XDI_EulerAngles, 400));
        cfgs.push_back(XsOutputConfiguration(XDI_VelocityXYZ, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_BaroPressure, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_LatLon, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_AltitudeMsl, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_GnssSatInfo, 1));

        if (!device->setOutputConfiguration(cfgs)) {
            std::cerr << "setOutputConfiguration failed (try lower rates)\n"; return;
        }

        device->addCallbackHandler(&cb);
        if (!device->gotoMeasurement()) { std::cerr << "gotoMeasurement failed\n"; return; }

        while (run) std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
