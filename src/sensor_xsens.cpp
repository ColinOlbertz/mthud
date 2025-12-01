#include "sensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
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
#include <xsensdeviceapi.h>
#include <xstypes/xsbaud.h>
#include <xstypes/xsbaudrate.h>

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
        // Construct Xsens control object
        control = XsControl::construct();
        if (!control) {
            std::cerr << "XsControl::construct failed\n";
            return;
        }

        // 1) Normal scan first, with logging
        XsPortInfoArray ports = XsScanner::scanPorts();
        std::cerr << "Xsens scan found " << ports.size() << " ports:\n";
        for (const auto& p : ports) {
            std::cerr << "  " << p.portName().toStdString()
                      << " id=" << p.deviceId().toString().toStdString()
                      << " baud=" << (int)p.baudrate()
                      << (p.deviceId().isMti() || p.deviceId().isMtig() ? " [MTi]" : "")
                      << "\n";
        }

        // Build a candidate list: prefer user override, then MTi/MTi-G, then other scanned ports, then common manual names.
        std::vector<XsPortInfo> candidates;
        if (const char* envPort = std::getenv("XSENS_PORT")) {
            if (*envPort) {
                XsPortInfo manual;
                manual.setPortName(XsString(envPort));
                // Allow env override of baud (numeric)
                if (const char* envBaud = std::getenv("XSENS_BAUD")) {
                    long b = std::strtol(envBaud, nullptr, 10);
                    if (b > 0) manual.setBaudrate(XsBaud::numericToRate(int(b)));
                }
                candidates.push_back(manual);
                std::cerr << "XSENS_PORT override: trying " << envPort;
                if (manual.baudrate() != XBR_Invalid)
                    std::cerr << " baud " << XsBaud::rateToNumeric(manual.baudrate());
                std::cerr << "\n";
            }
        }
        for (const auto& p : ports) {
            if (p.deviceId().isMti() || p.deviceId().isMtig()) {
                candidates.push_back(p);
            }
        }
        for (const auto& p : ports) {
            if (!(p.deviceId().isMti() || p.deviceId().isMtig())) {
                candidates.push_back(p);
            }
        }
        if (candidates.empty()) {
            std::cerr << "No MTi device found via scanner, trying common serial names\n";
            for (const char* name : { "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1" }) {
                XsPortInfo manual;
                manual.setPortName(XsString(name));
                manual.setBaudrate(XBR_Invalid);
                candidates.push_back(manual);
            }
        }

        // Try to open each candidate with its reported baud first, then a list of fallbacks.
        XsPortInfo chosen;
        XsBaudRate chosenBaud = XBR_Invalid;
        for (const auto& cand : candidates) {
            if (!cand.portName().empty()) {
                std::vector<XsBaudRate> baudChoices;
                auto toNumeric = [](XsBaudRate br) -> int {
                    int n = XsBaud::rateToNumeric(br);
                    return (n > 0) ? n : static_cast<int>(br); // fall back to enum value for logging
                };
                auto addBaud = [&baudChoices](XsBaudRate br) {
                    if (br != XBR_Invalid &&
                        std::find(baudChoices.begin(), baudChoices.end(), br) == baudChoices.end()) {
                        baudChoices.push_back(br);
                    }
                };

                addBaud(cand.baudrate()); // reported by scanner if available
                // Prefer user-configured 115200 first if not already present
                addBaud(XsBaud::numericToRate(115200));
                for (XsBaudRate b : { XBR_921k6, XBR_460k8, XBR_230k4, XBR_115k2, XBR_57k6, XBR_38k4 }) {
                    addBaud(b);
                }

                for (XsBaudRate br : baudChoices) {
                    std::cerr << "Trying Xsens port " << cand.portName().toStdString()
                              << " at baud " << toNumeric(br) << "\n";
                    if (control->openPort(cand.portName(), br)) {
                        chosen = cand;
                        chosenBaud = br;
                        goto port_opened;
                    }
                }
            }
        }

        std::cerr << "Failed to open any Xsens port (tried " << candidates.size() << " candidates)\n";
        return;

    port_opened:
        std::cerr << "Opened Xsens port " << chosen.portName().toStdString()
                  << " at baud " << (int)chosenBaud << "\n";

        // 4) Get device handle
        // In XDA, device(0) returns the first available main device,
        // so passing an invalid/default deviceId is allowed. :contentReference[oaicite:1]{index=1}
        XsDeviceId devId = chosen.deviceId();
        device = control->device(devId);
        if (!device) {
            std::cerr << "Failed to get device handle\n";
            return;
        }

        if (!device->gotoConfig()) {
            std::cerr << "gotoConfig failed\n";
            return;
        }

        // === Output configuration (your existing code) ===
        XsOutputConfigurationArray cfgs;
        cfgs.push_back(XsOutputConfiguration(XDI_EulerAngles, 400));
        cfgs.push_back(XsOutputConfiguration(XDI_VelocityXYZ, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_BaroPressure, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_LatLon, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_AltitudeMsl, 50));
        cfgs.push_back(XsOutputConfiguration(XDI_GnssSatInfo, 1));

        if (!device->setOutputConfiguration(cfgs)) {
            std::cerr << "setOutputConfiguration failed (try lower rates)\n";
            return;
        }

        device->addCallbackHandler(&cb);
        if (!device->gotoMeasurement()) {
            std::cerr << "gotoMeasurement failed\n";
            return;
        }

        while (run)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
};
#endif //USE_XSENS

//---------------- Factory ----------------
std::unique_ptr<ISensorSource> makeSensor() {
#if USE_XSENS
    return std::make_unique<XsensSensorImpl>();
#else
    return std::make_unique<DemoSensor>();
#endif
}
