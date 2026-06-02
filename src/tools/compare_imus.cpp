#include "../sensor.hpp"
#include "../moverio_sensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static constexpr double PI = 3.14159265358979323846;

static double nowSeconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static cv::Matx33d rotationFromQuaternion(const OrientationSample& q) {
    const double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return cv::Matx33d(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy));
}

static cv::Matx33d mapMoverioToCameraAxes(const cv::Matx33d& rotation) {
    const cv::Matx33d moverioToCamera(
        1.0,  0.0,  0.0,
        0.0, -1.0,  0.0,
        0.0,  0.0, -1.0);
    return moverioToCamera * rotation * moverioToCamera.t();
}

static cv::Matx33d rotationFromEulerDegrees(const SensorSample& sample) {
    const double roll = sample.bank_deg * PI / 180.0;
    const double pitch = sample.pitch_deg * PI / 180.0;
    const double yaw = sample.yaw_deg * PI / 180.0;
    const double cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    return cv::Matx33d(
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr);
}

static cv::Vec3d eulerZYXDegrees(const cv::Matx33d& rotation) {
    const double pitch = std::asin(std::clamp(-rotation(2, 0), -1.0, 1.0));
    const double cp = std::cos(pitch);
    double roll = 0.0;
    double yaw = 0.0;
    if (std::abs(cp) > 1e-6) {
        roll = std::atan2(rotation(2, 1), rotation(2, 2));
        yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    }
    else {
        roll = std::atan2(-rotation(1, 2), rotation(1, 1));
        yaw = 0.0;
    }
    return cv::Vec3d(roll * 180.0 / PI, pitch * 180.0 / PI, yaw * 180.0 / PI);
}

static double rotationAngleDegrees(const cv::Matx33d& rotation) {
    const double trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    const double c = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(c) * 180.0 / PI;
}

static cv::Point projectTriadPoint(const cv::Point& center, double scale, const cv::Vec3d& p) {
    const double x = p[0] + 0.35 * p[2];
    const double y = p[1] - 0.25 * p[2];
    return { int(std::lround(center.x + scale * x)), int(std::lround(center.y - scale * y)) };
}

static void drawTriad(cv::Mat& canvas, const cv::Point& center, const cv::Matx33d& delta,
                      const std::string& label, const cv::Scalar& textColor) {
    const double scale = 95.0;
    const cv::Point origin = projectTriadPoint(center, scale, {0.0, 0.0, 0.0});
    const cv::Vec3d axes[3] = {
        delta * cv::Vec3d(1.0, 0.0, 0.0),
        delta * cv::Vec3d(0.0, 1.0, 0.0),
        delta * cv::Vec3d(0.0, 0.0, 1.0)
    };
    const cv::Scalar colors[3] = {
        {40, 80, 255},   // X red-ish
        {80, 220, 80},   // Y green
        {255, 120, 60}   // Z blue-ish
    };
    const char* names[3] = {"X", "Y", "Z"};
    for (int i = 0; i < 3; ++i) {
        const cv::Point end = projectTriadPoint(center, scale, axes[i]);
        cv::arrowedLine(canvas, origin, end, colors[i], 3, cv::LINE_AA, 0, 0.12);
        cv::putText(canvas, names[i], end + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    colors[i], 2, cv::LINE_AA);
    }
    cv::circle(canvas, origin, 4, {230, 230, 230}, cv::FILLED, cv::LINE_AA);
    cv::putText(canvas, label, center + cv::Point(-115, 145), cv::FONT_HERSHEY_SIMPLEX,
                0.7, textColor, 2, cv::LINE_AA);
}

static void drawLine(cv::Mat& canvas, int& y, const std::string& text,
                     const cv::Scalar& color = {230, 230, 230}) {
    cv::putText(canvas, text, {24, y}, cv::FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv::LINE_AA);
    y += 28;
}

struct TimedRotation {
    double t{0.0};
    cv::Matx33d r{cv::Matx33d::eye()};
};

class RotationHistory {
public:
    void clear() {
        samples_.clear();
        last_t_ = 0.0;
    }

    void add(double t, const cv::Matx33d& rotation) {
        if (!std::isfinite(t) || t <= 0.0 || t == last_t_) return;
        samples_.push_back({t, rotation});
        last_t_ = t;
        while (!samples_.empty() && (samples_.back().t - samples_.front().t) > 5.0) {
            samples_.pop_front();
        }
    }

    bool windowDelta(double seconds, cv::Matx33d& delta) const {
        if (samples_.size() < 2) return false;
        const TimedRotation& now = samples_.back();
        const double target = now.t - seconds;
        const TimedRotation* base = &samples_.front();
        for (const auto& s : samples_) {
            if (s.t <= target) base = &s;
            else break;
        }
        const double actualWindow = now.t - base->t;
        if (actualWindow < std::min(0.15, seconds * 0.5)) return false;
        delta = now.r * base->r.t();
        return true;
    }

private:
    std::deque<TimedRotation> samples_;
    double last_t_{0.0};
};

int main() {
    auto xsens = makeSensor();
    xsens->start();

    MoverioOrientationSource moverio;
    moverio.start();

    bool axisFix = true;
    bool invertMoverioDelta = false;
    bool haveAnchor = false;
    cv::Matx33d moverioAnchor = cv::Matx33d::eye();
    cv::Matx33d xsensAnchor = cv::Matx33d::eye();
    RotationHistory moverioHistory;
    RotationHistory xsensHistory;
    double windowSeconds = 1.0;

    auto resetAnchor = [&](const OrientationSample& head, const SensorSample& body,
                           bool headOk, bool bodyOk) {
        if (headOk) {
            moverioAnchor = rotationFromQuaternion(head);
            if (axisFix) moverioAnchor = mapMoverioToCameraAxes(moverioAnchor);
        }
        if (bodyOk) xsensAnchor = rotationFromEulerDegrees(body);
        moverioHistory.clear();
        xsensHistory.clear();
        haveAnchor = headOk || bodyOk;
    };

    cv::namedWindow("IMU Compare", cv::WINDOW_NORMAL);
    cv::resizeWindow("IMU Compare", 1180, 720);

    while (true) {
        const double now = nowSeconds();
        OrientationSample head = moverio.latest();
        SensorSample body = xsens->latest();
        const bool headOk = head.valid && (now - head.t_host) < 0.5;
        const bool bodyOk = std::isfinite(body.t_host) && body.t_host > 0.0 &&
                            !body.xsens_calibrating && (now - body.t_host) < 0.5;

        if (!haveAnchor && (headOk || bodyOk)) resetAnchor(head, body, headOk, bodyOk);

        cv::Matx33d moverioNow = cv::Matx33d::eye();
        cv::Matx33d xsensNow = cv::Matx33d::eye();
        if (headOk) {
            moverioNow = rotationFromQuaternion(head);
            if (axisFix) moverioNow = mapMoverioToCameraAxes(moverioNow);
            moverioHistory.add(head.t_host, moverioNow);
        }
        if (bodyOk) {
            xsensNow = rotationFromEulerDegrees(body);
            xsensHistory.add(body.t_host, xsensNow);
        }

        cv::Matx33d moverioDelta = headOk ? moverioNow * moverioAnchor.t() : cv::Matx33d::eye();
        if (invertMoverioDelta) moverioDelta = moverioDelta.t();
        const cv::Matx33d xsensDelta = bodyOk ? xsensNow * xsensAnchor.t() : cv::Matx33d::eye();
        const cv::Matx33d relativeDelta = xsensDelta.t() * moverioDelta;
        cv::Matx33d moverioWindowDelta = cv::Matx33d::eye();
        cv::Matx33d xsensWindowDelta = cv::Matx33d::eye();
        const bool moverioWindowOk = moverioHistory.windowDelta(windowSeconds, moverioWindowDelta);
        const bool xsensWindowOk = xsensHistory.windowDelta(windowSeconds, xsensWindowDelta);
        if (invertMoverioDelta) moverioWindowDelta = moverioWindowDelta.t();
        const cv::Matx33d windowRelativeDelta =
            (moverioWindowOk && xsensWindowOk) ? xsensWindowDelta.t() * moverioWindowDelta : cv::Matx33d::eye();

        cv::Mat canvas(720, 1180, CV_8UC3, cv::Scalar(18, 18, 22));
        drawTriad(canvas, {230, 300}, moverioWindowDelta, "Moverio short delta",
                  moverioWindowOk ? cv::Scalar(230, 230, 230) : cv::Scalar(80, 80, 120));
        drawTriad(canvas, {590, 300}, xsensWindowDelta, "Xsens short delta",
                  xsensWindowOk ? cv::Scalar(230, 230, 230) : cv::Scalar(80, 80, 120));
        drawTriad(canvas, {950, 300}, windowRelativeDelta, "Short relative", cv::Scalar(230, 230, 230));

        const cv::Vec3d h = eulerZYXDegrees(moverioWindowDelta);
        const cv::Vec3d x = eulerZYXDegrees(xsensWindowDelta);
        const cv::Vec3d r = eulerZYXDegrees(windowRelativeDelta);
        const cv::Vec3d hl = eulerZYXDegrees(moverioDelta);
        const cv::Vec3d xl = eulerZYXDegrees(xsensDelta);

        int y = 36;
        drawLine(canvas, y, "IMU Compare: short-window movement | R reset | A axes | I invert | +/- window | Esc/Q quit",
                 {255, 255, 180});
        drawLine(canvas, y, std::string("Moverio: ") + (headOk ? "OK" : "missing/stale") +
                 " | Xsens onboard: " + (bodyOk ? "OK" : (body.xsens_calibrating ? "calibrating" : "missing/stale")) +
                 " | axisFix=" + (axisFix ? "on" : "off") +
                 " | invert=" + (invertMoverioDelta ? "on" : "off"));
        char buf[256];
        std::snprintf(buf, sizeof(buf), "Short window %.2fs | Moverio roll=%7.2f pitch=%7.2f yaw=%7.2f angle=%7.2f deg",
                      windowSeconds,
                      h[0], h[1], h[2], rotationAngleDegrees(moverioWindowDelta));
        drawLine(canvas, y, buf, {210, 235, 255});
        std::snprintf(buf, sizeof(buf), "Short window %.2fs | Xsens   roll=%7.2f pitch=%7.2f yaw=%7.2f angle=%7.2f deg",
                      windowSeconds,
                      x[0], x[1], x[2], rotationAngleDegrees(xsensWindowDelta));
        drawLine(canvas, y, buf, {210, 255, 210});
        std::snprintf(buf, sizeof(buf), "Short relative       roll=%7.2f pitch=%7.2f yaw=%7.2f angle=%7.2f deg",
                      r[0], r[1], r[2], rotationAngleDegrees(windowRelativeDelta));
        drawLine(canvas, y, buf, {255, 220, 190});
        std::snprintf(buf, sizeof(buf), "Long since reset | Moverio yaw=%7.2f angle=%7.2f | Xsens yaw=%7.2f angle=%7.2f",
                      hl[2], rotationAngleDegrees(moverioDelta), xl[2], rotationAngleDegrees(xsensDelta));
        drawLine(canvas, y, buf, {165, 165, 190});
        std::snprintf(buf, sizeof(buf), "Xsens gyro dps x=%8.2f y=%8.2f z=%8.2f | accel m/s2 x=%7.2f y=%7.2f z=%7.2f",
                      body.gyro_x_dps, body.gyro_y_dps, body.gyro_z_dps,
                      body.acc_x_ms2, body.acc_y_ms2, body.acc_z_ms2);
        drawLine(canvas, y, buf, {200, 220, 200});
        if (body.xsens_calibrating) {
            std::snprintf(buf, sizeof(buf), "Xsens startup calibration %.0f%%: keep sensor completely still",
                          body.xsens_calibration_progress * 100.0);
            drawLine(canvas, y, buf, {255, 210, 120});
        }

        cv::imshow("IMU Compare", canvas);
        const int key = cv::waitKey(10) & 0xff;
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == 'r' || key == 'R') resetAnchor(head, body, headOk, bodyOk);
        if (key == '+' || key == '=') windowSeconds = std::min(3.0, windowSeconds + 0.25);
        if (key == '-' || key == '_') windowSeconds = std::max(0.25, windowSeconds - 0.25);
        if (key == 'a' || key == 'A') {
            axisFix = !axisFix;
            resetAnchor(head, body, headOk, bodyOk);
        }
        if (key == 'i' || key == 'I') {
            invertMoverioDelta = !invertMoverioDelta;
        }
    }

    moverio.stop();
    xsens->stop();
    cv::destroyAllWindows();
    return 0;
}
