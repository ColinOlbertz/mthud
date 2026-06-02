#pragma once

#include <memory>

struct OrientationSample {
    bool valid{ false };
    double t_host{ 0.0 };
    double x{ 0.0 };
    double y{ 0.0 };
    double z{ 0.0 };
    double w{ 1.0 };
};

class MoverioOrientationSource {
public:
    MoverioOrientationSource();
    ~MoverioOrientationSource();

    bool start();
    void stop();
    OrientationSample latest() const;
    bool available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
