#include "moverio_sensor.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

static double nowSeconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sensorsapi.h>
#include <sensors.h>
#include <sensorsdef.h>
#include <sensorsstructures.h>
#include <propvarutil.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

struct MoverioOrientationSource::Impl {
    bool com_owned{ false };
    ComPtr<ISensorManager> manager;
    ComPtr<ISensor> sensor;
    const char* sensor_kind{ nullptr };
};

static bool selectOrientationSensor(ISensorManager* manager, REFSENSOR_TYPE_ID type,
                                    const char* kind, ComPtr<ISensor>& selectedSensor,
                                    const char*& selectedKind) {
    ComPtr<ISensorCollection> sensors;
    if (FAILED(manager->GetSensorsByType(type, &sensors)) || !sensors) return false;

    ULONG count = 0;
    if (FAILED(sensors->GetCount(&count))) return false;
    for (ULONG index = 0; index < count; ++index) {
        ComPtr<ISensor> sensor;
        if (FAILED(sensors->GetAt(index, &sensor)) || !sensor) continue;

        SensorState state = SENSOR_STATE_ERROR;
        if (FAILED(sensor->GetState(&state)) ||
            (state != SENSOR_STATE_READY && state != SENSOR_STATE_NO_DATA &&
             state != SENSOR_STATE_INITIALIZING)) {
            continue;
        }

        VARIANT_BOOL supportsQuaternion = VARIANT_FALSE;
        if (FAILED(sensor->SupportsDataField(SENSOR_DATA_TYPE_QUATERNION, &supportsQuaternion)) ||
            supportsQuaternion != VARIANT_TRUE) {
            continue;
        }

        selectedSensor = sensor;
        selectedKind = kind;
        return true;
    }
    return false;
}

#else
struct MoverioOrientationSource::Impl {};
#endif

MoverioOrientationSource::MoverioOrientationSource() : impl_(std::make_unique<Impl>()) {}
MoverioOrientationSource::~MoverioOrientationSource() { stop(); }

bool MoverioOrientationSource::start() {
#ifdef _WIN32
    if (impl_->sensor) return true;

    const HRESULT comHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    impl_->com_owned = SUCCEEDED(comHr);
    if (FAILED(comHr) && comHr != RPC_E_CHANGED_MODE) {
        std::cerr << "Moverio IMU: COM initialization failed (0x" << std::hex << comHr << std::dec << ")\n";
        return false;
    }

    const HRESULT managerHr = CoCreateInstance(CLSID_SensorManager, nullptr, CLSCTX_INPROC_SERVER,
                                               IID_PPV_ARGS(&impl_->manager));
    if (FAILED(managerHr) || !impl_->manager) {
        std::cerr << "Moverio IMU: Windows Sensor Manager is unavailable (0x"
                  << std::hex << managerHr << std::dec << ")\n";
        stop();
        return false;
    }

    // Prefer the fused device orientation when available; the pure relative
    // orientation sensor can accumulate noticeable yaw drift between visual anchors.
    if (!selectOrientationSensor(impl_->manager.Get(), SENSOR_TYPE_AGGREGATED_DEVICE_ORIENTATION,
                                 "device orientation", impl_->sensor, impl_->sensor_kind) &&
        !selectOrientationSensor(impl_->manager.Get(), GUID_SensorType_RelativeOrientation,
                                 "relative orientation", impl_->sensor, impl_->sensor_kind)) {
        std::cerr << "Moverio IMU: no Windows orientation quaternion sensor found.\n";
        stop();
        return false;
    }

    std::cerr << "Moverio IMU: using " << impl_->sensor_kind << " quaternion sensor.\n";
    return true;
#else
    return false;
#endif
}

void MoverioOrientationSource::stop() {
#ifdef _WIN32
    impl_->sensor.Reset();
    impl_->manager.Reset();
    impl_->sensor_kind = nullptr;
    if (impl_->com_owned) {
        CoUninitialize();
        impl_->com_owned = false;
    }
#endif
}

OrientationSample MoverioOrientationSource::latest() const {
    OrientationSample sample;
#ifdef _WIN32
    if (!impl_->sensor) return sample;

    ComPtr<ISensorDataReport> report;
    if (FAILED(impl_->sensor->GetData(&report)) || !report) return sample;

    PROPVARIANT value;
    PropVariantInit(&value);
    const HRESULT valueHr = report->GetSensorValue(SENSOR_DATA_TYPE_QUATERNION, &value);
    if (SUCCEEDED(valueHr) && value.vt == (VT_VECTOR | VT_UI1) &&
        value.caub.pElems && value.caub.cElems >= sizeof(QUATERNION)) {
        const auto* quaternion = reinterpret_cast<const QUATERNION*>(value.caub.pElems);
        const double norm = std::sqrt(
            double(quaternion->X) * quaternion->X +
            double(quaternion->Y) * quaternion->Y +
            double(quaternion->Z) * quaternion->Z +
            double(quaternion->W) * quaternion->W);
        if (std::isfinite(norm) && norm > 1e-9) {
            sample.valid = true;
            sample.t_host = nowSeconds();
            sample.x = quaternion->X / norm;
            sample.y = quaternion->Y / norm;
            sample.z = quaternion->Z / norm;
            sample.w = quaternion->W / norm;
        }
    }
    PropVariantClear(&value);
#endif
    return sample;
}

bool MoverioOrientationSource::available() const {
#ifdef _WIN32
    return bool(impl_->sensor);
#else
    return false;
#endif
}
