#pragma once

#include "aegis/core/frame_types.hpp"

#include <array>
#include <cstdint>

namespace aegis::control {

/**
 * @brief 6-DOF Constant-Velocity 3D Linear Kalman Filter with latency compensation
 *        and innovation outlier gating.
 *
 * State vector x = [px, py, pz, vx, vy, vz]^T in world coordinates.
 * Designed for zero heap allocation in the real-time flight loop.
 */
class KalmanTracker3D {
public:
    struct Config {
        double process_noise_pos{0.02};      // m^2 / s
        double process_noise_vel{1.50};      // (m/s)^2 / s (agile evasive target)
        double measurement_noise_pos{0.04};  // m^2 (camera + depth back-projection noise)
        double innovation_gate_m{3.5};       // Reject wild false-positive detections > 3.5 m jump
    };

    KalmanTracker3D() : KalmanTracker3D(Config{}) {}
    explicit KalmanTracker3D(const Config& cfg) noexcept;

    void reset(const core::Vec3& initial_pos, uint64_t timestamp_ns) noexcept;

    // Propagate state & covariance forward by dt seconds
    void predict(double dt_s) noexcept;

    // Fuse 3D world-space camera measurement captured at capture_timestamp_ns,
    // then forward-compensate to current_timestamp_ns for perception latency mitigation.
    bool update_with_latency_compensation(const core::Vec3& measured_pos_world,
                                          double confidence,
                                          uint64_t capture_timestamp_ns,
                                          uint64_t current_timestamp_ns) noexcept;

    [[nodiscard]] core::TargetEstimate3D estimate() const noexcept;
    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
    using Mat6 = std::array<double, 36>;

    Config cfg_;
    bool initialized_{false};
    uint64_t last_timestamp_ns_{0};
    double last_confidence_{0.0};
    std::array<double, 6> x_{};  // [px, py, pz, vx, vy, vz]
    Mat6 P_{};                   // 6x6 error covariance matrix
};

}  // namespace aegis::control
