#pragma once

#include "aegis/core/frame_types.hpp"

#include <array>
#include <cstdint>

namespace aegis::control {

/**
 * @brief 6-DOF Constant-Velocity 3D Linear Kalman Filter with latency-compensated
 *        measurement Jacobian, Joseph-form covariance update, Mahalanobis outlier gating,
 *        and NIS-driven adaptive process noise for agile maneuvering targets.
 *
 * State vector x = [px, py, pz, vx, vy, vz]^T in world coordinates.
 * Designed for zero heap allocation in the real-time flight loop.
 */
class KalmanTracker3D {
public:
    struct Config {
        double process_noise_pos{0.02};        // m^2 / s
        double process_noise_vel{1.50};        // (m/s)^2 / s (agile evasive target)
        double measurement_noise_pos{0.04};    // m^2 (camera + depth back-projection noise)
        double innovation_gate_m{3.5};         // Hard Euclidean jump ceiling [m] (<= 0 disables)
        double mahalanobis_gate_chi2{16.27};   // 3-DOF chi-squared 99.9% outlier gate (<= 0 disables)
        double maneuver_nis_threshold{4.50};   // NIS threshold to boost adaptive Q on maneuvers
        double adaptive_q_max_scale{8.0};      // Max velocity Q multiplier during target maneuvers
        double adaptive_q_decay{0.88};         // Exponential decay per nominal update toward 1.0
    };

    KalmanTracker3D() : KalmanTracker3D(Config{}) {}
    explicit KalmanTracker3D(const Config& cfg) noexcept;

    void reset(const core::Vec3& initial_pos, uint64_t timestamp_ns) noexcept;

    // Propagate state & covariance forward by dt seconds
    void predict(double dt_s) noexcept;

    // Fuse 3D world-space camera measurement captured at capture_timestamp_ns,
    // accounting for perception delay (current_timestamp_ns - capture_timestamp_ns)
    // in both the innovation and measurement Jacobian H = [I_3, -delay * I_3].
    bool update_with_latency_compensation(const core::Vec3& measured_pos_world,
                                          double confidence,
                                          uint64_t capture_timestamp_ns,
                                          uint64_t current_timestamp_ns) noexcept;

    [[nodiscard]] core::TargetEstimate3D estimate() const noexcept;
    [[nodiscard]] bool initialized() const noexcept { return initialized_; }
    [[nodiscard]] double last_nis() const noexcept { return last_nis_; }
    [[nodiscard]] double adaptive_q_scale() const noexcept { return adaptive_q_scale_; }
    [[nodiscard]] double position_covariance_trace() const noexcept;
    [[nodiscard]] double velocity_covariance_trace() const noexcept;
    [[nodiscard]] uint64_t accepted_updates() const noexcept { return accepted_updates_; }
    [[nodiscard]] uint64_t rejected_outliers() const noexcept { return rejected_outliers_; }

private:
    using Mat6 = std::array<double, 36>;

    Config cfg_;
    bool initialized_{false};
    uint64_t last_timestamp_ns_{0};
    double last_confidence_{0.0};
    double last_nis_{0.0};
    double adaptive_q_scale_{1.0};
    uint64_t accepted_updates_{0};
    uint64_t rejected_outliers_{0};
    std::array<double, 6> x_{};  // [px, py, pz, vx, vy, vz]
    Mat6 P_{};                   // 6x6 error covariance matrix
};

}  // namespace aegis::control
