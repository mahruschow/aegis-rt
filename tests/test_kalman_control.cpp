#include "aegis/control/controller.hpp"
#include "aegis/control/kalman_tracker.hpp"
#include "aegis/control/rover_controller.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace aegis;

    // 1. Verify KalmanTracker3D converges on constant-velocity target & rejects outliers
    control::KalmanTracker3D kf;
    const core::Vec3 true_vel{1.2, -0.8, 0.1};
    core::Vec3 true_pos{3.0, 1.0, 1.2};
    uint64_t ts_ns = 1'000'000'000ULL;

    kf.reset(true_pos, ts_ns);
    constexpr double dt = 1.0 / 60.0;

    for (int step = 0; step < 90; ++step) {
        true_pos += true_vel * dt;
        ts_ns += static_cast<uint64_t>(dt * 1e9);
        kf.predict(dt);
        const bool accepted = kf.update_with_latency_compensation(
            true_pos, 0.95, ts_ns - 8'000'000ULL, ts_ns);
        assert(accepted);
    }

    const auto est = kf.estimate();
    assert(est.valid);
    const double pos_err = (est.pos_world - true_pos).norm();
    const double vel_err = (est.vel_world - true_vel).norm();
    assert(pos_err < 0.05);
    assert(vel_err < 0.15);

    // Outlier injection (+10m jump) must be gated/rejected
    const bool outlier_accepted = kf.update_with_latency_compensation(
        {true_pos.x + 10.0, true_pos.y, true_pos.z}, 0.99, ts_ns, ts_ns);
    assert(!outlier_accepted);
    assert(kf.accepted_updates() == 90);
    assert(kf.rejected_outliers() == 1);
    assert(kf.adaptive_q_scale() < 1.05);

    // Sub-Euclidean (1.5m) glitch at steady state must still be caught by Mahalanobis chi2 gate
    const bool mahalanobis_outlier = kf.update_with_latency_compensation(
        {true_pos.x + 1.5, true_pos.y, true_pos.z}, 0.99, ts_ns, ts_ns);
    assert(!mahalanobis_outlier);
    assert(kf.rejected_outliers() == 2);

    // 1b. Verify occlusion coasting expands covariance & permits valid re-acquisition
    const double pre_coast_trace = kf.position_covariance_trace();
    for (int step = 0; step < 36; ++step) {  // 0.6s occlusion coast
        true_pos += true_vel * dt;
        ts_ns += static_cast<uint64_t>(dt * 1e9);
        kf.predict(dt);
    }
    assert(kf.position_covariance_trace() > pre_coast_trace * 3.0);
    // Target drifted 1.1m during occlusion: Mahalanobis accepts it because P grew
    const bool reacq_accepted = kf.update_with_latency_compensation(
        {true_pos.x + 1.1, true_pos.y, true_pos.z}, 0.90, ts_ns, ts_ns);
    assert(reacq_accepted);

    // 1c. Verify NIS-driven adaptive Q scaling spikes on sudden maneuver and decays afterward
    control::KalmanTracker3D::Config adapt_cfg{};
    adapt_cfg.process_noise_vel = 0.15;      // Tight CV tuning during cruise
    adapt_cfg.measurement_noise_pos = 0.01;  // 10 cm RGB-D std dev
    control::KalmanTracker3D kf_adapt(adapt_cfg);
    core::Vec3 p_m{0.0, 0.0, 1.0};
    core::Vec3 v_m{1.0, 0.0, 0.0};
    uint64_t t_m = 2'000'000'000ULL;
    kf_adapt.reset(p_m, t_m);
    for (int i = 0; i < 60; ++i) {
        p_m += v_m * dt;
        t_m += static_cast<uint64_t>(dt * 1e9);
        kf_adapt.predict(dt);
        assert(kf_adapt.update_with_latency_compensation(p_m, 0.95, t_m, t_m));
    }
    // Sudden lateral jink (step velocity change in Y) triggers maneuver NIS threshold (> 4.5)
    bool saw_q_boost = false;
    v_m = {1.0, 6.0, 0.0};
    for (int i = 0; i < 20; ++i) {
        p_m += v_m * dt;
        t_m += static_cast<uint64_t>(dt * 1e9);
        kf_adapt.predict(dt);
        kf_adapt.update_with_latency_compensation(p_m, 0.95, t_m, t_m);
        if (kf_adapt.adaptive_q_scale() > 1.2) {
            saw_q_boost = true;
        }
    }
    assert(saw_q_boost);
    const double peak_q_scale = kf_adapt.adaptive_q_scale();
    for (int i = 0; i < 40; ++i) {
        p_m += v_m * dt;
        t_m += static_cast<uint64_t>(dt * 1e9);
        kf_adapt.predict(dt);
        kf_adapt.update_with_latency_compensation(p_m, 0.95, t_m, t_m);
    }
    assert(kf_adapt.adaptive_q_scale() < peak_q_scale);

    // 2. Verify QuadrotorController produces hover equilibrium & symmetric rotor thrusts
    control::QuadrotorController ctrl;
    core::DroneState hover_drone{};
    hover_drone.pos = {0.0, 0.0, 1.55};
    hover_drone.quat = {1.0, 0.0, 0.0, 0.0};

    core::TargetEstimate3D stationary_target{};
    stationary_target.pos_world = {2.6, 0.0, 1.30};  // Exactly at standoff (2.6m) & desired_z = 1.30 + 0.25 = 1.55m
    stationary_target.valid = true;

    double clearance = 0.0;
    const auto cmd = ctrl.compute_command(hover_drone, stationary_target, nullptr, 0, clearance);
    const double expected_hover_thrust = 1.05 * 9.81;  // m * g
    assert(std::abs(cmd.collective_thrust_n - expected_hover_thrust) < 0.15);
    for (double f : cmd.motor_thrusts_n) {
        assert(std::abs(f - 0.25 * expected_hover_thrust) < 0.05);
    }

    // 2b. Verify tangential vortex term breaks collinear obstacle saddle point
    const control::ObstacleCylinder collinear_obs{1.0, 0.0, 0.32};
    core::TargetEstimate3D ahead_target{};
    ahead_target.pos_world = {4.0, 0.0, 1.30};
    ahead_target.valid = true;
    double obs_clearance = 0.0;
    const auto vortex_cmd = ctrl.compute_command(hover_drone, ahead_target, &collinear_obs, 1, obs_clearance);
    assert(obs_clearance < 1.0);
    // Lateral vortex force creates non-zero roll moment around body +X
    assert(std::abs(vortex_cmd.body_moment_nm.x) > 0.05);

    // 2c. Verify priority-preserving rotor desaturation keeps all motors in bounds and preserves roll/pitch
    core::DroneState sat_drone{};
    sat_drone.pos = {0.0, 0.0, 0.50};  // Large positive Z error -> high collective thrust
    // ~90 deg yaw error + ~20 deg roll error
    sat_drone.quat = {std::cos(0.785), std::sin(0.15), 0.0, std::sin(0.785)};
    const auto sat_cmd = ctrl.compute_command(sat_drone, ahead_target, nullptr, 0, obs_clearance);
    for (double f : sat_cmd.motor_thrusts_n) {
        assert(f >= 0.2 - 1e-9 && f <= 7.5 + 1e-9);
    }
    const double realized_tau_x = 0.15 * (
        -sat_cmd.motor_thrusts_n[0] + sat_cmd.motor_thrusts_n[1] +
         sat_cmd.motor_thrusts_n[2] - sat_cmd.motor_thrusts_n[3]);
    const double realized_tau_y = 0.15 * (
        -sat_cmd.motor_thrusts_n[0] - sat_cmd.motor_thrusts_n[1] +
         sat_cmd.motor_thrusts_n[2] + sat_cmd.motor_thrusts_n[3]);
    assert(std::abs(realized_tau_x - sat_cmd.body_moment_nm.x) < 1e-6);
    assert(std::abs(realized_tau_y - sat_cmd.body_moment_nm.y) < 1e-6);

    // 3. Verify AckermannRoverController inner/outer steering geometry (left turn: |steer_fl| > |steer_fr|)
    control::AckermannRoverController rover_ctrl;
    core::DroneState rover_st{};
    rover_st.pos = {0.0, 0.0, 0.14};
    rover_st.quat = {1.0, 0.0, 0.0, 0.0};
    core::TargetEstimate3D left_target{};
    left_target.pos_world = {3.0, 1.5, 0.32};
    left_target.valid = true;

    double rover_clearance = 0.0;
    const auto rcmd = rover_ctrl.compute_command(rover_st, left_target, nullptr, 0, rover_clearance);
    assert(rcmd.motor_thrusts_n[0] > 0.0);  // Steer left (+Z)
    assert(rcmd.motor_thrusts_n[1] > 0.0);
    assert(rcmd.motor_thrusts_n[0] > rcmd.motor_thrusts_n[1]);  // Inner (FL) turns sharper than outer (FR)!
    assert(rcmd.motor_thrusts_n[2] > 0.0 && rcmd.motor_thrusts_n[3] > 0.0);  // Forward drive torque

    std::cout << "[PASS] test_kalman_control: KF vel_err=" << vel_err
              << " m/s, SE(3) thrust=" << cmd.collective_thrust_n
              << " N, Ackermann inner/outer=(" << rcmd.motor_thrusts_n[0] << ", " << rcmd.motor_thrusts_n[1] << ") rad.\n";
    return 0;
}
