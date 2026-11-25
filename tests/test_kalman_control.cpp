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
