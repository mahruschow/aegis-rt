#pragma once

#include "aegis/core/frame_types.hpp"

#include <array>
#include <cstddef>

namespace aegis::control {

struct ObstacleCylinder {
    double x{0.0};
    double y{0.0};
    double radius{0.35};
};

/**
 * @brief Geometric SE(3) Cascaded Quadrotor Pursuit + Potential-Field Obstacle Avoidance Controller.
 *
 * Outer Loop:
 *   - Computes standoff pursuit setpoint at `standoff_distance_m` behind target velocity/bearing
 *     so the forward-facing camera maintains continuous line-of-sight framing.
 *   - Adds artificial potential-field repulsive acceleration away from arena obstacles.
 * Inner Loop:
 *   - Maps desired world force vector + target-tracking yaw onto SO(3) attitude error,
 *     computes body moments, and allocates individual thrusts across the 4 X-frame rotors.
 */
class QuadrotorController {
public:
    struct Config {
        double mass_kg{1.05};
        double gravity{9.81};
        double arm_length_m{0.15};       // X-frame motor distance along X and Y from CoM
        double yaw_torque_coeff{0.016};  // Counter-torque to thrust ratio k_m / k_f

        double standoff_distance_m{2.6}; // Desired horizontal distance from target
        double cruise_altitude_m{1.55};  // Nominal pursuit altitude [m]

        // Outer-loop position/velocity gains
        double kp_pos_xy{3.8};
        double kp_pos_z{6.5};
        double kd_vel_xy{2.9};
        double kd_vel_z{4.2};
        double max_tilt_rad{0.52};       // ~30 deg max roll/pitch angle
        double max_vel_xy{4.5};          // m/s saturation

        // Obstacle repulsive potential field
        double obstacle_influence_m{1.35};
        double obstacle_gain{3.2};

        // Inner-loop SO(3) attitude & angular rate gains
        double kr_roll_pitch{4.2};
        double kr_yaw{1.8};
        double kw_roll_pitch{0.65};
        double kw_yaw{0.35};

        double min_motor_thrust_n{0.2};
        double max_motor_thrust_n{7.5};
    };

    QuadrotorController() : QuadrotorController(Config{}) {}
    explicit QuadrotorController(const Config& cfg) noexcept : cfg_(cfg) {}

    [[nodiscard]] core::RotorCommand compute_command(
        const core::DroneState& drone,
        const core::TargetEstimate3D& target,
        const ObstacleCylinder* obstacles,
        std::size_t num_obstacles,
        double& out_min_obstacle_clearance_m) const noexcept;

private:
    Config cfg_;
};

}  // namespace aegis::control
