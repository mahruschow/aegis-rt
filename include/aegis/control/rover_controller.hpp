#pragma once

#include "aegis/control/controller.hpp"
#include "aegis/core/frame_types.hpp"

#include <cstddef>

namespace aegis::control {

/**
 * @brief Geometric Ackermann Pure-Pursuit + Potential-Field Obstacle Avoidance UGV Controller.
 *
 * Computes bicycle steering curvature toward the estimated target, applies repulsive lateral
 * avoidance from cylindrical obstacles, resolves exact inner/outer Ackermann steering angles,
 * and regulates longitudinal standoff distance via rear-wheel drive torque allocation.
 *
 * Output mapping into `core::RotorCommand::motor_thrusts_n`:
 *   [0] = steer_fl_rad, [1] = steer_fr_rad, [2] = torque_rl_nm, [3] = torque_rr_nm
 */
class AckermannRoverController {
public:
    struct Config {
        double wheelbase_m{0.36};
        double track_width_m{0.30};
        double wheel_radius_m{0.09};

        double standoff_distance_m{2.20};
        double min_lookahead_m{0.95};
        double max_steer_rad{0.62};  // ~35.5 deg
        double max_speed_mps{3.2};

        double kp_dist{2.2};
        double kp_speed{4.0};
        double max_wheel_torque_nm{4.5};

        double obstacle_influence_m{1.25};
        double obstacle_repulse_gain{1.8};
    };

    AckermannRoverController() : AckermannRoverController(Config{}) {}
    explicit AckermannRoverController(const Config& cfg) noexcept : cfg_(cfg) {}

    [[nodiscard]] core::RotorCommand compute_command(
        const core::DroneState& rover,
        const core::TargetEstimate3D& target,
        const ObstacleCylinder* obstacles,
        std::size_t num_obstacles,
        double& out_min_obstacle_clearance_m) const noexcept;

private:
    Config cfg_;
};

}  // namespace aegis::control
