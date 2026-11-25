#include "aegis/control/rover_controller.hpp"

#include <algorithm>
#include <cmath>

namespace aegis::control {

core::RotorCommand AckermannRoverController::compute_command(
    const core::DroneState& rover,
    const core::TargetEstimate3D& target,
    const ObstacleCylinder* obstacles,
    std::size_t num_obstacles,
    double& out_min_obstacle_clearance_m) const noexcept {

    out_min_obstacle_clearance_m = 99.0;
    core::RotorCommand cmd{};
    if (!target.valid) {
        return cmd;
    }

    // 1. World-frame vector to target + potential-field obstacle repulsion
    core::Vec3 goal_vec{
        target.pos_world.x - rover.pos.x,
        target.pos_world.y - rover.pos.y,
        0.0};
    const double target_dist = goal_vec.norm_xy();

    core::Vec3 repulse{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < num_obstacles; ++i) {
        const double dx = rover.pos.x - obstacles[i].x;
        const double dy = rover.pos.y - obstacles[i].y;
        const double center_d = std::sqrt(dx * dx + dy * dy);
        const double surf_d = std::max(0.05, center_d - obstacles[i].radius - 0.16);
        out_min_obstacle_clearance_m = std::min(out_min_obstacle_clearance_m, surf_d);

        if (surf_d < cfg_.obstacle_influence_m && center_d > 1e-4) {
            const double inv_d = 1.0 / surf_d;
            const double inv_d0 = 1.0 / cfg_.obstacle_influence_m;
            const double w = std::min(3.5, cfg_.obstacle_repulse_gain * (inv_d - inv_d0) * inv_d);
            repulse.x += w * (dx / center_d);
            repulse.y += w * (dy / center_d);
        }
    }

    const core::Vec3 steered_goal_world = goal_vec + repulse;

    // 2. Transform lookahead vector into rover horizontal body frame (+X forward, +Y left)
    const double yaw = rover.quat.to_euler_zyx().z;
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    const double x_body =  cy * steered_goal_world.x + sy * steered_goal_world.y;
    const double y_body = -sy * steered_goal_world.x + cy * steered_goal_world.y;

    // 3. Geometric Pure-Pursuit curvature & exact inner/outer Ackermann steering angles
    const double ld = std::max(cfg_.min_lookahead_m, std::hypot(x_body, y_body));
    double curvature = (2.0 * y_body) / (ld * ld);
    if (x_body < -0.1) {
        // Target is behind rover: command full lock toward target side to pivot into view
        curvature = (y_body >= 0.0) ? 2.5 : -2.5;
    }

    const double center_delta = std::clamp(
        std::atan(curvature * cfg_.wheelbase_m), -cfg_.max_steer_rad, cfg_.max_steer_rad);

    double steer_fl = center_delta;
    double steer_fr = center_delta;
    const double tan_d = std::tan(center_delta);
    if (std::abs(tan_d) > 1e-5) {
        // Exact Ackermann turning center geometry (inner wheel turns sharper than outer wheel)
        steer_fl = std::atan((cfg_.wheelbase_m * tan_d) /
                             (cfg_.wheelbase_m - 0.5 * cfg_.track_width_m * tan_d));
        steer_fr = std::atan((cfg_.wheelbase_m * tan_d) /
                             (cfg_.wheelbase_m + 0.5 * cfg_.track_width_m * tan_d));
    }
    steer_fl = std::clamp(steer_fl, -cfg_.max_steer_rad, cfg_.max_steer_rad);
    steer_fr = std::clamp(steer_fr, -cfg_.max_steer_rad, cfg_.max_steer_rad);

    // 4. Longitudinal standoff speed & differential torque allocation
    const double fwd_speed = cy * rover.vel.x + sy * rover.vel.y;
    const double dist_err = target_dist - cfg_.standoff_distance_m;
    double desired_speed = std::clamp(cfg_.kp_dist * dist_err, -1.2, cfg_.max_speed_mps);

    // Cornering speed modulation: slow slightly during tight turns to prevent lateral tire scrub
    desired_speed *= std::max(0.45, std::cos(center_delta));
    if (x_body < 0.0 && std::abs(dist_err) < 0.8) {
        // Keep creeping forward while turning around toward a target behind us
        desired_speed = 0.9;
    }

    const double drive_torque = std::clamp(
        cfg_.kp_speed * (desired_speed - fwd_speed),
        -cfg_.max_wheel_torque_nm,
        cfg_.max_wheel_torque_nm);

    // Electronic rear torque vectoring assist proportional to steering angle
    const double diff_bias = 0.20 * drive_torque * (center_delta / cfg_.max_steer_rad);
    const double torque_rl = std::clamp(drive_torque - diff_bias, -cfg_.max_wheel_torque_nm, cfg_.max_wheel_torque_nm);
    const double torque_rr = std::clamp(drive_torque + diff_bias, -cfg_.max_wheel_torque_nm, cfg_.max_wheel_torque_nm);

    cmd.motor_thrusts_n = {steer_fl, steer_fr, torque_rl, torque_rr};
    return cmd;
}

}  // namespace aegis::control
