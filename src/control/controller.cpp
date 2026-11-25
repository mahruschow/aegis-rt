#include "aegis/control/controller.hpp"

#include <algorithm>
#include <cmath>

namespace aegis::control {

namespace {
struct Mat3 {
    double m[3][3]{};

    static Mat3 from_quat(const core::Quat& q) noexcept {
        Mat3 R{};
        const double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        R.m[0][0] = 1.0 - 2.0 * (yy + zz);
        R.m[0][1] = 2.0 * (xy - wz);
        R.m[0][2] = 2.0 * (xz + wy);

        R.m[1][0] = 2.0 * (xy + wz);
        R.m[1][1] = 1.0 - 2.0 * (xx + zz);
        R.m[1][2] = 2.0 * (yz - wx);

        R.m[2][0] = 2.0 * (xz - wy);
        R.m[2][1] = 2.0 * (yz + wx);
        R.m[2][2] = 1.0 - 2.0 * (xx + yy);
        return R;
    }

    static Mat3 from_columns(const core::Vec3& c0, const core::Vec3& c1, const core::Vec3& c2) noexcept {
        Mat3 R{};
        R.m[0][0] = c0.x; R.m[0][1] = c1.x; R.m[0][2] = c2.x;
        R.m[1][0] = c0.y; R.m[1][1] = c1.y; R.m[1][2] = c2.y;
        R.m[2][0] = c0.z; R.m[2][1] = c1.z; R.m[2][2] = c2.z;
        return R;
    }
};
}  // namespace

core::RotorCommand QuadrotorController::compute_command(
    const core::DroneState& drone,
    const core::TargetEstimate3D& target,
    const ObstacleCylinder* obstacles,
    std::size_t num_obstacles,
    double& out_min_obstacle_clearance_m) const noexcept {

    out_min_obstacle_clearance_m = 99.0;

    // 1. Compute standoff pursuit setpoint + desired yaw looking directly at the target
    core::Vec3 desired_pos{drone.pos.x, drone.pos.y, cfg_.cruise_altitude_m};
    core::Vec3 desired_vel{0.0, 0.0, 0.0};
    double desired_yaw = drone.quat.to_euler_zyx().z;

    if (target.valid) {
        const core::Vec3 to_target{
            target.pos_world.x - drone.pos.x,
            target.pos_world.y - drone.pos.y,
            0.0};
        const double dist_xy = to_target.norm_xy();
        desired_yaw = std::atan2(to_target.y, to_target.x);

        if (dist_xy > 1e-3) {
            const core::Vec3 unit_to_target = to_target / dist_xy;
            desired_pos.x = target.pos_world.x - unit_to_target.x * cfg_.standoff_distance_m;
            desired_pos.y = target.pos_world.y - unit_to_target.y * cfg_.standoff_distance_m;
        }
        desired_pos.z = std::clamp(target.pos_world.z + 0.25, 1.10, 2.20);
        desired_vel = {target.vel_world.x * 0.90, target.vel_world.y * 0.90, target.vel_world.z * 0.80};
    }

    // 2. Artificial Potential Field repulsive acceleration from cylindrical obstacles
    core::Vec3 accel_repulsive{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < num_obstacles; ++i) {
        const double dx = drone.pos.x - obstacles[i].x;
        const double dy = drone.pos.y - obstacles[i].y;
        const double center_dist = std::sqrt(dx * dx + dy * dy);
        const double surface_dist = std::max(0.05, center_dist - obstacles[i].radius);
        out_min_obstacle_clearance_m = std::min(out_min_obstacle_clearance_m, surface_dist);

        if (surface_dist < cfg_.obstacle_influence_m && center_dist > 1e-4) {
            const double inv_d = 1.0 / surface_dist;
            const double inv_d0 = 1.0 / cfg_.obstacle_influence_m;
            const double mag = std::min(
                6.0, cfg_.obstacle_gain * (inv_d - inv_d0) * (inv_d * inv_d));
            accel_repulsive.x += mag * (dx / center_dist);
            accel_repulsive.y += mag * (dy / center_dist);
        }
    }

    // 3. Outer-loop PD velocity/acceleration command
    core::Vec3 pos_err = desired_pos - drone.pos;
    core::Vec3 vel_cmd{
        cfg_.kp_pos_xy * pos_err.x + desired_vel.x,
        cfg_.kp_pos_xy * pos_err.y + desired_vel.y,
        0.0};
    const double vel_xy_norm = vel_cmd.norm_xy();
    if (vel_xy_norm > cfg_.max_vel_xy) {
        vel_cmd.x *= cfg_.max_vel_xy / vel_xy_norm;
        vel_cmd.y *= cfg_.max_vel_xy / vel_xy_norm;
    }

    core::Vec3 accel_des{
        cfg_.kd_vel_xy * (vel_cmd.x - drone.vel.x) + accel_repulsive.x,
        cfg_.kd_vel_xy * (vel_cmd.y - drone.vel.y) + accel_repulsive.y,
        cfg_.kp_pos_z * pos_err.z + cfg_.kd_vel_z * (desired_vel.z - drone.vel.z) + cfg_.gravity};

    accel_des.z = std::max(2.0, accel_des.z);
    const double max_horiz_accel = accel_des.z * std::tan(cfg_.max_tilt_rad);
    const double horiz_accel = accel_des.norm_xy();
    if (horiz_accel > max_horiz_accel && horiz_accel > 1e-6) {
        accel_des.x *= max_horiz_accel / horiz_accel;
        accel_des.y *= max_horiz_accel / horiz_accel;
    }

    // 4. Construct desired SO(3) rotation matrix R_d = [b1d, b2d, b3d]
    const core::Vec3 f_world = accel_des * cfg_.mass_kg;
    const double f_norm = std::max(1e-4, f_world.norm());
    const core::Vec3 b3d = f_world / f_norm;

    const core::Vec3 b1_heading{std::cos(desired_yaw), std::sin(desired_yaw), 0.0};
    core::Vec3 b2d = b3d.cross(b1_heading);
    const double b2_norm = b2d.norm();
    b2d = (b2_norm > 1e-5) ? (b2d / b2_norm) : core::Vec3{-std::sin(desired_yaw), std::cos(desired_yaw), 0.0};
    const core::Vec3 b1d = b2d.cross(b3d);

    const Mat3 R = Mat3::from_quat(drone.quat);
    const Mat3 Rd = Mat3::from_columns(b1d, b2d, b3d);

    // Collective thrust projected onto current body Z-axis: T = f_world . b3
    const core::Vec3 b3_curr{R.m[0][2], R.m[1][2], R.m[2][2]};
    const double collective_thrust = std::max(0.0, f_world.dot(b3_curr));

    // Geometric attitude error e_R = 0.5 * vee(Rd^T * R - R^T * Rd)
    double M[3][3]{};  // Rd^T * R
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M[i][j] = Rd.m[0][i] * R.m[0][j] + Rd.m[1][i] * R.m[1][j] + Rd.m[2][i] * R.m[2][j];
        }
    }
    const core::Vec3 e_R{
        0.5 * (M[2][1] - M[1][2]),
        0.5 * (M[0][2] - M[2][0]),
        0.5 * (M[1][0] - M[0][1])};

    const core::Vec3 tau{
        -cfg_.kr_roll_pitch * e_R.x - cfg_.kw_roll_pitch * drone.omega_body.x,
        -cfg_.kr_roll_pitch * e_R.y - cfg_.kw_roll_pitch * drone.omega_body.y,
        -cfg_.kr_yaw * e_R.z - cfg_.kw_yaw * drone.omega_body.z};

    // 5. X-Configuration 4-Rotor Control Allocation Mixer
    // Rotor layout in body frame (+X forward, +Y left, +Z up):
    //   m0 (Front-Right): (+L, -L), CCW (+yaw reaction)
    //   m1 (Front-Left) : (+L, +L), CW  (-yaw reaction)
    //   m2 (Rear-Left)  : (-L, +L), CCW (+yaw reaction)
    //   m3 (Rear-Right) : (-L, -L), CW  (-yaw reaction)
    // Relations:
    //   T     = f0 + f1 + f2 + f3
    //   tau_x = L * (-f0 + f1 + f2 - f3)   (roll moment around +X)
    //   tau_y = L * (-f0 - f1 + f2 + f3)   (pitch moment around +Y)
    //   tau_z = c * ( f0 - f1 + f2 - f3)   (yaw reaction torque around +Z)
    const double t_part = 0.25 * collective_thrust;
    const double rx = 0.25 * tau.x / cfg_.arm_length_m;
    const double ry = 0.25 * tau.y / cfg_.arm_length_m;
    const double rz = 0.25 * tau.z / cfg_.yaw_torque_coeff;

    core::RotorCommand cmd{};
    cmd.collective_thrust_n = collective_thrust;
    cmd.body_moment_nm = tau;
    cmd.motor_thrusts_n[0] = std::clamp(t_part - rx - ry + rz, cfg_.min_motor_thrust_n, cfg_.max_motor_thrust_n);
    cmd.motor_thrusts_n[1] = std::clamp(t_part + rx - ry - rz, cfg_.min_motor_thrust_n, cfg_.max_motor_thrust_n);
    cmd.motor_thrusts_n[2] = std::clamp(t_part + rx + ry + rz, cfg_.min_motor_thrust_n, cfg_.max_motor_thrust_n);
    cmd.motor_thrusts_n[3] = std::clamp(t_part - rx + ry - rz, cfg_.min_motor_thrust_n, cfg_.max_motor_thrust_n);
    return cmd;
}

}  // namespace aegis::control
