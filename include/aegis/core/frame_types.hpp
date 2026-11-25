#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace aegis::core {

inline constexpr int kCameraWidth = 640;
inline constexpr int kCameraHeight = 480;
inline constexpr int kCameraChannels = 3;
inline constexpr std::size_t kRgbFrameBytes =
    static_cast<std::size_t>(kCameraWidth) * kCameraHeight * kCameraChannels;
inline constexpr std::size_t kDepthPixels =
    static_cast<std::size_t>(kCameraWidth) * kCameraHeight;

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vec3 operator+(const Vec3& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const noexcept { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(double s) const noexcept { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(double s) const noexcept { return {x / s, y / s, z / s}; }

    constexpr Vec3& operator+=(const Vec3& o) noexcept {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    [[nodiscard]] constexpr double dot(const Vec3& o) const noexcept {
        return x * o.x + y * o.y + z * o.z;
    }

    [[nodiscard]] constexpr Vec3 cross(const Vec3& o) const noexcept {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(x * x + y * y + z * z);
    }

    [[nodiscard]] double norm_xy() const noexcept {
        return std::sqrt(x * x + y * y);
    }
};

struct Quat {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    // Rotate vector v from body frame into world frame: R(q) * v
    [[nodiscard]] constexpr Vec3 rotate_body_to_world(const Vec3& v) const noexcept {
        const Vec3 qv{x, y, z};
        const Vec3 t = qv.cross(v) * 2.0;
        return v + (t * w) + qv.cross(t);
    }

    // Extract ZYX Euler angles (roll, pitch, yaw in radians)
    [[nodiscard]] Vec3 to_euler_zyx() const noexcept {
        const double sinr_cosp = 2.0 * (w * x + y * z);
        const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        const double roll = std::atan2(sinr_cosp, cosr_cosp);

        const double sinp = 2.0 * (w * y - z * x);
        const double pitch = (std::abs(sinp) >= 1.0)
                                 ? std::copysign(3.14159265358979323846 / 2.0, sinp)
                                 : std::asin(sinp);

        const double siny_cosp = 2.0 * (w * z + x * y);
        const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        const double yaw = std::atan2(siny_cosp, cosy_cosp);
        return {roll, pitch, yaw};
    }
};

struct DroneState {
    Vec3 pos{};          // World-frame position [m]
    Vec3 vel{};          // World-frame linear velocity [m/s]
    Quat quat{};         // Body-to-world orientation quaternion (w, x, y, z)
    Vec3 omega_body{};   // Body-frame angular velocity [rad/s]
    uint64_t timestamp_ns{0};
};

struct CameraIntrinsics {
    double fx{554.256};  // Derived from 45 deg vertical FOV at 640x480
    double fy{554.256};
    double cx{320.0};
    double cy{240.0};
};

/**
 * @brief Lightweight descriptor pointing to a pre-allocated slot in a frame pool.
 * Passing 64-byte descriptors over the SPSC queue instead of copying ~2.1 MB of RGB-D bytes
 * keeps producer/consumer handoff under 40 ns.
 */
struct FrameDescriptor {
    uint32_t slot_index{0};
    uint64_t sequence_id{0};
    uint64_t capture_timestamp_ns{0};
    double sensor_capture_us{0.0};
    DroneState drone_state{};
    Vec3 camera_pos_world{};
    Quat camera_quat_world{};
    Vec3 gt_target_pos_world{};  // Logged strictly for tracking RMSE ground-truth evaluation
};

/**
 * @brief Pre-allocated pixel buffer slot owned by the SensorFramePool.
 */
struct RawFrameSlot {
    alignas(64) std::array<uint8_t, kRgbFrameBytes> rgb{};
    alignas(64) std::array<float, kDepthPixels> depth_m{};
};

struct Detection2D {
    float cx_px{0.0f};       // Bounding-box center X in original camera pixels [0, kCameraWidth)
    float cy_px{0.0f};       // Bounding-box center Y in original camera pixels [0, kCameraHeight)
    float width_px{0.0f};
    float height_px{0.0f};
    float confidence{0.0f};
    int class_id{0};
    bool valid{false};
};

struct TargetEstimate3D {
    Vec3 pos_world{};
    Vec3 vel_world{};
    double confidence{0.0};
    bool valid{false};
    uint64_t timestamp_ns{0};
};

struct RotorCommand {
    // Collective thrust [N] and body moments [N*m] before motor allocation
    double collective_thrust_n{0.0};
    Vec3 body_moment_nm{};
    // Individual rotor thrusts [N] for 4 motors (FR, FL, RL, RR)
    std::array<double, 4> motor_thrusts_n{0.0, 0.0, 0.0, 0.0};
};

}  // namespace aegis::core
