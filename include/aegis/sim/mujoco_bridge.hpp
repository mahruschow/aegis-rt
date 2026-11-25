#pragma once

#include "aegis/control/controller.hpp"
#include "aegis/core/frame_types.hpp"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <array>
#include <string>
#include <vector>

namespace aegis::sim {

/**
 * @brief Deterministic MuJoCo physics plant + offscreen OpenGL RGB-D sensor bridge.
 *
 * Steps 6-DOF quadrotor dynamics at 200 Hz (dt = 5 ms) using RK4 integration while
 * capturing 640x480 RGB + linearized metric depth frames at 60 Hz.
 */
class MujocoBridge {
public:
    struct Config {
        std::string scene_xml_path{"assets/scenes/pursuit_arena.xml"};
        bool headless{true};
        int cam_width{core::kCameraWidth};
        int cam_height{core::kCameraHeight};
    };

    explicit MujocoBridge(const Config& cfg);
    ~MujocoBridge();

    MujocoBridge(const MujocoBridge&) = delete;
    MujocoBridge& operator=(const MujocoBridge&) = delete;

    void reset();

    // Advance target figure-8 trajectory, apply 4 motor thrusts, and step physics `substeps` times
    void step_physics(const core::RotorCommand& cmd, int substeps = 3);

    // Render FPV RGB + linearized metric depth [m] into `out_slot` and populate `out_desc`
    void capture_rgbd(core::RawFrameSlot& out_slot,
                      core::FrameDescriptor& out_desc,
                      uint64_t sequence_id);

    // Render interactive third-person chase viewer window if `headless == false`
    bool render_interactive_viewer();

    // Back-project 2D pixel detection + metric depth patch into 3D world coordinates
    [[nodiscard]] static core::Vec3 backproject_pixel_to_world(
        const core::Detection2D& det,
        const core::RawFrameSlot& slot,
        const core::FrameDescriptor& desc,
        const core::CameraIntrinsics& intrinsics) noexcept;

    [[nodiscard]] core::DroneState current_drone_state() const noexcept;
    [[nodiscard]] core::Vec3 current_target_gt_pos() const noexcept;
    [[nodiscard]] double sim_time_s() const noexcept { return data_ ? data_->time : 0.0; }
    [[nodiscard]] bool is_rover() const noexcept { return is_rover_; }
    [[nodiscard]] core::CameraIntrinsics camera_intrinsics() const noexcept { return intrinsics_; }

    [[nodiscard]] const std::vector<control::ObstacleCylinder>& obstacles() const noexcept {
        return obstacles_;
    }

private:
    void update_target_trajectory(double t_s) noexcept;

    Config cfg_;
    mjModel* model_{nullptr};
    mjData* data_{nullptr};
    bool is_rover_{false};

    GLFWwindow* window_{nullptr};
    mjvCamera fpv_cam_{};
    mjvCamera viewer_cam_{};
    mjvOption vopt_{};
    mjvScene scn_{};
    mjrContext con_{};

    int drone_body_id_{-1};
    int target_mocap_id_{-1};
    int fpv_cam_id_{-1};
    core::CameraIntrinsics intrinsics_{};
    std::vector<control::ObstacleCylinder> obstacles_;

    // Pre-allocated bottom-up OpenGL readback scratch buffers (zero runtime heap alloc)
    std::vector<uint8_t> gl_rgb_scratch_;
    std::vector<float> gl_depth_scratch_;
};

}  // namespace aegis::sim
