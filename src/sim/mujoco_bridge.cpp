#include "aegis/sim/mujoco_bridge.hpp"
#include "aegis/core/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aegis::sim {

MujocoBridge::MujocoBridge(const Config& cfg) : cfg_(cfg) {
    char err_buf[1024] = {0};
    model_ = mj_loadXML(cfg_.scene_xml_path.c_str(), nullptr, err_buf, sizeof(err_buf));
    if (!model_) {
        throw std::runtime_error(std::string("Failed to load MuJoCo XML '") +
                                 cfg_.scene_xml_path + "': " + err_buf);
    }
    data_ = mj_makeData(model_);
    if (!data_) {
        mj_deleteModel(model_);
        throw std::runtime_error("mj_makeData failed.");
    }

    drone_body_id_ = mj_name2id(model_, mjOBJ_BODY, "quadrotor");
    if (drone_body_id_ < 0) {
        drone_body_id_ = mj_name2id(model_, mjOBJ_BODY, "ackermann_rover");
        is_rover_ = (drone_body_id_ >= 0);
    }
    const int target_body_id = mj_name2id(model_, mjOBJ_BODY, "target_beacon");
    if (target_body_id >= 0) {
        target_mocap_id_ = model_->body_mocapid[target_body_id];
    }
    fpv_cam_id_ = mj_name2id(model_, mjOBJ_CAMERA, "fpv_rgb");
    if (drone_body_id_ < 0 || target_mocap_id_ < 0 || fpv_cam_id_ < 0) {
        throw std::runtime_error("Scene XML missing required bodies/cameras (quadrotor/ackermann_rover, target_beacon, fpv_rgb).");
    }

    // Compute exact pinhole intrinsics from MuJoCo camera vertical FOV
    const double fovy_rad = model_->cam_fovy[fpv_cam_id_] * (3.14159265358979323846 / 180.0);
    intrinsics_.fy = (0.5 * static_cast<double>(cfg_.cam_height)) / std::tan(0.5 * fovy_rad);
    intrinsics_.fx = intrinsics_.fy;
    intrinsics_.cx = 0.5 * static_cast<double>(cfg_.cam_width);
    intrinsics_.cy = 0.5 * static_cast<double>(cfg_.cam_height);

    // Extract cylindrical obstacles from model geoms ("obs_*")
    for (int g = 0; g < model_->ngeom; ++g) {
        const char* gname = mj_id2name(model_, mjOBJ_GEOM, g);
        if (gname && std::strncmp(gname, "obs_", 4) == 0) {
            control::ObstacleCylinder cyl{};
            cyl.x = model_->geom_pos[g * 3 + 0];
            cyl.y = model_->geom_pos[g * 3 + 1];
            cyl.radius = model_->geom_size[g * 3 + 0];
            obstacles_.push_back(cyl);
        }
    }

    // Initialize OpenGL context (hidden window for deterministic headless mode, visible for demo mode)
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW for OpenGL offscreen sensor rendering.");
    }
    glfwWindowHint(GLFW_VISIBLE, cfg_.headless ? GLFW_FALSE : GLFW_TRUE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    const int win_w = cfg_.headless ? cfg_.cam_width : 1120;
    const int win_h = cfg_.headless ? cfg_.cam_height : 720;
    window_ = glfwCreateWindow(win_w, win_h, "Aegis-RT: Autonomous Quadrotor Pursuit", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed.");
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // Never block real-time benchmark loop on display VSync

    mjv_defaultCamera(&fpv_cam_);
    fpv_cam_.type = mjCAMERA_FIXED;
    fpv_cam_.fixedcamid = fpv_cam_id_;

    mjv_defaultCamera(&viewer_cam_);
    viewer_cam_.type = mjCAMERA_TRACKING;
    viewer_cam_.trackbodyid = drone_body_id_;
    viewer_cam_.distance = 5.2;
    viewer_cam_.azimuth = 145.0;
    viewer_cam_.elevation = -28.0;

    mjv_defaultOption(&vopt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);
    mjv_makeScene(model_, &scn_, 2000);
    mjr_makeContext(model_, &con_, mjFONTSCALE_150);

    gl_rgb_scratch_.resize(static_cast<std::size_t>(cfg_.cam_width) * cfg_.cam_height * 3);
    gl_depth_scratch_.resize(static_cast<std::size_t>(cfg_.cam_width) * cfg_.cam_height);

    reset();
}

MujocoBridge::~MujocoBridge() {
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    if (window_) {
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
    if (data_) mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
}

void MujocoBridge::reset() {
    mj_resetData(model_, data_);
    update_target_trajectory(0.0);
    mj_forward(model_, data_);
}

void MujocoBridge::update_target_trajectory(double t_s) noexcept {
    if (is_rover_) {
        // Ground-level S-curve slalom trajectory for 4-wheel Ackermann UGV pursuit
        const double x = 2.5 + 1.15 * t_s;
        const double y = 0.85 * std::sin(0.75 * t_s);
        data_->mocap_pos[target_mocap_id_ * 3 + 0] = x;
        data_->mocap_pos[target_mocap_id_ * 3 + 1] = y;
        data_->mocap_pos[target_mocap_id_ * 3 + 2] = 0.32;
        return;
    }
    // Agile 3D figure-8 Lissajous evasive trajectory weaving around arena pillars
    constexpr double kOmega = 0.55;  // rad/s
    const double x = 2.8 + 1.65 * std::sin(kOmega * t_s);
    const double y = 2.15 * std::sin(2.0 * kOmega * t_s) * 0.5;
    const double z = 1.15 + 0.28 * std::sin(1.5 * kOmega * t_s);

    data_->mocap_pos[target_mocap_id_ * 3 + 0] = x;
    data_->mocap_pos[target_mocap_id_ * 3 + 1] = y;
    data_->mocap_pos[target_mocap_id_ * 3 + 2] = z;
}

void MujocoBridge::step_physics(const core::RotorCommand& cmd, int substeps) {
    AEGIS_NVTX_RANGE("MuJoCo_StepPhysics");
    for (int i = 0; i < 4 && i < model_->nu; ++i) {
        data_->ctrl[i] = cmd.motor_thrusts_n[static_cast<std::size_t>(i)];
    }
    for (int s = 0; s < substeps; ++s) {
        update_target_trajectory(data_->time);
        mj_step(model_, data_);
    }
}

core::DroneState MujocoBridge::current_drone_state() const noexcept {
    core::DroneState st{};
    // Freejoint qpos layout: [px, py, pz, qw, qx, qy, qz], qvel: [vx, vy, vz, wx, wy, wz]
    st.pos = {data_->qpos[0], data_->qpos[1], data_->qpos[2]};
    st.quat = {data_->qpos[3], data_->qpos[4], data_->qpos[5], data_->qpos[6]};
    st.vel = {data_->qvel[0], data_->qvel[1], data_->qvel[2]};
    st.omega_body = {data_->qvel[3], data_->qvel[4], data_->qvel[5]};
    st.timestamp_ns = core::now_ns();
    return st;
}

core::Vec3 MujocoBridge::current_target_gt_pos() const noexcept {
    return {
        data_->mocap_pos[target_mocap_id_ * 3 + 0],
        data_->mocap_pos[target_mocap_id_ * 3 + 1],
        data_->mocap_pos[target_mocap_id_ * 3 + 2]};
}

void MujocoBridge::capture_rgbd(core::RawFrameSlot& out_slot,
                                core::FrameDescriptor& out_desc,
                                uint64_t sequence_id) {
    AEGIS_NVTX_RANGE("Sensor_Capture_RGBD");
    mjr_setBuffer(mjFB_OFFSCREEN, &con_);
    const mjrRect viewport{0, 0, cfg_.cam_width, cfg_.cam_height};

    mjv_updateScene(model_, data_, &vopt_, nullptr, &fpv_cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);
    mjr_readPixels(gl_rgb_scratch_.data(), gl_depth_scratch_.data(), viewport, &con_);

    // Linearize OpenGL NDC depth buffer [0, 1] -> metric optical depth [m] AND flip bottom-up rows
    const float extent = static_cast<float>(model_->stat.extent);
    const float z_near = static_cast<float>(model_->vis.map.znear) * extent;
    const float z_far  = static_cast<float>(model_->vis.map.zfar) * extent;
    const float z_ratio = 1.0f - (z_near / z_far);

    const int w = cfg_.cam_width;
    const int h = cfg_.cam_height;
    const std::size_t row_rgb_bytes = static_cast<std::size_t>(w) * 3;

    for (int y = 0; y < h; ++y) {
        const int gl_y = (h - 1) - y;
        std::memcpy(out_slot.rgb.data() + static_cast<std::size_t>(y) * row_rgb_bytes,
                    gl_rgb_scratch_.data() + static_cast<std::size_t>(gl_y) * row_rgb_bytes,
                    row_rgb_bytes);

        const float* src_d = gl_depth_scratch_.data() + static_cast<std::size_t>(gl_y) * w;
        float* dst_d = out_slot.depth_m.data() + static_cast<std::size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const float z_ndc = src_d[x];
            dst_d[x] = z_near / std::max(1e-6f, 1.0f - z_ndc * z_ratio);
        }
    }

    out_desc.sequence_id = sequence_id;
    out_desc.capture_timestamp_ns = core::now_ns();
    out_desc.drone_state = current_drone_state();
    out_desc.gt_target_pos_world = current_target_gt_pos();

    // Record exact camera world pose (`cam_xpos` and `cam_xmat` converted to quaternion)
    out_desc.camera_pos_world = {
        data_->cam_xpos[fpv_cam_id_ * 3 + 0],
        data_->cam_xpos[fpv_cam_id_ * 3 + 1],
        data_->cam_xpos[fpv_cam_id_ * 3 + 2]};

    mjtNum cam_q[4];
    mju_mat2Quat(cam_q, data_->cam_xmat + fpv_cam_id_ * 9);
    out_desc.camera_quat_world = {cam_q[0], cam_q[1], cam_q[2], cam_q[3]};
}

bool MujocoBridge::render_interactive_viewer() {
    if (cfg_.headless || !window_) {
        return true;
    }
    if (glfwWindowShouldClose(window_)) {
        return false;
    }
    mjr_setBuffer(mjFB_WINDOW, &con_);
    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    const mjrRect viewport{0, 0, fb_w, fb_h};

    mjv_updateScene(model_, data_, &vopt_, nullptr, &viewer_cam_, mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);
    glfwSwapBuffers(window_);
    glfwPollEvents();
    return true;
}

core::Vec3 MujocoBridge::backproject_pixel_to_world(
    const core::Detection2D& det,
    const core::RawFrameSlot& slot,
    const core::FrameDescriptor& desc,
    const core::CameraIntrinsics& intrinsics) noexcept {

    const int cx = std::clamp(static_cast<int>( std::round(det.cx_px) ), 0, core::kCameraWidth - 1);
    const int cy = std::clamp(static_cast<int>( std::round(det.cy_px) ), 0, core::kCameraHeight - 1);

    // Sample minimum valid metric depth in a tight 5x5 patch around bbox center
    // (robust against background sky pixels at bbox edges; adds sphere radius 0.28m to estimate center)
    float min_depth = 50.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        const int py = std::clamp(cy + dy, 0, core::kCameraHeight - 1);
        for (int dx = -2; dx <= 2; ++dx) {
            const int px = std::clamp(cx + dx, 0, core::kCameraWidth - 1);
            const float d = slot.depth_m[static_cast<std::size_t>(py) * core::kCameraWidth + px];
            if (d > 0.20f && d < min_depth) {
                min_depth = d;
            }
        }
    }
    const double z_optical_m = static_cast<double>(min_depth) + 0.22;  // Surface-to-centroid offset

    // MuJoCo camera frame: +X right, +Y up, -Z forward optical axis
    const core::Vec3 p_cam{
        (static_cast<double>(det.cx_px) - intrinsics.cx) * z_optical_m / intrinsics.fx,
        (intrinsics.cy - static_cast<double>(det.cy_px)) * z_optical_m / intrinsics.fy,
        -z_optical_m};

    return desc.camera_pos_world + desc.camera_quat_world.rotate_body_to_world(p_cam);
}

}  // namespace aegis::sim
