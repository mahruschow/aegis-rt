#include "aegis/backends/i_backend.hpp"
#include "aegis/backends/ort_backend.hpp"
#if defined(AEGIS_ENABLE_TENSORRT)
#include "aegis/backends/trt_backend.hpp"
#endif
#include "aegis/control/controller.hpp"
#include "aegis/control/kalman_tracker.hpp"
#include "aegis/control/rover_controller.hpp"
#include "aegis/core/frame_types.hpp"
#include "aegis/core/spsc_queue.hpp"
#include "aegis/core/telemetry.hpp"
#include "aegis/sim/mujoco_bridge.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct CliOptions {
    std::string vehicle{"drone"};  // "rover" (Ackermann UGV) or "drone" (6-DOF Quadrotor UAV)
    std::string scene_path{};
    std::string model_path{"assets/models/aegis_detector.onnx"};
    std::string engine_dir{"assets/models"};
    std::string calib_dir{"assets/calib_frames"};
    std::string backend{"ort-cpu"};
    std::string csv_out{};
    std::string dump_dataset_dir{};
    int num_frames{600};
    bool gui{false};
    bool jetson_zerocopy{false};
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_val = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + arg);
            }
            return argv[++i];
        };
        if (arg == "--vehicle") opts.vehicle = next_val();
        else if (arg == "--scene") opts.scene_path = next_val();
        else if (arg == "--model") opts.model_path = next_val();
        else if (arg == "--engine-dir") opts.engine_dir = next_val();
        else if (arg == "--calib-dir") opts.calib_dir = next_val();
        else if (arg == "--backend") opts.backend = next_val();
        else if (arg == "--csv") opts.csv_out = next_val();
        else if (arg == "--dump-dataset") opts.dump_dataset_dir = next_val();
        else if (arg == "--frames") opts.num_frames = std::stoi(next_val());
        else if (arg == "--gui") opts.gui = true;
        else if (arg == "--jetson-zerocopy") opts.jetson_zerocopy = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: aegis_rt [--vehicle rover|drone] [--backend gt|ort-cpu|ort-coreml|ort-cuda|trt-fp32|trt-fp16|trt-int8]\n"
                      << "                [--frames N] [--scene XML] [--model ONNX] [--csv OUT.csv]\n"
                      << "                [--dump-dataset DIR] [--gui] [--jetson-zerocopy]\n";
            std::exit(0);
        }
    }
    if (opts.scene_path.empty()) {
        opts.scene_path = (opts.vehicle == "rover")
                              ? "assets/scenes/rover_arena.xml"
                              : "assets/scenes/pursuit_arena.xml";
    }
    return opts;
}

// Project ground-truth 3D sphere into 2D camera bounding box for dataset & calibration generation
bool project_sphere_to_bbox(const aegis::core::Vec3& sphere_world,
                            double radius_m,
                            const aegis::core::FrameDescriptor& desc,
                            const aegis::core::CameraIntrinsics& intr,
                            float& out_cx,
                            float& out_cy,
                            float& out_w,
                            float& out_h) noexcept {
    // Transform world -> camera frame: p_cam = R_cam^T * (p_world - cam_pos)
    const aegis::core::Quat q_inv{
        desc.camera_quat_world.w,
        -desc.camera_quat_world.x,
        -desc.camera_quat_world.y,
        -desc.camera_quat_world.z};
    const aegis::core::Vec3 p_cam = q_inv.rotate_body_to_world(sphere_world - desc.camera_pos_world);

    // MuJoCo optical axis is -Z_cam
    const double z_depth = -p_cam.z;
    if (z_depth <= 0.35) {
        return false;
    }

    const double u = intr.cx + (p_cam.x / z_depth) * intr.fx;
    const double v = intr.cy - (p_cam.y / z_depth) * intr.fy;
    const double box_px = (2.0 * radius_m / z_depth) * intr.fy;

    if (u < -20.0 || u >= aegis::core::kCameraWidth + 20.0 ||
        v < -20.0 || v >= aegis::core::kCameraHeight + 20.0) {
        return false;
    }

    out_cx = static_cast<float>(std::clamp(u, 0.0, static_cast<double>(aegis::core::kCameraWidth - 1)));
    out_cy = static_cast<float>(std::clamp(v, 0.0, static_cast<double>(aegis::core::kCameraHeight - 1)));
    out_w  = static_cast<float>(std::clamp(box_px, 8.0, static_cast<double>(aegis::core::kCameraWidth)));
    out_h  = static_cast<float>(std::clamp(box_px, 8.0, static_cast<double>(aegis::core::kCameraHeight)));
    return true;
}

std::unique_ptr<aegis::backends::IPerceptionBackend> create_backend(const CliOptions& opts) {
    using namespace aegis::backends;
    if (opts.backend == "gt") {
        return nullptr;
    }
    if (opts.backend == "ort-cpu") {
        return std::make_unique<OrtBackend>(opts.model_path, OrtBackend::ExecutionProvider::kCpu);
    }
    if (opts.backend == "ort-coreml") {
        return std::make_unique<OrtBackend>(opts.model_path, OrtBackend::ExecutionProvider::kCoreML);
    }
    if (opts.backend == "ort-cuda") {
        return std::make_unique<OrtBackend>(opts.model_path, OrtBackend::ExecutionProvider::kCuda);
    }
#if defined(AEGIS_ENABLE_TENSORRT)
    if (opts.backend == "trt-fp32") {
        return std::make_unique<TrtBackend>(
            opts.model_path, opts.engine_dir + "/aegis_fp32.engine",
            TrtBackend::Precision::kFp32, opts.calib_dir, opts.jetson_zerocopy);
    }
    if (opts.backend == "trt-fp16") {
        return std::make_unique<TrtBackend>(
            opts.model_path, opts.engine_dir + "/aegis_fp16.engine",
            TrtBackend::Precision::kFp16, opts.calib_dir, opts.jetson_zerocopy);
    }
    if (opts.backend == "trt-int8") {
        return std::make_unique<TrtBackend>(
            opts.model_path, opts.engine_dir + "/aegis_int8.engine",
            TrtBackend::Precision::kInt8, opts.calib_dir, opts.jetson_zerocopy);
    }
#endif
    throw std::runtime_error("Unsupported or uncompiled backend: " + opts.backend);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions opts = parse_args(argc, argv);

        aegis::sim::MujocoBridge::Config sim_cfg{};
        sim_cfg.scene_xml_path = opts.scene_path;
        sim_cfg.headless = !opts.gui;
        aegis::sim::MujocoBridge sim(sim_cfg);

        auto backend = create_backend(opts);
        const std::string vehicle_prefix = sim.is_rover() ? "[Ackermann-UGV] " : "[Quadrotor-UAV] ";
        const std::string backend_label = vehicle_prefix + (backend ? backend->name() : "GroundTruth-Oracle");

        // Pre-allocate static ring buffer pool (8 slots) + lock-free SPSC descriptor queue
        constexpr std::size_t kPoolSize = 8;
        std::vector<aegis::core::RawFrameSlot> frame_pool(kPoolSize);
        aegis::core::SpscQueue<aegis::core::FrameDescriptor, kPoolSize> spsc_queue;

        aegis::control::KalmanTracker3D tracker;
        aegis::control::QuadrotorController drone_controller;
        aegis::control::AckermannRoverController rover_controller;
        aegis::core::TelemetryRecorder telemetry(static_cast<std::size_t>(opts.num_frames));

        auto compute_vehicle_cmd = [&](const aegis::core::DroneState& st,
                                       const aegis::core::TargetEstimate3D& est,
                                       double& out_clearance) -> aegis::core::RotorCommand {
            if (sim.is_rover()) {
                return rover_controller.compute_command(
                    st, est, sim.obstacles().data(), sim.obstacles().size(), out_clearance);
            }
            return drone_controller.compute_command(
                st, est, sim.obstacles().data(), sim.obstacles().size(), out_clearance);
        };

        // Shared atomic/latest target estimate consumed by the 200 Hz flight controller
        std::mutex est_mutex;
        aegis::core::TargetEstimate3D latest_estimate{};
        latest_estimate.pos_world = sim.current_target_gt_pos();
        latest_estimate.valid = true;
        tracker.reset(latest_estimate.pos_world, aegis::core::now_ns());

        std::ofstream labels_csv;
        if (!opts.dump_dataset_dir.empty()) {
            std::filesystem::create_directories(opts.dump_dataset_dir);
            labels_csv.open(opts.dump_dataset_dir + "/labels.csv");
            labels_csv << "frame_id,visible,cx_px,cy_px,w_px,h_px\n";
        }

        std::atomic<bool> stop_worker{false};
        const auto intrinsics = sim.camera_intrinsics();

        // Dedicated Asynchronous Perception Consumer Thread (reads from lock-free SPSC queue)
        std::thread perception_worker([&]() {
            aegis::core::FrameDescriptor desc{};
            constexpr double kSimFrameDt = 0.015;  // 3 physics substeps * 5ms RK4 = 15ms sim step

            while (!stop_worker.load(std::memory_order_relaxed) || !spsc_queue.empty()) {
                if (!spsc_queue.try_pop(desc)) {
                    std::this_thread::yield();
                    continue;
                }

                const uint64_t t_pop = aegis::core::now_ns();
                const auto& slot = frame_pool[desc.slot_index];

                aegis::backends::BackendTimingsUs timings{};
                aegis::core::Detection2D det{};

                if (backend) {
                    det = backend->infer(slot.rgb.data(),
                                         aegis::core::kCameraWidth,
                                         aegis::core::kCameraHeight,
                                         timings);
                } else {
                    // Ground-truth oracle mode (Phase 1 baseline / dataset collection)
                    det.valid = project_sphere_to_bbox(
                        desc.gt_target_pos_world, 0.28, desc, intrinsics,
                        det.cx_px, det.cy_px, det.width_px, det.height_px);
                    det.confidence = det.valid ? 1.0f : 0.0f;
                }

                const uint64_t t_ctrl_start = aegis::core::now_ns();
                tracker.predict(kSimFrameDt);

                bool accepted_meas = false;
                if (det.valid) {
                    const aegis::core::Vec3 measured_3d =
                        aegis::sim::MujocoBridge::backproject_pixel_to_world(
                            det, slot, desc, intrinsics);
                    const uint64_t sim_capture_ns = desc.sequence_id * 15'000'000ULL;
                    const uint64_t perc_delay_ns =
                        static_cast<uint64_t>((timings.preprocess_us + timings.inference_us) * 1000.0);
                    accepted_meas = tracker.update_with_latency_compensation(
                        measured_3d, det.confidence, sim_capture_ns, sim_capture_ns + perc_delay_ns);
                }

                const auto est = tracker.estimate();
                {
                    std::lock_guard<std::mutex> lk(est_mutex);
                    latest_estimate = est;
                }

                double min_obs_dist = 99.0;
                (void)compute_vehicle_cmd(desc.drone_state, est, min_obs_dist);
                const uint64_t t_done = aegis::core::now_ns();

                aegis::core::FrameTelemetrySample sample{};
                sample.sequence_id = desc.sequence_id;
                sample.sim_time_s = static_cast<double>(desc.sequence_id) * kSimFrameDt;
                sample.sensor_capture_us = desc.sensor_capture_us;
                sample.queue_wait_us = static_cast<double>(t_pop - desc.capture_timestamp_ns) * 1e-3;
                sample.preprocess_us = timings.preprocess_us;
                sample.inference_us = timings.inference_us;
                sample.control_us = static_cast<double>(t_done - t_ctrl_start) * 1e-3;
                sample.total_e2e_us = sample.preprocess_us + sample.inference_us + sample.control_us;
                sample.tracking_error_m = (est.pos_world - desc.gt_target_pos_world).norm();
                sample.drone_target_dist_m = (desc.drone_state.pos - desc.gt_target_pos_world).norm();
                sample.min_obstacle_dist_m = min_obs_dist;
                sample.target_detected = det.valid && accepted_meas;
                telemetry.record(sample);
            }
        });

        // Main Thread: MuJoCo OpenGL Sensor Producer (60 Hz) + High-Rate Flight Control (200 Hz)
        // Note: macOS Cocoa/NSOpenGL requires OpenGL rendering calls to remain on the main thread.
        for (int frame = 0; frame < opts.num_frames; ++frame) {
            const uint32_t slot_idx = static_cast<uint32_t>(frame % kPoolSize);
            aegis::core::FrameDescriptor desc{};
            desc.slot_index = slot_idx;

            const uint64_t t_cap0 = aegis::core::now_ns();
            sim.capture_rgbd(frame_pool[slot_idx], desc, static_cast<uint64_t>(frame));
            const uint64_t t_cap1 = aegis::core::now_ns();
            desc.sensor_capture_us = static_cast<double>(t_cap1 - t_cap0) * 1e-3;

            // Optional dataset / calibration frame dump
            if (!opts.dump_dataset_dir.empty()) {
                char fname[64];
                std::snprintf(fname, sizeof(fname), "/frame_%04d.rgb", frame);
                std::ofstream rgb_out(opts.dump_dataset_dir + fname, std::ios::binary);
                rgb_out.write(reinterpret_cast<const char*>(frame_pool[slot_idx].rgb.data()),
                              static_cast<std::streamsize>(aegis::core::kRgbFrameBytes));

                float cx = 0.0f, cy = 0.0f, bw = 0.0f, bh = 0.0f;
                const bool vis = project_sphere_to_bbox(
                    desc.gt_target_pos_world, 0.28, desc, intrinsics, cx, cy, bw, bh);
                labels_csv << frame << "," << (vis ? 1 : 0) << ","
                           << cx << "," << cy << "," << bw << "," << bh << "\n";
            }

            // Push sensor descriptor into lock-free SPSC ring buffer (wait if consumer busy during benchmark)
            while (!spsc_queue.try_push(desc)) {
                std::this_thread::yield();
            }

            // Wait for perception worker on deterministic frame-accurate benchmark mode so closed-loop
            // physics reflects exact perception accuracy, while measuring pure pipeline latency.
            while (!spsc_queue.empty()) {
                std::this_thread::yield();
            }

            // Step 200 Hz inner flight control & RK4 physics (3 substeps x 5ms = 15ms ~ 66.7 Hz)
            aegis::core::TargetEstimate3D est_copy{};
            {
                std::lock_guard<std::mutex> lk(est_mutex);
                est_copy = latest_estimate;
            }
            for (int sub = 0; sub < 3; ++sub) {
                double obs_clearance = 0.0;
                const auto drone_st = sim.current_drone_state();
                const auto rotor_cmd = compute_vehicle_cmd(drone_st, est_copy, obs_clearance);
                sim.step_physics(rotor_cmd, 1);
            }

            if (!sim.render_interactive_viewer()) {
                break;
            }
        }

        stop_worker.store(true, std::memory_order_relaxed);
        perception_worker.join();

        telemetry.print_summary(backend_label);
        if (!opts.csv_out.empty()) {
            std::filesystem::create_directories(std::filesystem::path(opts.csv_out).parent_path());
            if (telemetry.export_csv(opts.csv_out)) {
                std::cout << "[Telemetry] Saved CSV trace to: " << opts.csv_out << "\n";
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }
}
