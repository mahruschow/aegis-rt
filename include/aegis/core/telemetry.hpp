#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(AEGIS_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
namespace aegis::core {
class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* label) noexcept { nvtxRangePushA(label); }
    ~ScopedNvtxRange() noexcept { nvtxRangePop(); }
    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};
}  // namespace aegis::core
#define AEGIS_CONCAT_INNER(a, b) a##b
#define AEGIS_CONCAT(a, b) AEGIS_CONCAT_INNER(a, b)
#define AEGIS_NVTX_RANGE(label) ::aegis::core::ScopedNvtxRange AEGIS_CONCAT(_nvtx_guard_, __LINE__)(label)
#else
#define AEGIS_NVTX_RANGE(label) ((void)0)
#endif

namespace aegis::core {

[[nodiscard]] inline uint64_t now_ns() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

struct FrameTelemetrySample {
    uint64_t sequence_id{0};
    double sim_time_s{0.0};
    double sensor_capture_us{0.0};
    double queue_wait_us{0.0};
    double preprocess_us{0.0};
    double inference_us{0.0};
    double control_us{0.0};
    double total_e2e_us{0.0};
    double tracking_error_m{0.0};
    double drone_target_dist_m{0.0};
    double min_obstacle_dist_m{0.0};
    bool target_detected{false};
    bool deadline_missed{false};
};

struct PercentileStats {
    double min{0.0};
    double mean{0.0};
    double p50{0.0};
    double p95{0.0};
    double p99{0.0};
    double p99_9{0.0};
    double max{0.0};
    double std_dev{0.0};
};

class TelemetryRecorder {
public:
    explicit TelemetryRecorder(std::size_t max_frames, double frame_deadline_us = 16666.7)
        : deadline_us_(frame_deadline_us) {
        samples_.reserve(max_frames);
    }

    void record(FrameTelemetrySample sample) noexcept {
        sample.deadline_missed = (sample.total_e2e_us > deadline_us_);
        if (samples_.size() < samples_.capacity()) {
            samples_.push_back(sample);
        }
    }

    [[nodiscard]] const std::vector<FrameTelemetrySample>& samples() const noexcept {
        return samples_;
    }

    [[nodiscard]] static PercentileStats compute_stats(std::vector<double> values) {
        PercentileStats stats{};
        if (values.empty()) {
            return stats;
        }
        std::sort(values.begin(), values.end());
        const std::size_t n = values.size();
        stats.min = values.front();
        stats.max = values.back();

        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        stats.mean = sum / static_cast<double>(n);

        double sq_diff_sum = 0.0;
        for (double v : values) {
            const double d = v - stats.mean;
            sq_diff_sum += d * d;
        }
        stats.std_dev = std::sqrt(sq_diff_sum / static_cast<double>(n));

        auto pct = [&](double p) -> double {
            const double idx = p * static_cast<double>(n - 1);
            const auto lo = static_cast<std::size_t>(std::floor(idx));
            const auto hi = static_cast<std::size_t>(std::ceil(idx));
            const double frac = idx - static_cast<double>(lo);
            return values[lo] * (1.0 - frac) + values[hi] * frac;
        };

        stats.p50 = pct(0.50);
        stats.p95 = pct(0.95);
        stats.p99 = pct(0.99);
        stats.p99_9 = pct(0.999);
        return stats;
    }

    void print_summary(const std::string& backend_name) const {
        if (samples_.empty()) {
            std::cout << "[Telemetry] No samples recorded.\n";
            return;
        }

        auto extract = [&](auto member_ptr) {
            std::vector<double> vals;
            vals.reserve(samples_.size());
            for (const auto& s : samples_) {
                vals.push_back(s.*member_ptr);
            }
            return compute_stats( std::move(vals) );
        };

        const auto pre_s = extract(&FrameTelemetrySample::preprocess_us);
        const auto inf_s = extract(&FrameTelemetrySample::inference_us);
        const auto ctl_s = extract(&FrameTelemetrySample::control_us);
        const auto e2e_s = extract(&FrameTelemetrySample::total_e2e_us);

        double sq_err_sum = 0.0;
        std::size_t detected_count = 0;
        std::size_t missed_deadlines = 0;
        double min_obs_clearance = 1e9;

        for (const auto& s : samples_) {
            sq_err_sum += s.tracking_error_m * s.tracking_error_m;
            if (s.target_detected) ++detected_count;
            if (s.deadline_missed) ++missed_deadlines;
            min_obs_clearance = std::min(min_obs_clearance, s.min_obstacle_dist_m);
        }

        const double n = static_cast<double>(samples_.size());
        const double tracking_rmse = std::sqrt(sq_err_sum / n);
        const double detection_rate = 100.0 * static_cast<double>(detected_count) / n;
        const double effective_hz = (e2e_s.mean > 0.0) ? (1e6 / e2e_s.mean) : 0.0;

        std::cout << "\n========================================================================\n";
        std::cout << "  AEGIS-RT FLIGHT TELEMETRY SUMMARY — Backend: " << backend_name << "\n";
        std::cout << "========================================================================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Frames Evaluated       : " << samples_.size() << "\n";
        std::cout << "  Preprocess (p50 / p99) : " << pre_s.p50 / 1000.0 << " ms / " << pre_s.p99 / 1000.0 << " ms\n";
        std::cout << "  Inference  (p50 / p99) : " << inf_s.p50 / 1000.0 << " ms / " << inf_s.p99 / 1000.0 << " ms\n";
        std::cout << "  Control    (p50 / p99) : " << ctl_s.p50 / 1000.0 << " ms / " << ctl_s.p99 / 1000.0 << " ms\n";
        std::cout << "  Total E2E  (p50 / p99) : " << e2e_s.p50 / 1000.0 << " ms / " << e2e_s.p99 / 1000.0
                  << " ms (jitter std: " << e2e_s.std_dev / 1000.0 << " ms)\n";
        std::cout << "  Sustained Pipeline Hz  : " << std::setprecision(1) << effective_hz << " Hz\n";
        std::cout << "  Deadline Misses (>16.7): " << missed_deadlines << " / " << samples_.size() << "\n";
        std::cout << "  Target Detection Rate  : " << detection_rate << " %\n";
        std::cout << std::setprecision(3);
        std::cout << "  3D State Est. RMSE     : " << tracking_rmse << " m\n";
        std::cout << "  Min Obstacle Clearance : " << min_obs_clearance << " m\n";
        std::cout << "========================================================================\n\n";
    }

    bool export_csv(const std::string& path) const {
        std::ofstream out(path);
        if (!out.is_open()) {
            return false;
        }
        out << "sequence_id,sim_time_s,sensor_capture_us,queue_wait_us,preprocess_us,"
               "inference_us,control_us,total_e2e_us,tracking_error_m,drone_target_dist_m,"
               "min_obstacle_dist_m,target_detected,deadline_missed\n";
        out << std::fixed << std::setprecision(4);
        for (const auto& s : samples_) {
            out << s.sequence_id << ","
                << s.sim_time_s << ","
                << s.sensor_capture_us << ","
                << s.queue_wait_us << ","
                << s.preprocess_us << ","
                << s.inference_us << ","
                << s.control_us << ","
                << s.total_e2e_us << ","
                << s.tracking_error_m << ","
                << s.drone_target_dist_m << ","
                << s.min_obstacle_dist_m << ","
                << (s.target_detected ? 1 : 0) << ","
                << (s.deadline_missed ? 1 : 0) << "\n";
        }
        return true;
    }

private:
    double deadline_us_;
    std::vector<FrameTelemetrySample> samples_;
};

}  // namespace aegis::core
