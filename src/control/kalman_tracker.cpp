#include "aegis/control/kalman_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace aegis::control {

namespace {
// Invert symmetric positive-definite 3x3 matrix S
bool invert_3x3(const std::array<double, 9>& m, std::array<double, 9>& inv_out) noexcept {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];

    const double A = (e * i - f * h);
    const double B = -(d * i - f * g);
    const double C = (d * h - e * g);
    const double det = a * A + b * B + c * C;

    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double inv_det = 1.0 / det;
    inv_out[0] = A * inv_det;
    inv_out[1] = -(b * i - c * h) * inv_det;
    inv_out[2] = (b * f - c * e) * inv_det;
    inv_out[3] = B * inv_det;
    inv_out[4] = (a * i - c * g) * inv_det;
    inv_out[5] = -(a * f - c * d) * inv_det;
    inv_out[6] = C * inv_det;
    inv_out[7] = -(a * h - b * g) * inv_det;
    inv_out[8] = (a * e - b * d) * inv_det;
    return true;
}
}  // namespace

KalmanTracker3D::KalmanTracker3D(const Config& cfg) noexcept : cfg_(cfg) {}

void KalmanTracker3D::reset(const core::Vec3& initial_pos, uint64_t timestamp_ns) noexcept {
    x_ = {initial_pos.x, initial_pos.y, initial_pos.z, 0.0, 0.0, 0.0};
    P_.fill(0.0);
    for (int i = 0; i < 3; ++i) {
        P_[i * 6 + i] = 0.25;              // Initial position variance
        P_[(i + 3) * 6 + (i + 3)] = 2.0;   // Initial velocity variance
    }
    last_timestamp_ns_ = timestamp_ns;
    last_confidence_ = 1.0;
    initialized_ = true;
}

void KalmanTracker3D::predict(double dt_s) noexcept {
    if (!initialized_ || dt_s <= 0.0) {
        return;
    }
    dt_s = std::clamp(dt_s, 1e-5, 0.25);

    // State transition: p_k+1 = p_k + v_k * dt
    x_[0] += x_[3] * dt_s;
    x_[1] += x_[4] * dt_s;
    x_[2] += x_[5] * dt_s;

    // Covariance transition: P = F * P * F^T + Q
    // Since F = [[I, dt*I], [0, I]], expand analytically in O(1) without heap matrices:
    Mat6 FP = P_;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 6; ++c) {
            FP[r * 6 + c] += dt_s * P_[(r + 3) * 6 + c];
        }
    }
    Mat6 FPFt = FP;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            FPFt[r * 6 + c] += FP[r * 6 + (c + 3)] * dt_s;
        }
    }

    for (int i = 0; i < 3; ++i) {
        FPFt[i * 6 + i] += cfg_.process_noise_pos * dt_s;
        FPFt[(i + 3) * 6 + (i + 3)] += cfg_.process_noise_vel * dt_s;
    }
    P_ = FPFt;
}

bool KalmanTracker3D::update_with_latency_compensation(
    const core::Vec3& measured_pos_world,
    double confidence,
    uint64_t capture_timestamp_ns,
    uint64_t current_timestamp_ns) noexcept {

    const uint64_t effective_ts = std::max(capture_timestamp_ns, current_timestamp_ns);
    if (!initialized_) {
        reset(measured_pos_world, effective_ts);
        last_confidence_ = confidence;
        return true;
    }

    // Innovation residual y = z - H*x (where H = [I_3x3, 0_3x3])
    // Note: Account for perception pipeline delay (current_ts - capture_ts) by comparing
    // against back-projected state at capture time: p_at_capture ~= p_now - v_now * delay
    const double delay_s = (current_timestamp_ns > capture_timestamp_ns)
                               ? std::clamp(static_cast<double>(current_timestamp_ns - capture_timestamp_ns) * 1e-9, 0.0, 0.10)
                               : 0.0;

    const core::Vec3 expected_at_capture{
        x_[0] - x_[3] * delay_s,
        x_[1] - x_[4] * delay_s,
        x_[2] - x_[5] * delay_s,
    };

    const core::Vec3 y = measured_pos_world - expected_at_capture;
    if (y.norm() > cfg_.innovation_gate_m) {
        // Outlier gate triggered — reject spurious detection
        return false;
    }

    // Scale measurement noise inversely with detector confidence
    const double conf_clamped = std::clamp(confidence, 0.2, 1.0);
    const double r_var = cfg_.measurement_noise_pos / (conf_clamped * conf_clamped);

    // S = H * P * H^T + R (top-left 3x3 of P + r_var * I)
    std::array<double, 9> S{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            S[r * 3 + c] = P_[r * 6 + c] + ((r == c) ? r_var : 0.0);
        }
    }

    std::array<double, 9> S_inv{};
    if (!invert_3x3(S, S_inv)) {
        return false;
    }

    // Kalman gain K = P * H^T * S^-1 (6x3 matrix)
    std::array<double, 18> K{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += P_[r * 6 + k] * S_inv[k * 3 + c];
            }
            K[r * 3 + c] = sum;
        }
    }

    // State update: x = x + K * y
    const std::array<double, 3> y_arr{y.x, y.y, y.z};
    for (int r = 0; r < 6; ++r) {
        x_[r] += K[r * 3 + 0] * y_arr[0] + K[r * 3 + 1] * y_arr[1] + K[r * 3 + 2] * y_arr[2];
    }

    // Covariance update: P = (I - K * H) * P
    Mat6 I_KH{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double val = (r == c) ? 1.0 : 0.0;
            if (c < 3) {
                val -= K[r * 3 + c];
            }
            I_KH[r * 6 + c] = val;
        }
    }

    Mat6 P_new{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += I_KH[r * 6 + k] * P_[k * 6 + c];
            }
            P_new[r * 6 + c] = sum;
        }
    }
    P_ = P_new;
    last_timestamp_ns_ = effective_ts;
    last_confidence_ = confidence;
    return true;
}

core::TargetEstimate3D KalmanTracker3D::estimate() const noexcept {
    core::TargetEstimate3D est{};
    est.pos_world = {x_[0], x_[1], x_[2]};
    est.vel_world = {x_[3], x_[4], x_[5]};
    est.confidence = last_confidence_;
    est.valid = initialized_;
    est.timestamp_ns = last_timestamp_ns_;
    return est;
}

}  // namespace aegis::control
