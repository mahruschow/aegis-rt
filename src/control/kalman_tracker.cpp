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
    last_nis_ = 0.0;
    adaptive_q_scale_ = 1.0;
    accepted_updates_ = 0;
    rejected_outliers_ = 0;
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

    const double q_vel_eff = cfg_.process_noise_vel * adaptive_q_scale_;
    for (int i = 0; i < 3; ++i) {
        FPFt[i * 6 + i] += cfg_.process_noise_pos * dt_s;
        FPFt[(i + 3) * 6 + (i + 3)] += q_vel_eff * dt_s;
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

    // Account for perception pipeline delay (current_ts - capture_ts) using the
    // latency-compensated linear measurement model z_k = p_now - delay_s * v_now,
    // whose exact Jacobian is H = [I_3x3, -delay_s * I_3x3].
    const double delay_s = (current_timestamp_ns > capture_timestamp_ns)
                               ? std::clamp(static_cast<double>(current_timestamp_ns - capture_timestamp_ns) * 1e-9, 0.0, 0.10)
                               : 0.0;

    const core::Vec3 expected_at_capture{
        x_[0] - x_[3] * delay_s,
        x_[1] - x_[4] * delay_s,
        x_[2] - x_[5] * delay_s,
    };

    const core::Vec3 y = measured_pos_world - expected_at_capture;
    if (cfg_.innovation_gate_m > 0.0 && y.norm() > cfg_.innovation_gate_m) {
        ++rejected_outliers_;
        return false;
    }

    // Scale measurement noise inversely with detector confidence
    const double conf_clamped = std::clamp(confidence, 0.2, 1.0);
    const double r_var = cfg_.measurement_noise_pos / (conf_clamped * conf_clamped);

    // P_HT = P * H^T (6x3): column c is P[:, c] - delay_s * P[:, c + 3]
    std::array<double, 18> P_HT{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            P_HT[r * 3 + c] = P_[r * 6 + c] - delay_s * P_[r * 6 + (c + 3)];
        }
    }

    // Innovation covariance S = H * P * H^T + R (3x3)
    std::array<double, 9> S{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            S[r * 3 + c] = P_HT[r * 3 + c] - delay_s * P_HT[(r + 3) * 3 + c] + ((r == c) ? r_var : 0.0);
        }
    }

    std::array<double, 9> S_inv{};
    if (!invert_3x3(S, S_inv)) {
        return false;
    }

    // Normalized Innovation Squared (squared Mahalanobis distance): d^2 = y^T * S^-1 * y
    const std::array<double, 3> y_arr{y.x, y.y, y.z};
    double nis = 0.0;
    for (int r = 0; r < 3; ++r) {
        double s_inv_y = 0.0;
        for (int c = 0; c < 3; ++c) {
            s_inv_y += S_inv[r * 3 + c] * y_arr[c];
        }
        nis += y_arr[r] * s_inv_y;
    }
    last_nis_ = nis;

    // Statistical Mahalanobis outlier gate (3-DOF chi-squared)
    if (cfg_.mahalanobis_gate_chi2 > 0.0 && nis > cfg_.mahalanobis_gate_chi2) {
        ++rejected_outliers_;
        return false;
    }

    // NIS-driven adaptive process noise scaling for maneuvering targets
    if (cfg_.maneuver_nis_threshold > 0.0 && nis > cfg_.maneuver_nis_threshold) {
        const double ratio = nis / cfg_.maneuver_nis_threshold;
        const double target_scale = std::clamp(ratio, 1.0, cfg_.adaptive_q_max_scale);
        adaptive_q_scale_ = std::max(adaptive_q_scale_, target_scale);
    } else {
        adaptive_q_scale_ = 1.0 + cfg_.adaptive_q_decay * (adaptive_q_scale_ - 1.0);
    }

    // Kalman gain K = P * H^T * S^-1 (6x3 matrix)
    std::array<double, 18> K{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += P_HT[r * 3 + k] * S_inv[k * 3 + c];
            }
            K[r * 3 + c] = sum;
        }
    }

    // State update: x = x + K * y
    for (int r = 0; r < 6; ++r) {
        x_[r] += K[r * 3 + 0] * y_arr[0] + K[r * 3 + 1] * y_arr[1] + K[r * 3 + 2] * y_arr[2];
    }

    // Joseph-form covariance update: P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
    // Preserves symmetry and positive-definiteness across long flight horizons.
    Mat6 A{};  // I - K * H, where H = [I_3, -delay_s * I_3]
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double val = (r == c) ? 1.0 : 0.0;
            if (c < 3) {
                val -= K[r * 3 + c];
            } else {
                val += delay_s * K[r * 3 + (c - 3)];
            }
            A[r * 6 + c] = val;
        }
    }

    Mat6 AP{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += A[r * 6 + k] * P_[k * 6 + c];
            }
            AP[r * 6 + c] = sum;
        }
    }

    Mat6 P_new{};
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                sum += AP[r * 6 + k] * A[c * 6 + k];
            }
            const double krk = r_var * (K[r * 3 + 0] * K[c * 3 + 0] +
                                        K[r * 3 + 1] * K[c * 3 + 1] +
                                        K[r * 3 + 2] * K[c * 3 + 2]);
            const double sym_val = sum + krk;
            P_new[r * 6 + c] = sym_val;
            P_new[c * 6 + r] = sym_val;
        }
    }
    P_ = P_new;
    last_timestamp_ns_ = effective_ts;
    last_confidence_ = confidence;
    ++accepted_updates_;
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

double KalmanTracker3D::position_covariance_trace() const noexcept {
    return P_[0] + P_[7] + P_[14];
}

double KalmanTracker3D::velocity_covariance_trace() const noexcept {
    return P_[21] + P_[28] + P_[35];
}

}  // namespace aegis::control
