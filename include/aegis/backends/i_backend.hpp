#pragma once

#include "aegis/core/frame_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace aegis::backends {

struct LetterboxMeta {
    int src_w{640};
    int src_h{480};
    int dst_w{320};
    int dst_h{320};
    float scale{0.5f};
    int pad_x{0};
    int pad_y{40};
    int resized_w{320};
    int resized_h{240};
};

[[nodiscard]] inline LetterboxMeta compute_letterbox_meta(int src_w, int src_h, int dst_w, int dst_h) noexcept {
    LetterboxMeta m{};
    m.src_w = src_w;
    m.src_h = src_h;
    m.dst_w = dst_w;
    m.dst_h = dst_h;
    const float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
    m.scale = std::min(scale_w, scale_h);
    m.resized_w = static_cast<int>(std::round(static_cast<float>(src_w) * m.scale));
    m.resized_h = static_cast<int>(std::round(static_cast<float>(src_h) * m.scale));
    m.pad_x = (dst_w - m.resized_w) / 2;
    m.pad_y = (dst_h - m.resized_h) / 2;
    return m;
}

/**
 * @brief Precomputed separable bilinear + letterbox plan for zero-allocation CPU preprocessing.
 *
 * Hoists all floating-point coordinate mapping, clamping, and floor operations out of the
 * per-pixel loop into 1D horizontal/vertical lookup tables, folds `1/255` scaling into vertical
 * weights, and uses `std::fill_n` for SIMD-vectorizable letterbox border fills.
 */
class CpuPreprocessPlan {
public:
    CpuPreprocessPlan() = default;

    CpuPreprocessPlan(int src_w, int src_h, int dst_w, int dst_h) {
        configure(src_w, src_h, dst_w, dst_h);
    }

    void configure(int src_w, int src_h, int dst_w, int dst_h) {
        if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return;
        }
        if (configured_ && meta_.src_w == src_w && meta_.src_h == src_h &&
            meta_.dst_w == dst_w && meta_.dst_h == dst_h) {
            return;
        }

        meta_ = compute_letterbox_meta(src_w, src_h, dst_w, dst_h);
        dx_begin_ = std::max(0, meta_.pad_x);
        dx_end_   = std::min(meta_.dst_w, meta_.pad_x + meta_.resized_w);
        dy_begin_ = std::max(0, meta_.pad_y);
        dy_end_   = std::min(meta_.dst_h, meta_.pad_y + meta_.resized_h);

        constexpr float kInv255 = 1.0f / 255.0f;
        const std::size_t row_stride = static_cast<std::size_t>(src_w) * 3;

        x_lut_.resize(static_cast<std::size_t>(meta_.resized_w));
        for (int inner_x = 0; inner_x < meta_.resized_w; ++inner_x) {
            const float src_xf = std::clamp(
                (static_cast<float>(inner_x) + 0.5f) / meta_.scale - 0.5f,
                0.0f, static_cast<float>(src_w - 1));
            const int x0 = static_cast<int>(std::floor(src_xf));
            const int x1 = std::min(x0 + 1, src_w - 1);
            const float wx1 = src_xf - static_cast<float>(x0);
            const float wx0 = 1.0f - wx1;
            x_lut_[static_cast<std::size_t>(inner_x)] = HorizontalSample{
                static_cast<std::size_t>(x0) * 3,
                static_cast<std::size_t>(x1) * 3,
                wx0,
                wx1
            };
        }

        y_lut_.resize(static_cast<std::size_t>(meta_.resized_h));
        for (int inner_y = 0; inner_y < meta_.resized_h; ++inner_y) {
            const float src_yf = std::clamp(
                (static_cast<float>(inner_y) + 0.5f) / meta_.scale - 0.5f,
                0.0f, static_cast<float>(src_h - 1));
            const int y0 = static_cast<int>(std::floor(src_yf));
            const int y1 = std::min(y0 + 1, src_h - 1);
            const float wy1 = src_yf - static_cast<float>(y0);
            const float wy0 = 1.0f - wy1;
            y_lut_[static_cast<std::size_t>(inner_y)] = VerticalSample{
                static_cast<std::size_t>(y0) * row_stride,
                static_cast<std::size_t>(y1) * row_stride,
                wy0 * kInv255,
                wy1 * kInv255
            };
        }

        configured_ = true;
    }

    [[nodiscard]] const LetterboxMeta& meta() const noexcept { return meta_; }
    [[nodiscard]] bool is_configured() const noexcept { return configured_; }

    void execute(const uint8_t* src_rgb_hwc, float* dst_nchw) const noexcept {
        if (!configured_) {
            return;
        }
        constexpr float kPadVal = 114.0f / 255.0f;
        const int dst_w = meta_.dst_w;
        const int dst_h = meta_.dst_h;
        const std::size_t plane_area = static_cast<std::size_t>(dst_w) * dst_h;

        float* ch_r = dst_nchw;
        float* ch_g = dst_nchw + plane_area;
        float* ch_b = dst_nchw + 2 * plane_area;

        const std::size_t top_pixels = static_cast<std::size_t>(dy_begin_) * dst_w;
        if (top_pixels > 0) {
            std::fill_n(ch_r, top_pixels, kPadVal);
            std::fill_n(ch_g, top_pixels, kPadVal);
            std::fill_n(ch_b, top_pixels, kPadVal);
        }

        const std::size_t bot_offset = static_cast<std::size_t>(dy_end_) * dst_w;
        const std::size_t bot_pixels = static_cast<std::size_t>(dst_h - dy_end_) * dst_w;
        if (bot_pixels > 0) {
            std::fill_n(ch_r + bot_offset, bot_pixels, kPadVal);
            std::fill_n(ch_g + bot_offset, bot_pixels, kPadVal);
            std::fill_n(ch_b + bot_offset, bot_pixels, kPadVal);
        }

        const int left_pad = dx_begin_;
        const int right_pad = dst_w - dx_end_;
        const int active_w = dx_end_ - dx_begin_;
        const HorizontalSample* xs_ptr = x_lut_.data() + (dx_begin_ - meta_.pad_x);

        for (int dy = dy_begin_; dy < dy_end_; ++dy) {
            const std::size_t row_off = static_cast<std::size_t>(dy) * dst_w;
            if (left_pad > 0) {
                std::fill_n(ch_r + row_off, left_pad, kPadVal);
                std::fill_n(ch_g + row_off, left_pad, kPadVal);
                std::fill_n(ch_b + row_off, left_pad, kPadVal);
            }
            if (right_pad > 0) {
                std::fill_n(ch_r + row_off + dx_end_, right_pad, kPadVal);
                std::fill_n(ch_g + row_off + dx_end_, right_pad, kPadVal);
                std::fill_n(ch_b + row_off + dx_end_, right_pad, kPadVal);
            }

            const VerticalSample ys = y_lut_[static_cast<std::size_t>(dy - meta_.pad_y)];
            const uint8_t* row0 = src_rgb_hwc + ys.row0_off;
            const uint8_t* row1 = src_rgb_hwc + ys.row1_off;
            const float wy0 = ys.wy0_norm;
            const float wy1 = ys.wy1_norm;

            float* dst_r = ch_r + row_off + dx_begin_;
            float* dst_g = ch_g + row_off + dx_begin_;
            float* dst_b = ch_b + row_off + dx_begin_;

            for (int i = 0; i < active_w; ++i) {
                const HorizontalSample xs = xs_ptr[i];
                const uint8_t* p00 = row0 + xs.off0;
                const uint8_t* p01 = row0 + xs.off1;
                const uint8_t* p10 = row1 + xs.off0;
                const uint8_t* p11 = row1 + xs.off1;

                const float h0_r = xs.wx0 * static_cast<float>(p00[0]) + xs.wx1 * static_cast<float>(p01[0]);
                const float h1_r = xs.wx0 * static_cast<float>(p10[0]) + xs.wx1 * static_cast<float>(p11[0]);
                const float h0_g = xs.wx0 * static_cast<float>(p00[1]) + xs.wx1 * static_cast<float>(p01[1]);
                const float h1_g = xs.wx0 * static_cast<float>(p10[1]) + xs.wx1 * static_cast<float>(p11[1]);
                const float h0_b = xs.wx0 * static_cast<float>(p00[2]) + xs.wx1 * static_cast<float>(p01[2]);
                const float h1_b = xs.wx0 * static_cast<float>(p10[2]) + xs.wx1 * static_cast<float>(p11[2]);

                dst_r[i] = wy0 * h0_r + wy1 * h1_r;
                dst_g[i] = wy0 * h0_g + wy1 * h1_g;
                dst_b[i] = wy0 * h0_b + wy1 * h1_b;
            }
        }
    }

private:
    struct HorizontalSample {
        std::size_t off0;
        std::size_t off1;
        float wx0;
        float wx1;
    };

    struct VerticalSample {
        std::size_t row0_off;
        std::size_t row1_off;
        float wy0_norm;
        float wy1_norm;
    };

    LetterboxMeta meta_{};
    int dx_begin_{0};
    int dx_end_{0};
    int dy_begin_{0};
    int dy_end_{0};
    bool configured_{false};
    std::vector<HorizontalSample> x_lut_{};
    std::vector<VerticalSample> y_lut_{};
};

/**
 * @brief Golden CPU reference implementation of fused letterbox + bilinear resize +
 *        [0, 1] normalization + HWC -> NCHW transpose.
 *
 * Used by OrtBackend on CPU/CoreML AND as the numerical ground truth in test_cuda_preprocess.cu
 * to verify <1e-5 parity of our custom CUDA kernel.
 */
inline void cpu_preprocess_letterbox_nchw(
    const uint8_t* src_rgb_hwc,
    const LetterboxMeta& meta,
    float* dst_nchw) noexcept {

    const int plane_Area = meta.dst_w * meta.dst_h;
    constexpr float kInv255 = 1.0f / 255.0f;
    constexpr float kPadVal = 114.0f / 255.0f;

    float* ch_r = dst_nchw;
    float* ch_g = dst_nchw + plane_Area;
    float* ch_b = dst_nchw + 2 * plane_Area;

    for (int dy = 0; dy < meta.dst_h; ++dy) {
        const int row_offset = dy * meta.dst_w;
        const int inner_y = dy - meta.pad_y;

        if (inner_y < 0 || inner_y >= meta.resized_h) {
            for (int dx = 0; dx < meta.dst_w; ++dx) {
                const int out_idx = row_offset + dx;
                ch_r[out_idx] = kPadVal;
                ch_g[out_idx] = kPadVal;
                ch_b[out_idx] = kPadVal;
            }
            continue;
        }

        // Pixel-center aligned inverse bilinear mapping
        const float src_yf = std::clamp(
            (static_cast<float>(inner_y) + 0.5f) / meta.scale - 0.5f,
            0.0f, static_cast<float>(meta.src_h - 1));
        const int y0 = static_cast<int>(std::floor(src_yf));
        const int y1 = std::min(y0 + 1, meta.src_h - 1);
        const float wy1 = src_yf - static_cast<float>(y0);
        const float wy0 = 1.0f - wy1;

        for (int dx = 0; dx < meta.dst_w; ++dx) {
            const int out_idx = row_offset + dx;
            const int inner_x = dx - meta.pad_x;

            if (inner_x < 0 || inner_x >= meta.resized_w) {
                ch_r[out_idx] = kPadVal;
                ch_g[out_idx] = kPadVal;
                ch_b[out_idx] = kPadVal;
                continue;
            }

            const float src_xf = std::clamp(
                (static_cast<float>(inner_x) + 0.5f) / meta.scale - 0.5f,
                0.0f, static_cast<float>(meta.src_w - 1));
            const int x0 = static_cast<int>(std::floor(src_xf));
            const int x1 = std::min(x0 + 1, meta.src_w - 1);
            const float wx1 = src_xf - static_cast<float>(x0);
            const float wx0 = 1.0f - wx1;

            const uint8_t* p00 = src_rgb_hwc + (y0 * meta.src_w + x0) * 3;
            const uint8_t* p01 = src_rgb_hwc + (y0 * meta.src_w + x1) * 3;
            const uint8_t* p10 = src_rgb_hwc + (y1 * meta.src_w + x0) * 3;
            const uint8_t* p11 = src_rgb_hwc + (y1 * meta.src_w + x1) * 3;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            ch_r[out_idx] = (w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]) * kInv255;
            ch_g[out_idx] = (w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]) * kInv255;
            ch_b[out_idx] = (w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]) * kInv255;
        }
    }
}

/**
 * @brief Decode single-stage anchor-free detection grid [1, 5, GridH, GridW]
 *        Channels: [0]=logit_conf, [1]=dx_cell, [2]=dy_cell, [3]=w_norm, [4]=h_norm
 *        Maps model-space coordinates back through inverse letterbox transform to camera pixels.
 */
[[nodiscard]] inline core::Detection2D decode_anchor_free_grid(
    const float* output_tensor,
    int grid_h,
    int grid_w,
    const LetterboxMeta& meta,
    float conf_threshold = 0.40f) noexcept {

    const int cells = grid_h * grid_w;
    const float* ch_logit = output_tensor;
    const float* ch_dx    = output_tensor + cells;
    const float* ch_dy    = output_tensor + 2 * cells;
    const float* ch_w     = output_tensor + 3 * cells;
    const float* ch_h     = output_tensor + 4 * cells;

    int best_idx = -1;
    float best_logit = -1e9f;
    for (int i = 0; i < cells; ++i) {
        if (ch_logit[i] > best_logit) {
            best_logit = ch_logit[i];
            best_idx = i;
        }
    }

    core::Detection2D det{};
    if (best_idx < 0) {
        return det;
    }

    const float conf = 1.0f / (1.0f + std::exp(-best_logit));
    if (conf < conf_threshold) {
        return det;
    }

    const int gy = best_idx / grid_w;
    const int gx = best_idx % grid_w;
    const float stride_x = static_cast<float>(meta.dst_w) / static_cast<float>(grid_w);
    const float stride_y = static_cast<float>(meta.dst_h) / static_cast<float>(grid_h);

    const float sigmoid_dx = 1.0f / (1.0f + std::exp(-ch_dx[best_idx]));
    const float sigmoid_dy = 1.0f / (1.0f + std::exp(-ch_dy[best_idx]));
    const float sigmoid_w  = 1.0f / (1.0f + std::exp(-ch_w[best_idx]));
    const float sigmoid_h  = 1.0f / (1.0f + std::exp(-ch_h[best_idx]));

    const float model_cx = (static_cast<float>(gx) + sigmoid_dx) * stride_x;
    const float model_cy = (static_cast<float>(gy) + sigmoid_dy) * stride_y;
    const float model_w  = sigmoid_w * static_cast<float>(meta.dst_w);
    const float model_h  = sigmoid_h * static_cast<float>(meta.dst_h);

    // Undo letterbox padding & scaling back to original camera frame
    det.cx_px = std::clamp((model_cx - static_cast<float>(meta.pad_x)) / meta.scale,
                           0.0f, static_cast<float>(meta.src_w - 1));
    det.cy_px = std::clamp((model_cy - static_cast<float>(meta.pad_y)) / meta.scale,
                           0.0f, static_cast<float>(meta.src_h - 1));
    det.width_px  = model_w / meta.scale;
    det.height_px = model_h / meta.scale;
    det.confidence = conf;
    det.class_id = 0;
    det.valid = true;
    return det;
}

struct BackendTimingsUs {
    double preprocess_us{0.0};
    double inference_us{0.0};
};

class IPerceptionBackend {
public:
    virtual ~IPerceptionBackend() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    [[nodiscard]] virtual core::Detection2D infer(
        const uint8_t* rgb_hwc,
        int src_width,
        int src_height,
        BackendTimingsUs& out_timings) = 0;
};

}  // namespace aegis::backends
