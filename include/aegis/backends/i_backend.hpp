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
