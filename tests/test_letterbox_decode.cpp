#include "aegis/backends/i_backend.hpp"
#include "aegis/core/frame_types.hpp"
#include "aegis/core/telemetry.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
float logit(float p) {
    return std::log(p / (1.0f - p));
}
}  // namespace

int main() {
    using namespace aegis;

    // 1. Verify aspect-preserving letterbox geometry (640x480 -> 320x320)
    const auto meta = backends::compute_letterbox_meta(640, 480, 320, 320);
    assert(std::abs(meta.scale - 0.5f) < 1e-6f);
    assert(meta.resized_w == 320);
    assert(meta.resized_h == 240);
    assert(meta.pad_x == 0);
    assert(meta.pad_y == 40);

    // 2. Verify NCHW planar layout, [0, 1] normalization, and 114/255 letterbox border padding
    std::vector<uint8_t> rgb(640 * 480 * 3, 0);
    for (int i = 0; i < 640 * 480; ++i) {
        rgb[i * 3 + 0] = 255;  // Pure R
        rgb[i * 3 + 1] = 128;  // Mid G
        rgb[i * 3 + 2] = 0;    // Zero B
    }
    std::vector<float> nchw(3 * 320 * 320, -1.0f);
    backends::cpu_preprocess_letterbox_nchw(rgb.data(), meta, nchw.data());

    constexpr int kPlane = 320 * 320;
    constexpr float kPadExpected = 114.0f / 255.0f;
    // Top letterbox pad row (y = 10) must equal 114/255 across all 3 channels
    assert(std::abs(nchw[0 * kPlane + 10 * 320 + 160] - kPadExpected) < 1e-5f);
    assert(std::abs(nchw[1 * kPlane + 10 * 320 + 160] - kPadExpected) < 1e-5f);
    assert(std::abs(nchw[2 * kPlane + 10 * 320 + 160] - kPadExpected) < 1e-5f);

    // Active interior pixel (y = 160, x = 160) must reflect normalized RGB = (1.0, 128/255, 0.0)
    assert(std::abs(nchw[0 * kPlane + 160 * 320 + 160] - 1.0f) < 1e-5f);
    assert(std::abs(nchw[1 * kPlane + 160 * 320 + 160] - (128.0f / 255.0f)) < 1e-5f);
    assert(std::abs(nchw[2 * kPlane + 160 * 320 + 160] - 0.0f) < 1e-5f);

    // 3. Verify round-trip anchor-free grid encode -> inverse-letterbox decode (<1e-3 px error)
    constexpr float kTrueCx = 412.5f;
    constexpr float kTrueCy = 188.0f;
    constexpr float kTrueW  = 72.0f;
    constexpr float kTrueH  = 72.0f;

    const float model_cx = kTrueCx * meta.scale + static_cast<float>(meta.pad_x);
    const float model_cy = kTrueCy * meta.scale + static_cast<float>(meta.pad_y);
    const float stride = 16.0f;  // 320 / 20
    const int gx = static_cast<int>(std::floor(model_cx / stride));
    const int gy = static_cast<int>(std::floor(model_cy / stride));
    const float dx = (model_cx / stride) - static_cast<float>(gx);
    const float dy = (model_cy / stride) - static_cast<float>(gy);

    constexpr int kCells = 20 * 20;
    std::vector<float> grid_out(5 * kCells, -10.0f);  // Low background logits
    const int cell = gy * 20 + gx;
    grid_out[0 * kCells + cell] = logit(0.96f);
    grid_out[1 * kCells + cell] = logit(dx);
    grid_out[2 * kCells + cell] = logit(dy);
    grid_out[3 * kCells + cell] = logit((kTrueW * meta.scale) / 320.0f);
    grid_out[4 * kCells + cell] = logit((kTrueH * meta.scale) / 320.0f);

    const auto det = backends::decode_anchor_free_grid(grid_out.data(), 20, 20, meta);
    assert(det.valid);
    assert(std::abs(det.confidence - 0.96f) < 1e-4f);
    assert(std::abs(det.cx_px - kTrueCx) < 1e-3f);
    assert(std::abs(det.cy_px - kTrueCy) < 1e-3f);
    assert(std::abs(det.width_px - kTrueW) < 1e-2f);
    assert(std::abs(det.height_px - kTrueH) < 1e-2f);

    // 4. Verify Quat 90-deg yaw rotation (+X body -> +Y world)
    const double half_pi4 = 3.14159265358979323846 / 4.0;
    const core::Quat q_yaw90{std::cos(half_pi4), 0.0, 0.0, std::sin(half_pi4)};
    const core::Vec3 rotated = q_yaw90.rotate_body_to_world({1.0, 0.0, 0.0});
    assert(std::abs(rotated.x) < 1e-9);
    assert(std::abs(rotated.y - 1.0) < 1e-9);

    // 5. Verify CpuPreprocessPlan matches golden cpu_preprocess_letterbox_nchw (<1e-5) across aspect ratios
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> pix_dist(0, 255);
    struct Case { int sw, sh, dw, dh; };
    const Case cases[] = {
        {640, 480, 320, 320},   // Landscape 4:3 -> square letterbox
        {480, 640, 320, 320},   // Portrait 3:4 -> square pillarbox
        {1280, 720, 416, 416},  // Non-integer 16:9 downsample
        {320, 320, 320, 320},   // 1:1 identity geometry
    };

    float worst_diff = 0.0f;
    for (const auto& c : cases) {
        std::vector<uint8_t> img(static_cast<std::size_t>(c.sw) * c.sh * 3);
        for (auto& b : img) {
            b = static_cast<uint8_t>(pix_dist(rng));
        }
        const auto m = backends::compute_letterbox_meta(c.sw, c.sh, c.dw, c.dh);
        const std::size_t out_floats = static_cast<std::size_t>(3) * c.dw * c.dh;
        std::vector<float> ref_buf(out_floats, -1.0f);
        std::vector<float> plan_buf(out_floats, -2.0f);

        backends::cpu_preprocess_letterbox_nchw(img.data(), m, ref_buf.data());
        backends::CpuPreprocessPlan plan(c.sw, c.sh, c.dw, c.dh);
        plan.execute(img.data(), plan_buf.data());

        for (std::size_t i = 0; i < out_floats; ++i) {
            const float d = std::abs(ref_buf[i] - plan_buf[i]);
            worst_diff = std::max(worst_diff, d);
            assert(d < 1e-5f);
        }
    }

    // 6. Microbenchmark CpuPreprocessPlan vs scalar reference on 640x480 -> 320x320
    {
        std::vector<uint8_t> bench_rgb(640 * 480 * 3, 127);
        std::vector<float> bench_out(3 * 320 * 320, 0.0f);
        const auto bench_meta = backends::compute_letterbox_meta(640, 480, 320, 320);
        backends::CpuPreprocessPlan bench_plan(640, 480, 320, 320);

        constexpr int kIters = 100;
        const uint64_t t_ref0 = core::now_ns();
        for (int i = 0; i < kIters; ++i) {
            backends::cpu_preprocess_letterbox_nchw(bench_rgb.data(), bench_meta, bench_out.data());
        }
        const uint64_t t_ref1 = core::now_ns();

        const uint64_t t_lut0 = core::now_ns();
        for (int i = 0; i < kIters; ++i) {
            bench_plan.execute(bench_rgb.data(), bench_out.data());
        }
        const uint64_t t_lut1 = core::now_ns();

        const double ref_us = static_cast<double>(t_ref1 - t_ref0) / (1000.0 * kIters);
        const double lut_us = static_cast<double>(t_lut1 - t_lut0) / (1000.0 * kIters);
        std::cout << "[PASS] test_letterbox_decode: NCHW pad/norm verified, bbox err="
                  << std::hypot(det.cx_px - kTrueCx, det.cy_px - kTrueCy)
                  << " px, LUT max_diff=" << worst_diff
                  << ", ref=" << ref_us << " us vs LUT=" << lut_us << " us.\n";
    }
    return 0;
}
