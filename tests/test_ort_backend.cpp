#include "aegis/backends/ort_backend.hpp"
#include "aegis/core/frame_types.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string resolve_onnx_path() {
#if defined(AEGIS_DEFAULT_ONNX_PATH)
    if (std::filesystem::exists(AEGIS_DEFAULT_ONNX_PATH)) {
        return AEGIS_DEFAULT_ONNX_PATH;
    }
#endif
    const std::string rel = "assets/models/aegis_detector.onnx";
    if (std::filesystem::exists(rel)) {
        return rel;
    }
    return {};
}

void verify_backend(aegis::backends::OrtBackend& backend, const std::string& label) {
    using namespace aegis;
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> pix_dist(0, 255);

    std::vector<uint8_t> frame_640(core::kRgbFrameBytes);
    for (auto& b : frame_640) {
        b = static_cast<uint8_t>(pix_dist(rng));
    }

    backends::BackendTimingsUs t0{};
    const auto det0 = backend.infer(frame_640.data(), core::kCameraWidth, core::kCameraHeight, t0);
    assert(std::isfinite(t0.preprocess_us) && t0.preprocess_us >= 0.0);
    assert(std::isfinite(t0.inference_us) && t0.inference_us > 0.0);

    // Repeat inference on identical frame: pre-bound input/output tensors must produce identical output
    double pre_sum = 0.0;
    double inf_sum = 0.0;
    constexpr int kBenchRuns = 20;
    for (int i = 0; i < kBenchRuns; ++i) {
        backends::BackendTimingsUs t{};
        const auto det_i = backend.infer(frame_640.data(), core::kCameraWidth, core::kCameraHeight, t);
        assert(det_i.valid == det0.valid);
        assert(std::abs(det_i.confidence - det0.confidence) < 1e-5f);
        assert(std::abs(det_i.cx_px - det0.cx_px) < 1e-4f);
        assert(std::abs(det_i.cy_px - det0.cy_px) < 1e-4f);
        pre_sum += t.preprocess_us;
        inf_sum += t.inference_us;
    }

    // Exercise dynamic resolution change (320x240) and recovery back to 640x480
    std::vector<uint8_t> frame_320(320 * 240 * 3, 180);
    backends::BackendTimingsUs t_small{};
    (void)backend.infer(frame_320.data(), 320, 240, t_small);

    backends::BackendTimingsUs t_restored{};
    const auto det_restored = backend.infer(frame_640.data(), core::kCameraWidth, core::kCameraHeight, t_restored);
    assert(det_restored.valid == det0.valid);
    assert(std::abs(det_restored.confidence - det0.confidence) < 1e-5f);

    std::cout << "[PASS] test_ort_backend (" << label << "): pre_avg="
              << (pre_sum / kBenchRuns) << " us, inf_avg="
              << (inf_sum / kBenchRuns) << " us.\n";
}

}  // namespace

int main() {
    const std::string onnx_path = resolve_onnx_path();
    if (onnx_path.empty()) {
        std::cout << "[SKIP] test_ort_backend: aegis_detector.onnx not present on disk.\n";
        return 0;
    }

    aegis::backends::OrtBackend cpu_backend(
        onnx_path, aegis::backends::OrtBackend::ExecutionProvider::kCpu);
    verify_backend(cpu_backend, cpu_backend.name());

#if defined(__APPLE__)
    aegis::backends::OrtBackend coreml_backend(
        onnx_path, aegis::backends::OrtBackend::ExecutionProvider::kCoreML);
    verify_backend(coreml_backend, coreml_backend.name());
#endif

    return 0;
}
