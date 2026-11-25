#pragma once

#include "aegis/backends/i_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace aegis::backends {

/**
 * @brief Zero-allocation C++ ONNX Runtime perception backend.
 *
 * Supports:
 *  - CPU Execution Provider (deterministic baseline on macOS ARM64 & Linux x86_64)
 *  - CoreML Execution Provider (Apple Silicon Neural Engine / Metal GPU acceleration)
 *  - CUDA Execution Provider (Linux baseline comparison against TensorRT)
 */
class OrtBackend final : public IPerceptionBackend {
public:
    enum class ExecutionProvider {
        kCpu,
        kCoreML,
        kCuda
    };

    OrtBackend(const std::string& onnx_path,
               ExecutionProvider provider = ExecutionProvider::kCpu,
               int input_width = 320,
               int input_height = 320,
               int intra_op_threads = 2);

    [[nodiscard]] std::string name() const override { return backend_name_; }

    [[nodiscard]] core::Detection2D infer(
        const uint8_t* rgb_hwc,
        int src_width,
        int src_height,
        BackendTimingsUs& out_timings) override;

private:
    int input_w_;
    int input_h_;
    int grid_w_{20};
    int grid_h_{20};
    std::string backend_name_;

    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo mem_info_;

    std::string input_name_;
    std::string output_name_;
    std::array<const char*, 1> input_names_{};
    std::array<const char*, 1> output_names_{};

    // Pre-allocated input tensor buffer — zero heap allocations during infer()
    std::vector<float> input_nchw_buffer_;
    Ort::Value input_tensor_{nullptr};
};

}  // namespace aegis::backends
