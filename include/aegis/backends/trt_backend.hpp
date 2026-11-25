#pragma once

#if defined(AEGIS_ENABLE_TENSORRT)

#include "aegis/backends/i_backend.hpp"
#include "aegis/kernels/preprocess.cuh"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace aegis::backends {

/**
 * @brief TensorRT 10.x C++ Inference Engine with fused CUDA preprocessing & INT8 PTQ Calibration.
 */
class TrtBackend final : public IPerceptionBackend {
public:
    enum class Precision {
        kFp32,
        kFp16,
        kInt8
    };

    TrtBackend(const std::string& onnx_path,
               const std::string& engine_cache_path,
               Precision precision,
               const std::string& calib_dir = "",
               bool zero_copy_unified = false,
               int input_width = 320,
               int input_height = 320);

    ~TrtBackend() override;

    [[nodiscard]] std::string name() const override { return backend_name_; }

    [[nodiscard]] core::Detection2D infer(
        const uint8_t* rgb_hwc,
        int src_width,
        int src_height,
        BackendTimingsUs& out_timings) override;

private:
    void build_or_load_engine(const std::string& onnx_path,
                              const std::string& engine_cache_path,
                              Precision precision,
                              const std::string& calib_dir);

    int input_w_{320};
    int input_h_{320};
    int grid_w_{20};
    int grid_h_{20};
    std::string backend_name_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<kernels::CudaPreprocessor> cuda_preproc_;

    cudaStream_t stream_{nullptr};
    cudaEvent_t ev_start_{nullptr};
    cudaEvent_t ev_pre_done_{nullptr};
    cudaEvent_t ev_inf_done_{nullptr};

    float* d_input_{nullptr};
    float* d_output_{nullptr};
    float* h_output_pinned_{nullptr};
    std::size_t output_floats_{0};
};

}  // namespace aegis::backends

#endif  // AEGIS_ENABLE_TENSORRT
